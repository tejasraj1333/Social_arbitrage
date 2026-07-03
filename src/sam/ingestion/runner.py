"""Ingestion orchestration: run collectors with full run bookkeeping.

For each source the runner:
  1. resolves the `sources` row (get_or_create)
  2. opens an `ingestion_runs` row (status=running, committed immediately so a
     hung collector is visible to observers)
  3. fetch -> bronze lake (raw preserved even if persistence later fails)
  4. normalize -> persist (idempotent upserts via the repositories)
  5. finishes the run row with metrics (success) or error detail

One broken source never sinks the others: failures are captured as an
IngestResult with status="error" (same contract as recon), and the CLI maps
that to a non-zero exit code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from sam.core.db import default_session
from sam.core.logging import get_logger
from sam.ingestion import lake
from sam.ingestion.base import Collector
from sam.ingestion.hackernews import HackerNewsIngestionCollector
from sam.ingestion.rss import RSSIngestionCollector
from sam.ingestion.yahoo import YahooIngestionCollector
from sam.storage.repositories import IngestionRunRepository, SourceRepository

log = get_logger("ingestion.runner")

# (session, source_id, backfill) -> collector. backfill only matters for
# window-based sources (Yahoo); streaming sources ignore it.
CollectorFactory = Callable[[Session, int, bool], Collector]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Registry entry wiring a source name to its collector."""

    name: str
    type: str
    config_ref: str | None
    factory: CollectorFactory


SOURCES: dict[str, SourceSpec] = {
    "rss": SourceSpec(
        name="rss",
        type="rss",
        config_ref="config/sources.yaml:rss",
        factory=lambda session, sid, backfill: RSSIngestionCollector(session, sid),
    ),
    "yahoo": SourceSpec(
        name="yahoo",
        type="yahoo",
        config_ref="config/sources.yaml:yahoo",
        factory=lambda session, sid, backfill: YahooIngestionCollector(
            session, sid, backfill=backfill
        ),
    ),
    "hackernews": SourceSpec(
        name="hackernews",
        type="hackernews",
        config_ref="config/sources.yaml:hackernews",
        factory=lambda session, sid, backfill: HackerNewsIngestionCollector(session, sid),
    ),
}


@dataclass(slots=True)
class IngestResult:
    """Outcome of one collector run (mirror of its ingestion_runs row)."""

    source_name: str
    status: str  # "success" | "error"
    rows_fetched: int = 0
    rows_inserted: int = 0
    raw_path: str | None = None
    run_id: int | None = None
    detail: str = ""


__all__ = ["SOURCES", "IngestResult", "IngestionRunner", "SourceSpec", "default_session"]


class IngestionRunner:
    def __init__(self, session_factory: Callable[[], Session] = default_session) -> None:
        self._session_factory = session_factory

    def run(self, name: str, *, backfill: bool = False) -> IngestResult:
        spec = SOURCES[name]
        session = self._session_factory()
        try:
            return self._run_in_session(session, spec, backfill)
        finally:
            session.close()

    def run_many(self, names: list[str], *, backfill: bool = False) -> list[IngestResult]:
        return [self.run(name, backfill=backfill) for name in names]

    def _run_in_session(self, session: Session, spec: SourceSpec, backfill: bool) -> IngestResult:
        source = SourceRepository(session).get_or_create(spec.type, spec.name, spec.config_ref)
        runs = IngestionRunRepository(session)
        run_row = runs.start(source.id)
        session.commit()  # 'running' row visible before any network work
        log.info("ingest_start", source=spec.name, run_id=run_row.id, backfill=backfill)

        rows_fetched = 0
        raw_path: str | None = None
        try:
            collector = spec.factory(session, source.id, backfill)
            raw = list(collector.fetch())
            rows_fetched = len(raw)
            artifact = lake.write_raw(spec.name, raw)
            raw_path = artifact.path if artifact else None
            documents = [collector.normalize(record) for record in raw]
            inserted = collector.persist(documents)
            runs.finish(
                run_row,
                status="success",
                rows_fetched=rows_fetched,
                rows_inserted=inserted,
                raw_path=raw_path,
            )
            session.commit()
            log.info(
                "ingest_done",
                source=spec.name,
                run_id=run_row.id,
                fetched=rows_fetched,
                inserted=inserted,
                raw=raw_path,
            )
            return IngestResult(
                source_name=spec.name,
                status="success",
                rows_fetched=rows_fetched,
                rows_inserted=inserted,
                raw_path=raw_path,
                run_id=run_row.id,
            )
        except Exception as exc:
            session.rollback()  # discard partial writes; the lake file remains
            detail = f"{type(exc).__name__}: {exc}"
            log.error(
                "ingest_error", source=spec.name, run_id=run_row.id, error=detail, exc_info=True
            )
            runs.finish(
                run_row,
                status="error",
                rows_fetched=rows_fetched,
                raw_path=raw_path,
                error=detail,
            )
            session.commit()
            return IngestResult(
                source_name=spec.name,
                status="error",
                rows_fetched=rows_fetched,
                raw_path=raw_path,
                run_id=run_row.id,
                detail=detail,
            )
