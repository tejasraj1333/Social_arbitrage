"""Batch entity-resolution pipeline (Phase 3 / architecture M2).

Scans documents, matches them against the active entity universe (see
sam.processing.resolver), and persists document→entity links. Incremental by
default via the ``documents.resolved_at`` watermark — every scanned document
is stamped, matched or not, so a run only ever touches new documents.
``re_resolve=True`` (``sam resolve --all``) re-scans everything after a
dictionary change; links refresh via DO UPDATE.

Commits per batch, so an interrupted run loses at most one batch of progress
and never leaves a document stamped without its links (same transaction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from sam.core.db import default_session
from sam.core.logging import get_logger
from sam.processing.resolver import EntityMatcher, refs_from_rows
from sam.storage.repositories import (
    DocumentEntityRepository,
    DocumentRepository,
    EntityRepository,
)

log = get_logger("processing.pipeline")

_BATCH_SIZE = 500


@dataclass(slots=True)
class ResolveResult:
    """Outcome of one resolution run."""

    docs_scanned: int = 0
    docs_matched: int = 0
    links_written: int = 0


class ResolutionPipeline:
    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        # Resolved lazily so tests can monkeypatch module-level default_session.
        self._session_factory = session_factory or default_session

    def run(self, *, re_resolve: bool = False, batch_size: int = _BATCH_SIZE) -> ResolveResult:
        session = self._session_factory()
        try:
            return self._run_in_session(session, re_resolve=re_resolve, batch_size=batch_size)
        finally:
            session.close()

    def _run_in_session(
        self, session: Session, *, re_resolve: bool, batch_size: int
    ) -> ResolveResult:
        entities = EntityRepository(session).active()
        if not entities:
            log.warning("resolve_empty_universe", hint="run `sam seed` first")
        matcher = EntityMatcher(
            refs_from_rows([(e.id, e.ticker, e.name, list(e.aliases or [])) for e in entities])
        )
        docs = DocumentRepository(session)
        links_repo = DocumentEntityRepository(session)
        result = ResolveResult()
        last_id = 0

        while True:
            batch = docs.resolution_batch(
                after_id=last_id, limit=batch_size, include_resolved=re_resolve
            )
            if not batch:
                break
            now = datetime.now(tz=UTC)
            links: list[dict[str, object]] = []
            for doc in batch:
                matches = matcher.match(doc.title, doc.raw_text)
                if matches:
                    result.docs_matched += 1
                links.extend(
                    {
                        "document_id": doc.id,
                        "entity_id": m.entity_id,
                        "confidence": m.confidence,
                        "method": m.method,
                        "resolved_at": now,
                    }
                    for m in matches
                )
            result.links_written += links_repo.upsert_many(links)
            docs.mark_resolved([doc.id for doc in batch], at=now)
            session.commit()  # per-batch: progress survives interruption
            result.docs_scanned += len(batch)
            last_id = batch[-1].id

        log.info(
            "resolve_done",
            scanned=result.docs_scanned,
            matched=result.docs_matched,
            links=result.links_written,
            re_resolve=re_resolve,
            universe=len(entities),
        )
        return result
