"""Batch SAI pipeline (Phase 5 / architecture M4).

Computes the dense sai_daily panel — every active entity × every *closed*
UTC day — from resolved links, sentiment rows and topic assignments already
in the database. Incremental by default via the panel's own watermark
(``max(sai_daily.date)``); ``rebuild=True`` (``sam sai --rebuild``) recomputes
every closed day, which must reproduce identical values — the P5
"deterministic rebuild from raw" gate.

Only closed days are computed (through yesterday, UTC): a partial day would
change on re-run, and closed days never do — documents can only be ingested
"now", never backdated. Two operational rules follow (docs/sai_methodology.md):
run ``sam sai`` *after* resolve/enrich in the daily chain, and rebuild after
changing ``nlp.sentiment_model`` or any ``signals.*`` setting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from sam.core.config import SignalSettings, get_settings
from sam.core.db import default_session
from sam.core.logging import get_logger
from sam.signals.compute import (
    ENGAGEMENT,
    MENTIONS,
    SENTIMENT,
    TOPICS,
    LinkRow,
    TopicRow,
    build_panel,
    first_activity_day,
)
from sam.storage.repositories import (
    EntityRepository,
    SaiRepository,
    SignalInputRepository,
    TopicRepository,
)

log = get_logger("signals.pipeline")


@dataclass(slots=True)
class SaiResult:
    """Outcome of one SAI run."""

    days_computed: int = 0
    rows_written: int = 0
    first_day: date | None = None
    last_day: date | None = None
    skipped: str | None = None  # honest no-op reason (cron-safe), else None


class SaiPipeline:
    """Reads signal inputs, computes the panel, upserts sai_daily.

    Settings and the sentiment-model id are injectable for tests; production
    wiring reads both from configuration. All math is delegated to
    :mod:`sam.signals.compute` (pure); this class only orchestrates.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        settings: SignalSettings | None = None,
        sentiment_model: str | None = None,
    ) -> None:
        # Resolved lazily so tests can monkeypatch module-level default_session.
        self._session_factory = session_factory or default_session
        self._settings = settings
        self._sentiment_model = sentiment_model

    def run(self, *, rebuild: bool = False, as_of: date | None = None) -> SaiResult:
        """Compute closed days up to (excluding) ``as_of`` (default: today UTC)."""
        session = self._session_factory()
        try:
            return self._run_in_session(session, rebuild=rebuild, as_of=as_of)
        finally:
            session.close()

    def _run_in_session(self, session: Session, *, rebuild: bool, as_of: date | None) -> SaiResult:
        settings = self._settings if self._settings is not None else get_settings().signals
        model = (
            self._sentiment_model
            if self._sentiment_model is not None
            else get_settings().nlp.sentiment_model
        )

        inputs = SignalInputRepository(session)
        links = [LinkRow(*row) for row in inputs.linked_document_rows(model)]
        panel_start = first_activity_day(links, max_doc_age_days=settings.max_doc_age_days)
        if panel_start is None:
            log.warning("sai_skipped", reason="no linked documents")
            return SaiResult(skipped="no linked documents (run `sam ingest` + `sam resolve` first)")

        end_day = (as_of or datetime.now(tz=UTC).date()) - timedelta(days=1)
        if end_day < panel_start:
            log.warning("sai_skipped", reason="no closed days yet")
            return SaiResult(skipped="no closed days yet (first documents arrived today)")

        watermark = None if rebuild else SaiRepository(session).latest_date()
        start_day = (
            panel_start if watermark is None else max(panel_start, watermark + timedelta(days=1))
        )
        if start_day > end_day:
            log.info("sai_up_to_date", watermark=str(watermark))
            return SaiResult(first_day=watermark, last_day=watermark)

        days = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
        entity_ids = [entity.id for entity in EntityRepository(session).active()]
        topic_rows = [TopicRow(*row) for row in inputs.topic_assignment_rows()]
        versions = TopicRepository(session).versions()

        panel = build_panel(
            links,
            topic_rows=topic_rows,
            versions=versions,
            entity_ids=entity_ids,
            days=days,
            panel_start=panel_start,
            window_days=settings.window_days,
            min_history_days=settings.min_history_days,
            max_doc_age_days=settings.max_doc_age_days,
            weights={
                MENTIONS: settings.weight_mentions,
                SENTIMENT: settings.weight_sentiment,
                TOPICS: settings.weight_topics,
                ENGAGEMENT: settings.weight_engagement,
            },
        )

        computed_at = datetime.now(tz=UTC)
        written = SaiRepository(session).upsert_many(
            [
                {
                    "entity_id": row.entity_id,
                    "date": row.day,
                    "mention_growth": row.mention_growth,
                    "sentiment_momentum": row.sentiment_momentum,
                    "topic_velocity": row.topic_velocity,
                    "engagement_growth": row.engagement_growth,
                    "sai_score": row.sai_score,
                    "sai_rank": row.sai_rank,
                    "computed_at": computed_at,
                }
                for row in panel
            ]
        )
        session.commit()

        result = SaiResult(
            days_computed=len(days),
            rows_written=written,
            first_day=days[0],
            last_day=days[-1],
        )
        log.info(
            "sai_done",
            days=result.days_computed,
            rows=result.rows_written,
            first_day=str(result.first_day),
            last_day=str(result.last_day),
            rebuild=rebuild,
            sentiment_model=model,
            entities=len(entity_ids),
        )
        return result
