"""SAI pipeline tests (Phase 5) — plumbing over in-memory SQLite.

The math is proven in test_signals_compute.py; here we prove orchestration:
dense rows land in sai_daily, the watermark makes runs incremental, rebuilds
reproduce identical values (the P5 gate), and the sentiment model id is
pinned. ``ingested_at`` is set explicitly (never server-now) so every test
is clock-independent via ``as_of``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.core.config import SignalSettings
from sam.signals.pipeline import SaiPipeline
from sam.storage.models import Document
from sam.storage.repositories import (
    DocumentEntityRepository,
    DocumentRepository,
    EntityRepository,
    SaiRepository,
    SentimentRepository,
    SourceRepository,
    TopicRepository,
)

UNIVERSE = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
]
SETTINGS = SignalSettings(window_days=7, min_history_days=3, max_doc_age_days=7)
MODEL = "test-sentiment-model"


def _pipeline(db_session: Session) -> SaiPipeline:
    return SaiPipeline(session_factory=lambda: db_session, settings=SETTINGS, sentiment_model=MODEL)


def _seed_linked_doc(
    session: Session,
    *,
    doc_no: int,
    ticker: str,
    day: date,
    sentiment: tuple[str, float] | None = ("positive", 0.5),  # signed +0.5
    confidence: float = 1.0,
    engagement: dict[str, object] | None = None,
    model: str = MODEL,
) -> int:
    """One document ingested at noon UTC on ``day``, linked (+ scored)."""
    ingested = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
    DocumentRepository(session).upsert_many(
        [
            {
                "source_id": 1,
                "external_id": f"doc-{doc_no}",
                "url": f"https://example.com/{doc_no}",
                "author": None,
                "title": f"Headline {doc_no}",
                "raw_text": None,
                "lang": None,
                "content_hash": f"{doc_no:064x}",
                "published_at": ingested,
                "ingested_at": ingested,
                "engagement": engagement or {},
            }
        ]
    )
    doc_id = session.execute(
        select(Document.id).where(Document.external_id == f"doc-{doc_no}")
    ).scalar_one()
    entity_id = EntityRepository(session).by_ticker()[ticker]
    DocumentEntityRepository(session).upsert_many(
        [
            {
                "document_id": doc_id,
                "entity_id": entity_id,
                "confidence": confidence,
                "method": "ticker",
                "resolved_at": ingested,
            }
        ]
    )
    if sentiment is not None:
        label, score = sentiment
        SentimentRepository(session).upsert_many(
            [{"document_id": doc_id, "model": model, "label": label, "score": score}]
        )
    return doc_id


def _seed_panel(db_session: Session) -> None:
    """Two entities; AAPL active 7/01-7/04 with a positive turn on 7/04."""
    SourceRepository(db_session).get_or_create("rss", "rss")
    EntityRepository(db_session).seed(UNIVERSE)
    for i, day in enumerate([date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)], start=1):
        _seed_linked_doc(db_session, doc_no=i, ticker="AAPL", day=day)
    _seed_linked_doc(
        db_session,
        doc_no=4,
        ticker="AAPL",
        day=date(2026, 7, 4),
        sentiment=("positive", 0.9),
        engagement={"score": 40, "comments": 10},
    )
    _seed_linked_doc(
        db_session, doc_no=5, ticker="NVDA", day=date(2026, 7, 4), sentiment=None, confidence=0.5
    )
    db_session.commit()


def test_sai_pipeline_writes_dense_panel(db_session: Session) -> None:
    _seed_panel(db_session)
    result = _pipeline(db_session).run(as_of=date(2026, 7, 5))

    assert result.skipped is None
    assert (result.first_day, result.last_day) == (date(2026, 7, 1), date(2026, 7, 4))
    assert result.days_computed == 4
    assert result.rows_written == 8  # dense: 2 entities x 4 days

    rows = SaiRepository(db_session).rows_ordered()
    by_key = {(r.entity_id, r.date): r for r in rows}
    ids = EntityRepository(db_session).by_ticker()

    # Inside the min-history gate: all components NULL, no score, no rank.
    early = by_key[(ids["AAPL"], date(2026, 7, 1))]
    assert (early.mention_growth, early.sai_score, early.sai_rank) == (None, None, None)

    # 7/04: AAPL mentions flat vs (1,1,1) baseline; signed sentiment jumps
    # +0.9 against a +0.5 trailing mean -> momentum 0.4.
    aapl = by_key[(ids["AAPL"], date(2026, 7, 4))]
    assert aapl.mention_growth == 0.0
    assert aapl.sentiment_momentum is not None and abs(aapl.sentiment_momentum - 0.4) < 1e-9
    assert aapl.topic_velocity is None  # no topic model fitted
    assert aapl.engagement_growth == 50.0  # conf 1.0 * (40+10) vs zero baseline

    # NVDA first appears 7/04: 0.5 mentions vs zero baseline; unscored -> no momentum.
    nvda = by_key[(ids["NVDA"], date(2026, 7, 4))]
    assert nvda.mention_growth == 0.5
    assert nvda.sentiment_momentum is None
    assert {aapl.sai_rank, nvda.sai_rank} == {1, 2}  # both scored, cross-ranked


def test_sai_pipeline_is_incremental_and_idempotent(db_session: Session) -> None:
    _seed_panel(db_session)
    pipeline = _pipeline(db_session)
    assert pipeline.run(as_of=date(2026, 7, 5)).days_computed == 4

    # Same as_of again: watermark says up to date -> nothing recomputed.
    again = pipeline.run(as_of=date(2026, 7, 5))
    assert (again.days_computed, again.rows_written, again.skipped) == (0, 0, None)

    # A day later with no new documents: the zero-activity day still lands
    # (an attention collapse is signal, not missing data).
    next_day = pipeline.run(as_of=date(2026, 7, 6))
    assert (next_day.first_day, next_day.days_computed) == (date(2026, 7, 5), 1)
    ids = EntityRepository(db_session).by_ticker()
    rows = {(r.entity_id, r.date): r for r in SaiRepository(db_session).rows_ordered()}
    collapse = rows[(ids["AAPL"], date(2026, 7, 5))]
    assert collapse.mention_growth == -1.0  # 0 today vs positive baseline


def test_sai_pipeline_rebuild_reproduces_identical_values(db_session: Session) -> None:
    _seed_panel(db_session)
    # Topics too, so the rebuild covers all four components.
    topics = TopicRepository(db_session)
    (topic,) = topics.create_topics("v1", [{"label": "ai"}])
    topic.created_at = datetime(2026, 7, 1, 6, tzinfo=UTC)  # fitted before the panel
    topics.assign_documents(
        [{"document_id": doc_id, "topic_id": topic.id, "probability": 0.9} for doc_id in (1, 4)]
    )
    db_session.commit()

    pipeline = _pipeline(db_session)
    pipeline.run(as_of=date(2026, 7, 5))

    def snapshot() -> list[tuple[object, ...]]:
        return [
            (
                r.entity_id,
                r.date,
                r.mention_growth,
                r.sentiment_momentum,
                r.topic_velocity,
                r.engagement_growth,
                r.sai_score,
                r.sai_rank,
            )
            for r in SaiRepository(db_session).rows_ordered()
        ]

    first = snapshot()
    assert any(row[4] is not None for row in first)  # topic velocity actually computed
    rebuild = pipeline.run(rebuild=True, as_of=date(2026, 7, 5))
    assert rebuild.days_computed == 4
    assert snapshot() == first  # the P5 gate: identical values, byte for byte


def test_sai_pipeline_skips_honestly_without_inputs(db_session: Session) -> None:
    result = _pipeline(db_session).run(as_of=date(2026, 7, 5))
    assert result.skipped is not None and "no linked documents" in result.skipped
    assert SaiRepository(db_session).rows_ordered() == []

    # Linked docs exist but only from today: no closed day to compute yet.
    SourceRepository(db_session).get_or_create("rss", "rss")
    EntityRepository(db_session).seed(UNIVERSE)
    _seed_linked_doc(db_session, doc_no=1, ticker="AAPL", day=date(2026, 7, 5))
    db_session.commit()
    result = _pipeline(db_session).run(as_of=date(2026, 7, 5))
    assert result.skipped is not None and "no closed days" in result.skipped


def test_sai_pipeline_pins_the_sentiment_model(db_session: Session) -> None:
    _seed_panel(db_session)
    # Doc 6 on 7/04 scored *only* by another model: its tone must not leak in.
    _seed_linked_doc(
        db_session,
        doc_no=6,
        ticker="NVDA",
        day=date(2026, 7, 4),
        sentiment=("negative", 0.99),
        model="other-model",
    )
    db_session.commit()

    _pipeline(db_session).run(as_of=date(2026, 7, 5))
    ids = EntityRepository(db_session).by_ticker()
    rows = {(r.entity_id, r.date): r for r in SaiRepository(db_session).rows_ordered()}
    nvda = rows[(ids["NVDA"], date(2026, 7, 4))]
    assert nvda.sentiment_momentum is None  # other-model's score ignored
    assert nvda.mention_growth == 1.5  # but the mention itself counts (0.5 + 1.0)
