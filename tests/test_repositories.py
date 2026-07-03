"""Repository-layer tests on in-memory SQLite (same ON CONFLICT paths as prod)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.storage.models import (
    Document,
    DocumentTopic,
    Entity,
    MarketData,
    SentimentScore,
    Source,
)
from sam.storage.repositories import (
    DocumentRepository,
    EmbeddingRepository,
    EntityRepository,
    IngestionRunRepository,
    MarketDataRepository,
    SentimentRepository,
    SourceRepository,
    TopicRepository,
)

UNIVERSE = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
]


def _doc(hash_: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source_id": 1,
        "external_id": f"ext-{hash_[:8]}",
        "url": f"https://example.com/{hash_[:8]}",
        "author": None,
        "title": "Title",
        "raw_text": "Body",
        "lang": None,
        "content_hash": hash_,
        "published_at": datetime(2026, 7, 1, tzinfo=UTC),
        "engagement": {"score": 1},
    }
    base.update(overrides)
    return base


def test_source_get_or_create_idempotent(db_session: Session) -> None:
    repo = SourceRepository(db_session)
    first = repo.get_or_create("rss", "rss", "sources.yaml:rss")
    second = repo.get_or_create("rss", "rss")
    assert first.id == second.id
    assert len(db_session.execute(select(Source)).scalars().all()) == 1


def test_entity_seed_idempotent_and_preserves_existing(db_session: Session) -> None:
    repo = EntityRepository(db_session)
    assert repo.seed(UNIVERSE) == 2
    assert repo.seed(UNIVERSE) == 0  # re-seed is a no-op

    # Curated edits survive a re-seed (DO NOTHING, not DO UPDATE).
    apple = db_session.execute(select(Entity).where(Entity.ticker == "AAPL")).scalar_one()
    apple.aliases = ["Apple", "$AAPL"]
    db_session.flush()
    repo.seed(UNIVERSE)
    refreshed = db_session.execute(select(Entity).where(Entity.ticker == "AAPL")).scalar_one()
    assert refreshed.aliases == ["Apple", "$AAPL"]

    assert set(repo.by_ticker()) == {"AAPL", "MSFT"}


def test_entity_seed_update_refreshes_curated_fields(db_session: Session) -> None:
    repo = EntityRepository(db_session)
    repo.seed(UNIVERSE)
    db_session.commit()

    # Config gains aliases; --update pushes them onto existing rows and still
    # inserts brand-new tickers in the same statement.
    curated = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "aliases": ["Apple"]},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "aliases": ["Microsoft"]},
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "aliases": ["Nvidia"]},
    ]
    written = repo.seed(curated, update=True)
    db_session.commit()
    assert written == 3  # 2 updated + 1 inserted

    apple = db_session.execute(select(Entity).where(Entity.ticker == "AAPL")).scalar_one()
    assert apple.aliases == ["Apple"]
    assert set(repo.by_ticker()) == {"AAPL", "MSFT", "NVDA"}


def test_entity_active_returns_ticker_ordered_active_rows(db_session: Session) -> None:
    repo = EntityRepository(db_session)
    repo.seed(UNIVERSE)
    msft = db_session.execute(select(Entity).where(Entity.ticker == "MSFT")).scalar_one()
    msft.active = False
    db_session.flush()

    active = repo.active()
    assert [e.ticker for e in active] == ["AAPL"]


def test_document_upsert_is_idempotent(db_session: Session) -> None:
    SourceRepository(db_session).get_or_create("rss", "rss")
    repo = DocumentRepository(db_session)

    batch = [_doc("a" * 64), _doc("b" * 64)]
    assert repo.upsert_many(batch) == 2
    assert repo.upsert_many(batch) == 0  # full re-ingest -> no-op
    assert repo.upsert_many([_doc("b" * 64), _doc("c" * 64)]) == 1  # partial overlap
    assert len(db_session.execute(select(Document)).scalars().all()) == 3


def test_document_upsert_collapses_intra_batch_duplicates(db_session: Session) -> None:
    SourceRepository(db_session).get_or_create("rss", "rss")
    repo = DocumentRepository(db_session)
    batch = [_doc("d" * 64, title="first"), _doc("d" * 64, title="second")]
    assert repo.upsert_many(batch) == 1
    row = db_session.execute(select(Document)).scalar_one()
    assert row.title == "first"  # first occurrence wins (DO NOTHING semantics)


def test_document_upsert_empty_batch(db_session: Session) -> None:
    assert DocumentRepository(db_session).upsert_many([]) == 0


def test_market_data_upsert_updates_restated_bars(db_session: Session) -> None:
    EntityRepository(db_session).seed(UNIVERSE)
    ids = EntityRepository(db_session).by_ticker()
    repo = MarketDataRepository(db_session)

    day = date(2026, 7, 1)
    bar = {
        "entity_id": ids["AAPL"],
        "date": day,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 100,
    }
    assert repo.upsert_many([bar]) == 1

    restated = dict(bar, adj_close=9.8)  # e.g. dividend adjustment
    repo.upsert_many([restated])
    stored = db_session.execute(select(MarketData)).scalar_one()
    assert stored.adj_close == 9.8  # freshest vendor value wins


def test_market_data_intra_batch_last_wins(db_session: Session) -> None:
    EntityRepository(db_session).seed(UNIVERSE)
    ids = EntityRepository(db_session).by_ticker()
    repo = MarketDataRepository(db_session)
    day = date(2026, 7, 1)
    first = {"entity_id": ids["AAPL"], "date": day, "close": 1.0}
    second = {"entity_id": ids["AAPL"], "date": day, "close": 2.0}
    assert repo.upsert_many([first, second]) == 1
    assert db_session.execute(select(MarketData)).scalar_one().close == 2.0


def test_ingestion_run_start_finish_and_recent(db_session: Session) -> None:
    source = SourceRepository(db_session).get_or_create("rss", "rss")
    repo = IngestionRunRepository(db_session)

    run = repo.start(source.id)
    assert run.id is not None and run.status == "running"

    repo.finish(
        run,
        status="success",
        rows_fetched=100,
        rows_inserted=40,
        raw_path="data/00_raw/rss/dt=2026-07-02/rss.jsonl.gz",
    )
    assert run.finished_at is not None

    failed = repo.start(source.id)
    repo.finish(failed, status="error", error="ConnectError: boom")

    recent = repo.recent(limit=10)
    assert len(recent) == 2
    assert {r.status for r in recent} == {"success", "error"}


def _seed_docs(session: Session, n: int = 3) -> list[int]:
    """Insert n documents (via the repo, like production) and return their ids."""
    SourceRepository(session).get_or_create("rss", "rss")
    docs = [_doc(f"{i:064x}", title=f"Headline number {i}") for i in range(n)]
    DocumentRepository(session).upsert_many(docs)
    return list(session.execute(select(Document.id).order_by(Document.id)).scalars())


def test_sentiment_upsert_refreshes_and_models_coexist(db_session: Session) -> None:
    (doc_id,) = _seed_docs(db_session, n=1)
    repo = SentimentRepository(db_session)

    row = {"document_id": doc_id, "model": "finbert-v1", "label": "neutral", "score": 0.6}
    assert repo.upsert_many([row]) == 1
    # Re-scoring the same (doc, model) refreshes in place (DO UPDATE).
    assert repo.upsert_many([dict(row, label="positive", score=0.9)]) == 1
    stored = db_session.execute(select(SentimentScore)).scalar_one()
    assert (stored.label, stored.score) == ("positive", 0.9)

    # A different model id is a new row, not a conflict.
    assert repo.upsert_many([dict(row, model="finbert-v2")]) == 1
    assert len(db_session.execute(select(SentimentScore)).scalars().all()) == 2

    # Intra-batch duplicates collapse last-wins; empty batch is a no-op.
    assert repo.upsert_many([row, dict(row, label="negative")]) == 1
    assert repo.upsert_many([]) == 0


def test_embedding_upsert_and_document_join(db_session: Session) -> None:
    ids = _seed_docs(db_session, n=2)
    repo = EmbeddingRepository(db_session)

    rows = [
        {"document_id": ids[0], "model": "minilm", "vector": [0.1, 0.2]},
        {"document_id": ids[1], "model": "minilm", "vector": [0.3, 0.4]},
        {"document_id": ids[0], "model": "other-model", "vector": [9.9]},
    ]
    assert repo.upsert_many(rows) == 3
    # Refresh wins (DO UPDATE on vector).
    assert repo.upsert_many([dict(rows[0], vector=[0.5, 0.6])]) == 1

    joined = repo.rows_with_documents("minilm")
    assert [(doc_id, list(vec)) for doc_id, _t, _r, vec in joined] == [
        (ids[0], [0.5, 0.6]),
        (ids[1], [0.3, 0.4]),
    ]
    assert joined[0][1] == "Headline number 0"  # title comes along for topic fitting


def test_enrichment_batch_watermark_and_pagination(db_session: Session) -> None:
    ids = _seed_docs(db_session, n=3)
    repo = DocumentRepository(db_session)

    assert [d.id for d in repo.enrichment_batch()] == ids
    repo.mark_enriched(ids[:2], at=datetime(2026, 7, 3, tzinfo=UTC))
    assert [d.id for d in repo.enrichment_batch()] == ids[2:]  # watermark skips done
    assert [d.id for d in repo.enrichment_batch(include_enriched=True)] == ids  # --all
    assert [d.id for d in repo.enrichment_batch(after_id=ids[2])] == []  # keyset
    repo.mark_enriched([], at=datetime(2026, 7, 3, tzinfo=UTC))  # no-op, no error


def test_enrichment_stats_counts(db_session: Session) -> None:
    ids = _seed_docs(db_session, n=3)
    repo = DocumentRepository(db_session)
    repo.mark_enriched(ids[:2], at=datetime(2026, 7, 3, tzinfo=UTC))
    SentimentRepository(db_session).upsert_many(
        [{"document_id": ids[0], "model": "m", "label": "neutral", "score": 0.5}]
    )
    assert repo.enrichment_stats() == (3, 1, 1)  # total, unenriched, with_sentiment


def test_topic_create_assign_and_versioning(db_session: Session) -> None:
    ids = _seed_docs(db_session, n=2)
    repo = TopicRepository(db_session)

    v1 = repo.create_topics(
        "v1", [{"label": "ai_chips", "keywords": ["nvidia", "gpu"]}, {"label": "macro"}]
    )
    assert [t.id is not None for t in v1] == [True, True]
    assert repo.latest_version() == "v1"

    assignments = [
        {"document_id": ids[0], "topic_id": v1[0].id, "probability": 0.8},
        {"document_id": ids[1], "topic_id": v1[1].id, "probability": 0.7},
    ]
    assert repo.assign_documents(assignments) == 2
    # Re-running the same version refreshes probabilities idempotently.
    assert repo.assign_documents([dict(assignments[0], probability=0.9)]) == 1
    stored = {
        dt.document_id: dt.probability for dt in db_session.execute(select(DocumentTopic)).scalars()
    }
    assert stored == {ids[0]: 0.9, ids[1]: 0.7}
    assert repo.assign_documents([]) == 0

    repo.create_topics("v2", [{"label": "ai_chips_v2", "keywords": ["nvidia"]}])
    assert repo.latest_version() == "v2"
    assert [t.label for t in repo.topics_for_version("v1")] == ["ai_chips", "macro"]
