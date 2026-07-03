"""Repository-layer tests on in-memory SQLite (same ON CONFLICT paths as prod)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.storage.models import Document, Entity, MarketData, Source
from sam.storage.repositories import (
    DocumentRepository,
    EntityRepository,
    IngestionRunRepository,
    MarketDataRepository,
    SourceRepository,
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
