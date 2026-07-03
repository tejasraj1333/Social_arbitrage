"""Postgres integration tests — skipped unless SAM_TEST_DB_URL is set.

Run locally against the compose DB:

    docker compose up -d db
    SAM_TEST_DB_URL=postgresql+psycopg://sam:sam@localhost:5433/sam \
        uv run pytest tests/test_integration_pg.py

Everything runs in an isolated `sam_test` schema (created/dropped per session)
so a real `sam` database is never touched. These tests prove the pieces SQLite
can only approximate: JSONB round-trip and Postgres ON CONFLICT semantics.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from sam.core.db import Base
from sam.storage.models import Document, MarketData
from sam.storage.repositories import (
    DocumentRepository,
    EntityRepository,
    MarketDataRepository,
    SourceRepository,
)

pytestmark = pytest.mark.skipif(
    "SAM_TEST_DB_URL" not in os.environ,
    reason="Postgres integration tests need SAM_TEST_DB_URL",
)

_SCHEMA = "sam_test"


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[object]:
    url = os.environ["SAM_TEST_DB_URL"]
    admin = create_engine(url)
    with admin.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {_SCHEMA}"))
        conn.commit()
    engine = create_engine(url, connect_args={"options": f"-csearch_path={_SCHEMA}"})
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    with admin.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE"))
        conn.commit()
    admin.dispose()


@pytest.fixture
def pg_session(pg_engine) -> Iterator[Session]:
    factory = sessionmaker(bind=pg_engine, expire_on_commit=False)
    with factory() as session:
        yield session
        session.rollback()
        # Clean tables between tests (order respects FKs).
        for table in ("ingestion_runs", "documents", "market_data", "sources", "entities"):
            session.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
        session.commit()


def test_document_upsert_idempotent_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    repo = DocumentRepository(pg_session)
    docs = [
        {
            "source_id": source.id,
            "external_id": "https://x/a",
            "url": "https://x/a",
            "author": None,
            "title": "Headline",
            "raw_text": "Body",
            "lang": None,
            "content_hash": "a" * 64,
            "published_at": datetime(2026, 7, 1, tzinfo=UTC),
            "engagement": {"score": 10, "feed": "CNBC"},
        }
    ]
    assert repo.upsert_many(docs) == 1
    assert repo.upsert_many(docs) == 0  # real Postgres ON CONFLICT DO NOTHING
    pg_session.commit()

    loaded = pg_session.execute(select(Document)).scalar_one()
    assert loaded.engagement == {"score": 10, "feed": "CNBC"}  # JSONB round-trip
    assert loaded.published_at is not None
    assert loaded.published_at.tzinfo is not None  # timestamptz comes back aware


def test_market_upsert_do_update_on_postgres(pg_session: Session) -> None:
    EntityRepository(pg_session).seed([{"ticker": "AAPL", "name": "Apple Inc."}])
    ids = EntityRepository(pg_session).by_ticker()
    repo = MarketDataRepository(pg_session)

    bar = {
        "entity_id": ids["AAPL"],
        "date": date(2026, 7, 1),
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 100,
    }
    assert repo.upsert_many([bar]) == 1
    assert repo.upsert_many([dict(bar, adj_close=9.8)]) == 1  # DO UPDATE path
    pg_session.commit()

    stored = pg_session.execute(select(MarketData)).scalar_one()
    assert stored.adj_close == 9.8
    assert stored.volume == 100


def test_entity_seed_idempotent_on_postgres(pg_session: Session) -> None:
    repo = EntityRepository(pg_session)
    universe = [{"ticker": "AAPL", "name": "Apple Inc.", "aliases": ["Apple"]}]
    assert repo.seed(universe) == 1
    assert repo.seed(universe) == 0
    pg_session.commit()
