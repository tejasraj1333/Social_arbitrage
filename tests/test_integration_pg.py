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
from sam.processing.pipeline import ResolutionPipeline
from sam.processing.quality import DataQualityRunner
from sam.storage.models import (
    EMBEDDING_DIM,
    DataQualityCheck,
    Document,
    DocumentEntity,
    Embedding,
    MarketData,
    SaiDaily,
    SentimentScore,
)
from sam.storage.repositories import (
    DocumentEntityRepository,
    DocumentRepository,
    EmbeddingRepository,
    EntityRepository,
    IngestionRunRepository,
    MarketDataRepository,
    SaiRepository,
    SentimentRepository,
    SourceRepository,
    TopicRepository,
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
    # public stays on the search path so the pgvector *type* (installed in
    # public) resolves; tables are created in the first schema (sam_test).
    engine = create_engine(url, connect_args={"options": f"-csearch_path={_SCHEMA},public"})
    # checkfirst=False: with public visible, the existence check would find
    # the *production* tables and skip creation; sam_test was just recreated
    # empty, so unconditional CREATE TABLE is both safe and required.
    Base.metadata.create_all(engine, checkfirst=False)
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
        # Clean tables between tests (order respects FKs). Schema-qualified so
        # the search_path can never resolve these names to real public tables.
        for table in (
            "data_quality_checks",
            "sai_daily",
            "document_topics",
            "topics",
            "embeddings",
            "sentiment_scores",
            "document_entities",
            "ingestion_runs",
            "documents",
            "market_data",
            "sources",
            "entities",
        ):
            session.execute(text(f"TRUNCATE TABLE {_SCHEMA}.{table} RESTART IDENTITY CASCADE"))
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


def _pg_doc(source_id: int, hash_: str, title: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "external_id": hash_[:8],
        "url": f"https://x/{hash_[:8]}",
        "author": None,
        "title": title,
        "raw_text": None,
        "lang": None,
        "content_hash": hash_,
        "published_at": None,
        "engagement": {},
    }


def test_document_entity_upsert_do_update_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    EntityRepository(pg_session).seed([{"ticker": "NVDA", "name": "NVIDIA Corporation"}])
    ids = EntityRepository(pg_session).by_ticker()
    DocumentRepository(pg_session).upsert_many([_pg_doc(source.id, "a" * 64, "Nvidia rises")])
    doc_id = pg_session.execute(select(Document.id)).scalar_one()

    repo = DocumentEntityRepository(pg_session)
    now = datetime.now(tz=UTC)
    link = {
        "document_id": doc_id,
        "entity_id": ids["NVDA"],
        "confidence": 0.8,
        "method": "alias",
        "resolved_at": now,
    }
    assert repo.upsert_many([link]) == 1
    # Re-resolution refreshes the link (real Postgres ON CONFLICT DO UPDATE).
    assert repo.upsert_many([dict(link, confidence=1.0, method="cashtag")]) == 1
    pg_session.commit()

    stored = pg_session.execute(select(DocumentEntity)).scalar_one()
    assert (stored.confidence, stored.method) == (1.0, "cashtag")


def test_resolution_pipeline_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    # aliases exercise the Postgres text[] column end-to-end through the matcher.
    EntityRepository(pg_session).seed(
        [{"ticker": "NVDA", "name": "NVIDIA Corporation", "aliases": ["Nvidia"]}]
    )
    DocumentRepository(pg_session).upsert_many(
        [
            _pg_doc(source.id, "b" * 64, "Nvidia extends its AI lead"),
            _pg_doc(source.id, "c" * 64, "Fed leaves rates unchanged"),
        ]
    )
    pg_session.commit()

    pipeline = ResolutionPipeline(session_factory=lambda: pg_session)
    result = pipeline.run()
    assert result.docs_scanned == 2
    assert result.links_written == 1
    assert pipeline.run().docs_scanned == 0  # watermark works on timestamptz

    unresolved = pg_session.execute(
        select(Document).where(Document.resolved_at.is_(None))
    ).scalars()
    assert list(unresolved) == []


def test_dq_runner_persists_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    runs = IngestionRunRepository(pg_session)
    runs.finish(runs.start(source.id), status="success", rows_fetched=7)
    DocumentRepository(pg_session).upsert_many(
        [_pg_doc(source.id, "d" * 64, "Nvidia rises on record data-center demand")]
    )
    pg_session.commit()

    outcomes = DataQualityRunner(session_factory=lambda: pg_session).run()
    assert {o.check_name for o in outcomes} == {
        "duplicate_rate",
        "freshness",
        "volume_anomaly",
        "resolution_coverage",
        "enrichment_coverage",
        "sai_freshness",
    }
    rows = pg_session.execute(select(DataQualityCheck)).scalars().all()
    assert len(rows) == len(outcomes)
    assert all(isinstance(row.details, dict) for row in rows)  # JSONB round-trip


def test_embedding_vector_round_trip_on_postgres(pg_session: Session) -> None:
    """Real pgvector semantics: list[float] in, numpy array out, DO UPDATE wins."""
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    DocumentRepository(pg_session).upsert_many([_pg_doc(source.id, "e" * 64, "Nvidia rises")])
    doc_id = pg_session.execute(select(Document.id)).scalar_one()

    repo = EmbeddingRepository(pg_session)
    vector = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
    assert repo.upsert_many([{"document_id": doc_id, "model": "minilm", "vector": vector}]) == 1
    refreshed = [v + 1.0 for v in vector]
    assert repo.upsert_many([{"document_id": doc_id, "model": "minilm", "vector": refreshed}]) == 1
    pg_session.commit()

    stored = pg_session.execute(select(Embedding)).scalar_one()
    values = [float(x) for x in stored.vector]  # pgvector returns a numpy array
    assert len(values) == EMBEDDING_DIM
    assert values[0] == 1.0 and abs(values[-1] - (1.0 + (EMBEDDING_DIM - 1) / EMBEDDING_DIM)) < 1e-6

    joined = repo.rows_with_documents("minilm")
    assert [(row[0], row[1]) for row in joined] == [(doc_id, "Nvidia rises")]


def test_sentiment_upsert_do_update_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    DocumentRepository(pg_session).upsert_many([_pg_doc(source.id, "f" * 64, "Apple slides")])
    doc_id = pg_session.execute(select(Document.id)).scalar_one()

    repo = SentimentRepository(pg_session)
    row = {"document_id": doc_id, "model": "finbert", "label": "neutral", "score": 0.5}
    assert repo.upsert_many([row]) == 1
    assert repo.upsert_many([dict(row, label="negative", score=0.95)]) == 1  # DO UPDATE
    pg_session.commit()

    stored = pg_session.execute(select(SentimentScore)).scalar_one()
    assert (stored.label, stored.score) == ("negative", 0.95)
    assert stored.scored_at.tzinfo is not None  # timestamptz round-trip


def test_sai_daily_upsert_do_update_on_postgres(pg_session: Session) -> None:
    """Panel rebuild semantics on real Postgres: DO UPDATE refreshes values,
    NULL components round-trip, and the date watermark reads back correctly."""
    EntityRepository(pg_session).seed([{"ticker": "NVDA", "name": "NVIDIA Corporation"}])
    ids = EntityRepository(pg_session).by_ticker()
    repo = SaiRepository(pg_session)

    row = {
        "entity_id": ids["NVDA"],
        "date": date(2026, 7, 3),
        "mention_growth": 1.25,
        "sentiment_momentum": None,  # insufficient history round-trips as NULL
        "topic_velocity": None,
        "engagement_growth": 0.0,
        "sai_score": 0.5,
        "sai_rank": 1,
        "computed_at": datetime.now(tz=UTC),
    }
    assert repo.upsert_many([row]) == 1
    # A rebuild refreshes in place (real Postgres ON CONFLICT DO UPDATE).
    assert repo.upsert_many([dict(row, sai_score=-0.5, sai_rank=1)]) == 1
    pg_session.commit()

    stored = pg_session.execute(select(SaiDaily)).scalar_one()
    assert (stored.sai_score, stored.sai_rank) == (-0.5, 1)
    assert stored.sentiment_momentum is None
    assert stored.computed_at.tzinfo is not None  # timestamptz round-trip
    assert repo.latest_date() == date(2026, 7, 3)


def test_topic_versioning_on_postgres(pg_session: Session) -> None:
    source = SourceRepository(pg_session).get_or_create("rss", "rss")
    DocumentRepository(pg_session).upsert_many([_pg_doc(source.id, "9" * 64, "Chips rally")])
    doc_id = pg_session.execute(select(Document.id)).scalar_one()

    repo = TopicRepository(pg_session)
    topics = repo.create_topics("v1", [{"label": "chips", "keywords": ["gpu", "fab"]}])
    assert (
        repo.assign_documents(
            [{"document_id": doc_id, "topic_id": topics[0].id, "probability": 0.7}]
        )
        == 1
    )
    pg_session.commit()

    assert repo.latest_version() == "v1"
    (loaded,) = repo.topics_for_version("v1")
    assert loaded.keywords == ["gpu", "fab"]  # text[] round-trip
