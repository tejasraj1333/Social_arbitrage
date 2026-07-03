"""Schema round-trip tests for the Phase-2/3 tables (in-memory SQLite).

The models declare portable column types, so the same ORM definitions run on
SQLite here and on Postgres in production (DDL canonicalized by migrations
0002/0003).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sam.storage.models import (
    DataQualityCheck,
    Document,
    DocumentEntity,
    Entity,
    IngestionRun,
    MarketData,
    Source,
)


@pytest.fixture
def session(db_session: Session) -> Session:
    return db_session


def test_all_tables_create_on_sqlite(session: Session) -> None:
    # create_all in the fixture is the real assertion; sanity-check emptiness.
    assert session.execute(select(Source)).all() == []
    assert session.execute(select(Document)).all() == []


def test_document_round_trip_with_engagement_json(session: Session) -> None:
    src = Source(type="rss", name="rss", config_ref="sources.yaml:rss")
    session.add(src)
    session.flush()

    doc = Document(
        source_id=src.id,
        external_id="https://example.com/a",
        url="https://example.com/a",
        title="Headline",
        raw_text="Summary text",
        content_hash="a" * 64,
        published_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        engagement={"score": 10, "comments": 2},
    )
    session.add(doc)
    session.commit()

    loaded = session.execute(select(Document)).scalar_one()
    assert loaded.engagement == {"score": 10, "comments": 2}
    assert loaded.ingested_at is not None  # server default applied


def test_document_content_hash_unique(session: Session) -> None:
    src = Source(type="rss", name="rss")
    session.add(src)
    session.flush()
    session.add(Document(source_id=src.id, content_hash="h" * 64))
    session.commit()
    session.add(Document(source_id=src.id, content_hash="h" * 64))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_document_requires_valid_source_fk(session: Session) -> None:
    session.add(Document(source_id=999, content_hash="f" * 64))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_market_data_composite_pk(session: Session) -> None:
    ent = Entity(ticker="AAPL", name="Apple Inc.", aliases=[])
    session.add(ent)
    session.flush()
    bar = MarketData(
        entity_id=ent.id,
        date=date(2026, 7, 1),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        adj_close=1.5,
        volume=1000,
    )
    session.add(bar)
    session.commit()
    # Core insert: dodges the ORM identity map so the DB constraint itself fires.
    dup = insert(MarketData).values(entity_id=ent.id, date=date(2026, 7, 1), close=9.9)
    with pytest.raises(IntegrityError):  # same (entity_id, date) key
        session.execute(dup)
    session.rollback()


def test_ingestion_run_lifecycle_fields(session: Session) -> None:
    src = Source(type="hackernews", name="hackernews")
    session.add(src)
    session.flush()
    run = IngestionRun(source_id=src.id)
    session.add(run)
    session.commit()
    assert run.status == "running"
    assert run.started_at is not None

    run.status = "success"
    run.finished_at = datetime.now(tz=UTC)
    run.rows_fetched = 100
    run.rows_inserted = 42
    run.raw_path = "data/00_raw/hackernews/dt=2026-07-02/x.jsonl.gz"
    session.commit()
    loaded = session.execute(select(IngestionRun)).scalar_one()
    assert loaded.rows_inserted == 42


def _doc_and_entity(session: Session) -> tuple[Document, Entity]:
    src = Source(type="rss", name="rss")
    ent = Entity(ticker="NVDA", name="NVIDIA Corporation", aliases=["Nvidia"])
    session.add_all([src, ent])
    session.flush()
    doc = Document(source_id=src.id, title="Nvidia pops", content_hash="d" * 64)
    session.add(doc)
    session.flush()
    return doc, ent


def test_document_entity_round_trip_and_composite_pk(session: Session) -> None:
    doc, ent = _doc_and_entity(session)
    link = DocumentEntity(document_id=doc.id, entity_id=ent.id, confidence=0.9, method="ticker")
    session.add(link)
    session.commit()

    loaded = session.execute(select(DocumentEntity)).scalar_one()
    assert (loaded.document_id, loaded.entity_id) == (doc.id, ent.id)
    assert loaded.resolved_at is not None  # server default applied

    # Core insert: dodges the ORM identity map so the DB constraint itself fires.
    dup = insert(DocumentEntity).values(
        document_id=doc.id, entity_id=ent.id, confidence=1.0, method="cashtag"
    )
    with pytest.raises(IntegrityError):  # same (document_id, entity_id) key
        session.execute(dup)
    session.rollback()


def test_document_entity_requires_valid_fks(session: Session) -> None:
    session.add(DocumentEntity(document_id=999, entity_id=999, confidence=0.5, method="alias"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_document_resolved_at_defaults_null(session: Session) -> None:
    doc, _ = _doc_and_entity(session)
    session.commit()
    assert doc.resolved_at is None  # unresolved until the resolver scans it


def test_data_quality_check_round_trip(session: Session) -> None:
    check = DataQualityCheck(
        check_name="duplicate_rate",
        source_name="rss",
        status="pass",
        value=0.004,
        threshold=0.02,
        details={"near_dup_pairs": [[1, 2]]},
    )
    session.add(check)
    session.commit()

    loaded = session.execute(select(DataQualityCheck)).scalar_one()
    assert loaded.status == "pass"
    assert loaded.details == {"near_dup_pairs": [[1, 2]]}
    assert loaded.ran_at is not None


def test_data_quality_check_rejects_unknown_status(session: Session) -> None:
    session.add(DataQualityCheck(check_name="freshness", status="bogus"))
    with pytest.raises(IntegrityError):  # CHECK constraint
        session.commit()
    session.rollback()
