"""Shared fixtures.

Tests stay self-contained so they run anywhere (incl. CI): no external
services. DB-backed tests use in-memory SQLite via the portable model types —
the ON CONFLICT semantics exercised here are the same mechanism production
uses on Postgres.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from sam.api.app import create_app
from sam.core.db import Base


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def db_session() -> Iterator[Session]:
    """In-memory SQLite session with the full schema and FK enforcement on."""
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):  # SQLite needs FK enforcement opted in
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session
    engine.dispose()
