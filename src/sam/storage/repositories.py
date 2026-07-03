"""Repository layer — the only module that writes ingestion SQL.

Collectors and the runner talk to these classes, never to the session/tables
directly, so persistence semantics (upserts, dedup, run bookkeeping) live in
exactly one place.

Upserts use native ON CONFLICT on both Postgres (production) and SQLite
(unit tests), dispatched by the session's bind dialect — the idempotency the
tests prove is the same mechanism production runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from sam.core.db import Base
from sam.core.errors import IngestionError
from sam.core.logging import get_logger
from sam.storage.models import Document, Entity, IngestionRun, MarketData, Source

log = get_logger("storage.repositories")

_CHUNK = 500  # rows per multi-VALUES insert statement


def _insert_for(session: Session, model: type[Base]) -> Any:
    """Return a dialect-specific insert() supporting on_conflict_* clauses."""
    table = cast(Table, model.__table__)
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return postgresql.insert(table)
    if dialect == "sqlite":
        return sqlite.insert(table)
    raise IngestionError(f"Unsupported database dialect for upserts: {dialect}")


def _written_count(session: Session, stmt: Any, pk_column: Any) -> int:
    """Execute an upsert and count the rows it actually wrote, via RETURNING.

    cursor.rowcount is NOT trustworthy for INSERTs here (psycopg3 reports -1;
    caught live by the Postgres integration tests). ON CONFLICT ... RETURNING
    returns exactly the written rows on both Postgres and SQLite (>=3.35), so
    counting them is the one mechanism with correct semantics on both.
    """
    result = session.execute(stmt.returning(pk_column))
    return len(result.fetchall())


def _chunks(rows: list[dict[str, Any]], size: int = _CHUNK) -> list[list[dict[str, Any]]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


class SourceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create(self, type_: str, name: str, config_ref: str | None = None) -> Source:
        """Idempotently resolve the source row for a collector."""
        existing = self.session.execute(
            select(Source).where(Source.name == name)
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        source = Source(type=type_, name=name, config_ref=config_ref)
        self.session.add(source)
        self.session.flush()  # assign id without committing the caller's txn
        log.info("source_created", name=name, type=type_)
        return source


class EntityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def seed(self, universe: list[dict[str, Any]]) -> int:
        """Upsert the configured ticker universe; returns newly inserted count.

        Existing rows are left untouched (curated fields like aliases must not
        be clobbered by a re-seed).
        """
        rows = [
            {
                "ticker": item["ticker"],
                "name": item.get("name", item["ticker"]),
                "sector": item.get("sector"),
                "aliases": item.get("aliases", []),
                "active": True,
            }
            for item in universe
        ]
        if not rows:
            return 0
        stmt = _insert_for(self.session, Entity).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["ticker"])
        inserted = _written_count(self.session, stmt, Entity.__table__.c.id)
        log.info("entities_seeded", requested=len(rows), inserted=inserted)
        return inserted

    def by_ticker(self) -> dict[str, int]:
        """Map ticker -> entity id for the active universe."""
        result = self.session.execute(select(Entity.ticker, Entity.id))
        return dict(result.tuples().all())


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, documents: list[dict[str, Any]]) -> int:
        """Insert documents, skipping any whose content_hash already exists.

        Returns the number of *newly inserted* rows — the idempotency signal
        (re-ingesting the same fetch returns 0). Intra-batch duplicates are
        collapsed first (first occurrence wins, matching DO NOTHING semantics).
        """
        deduped: dict[str, dict[str, Any]] = {}
        for doc in documents:
            deduped.setdefault(doc["content_hash"], doc)
        rows = list(deduped.values())
        if not rows:
            return 0

        inserted = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, Document).values(chunk)
            stmt = stmt.on_conflict_do_nothing(index_elements=["content_hash"])
            inserted += _written_count(self.session, stmt, Document.__table__.c.id)
        return inserted


class MarketDataRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, bars: list[dict[str, Any]]) -> int:
        """Upsert bars on (entity_id, date); vendor restatements win.

        Returns the number of rows written (inserted or updated). Intra-batch
        duplicates are collapsed last-wins first — Postgres rejects a DO UPDATE
        that touches the same key twice in one statement.
        """
        deduped: dict[tuple[int, Any], dict[str, Any]] = {
            (bar["entity_id"], bar["date"]): bar for bar in bars
        }
        rows = list(deduped.values())
        if not rows:
            return 0

        price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        written = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, MarketData).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["entity_id", "date"],
                set_={col: stmt.excluded[col] for col in price_cols},
            )
            written += _written_count(self.session, stmt, MarketData.__table__.c.entity_id)
        return written


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, source_id: int) -> IngestionRun:
        run = IngestionRun(source_id=source_id, status="running")
        self.session.add(run)
        self.session.flush()
        return run

    def finish(
        self,
        run: IngestionRun,
        *,
        status: str,
        rows_fetched: int = 0,
        rows_inserted: int = 0,
        raw_path: str | None = None,
        error: str | None = None,
    ) -> IngestionRun:
        run.status = status
        run.finished_at = datetime.now(tz=UTC)
        run.rows_fetched = rows_fetched
        run.rows_inserted = rows_inserted
        run.raw_path = raw_path
        run.error = error
        self.session.flush()
        return run

    def recent(self, limit: int = 20) -> list[IngestionRun]:
        """Most recent runs first — the `sam runs` observability query."""
        return list(
            self.session.execute(
                select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
            ).scalars()
        )
