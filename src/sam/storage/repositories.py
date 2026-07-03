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

from sqlalchemy import Table, func, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from sam.core.db import Base
from sam.core.errors import IngestionError
from sam.core.logging import get_logger
from sam.storage.models import (
    DataQualityCheck,
    Document,
    DocumentEntity,
    DocumentTopic,
    Embedding,
    Entity,
    IngestionRun,
    MarketData,
    SentimentScore,
    Source,
    Topic,
)

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

    def all(self) -> list[Source]:
        """Every registered source (DQ iterates these), name-ordered."""
        return list(self.session.execute(select(Source).order_by(Source.name)).scalars())


class EntityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def seed(self, universe: list[dict[str, Any]], *, update: bool = False) -> int:
        """Upsert the configured ticker universe; returns written-row count.

        Default (``update=False``): existing rows are left untouched — a plain
        re-seed never clobbers the DB. With ``update=True`` the config is
        treated as the curation source of truth and name/sector/aliases of
        existing tickers are refreshed (``sam seed --update``, e.g. after
        adding resolver aliases).
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
        if update:
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker"],
                set_={
                    "name": stmt.excluded.name,
                    "sector": stmt.excluded.sector,
                    "aliases": stmt.excluded.aliases,
                    "active": stmt.excluded.active,
                },
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=["ticker"])
        written = _written_count(self.session, stmt, Entity.__table__.c.id)
        log.info("entities_seeded", requested=len(rows), written=written, update=update)
        return written

    def by_ticker(self) -> dict[str, int]:
        """Map ticker -> entity id for the active universe."""
        result = self.session.execute(select(Entity.ticker, Entity.id))
        return dict(result.tuples().all())

    def active(self) -> list[Entity]:
        """Active entities, resolver-dictionary source (ticker order = stable)."""
        return list(
            self.session.execute(
                select(Entity).where(Entity.active.is_(True)).order_by(Entity.ticker)
            ).scalars()
        )


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

    def resolution_batch(
        self, *, after_id: int = 0, limit: int = 500, include_resolved: bool = False
    ) -> list[Document]:
        """Next id-ordered batch for the entity resolver (keyset pagination).

        Default scans only unresolved documents (``resolved_at IS NULL``);
        ``include_resolved=True`` is the ``sam resolve --all`` path after a
        dictionary change. ``after_id`` guards against re-reading a batch
        even if the caller's resolved-marking failed.
        """
        stmt = select(Document).where(Document.id > after_id).order_by(Document.id).limit(limit)
        if not include_resolved:
            stmt = stmt.where(Document.resolved_at.is_(None))
        return list(self.session.execute(stmt).scalars())

    def mark_resolved(self, document_ids: list[int], *, at: datetime) -> None:
        """Stamp the resolver watermark — also for docs with zero matches."""
        if not document_ids:
            return
        self.session.execute(
            update(Document).where(Document.id.in_(document_ids)).values(resolved_at=at)
        )

    def enrichment_batch(
        self, *, after_id: int = 0, limit: int = 500, include_enriched: bool = False
    ) -> list[Document]:
        """Next id-ordered batch for the NLP enricher (keyset pagination).

        Mirror of :meth:`resolution_batch` on the ``enriched_at`` watermark;
        ``include_enriched=True`` is the ``sam enrich --all`` path after a
        model change.
        """
        stmt = select(Document).where(Document.id > after_id).order_by(Document.id).limit(limit)
        if not include_enriched:
            stmt = stmt.where(Document.enriched_at.is_(None))
        return list(self.session.execute(stmt).scalars())

    def mark_enriched(self, document_ids: list[int], *, at: datetime) -> None:
        """Stamp the enrichment watermark — also for docs with no usable text."""
        if not document_ids:
            return
        self.session.execute(
            update(Document).where(Document.id.in_(document_ids)).values(enriched_at=at)
        )

    def enrichment_stats(self) -> tuple[int, int, int]:
        """(total, unenriched, with_sentiment) document counts for DQ coverage."""
        total = self.session.execute(select(func.count(Document.id))).scalar_one()
        unenriched = self.session.execute(
            select(func.count(Document.id)).where(Document.enriched_at.is_(None))
        ).scalar_one()
        with_sentiment = self.session.execute(
            select(func.count(func.distinct(SentimentScore.document_id)))
        ).scalar_one()
        return total, unenriched, with_sentiment

    def recent_titles(self, limit: int = 1000) -> list[tuple[int, str | None]]:
        """(id, title) of the most recent documents — the near-dup check window.

        "Recent" by id, not timestamp: portable across SQLite/Postgres without
        tz-comparison pitfalls, and ingestion order is what dedup cares about.
        """
        rows = self.session.execute(
            select(Document.id, Document.title).order_by(Document.id.desc()).limit(limit)
        ).tuples()
        return list(rows)

    def resolution_stats(self) -> tuple[int, int, int]:
        """(total, unresolved, with_links) document counts for DQ coverage."""
        total = self.session.execute(select(func.count(Document.id))).scalar_one()
        unresolved = self.session.execute(
            select(func.count(Document.id)).where(Document.resolved_at.is_(None))
        ).scalar_one()
        with_links = self.session.execute(
            select(func.count(func.distinct(DocumentEntity.document_id)))
        ).scalar_one()
        return total, unresolved, with_links


class DocumentEntityRepository:
    """document→entity links — the resolver's output table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, links: list[dict[str, Any]]) -> int:
        """Upsert links on (document_id, entity_id); the latest resolution wins.

        DO UPDATE (not DO NOTHING): re-resolving after a dictionary change
        must refresh confidence/method/resolved_at. Intra-batch duplicates
        collapse last-wins to match that semantics.
        """
        deduped: dict[tuple[int, int], dict[str, Any]] = {
            (link["document_id"], link["entity_id"]): link for link in links
        }
        rows = list(deduped.values())
        if not rows:
            return 0

        set_cols = ["confidence", "method", "resolved_at"]
        written = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, DocumentEntity).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["document_id", "entity_id"],
                set_={col: stmt.excluded[col] for col in set_cols},
            )
            written += _written_count(self.session, stmt, DocumentEntity.__table__.c.document_id)
        return written


class SentimentRepository:
    """sentiment_scores — the enricher's sentiment output table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, scores: list[dict[str, Any]]) -> int:
        """Upsert scores on (document_id, model); the latest scoring wins.

        DO UPDATE (not DO NOTHING): re-enriching after ``--all`` must refresh
        label/score/scored_at. Rows from *different* models coexist — the
        model id is part of the key.
        """
        deduped: dict[tuple[int, str], dict[str, Any]] = {
            (row["document_id"], row["model"]): row for row in scores
        }
        rows = list(deduped.values())
        if not rows:
            return 0

        set_cols = ["label", "score", "scored_at"]
        written = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, SentimentScore).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["document_id", "model"],
                set_={col: stmt.excluded[col] for col in set_cols},
            )
            written += _written_count(self.session, stmt, SentimentScore.__table__.c.document_id)
        return written


class EmbeddingRepository:
    """embeddings — the enricher's vector output table (pgvector on Postgres)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, embeddings: list[dict[str, Any]]) -> int:
        """Upsert vectors on (document_id, model); the latest embedding wins."""
        deduped: dict[tuple[int, str], dict[str, Any]] = {
            (row["document_id"], row["model"]): row for row in embeddings
        }
        rows = list(deduped.values())
        if not rows:
            return 0

        set_cols = ["vector", "embedded_at"]
        written = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, Embedding).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["document_id", "model"],
                set_={col: stmt.excluded[col] for col in set_cols},
            )
            written += _written_count(self.session, stmt, Embedding.__table__.c.document_id)
        return written

    def rows_with_documents(self, model: str) -> list[tuple[int, str | None, str | None, Any]]:
        """(document_id, title, raw_text, vector) for every doc embedded by ``model``.

        The topic pipeline's input: precomputed vectors joined to their text.
        ``vector`` is dialect-typed (ndarray on Postgres, list on SQLite) —
        callers must not assume the concrete type.
        """
        rows = self.session.execute(
            select(Document.id, Document.title, Document.raw_text, Embedding.vector)
            .join(Embedding, Embedding.document_id == Document.id)
            .where(Embedding.model == model)
            .order_by(Document.id)
        ).tuples()
        return list(rows)


class TopicRepository:
    """topics + document_topics — versioned output of topic-model runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_topics(self, version: str, topics: list[dict[str, Any]]) -> list[Topic]:
        """Insert this run's topic rows (append-only; versions coexist)."""
        rows = [
            Topic(
                topic_model_version=version,
                label=item["label"],
                keywords=list(item.get("keywords", [])),
            )
            for item in topics
        ]
        self.session.add_all(rows)
        self.session.flush()  # assign ids for the assignment step
        return rows

    def assign_documents(self, assignments: list[dict[str, Any]]) -> int:
        """Upsert document→topic assignments on (document_id, topic_id).

        DO UPDATE on probability so re-running the same version is idempotent;
        assignments from older versions survive untouched (their topic ids
        belong to older topic rows).
        """
        deduped: dict[tuple[int, int], dict[str, Any]] = {
            (row["document_id"], row["topic_id"]): row for row in assignments
        }
        rows = list(deduped.values())
        if not rows:
            return 0

        written = 0
        for chunk in _chunks(rows):
            stmt = _insert_for(self.session, DocumentTopic).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["document_id", "topic_id"],
                set_={"probability": stmt.excluded["probability"]},
            )
            written += _written_count(self.session, stmt, DocumentTopic.__table__.c.document_id)
        return written

    def latest_version(self) -> str | None:
        """Most recent topic_model_version (by creation time), if any."""
        return (
            self.session.execute(
                select(Topic.topic_model_version).order_by(Topic.created_at.desc(), Topic.id.desc())
            )
            .scalars()
            .first()
        )

    def topics_for_version(self, version: str) -> list[Topic]:
        """Topic rows of one run, id-ordered."""
        return list(
            self.session.execute(
                select(Topic).where(Topic.topic_model_version == version).order_by(Topic.id)
            ).scalars()
        )


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

    def recent_successes_for(self, source_id: int, limit: int = 10) -> list[IngestionRun]:
        """Latest successful runs for one source (freshness/volume DQ inputs)."""
        return list(
            self.session.execute(
                select(IngestionRun)
                .where(IngestionRun.source_id == source_id, IngestionRun.status == "success")
                .order_by(IngestionRun.started_at.desc())
                .limit(limit)
            ).scalars()
        )


class DataQualityRepository:
    """Persisted DQ assertions — every check run leaves an auditable row."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, checks: list[DataQualityCheck]) -> int:
        """Append check rows (plain inserts — history is the point)."""
        self.session.add_all(checks)
        self.session.flush()
        return len(checks)

    def latest(self, limit: int = 50) -> list[DataQualityCheck]:
        """Most recent check rows first (the `sam dq --history` query)."""
        return list(
            self.session.execute(
                select(DataQualityCheck).order_by(DataQualityCheck.ran_at.desc()).limit(limit)
            ).scalars()
        )
