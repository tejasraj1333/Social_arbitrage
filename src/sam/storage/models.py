"""ORM models.

M0 shipped ``entities``; Phase 2 (ingestion backbone) added ``sources``,
``documents``, ``market_data`` and ``ingestion_runs``; Phase 3 (entity
resolution & quality) added ``document_entities`` and ``data_quality_checks``;
Phase 4 (NLP enrichment) added ``sentiment_scores``, ``embeddings``,
``topics`` and ``document_topics``; Phase 5 (SAI) adds ``sai_daily`` per the
target schema in docs/architecture.md. The rest (forecasts, reports, ...)
lands in later milestones.

Point-in-time rule: every fact row carries both the event time
(``published_at`` / ``date``) and the known time (``ingested_at``). Backtests
must join on known time only.

Column types are declared portably (JSONB/ARRAY on Postgres, JSON on SQLite)
so repository semantics — including ON CONFLICT idempotency — are exercised by
fast in-memory-SQLite unit tests while production DDL stays canonical Postgres.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from sam.core.db import Base

# Embedding dimensionality is part of the schema (pgvector columns are fixed
# width). all-MiniLM-L6-v2 = 384; switching to a different-width model is a
# migration, not a config change.
EMBEDDING_DIM = 384

# JSONB on Postgres, plain JSON elsewhere (SQLite unit tests).
PortableJSON = JSON().with_variant(JSONB(), "postgresql")
# Postgres text[] ; JSON-encoded list on SQLite.
PortableStringList = ARRAY(String).with_variant(JSON(), "sqlite")
# BIGINT pk on Postgres; SQLite needs plain INTEGER for rowid autoincrement.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")
# pgvector on Postgres; JSON-encoded float list on SQLite. Note the read-side
# asymmetry: Postgres returns a numpy array, SQLite a plain list — writers
# always pass list[float], readers must not assume the concrete type.
PortableVector = JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql")


class Entity(Base):
    """A tradable entity (company/ticker) in the watch universe."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    aliases: Mapped[list[str]] = mapped_column(PortableStringList, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Source(Base):
    """A configured data source (one row per collector, e.g. 'rss', 'yahoo')."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(128), unique=True)
    config_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    """One normalized text item (headline, post, story) from any source.

    ``content_hash`` is the global dedup anchor: canonical SHA-256 over the
    document's identity fields (see sam.ingestion.hashing). Re-ingesting the
    same content is a no-op (ON CONFLICT DO NOTHING), which makes collector
    runs idempotent. ``engagement`` is the first-seen snapshot (point-in-time);
    engagement time series are a later-phase concern.
    """

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_source_external", "source_id", "external_id"),
        Index("ix_documents_published_at", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    engagement: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict)
    # Entity-resolution watermark: NULL = not yet scanned by the resolver.
    # Set even when a scan finds no entities, so incremental runs skip the doc.
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    # NLP-enrichment watermark: NULL = not yet scored/embedded (Phase 4).
    # Independent of resolved_at — resolution and enrichment are decoupled
    # stages that may run in either order.
    enriched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class MarketData(Base):
    """Daily OHLCV bar per entity. Natural key (entity_id, date) is the PK.

    Bars are upserted with DO UPDATE (not DO NOTHING): adj_close legitimately
    restates after splits/dividends, and the freshest vendor value must win.
    """

    __tablename__ = "market_data"

    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IngestionRun(Base):
    """Audit row per collector run: timing, row metrics, outcome, lake pointer."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'success', 'error')", name="ck_ingestion_runs_status"
        ),
        Index("ix_ingestion_runs_source_started", "source_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    rows_fetched: Mapped[int] = mapped_column(Integer, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    raw_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentEntity(Base):
    """Resolved link: this document mentions this entity (Phase 3, M2 gate).

    ``confidence`` reflects the strength of the match rule that produced the
    link (cashtag > bare ticker > name/alias — see sam.processing.resolver);
    downstream signals weight by it instead of hard-filtering. ``resolved_at``
    is the *known* time of the link (point-in-time rule) — re-resolving with a
    newer dictionary refreshes it.
    """

    __tablename__ = "document_entities"
    __table_args__ = (Index("ix_document_entities_entity", "entity_id"),)

    document_id: Mapped[int] = mapped_column(BigIntPK, ForeignKey("documents.id"), primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
    confidence: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(16))  # 'cashtag' | 'ticker' | 'alias'
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SentimentScore(Base):
    """Document-level sentiment from a finance-domain model (Phase 4 / M3).

    PK (document_id, model): one score per document per model, so a model
    upgrade re-scores via DO UPDATE per model id while rows from other models
    coexist for comparison. ``score`` is the model's confidence in ``label``
    (a weight for downstream signals, not a filter). ``scored_at`` is the
    *known* time of the score (point-in-time rule).
    """

    __tablename__ = "sentiment_scores"
    __table_args__ = (
        CheckConstraint("label IN ('positive', 'negative', 'neutral')", name="ck_sentiment_label"),
    )

    document_id: Mapped[int] = mapped_column(BigIntPK, ForeignKey("documents.id"), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Embedding(Base):
    """Semantic embedding per document (pgvector column on Postgres).

    PK (document_id, model) mirrors sentiment_scores. Dimension is fixed by
    the schema (EMBEDDING_DIM); a different-width model requires a migration.
    No ANN index yet — vector search arrives with the API phase (M8), and
    unindexed writes keep enrichment cheap until then.
    """

    __tablename__ = "embeddings"

    document_id: Mapped[int] = mapped_column(BigIntPK, ForeignKey("documents.id"), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(PortableVector)
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Topic(Base):
    """One discovered topic cluster from a versioned topic-model run (Phase 4).

    Topic runs are append-only: each fit writes fresh rows under a new
    ``topic_model_version``; "current" topics are the rows of the latest
    version. Older versions are kept — past signal values were computed
    against past topic models (point-in-time rule).
    """

    __tablename__ = "topics"
    __table_args__ = (Index("ix_topics_version", "topic_model_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_model_version: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(256))
    keywords: Mapped[list[str]] = mapped_column(PortableStringList, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentTopic(Base):
    """Assignment of a document to a topic, with the model's probability.

    Versioning rides on ``topic_id`` (topics are per-version rows), so
    assignments from different model versions coexist without a version
    column here.
    """

    __tablename__ = "document_topics"
    __table_args__ = (Index("ix_document_topics_topic", "topic_id"),)

    document_id: Mapped[int] = mapped_column(BigIntPK, ForeignKey("documents.id"), primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), primary_key=True)
    probability: Mapped[float] = mapped_column(Float)


class SaiDaily(Base):
    """One row of the Social Arbitrage Index panel: entity × closed UTC day.

    Documents are bucketed by the *known* time (``ingested_at``), never the
    event time — closed days are immutable (new documents can only land
    "today"), which is what makes the P5 "deterministic rebuild from raw"
    gate achievable. Components are nullable: a young panel has no trailing
    baseline yet, and NULL means "insufficient history", not zero signal.
    ``sai_score`` is the weighted mean of the components' cross-sectional
    centered ranks; ``sai_rank`` ranks scores within the day (1 = strongest).
    ``computed_at`` is the known time of the row (point-in-time rule).
    """

    __tablename__ = "sai_daily"
    __table_args__ = (Index("ix_sai_daily_date", "date"),)

    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    mention_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_momentum: Mapped[float | None] = mapped_column(Float, nullable=True)
    topic_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_growth: Mapped[float | None] = mapped_column(Float, nullable=True)
    sai_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sai_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DataQualityCheck(Base):
    """One executed data-quality assertion (Phase 3 DQ framework).

    Every check run writes a row — pass or fail — so quality is a monitored
    time series, not a one-off script. ``value`` is the measured metric,
    ``threshold`` what it was compared against, ``details`` the evidence
    (e.g. offending document ids; quarantine-don't-delete).
    """

    __tablename__ = "data_quality_checks"
    __table_args__ = (
        CheckConstraint("status IN ('pass', 'warn', 'fail')", name="ck_dq_checks_status"),
        Index("ix_dq_checks_name_ran", "check_name", "ran_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    check_name: Mapped[str] = mapped_column(String(64))
    source_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(8))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(PortableJSON, default=dict)
    ran_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
