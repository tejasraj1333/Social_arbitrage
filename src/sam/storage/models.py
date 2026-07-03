"""ORM models.

M0 shipped ``entities``; Phase 2 (ingestion backbone) adds ``sources``,
``documents``, ``market_data`` and ``ingestion_runs`` per the target schema in
docs/architecture.md. The rest (document_entities, sentiment_scores, sai_daily,
...) lands in later milestones.

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

# JSONB on Postgres, plain JSON elsewhere (SQLite unit tests).
PortableJSON = JSON().with_variant(JSONB(), "postgresql")
# Postgres text[] ; JSON-encoded list on SQLite.
PortableStringList = ARRAY(String).with_variant(JSON(), "sqlite")
# BIGINT pk on Postgres; SQLite needs plain INTEGER for rowid autoincrement.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


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
