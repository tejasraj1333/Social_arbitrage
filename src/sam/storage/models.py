"""ORM models.

M0 ships only the ``entities`` table so migrations have something real to
generate and tests can exercise the DB. The full schema (documents,
sentiment_scores, sai_daily, ...) lands incrementally in later milestones —
see docs/architecture.md for the target schema.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from sam.core.db import Base


class Entity(Base):
    """A tradable entity (company/ticker) in the watch universe."""

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
