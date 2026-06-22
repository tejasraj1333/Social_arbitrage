"""initial schema: pgvector extension + entities

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-22

Hand-written so the project has a runnable migration from day one (no live
DB needed to generate it). Later milestones use `alembic revision --autogenerate`.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector for embeddings stored alongside metadata (chosen vector store).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "entities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("sector", sa.String(length=128), nullable=True),
        sa.Column("aliases", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("ticker", name="uq_entities_ticker"),
    )
    op.create_index("ix_entities_ticker", "entities", ["ticker"])


def downgrade() -> None:
    op.drop_index("ix_entities_ticker", table_name="entities")
    op.drop_table("entities")
