"""sai panel: sai_daily

Revision ID: 0005_sai
Revises: 0004_nlp
Create Date: 2026-07-04

Hand-written like 0001-0004 (no live DB needed). Purely additive: one new
table, so the migration is safe on a database that already holds enriched
documents.

Component columns are nullable on purpose: NULL means "insufficient trailing
history for this entity/day", which is different information than 0.0 (no
change vs baseline). The composite inherits the same semantics.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_sai"
down_revision: str | None = "0004_nlp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sai_daily",
        sa.Column(
            "entity_id",
            sa.Integer(),
            sa.ForeignKey("entities.id"),
            primary_key=True,
        ),
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("mention_growth", sa.Float(), nullable=True),
        sa.Column("sentiment_momentum", sa.Float(), nullable=True),
        sa.Column("topic_velocity", sa.Float(), nullable=True),
        sa.Column("engagement_growth", sa.Float(), nullable=True),
        sa.Column("sai_score", sa.Float(), nullable=True),
        sa.Column("sai_rank", sa.Integer(), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # Cross-sectional reads ("all entities on day D") are the dominant query.
    op.create_index("ix_sai_daily_date", "sai_daily", ["date"])


def downgrade() -> None:
    op.drop_index("ix_sai_daily_date", table_name="sai_daily")
    op.drop_table("sai_daily")
