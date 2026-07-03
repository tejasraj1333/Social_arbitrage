"""entity resolution & quality: document_entities, data_quality_checks

Revision ID: 0003_resolution
Revises: 0002_ingestion
Create Date: 2026-07-03

Hand-written like 0001/0002 (no live DB needed). Purely additive: two new
tables plus a nullable ``documents.resolved_at`` watermark column, so the
migration is safe on a database that already holds ingested documents.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003_resolution"
down_revision: str | None = "0002_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Resolver watermark: NULL = document not yet scanned for entity mentions.
    op.add_column(
        "documents",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_resolved_at", "documents", ["resolved_at"])

    op.create_table(
        "document_entities",
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            sa.Integer(),
            sa.ForeignKey("entities.id"),
            primary_key=True,
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column(
            "resolved_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_document_entities_entity", "document_entities", ["entity_id"])

    op.create_table(
        "data_quality_checks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("check_name", sa.String(length=64), nullable=False),
        sa.Column("source_name", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("details", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "ran_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("status IN ('pass', 'warn', 'fail')", name="ck_dq_checks_status"),
    )
    op.create_index("ix_dq_checks_name_ran", "data_quality_checks", ["check_name", "ran_at"])


def downgrade() -> None:
    op.drop_index("ix_dq_checks_name_ran", table_name="data_quality_checks")
    op.drop_table("data_quality_checks")
    op.drop_index("ix_document_entities_entity", table_name="document_entities")
    op.drop_table("document_entities")
    op.drop_index("ix_documents_resolved_at", table_name="documents")
    op.drop_column("documents", "resolved_at")
