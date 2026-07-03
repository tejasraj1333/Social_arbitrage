"""nlp enrichment: sentiment_scores, embeddings, topics, document_topics

Revision ID: 0004_nlp
Revises: 0003_resolution
Create Date: 2026-07-03

Hand-written like 0001-0003 (no live DB needed). Purely additive: four new
tables plus a nullable ``documents.enriched_at`` watermark column, so the
migration is safe on a database that already holds resolved documents.

The vector dimension (384) is pinned here on purpose — pgvector columns are
fixed width, so moving to a different-width embedding model is a schema
change, not a config change (see sam.storage.models.EMBEDDING_DIM).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004_nlp"
down_revision: str | None = "0003_resolution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enrichment watermark: NULL = document not yet scored/embedded.
    op.add_column(
        "documents",
        sa.Column("enriched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_enriched_at", "documents", ["enriched_at"])

    op.create_table(
        "sentiment_scores",
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id"),
            primary_key=True,
        ),
        sa.Column("model", sa.String(length=128), primary_key=True),
        sa.Column("label", sa.String(length=8), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "label IN ('positive', 'negative', 'neutral')", name="ck_sentiment_label"
        ),
    )

    op.create_table(
        "embeddings",
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id"),
            primary_key=True,
        ),
        sa.Column("model", sa.String(length=128), primary_key=True),
        sa.Column("vector", Vector(384), nullable=False),
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("topic_model_version", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False),
        sa.Column("keywords", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_topics_version", "topics", ["topic_model_version"])

    op.create_table(
        "document_topics",
        sa.Column(
            "document_id",
            sa.BigInteger(),
            sa.ForeignKey("documents.id"),
            primary_key=True,
        ),
        sa.Column("topic_id", sa.Integer(), sa.ForeignKey("topics.id"), primary_key=True),
        sa.Column("probability", sa.Float(), nullable=False),
    )
    op.create_index("ix_document_topics_topic", "document_topics", ["topic_id"])


def downgrade() -> None:
    op.drop_index("ix_document_topics_topic", table_name="document_topics")
    op.drop_table("document_topics")
    op.drop_index("ix_topics_version", table_name="topics")
    op.drop_table("topics")
    op.drop_table("embeddings")
    op.drop_table("sentiment_scores")
    op.drop_index("ix_documents_enriched_at", table_name="documents")
    op.drop_column("documents", "enriched_at")
