"""Create documents and chunks tables.

Revision ID: 0001
Revises:
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The Docker init script already does this; repeating it makes migrations self-contained
    # (e.g. for the test database or a managed Postgres).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("doc_id", sa.String(32), nullable=False, unique=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("source_path", sa.String(500), nullable=False),
        sa.Column("doc_type", sa.String(16), nullable=False),
        sa.Column("category", sa.String(50)),
        sa.Column("service", sa.String(100)),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("section", sa.String(500), nullable=False),
        sa.Column("page", sa.Integer),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    # The vector index for fast similarity search (HNSW) is added in Phase 2.


def downgrade() -> None:
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
