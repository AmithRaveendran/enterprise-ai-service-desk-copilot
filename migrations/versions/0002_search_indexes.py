"""Add search indexes: HNSW for vector search, a tsvector column + GIN for keyword search.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # HNSW: an approximate nearest-neighbour graph, so a similarity query visits a few
    # hundred vectors instead of all of them. vector_cosine_ops matches the <=> (cosine
    # distance) operator used by the search queries; an index only helps queries that use
    # the same operator it was built for.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    # Keyword search: Postgres full-text search turns text into a tsvector of normalised
    # words ("connecting" -> "connect", stop words like "the" dropped). A GENERATED column is
    # kept up to date by Postgres itself, so the ingestion code doesn't change. Words in the
    # section heading path (which starts with the document title) weigh more (A) than words
    # in the body (B).
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', section), 'A') ||
            setweight(to_tsvector('english', content), 'B')
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunks_search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN search_vector")
    op.execute("DROP INDEX ix_chunks_embedding_hnsw")
