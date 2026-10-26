"""SQLAlchemy ORM models.

One row in `documents` per source file, and many rows in `chunks` per document. Each chunk
stores its embedding in a pgvector `vector` column, so the text, its metadata and its
embedding live together and are updated in the same transaction.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Must match the embedding model's output size (settings.embedding_dimensions).
# Changing it needs a migration and a full re-ingest.
EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), unique=True)  # e.g. "KB-0001"
    title: Mapped[str] = mapped_column(String(300))
    source_path: Mapped[str] = mapped_column(String(500))
    doc_type: Mapped[str] = mapped_column(String(16))
    category: Mapped[str | None] = mapped_column(String(50))
    service: Mapped[str | None] = mapped_column(String(100))
    # SHA-256 of the file bytes, plus the model used: if neither changed, skip re-embedding.
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(100))
    # "metadata" is reserved by SQLAlchemy, so the Python attribute gets a trailing underscore.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]
    section: Mapped[str] = mapped_column(String(500))
    page: Mapped[int | None]
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))

    document: Mapped[Document] = relationship(back_populates="chunks")
