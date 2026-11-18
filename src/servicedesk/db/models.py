"""SQLAlchemy ORM models.

Knowledge base: one row in `documents` per source file, and many rows in `chunks` per
document. Each chunk stores its embedding in a pgvector `vector` column, so the text, its
metadata and its embedding live together and are updated in the same transaction.

Service desk (Phase 4): `services` (the IT service catalogue and each service's current
health), `tickets` and `ticket_comments`. Enum-like columns (status, priority...) are plain
strings guarded by CHECK constraints in the migration; Python-side validation lives in
desk/schemas.py.

Phase 6 adds `tool_executions`: idempotency keys for writes, so a resumed agent run can't
create the same ticket twice.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
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
    # Full-text search words, computed by Postgres from section + content (migration 0002).
    # Computed tells SQLAlchemy never to write it; deferred skips loading it with each chunk.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', section), 'A') || "
            "setweight(to_tsvector('english', content), 'B')",
            persisted=True,
        ),
        deferred=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Service(Base):
    __tablename__ = "services"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # slug, e.g. "vpn"
    name: Mapped[str] = mapped_column(String(100), unique=True)  # e.g. "VPN"
    description: Mapped[str] = mapped_column(String(300))
    owning_team: Mapped[str] = mapped_column(String(100))
    criticality: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32))
    status_message: Mapped[str | None] = mapped_column(Text)
    status_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    # Numbers start at 1001 so they look like real ticket numbers (shown as INC-1001).
    id: Mapped[int] = mapped_column(Identity(start=1001), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32))
    priority: Mapped[str] = mapped_column(String(2))
    status: Mapped[str] = mapped_column(String(32), default="new")
    requester: Mapped[str] = mapped_column(String(254), index=True)
    assignee: Mapped[str | None] = mapped_column(String(254))
    service_id: Mapped[str | None] = mapped_column(ForeignKey("services.id"), index=True)
    resolution: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    service: Mapped[Service | None] = relationship()
    comments: Mapped[list["TicketComment"]] = relationship(
        back_populates="ticket",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TicketComment.id",
    )


class TicketComment(Base):
    __tablename__ = "ticket_comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    author: Mapped[str] = mapped_column(String(254))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="comments")


class ToolExecution(Base):
    """One write that was carried out with an idempotency key, and what it returned
    (Phase 6). Saved in the same transaction as the write itself."""

    __tablename__ = "tool_executions"

    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    tool: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
