"""The ingestion pipeline: discover -> load -> chunk -> embed -> store.

It is idempotent: running it twice on the same files changes nothing and costs nothing the
second time. A document is identified by its doc_id (e.g. "KB-0001"), not its file path,
so renaming or moving a file doesn't create a duplicate.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.db.models import EMBEDDING_DIMENSIONS, Document
from servicedesk.db.models import Chunk as ChunkRow
from servicedesk.ingestion.chunker import Chunker
from servicedesk.ingestion.embeddings import Embedder
from servicedesk.ingestion.loaders import discover_documents, display_path, load_document

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (path, error)
    chunks_written: int = 0
    tokens_embedded: int = 0
    pruning_skipped: bool = False


@dataclass
class PreviewRow:
    doc_id: str
    doc_type: str
    source_path: str
    sections: int
    chunks: int
    tokens: int


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preview(root: Path, chunker: Chunker) -> tuple[list[PreviewRow], list[tuple[str, str]]]:
    """Load and chunk without embedding or writing anything: a free dry run."""
    rows, failed = [], []
    for path in discover_documents(root):
        try:
            doc = load_document(path)
            chunks = chunker.split(doc)
        except Exception as exc:
            failed.append((display_path(path), f"{type(exc).__name__}: {exc}"))
            continue
        rows.append(
            PreviewRow(
                doc.doc_id,
                doc.doc_type,
                doc.source_path,
                len(doc.sections),
                len(chunks),
                sum(c.token_count for c in chunks),
            )
        )
    return rows, failed


class IngestionPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        chunker: Chunker,
    ):
        if embedder.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedder produces {embedder.dimensions}-d vectors but the database column "
                f"is vector({EMBEDDING_DIMENSIONS})."
            )
        self.session_factory = session_factory
        self.embedder = embedder
        self.chunker = chunker

    def run(self, root: Path, *, force: bool = False, prune: bool = True) -> IngestReport:
        report = IngestReport()
        seen: set[str] = set()

        for path in discover_documents(root):
            # One bad file must not stop the rest, so failures are recorded, not raised.
            try:
                self._ingest_file(path, force, seen, report)
            except Exception as exc:
                logger.exception("Failed to ingest %s", path)
                report.failed.append((display_path(path), f"{type(exc).__name__}: {exc}"))

        if prune and root.is_dir():
            if report.failed:
                # A file that failed to load has an unknown doc_id, so pruning could delete
                # its last good version. Be safe and skip pruning this run.
                report.pruning_skipped = True
            else:
                self._prune(root, seen, report)
        return report

    def _ingest_file(self, path: Path, force: bool, seen: set[str], report: IngestReport) -> None:
        doc = load_document(path)
        if doc.doc_id in seen:
            raise ValueError(f"Duplicate doc_id {doc.doc_id}: already used by another file")
        seen.add(doc.doc_id)
        digest = file_sha256(path)

        with self.session_factory() as session:
            existing = session.scalar(select(Document).where(Document.doc_id == doc.doc_id))
            unchanged = (
                existing is not None
                and not force
                and existing.content_hash == digest
                and existing.embedding_model == self.embedder.model
            )
            if unchanged:
                if existing.source_path != doc.source_path:  # renamed/moved: no re-embed needed
                    existing.source_path = doc.source_path
                    session.commit()
                report.unchanged.append(doc.doc_id)
                return

        chunks = self.chunker.split(doc)
        if not chunks:
            raise ValueError("No text could be extracted from the document")

        # Call the API *before* opening the write transaction: if embedding fails, the
        # previously stored version of this document is left untouched.
        vectors = self.embedder.embed([c.embedding_text for c in chunks])

        # session_factory.begin() commits on success and rolls back on any error, so readers
        # never see a document with half of its chunks replaced.
        with self.session_factory.begin() as session:
            row = session.scalar(select(Document).where(Document.doc_id == doc.doc_id))
            is_new = row is None
            if is_new:
                row = Document(doc_id=doc.doc_id)
                session.add(row)
            else:
                session.execute(delete(ChunkRow).where(ChunkRow.document_id == row.id))

            row.title = doc.title
            row.source_path = doc.source_path
            row.doc_type = doc.doc_type
            row.category = doc.category
            row.service = doc.service
            row.content_hash = digest
            row.embedding_model = self.embedder.model
            row.metadata_ = doc.metadata

            session.add_all(
                ChunkRow(
                    document=row,
                    chunk_index=chunk.index,
                    section=chunk.section,
                    page=chunk.page,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            )

        (report.added if is_new else report.updated).append(doc.doc_id)
        report.chunks_written += len(chunks)
        report.tokens_embedded += sum(c.token_count for c in chunks)

    def _prune(self, root: Path, seen: set[str], report: IngestReport) -> None:
        """Delete documents that used to live under root but whose files are gone."""
        prefix = display_path(root).rstrip("/") + "/"
        with self.session_factory.begin() as session:
            stale = session.scalars(
                select(Document).where(
                    Document.source_path.startswith(prefix, autoescape=True),
                    Document.doc_id.not_in(seen),
                )
            ).all()
            for row in stale:
                session.delete(row)  # chunks go too, via ON DELETE CASCADE
                report.removed.append(row.doc_id)
