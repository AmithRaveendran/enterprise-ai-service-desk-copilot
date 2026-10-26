"""Integration tests: the full pipeline against a real Postgres + pgvector database."""

import shutil

import pytest
from sqlalchemy import func, select

from servicedesk.db.models import Chunk, Document
from servicedesk.ingestion.chunker import Chunker
from servicedesk.ingestion.pipeline import IngestionPipeline
from tests.conftest import KB_DIR, FakeEmbedder

pytestmark = pytest.mark.integration

SAMPLE_FILES = [
    "how-to/vpn-setup.md",
    "policies/incident-severity-and-sla.html",
    "policies/information-security-policy.pdf",
]


@pytest.fixture
def kb(tmp_path):
    """A small knowledge base copied into a temp folder, so tests can edit and delete files."""
    root = tmp_path / "kb"
    for rel in SAMPLE_FILES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(KB_DIR / rel, dest)
    return root


def make_pipeline(session_factory, embedder):
    return IngestionPipeline(session_factory, embedder, Chunker())


def count(session_factory, model) -> int:
    with session_factory() as session:
        return session.scalar(select(func.count()).select_from(model))


def test_first_run_stores_documents_chunks_and_embeddings(session_factory, fake_embedder, kb):
    report = make_pipeline(session_factory, fake_embedder).run(kb)

    assert sorted(report.added) == ["KB-0001", "POL-0002", "POL-0005"]
    assert report.failed == []
    assert count(session_factory, Chunk) == report.chunks_written > 0

    with session_factory() as session:
        doc = session.scalar(select(Document).where(Document.doc_id == "POL-0005"))
        assert doc.doc_type == "pdf"
        assert doc.embedding_model == "fake-hashing-embedder"
        assert len(doc.content_hash) == 64
        assert [c.chunk_index for c in doc.chunks] == list(range(len(doc.chunks)))
        assert len(doc.chunks[0].embedding) == 1536
        assert any(c.page == 2 for c in doc.chunks)


def test_second_run_is_a_no_op(session_factory, fake_embedder, kb):
    pipeline = make_pipeline(session_factory, fake_embedder)
    pipeline.run(kb)
    texts_after_first_run = fake_embedder.texts_embedded

    report = pipeline.run(kb)

    assert sorted(report.unchanged) == ["KB-0001", "POL-0002", "POL-0005"]
    assert report.added == report.updated == report.removed == []
    assert fake_embedder.texts_embedded == texts_after_first_run  # no new API calls


def test_changed_file_is_re_embedded_and_its_chunks_replaced(session_factory, fake_embedder, kb):
    pipeline = make_pipeline(session_factory, fake_embedder)
    pipeline.run(kb)
    md = kb / "how-to" / "vpn-setup.md"
    md.write_text(md.read_text(encoding="utf-8") + "\n## New section\n\nCall 4357.\n", "utf-8")

    report = pipeline.run(kb)

    assert report.updated == ["KB-0001"]
    assert sorted(report.unchanged) == ["POL-0002", "POL-0005"]
    with session_factory() as session:
        doc = session.scalar(select(Document).where(Document.doc_id == "KB-0001"))
        assert doc.chunks[-1].section.endswith("> New section")
        assert [c.chunk_index for c in doc.chunks] == list(range(len(doc.chunks)))


def test_deleted_file_is_pruned_with_its_chunks(session_factory, fake_embedder, kb):
    pipeline = make_pipeline(session_factory, fake_embedder)
    pipeline.run(kb)
    (kb / "policies" / "incident-severity-and-sla.html").unlink()

    report = pipeline.run(kb)

    assert report.removed == ["POL-0002"]
    with session_factory() as session:
        assert session.scalar(select(Document).where(Document.doc_id == "POL-0002")) is None
        orphans = session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id.not_in(select(Document.id)))
        )
        assert orphans == 0


def test_renamed_file_keeps_its_embeddings(session_factory, fake_embedder, kb):
    pipeline = make_pipeline(session_factory, fake_embedder)
    pipeline.run(kb)
    texts_before = fake_embedder.texts_embedded
    (kb / "how-to" / "vpn-setup.md").rename(kb / "how-to" / "vpn-guide.md")

    report = pipeline.run(kb)

    assert "KB-0001" in report.unchanged and report.removed == []
    assert fake_embedder.texts_embedded == texts_before
    with session_factory() as session:
        doc = session.scalar(select(Document).where(Document.doc_id == "KB-0001"))
        assert doc.source_path.endswith("vpn-guide.md")


def test_changing_embedding_model_re_embeds_everything(session_factory, fake_embedder, kb):
    make_pipeline(session_factory, fake_embedder).run(kb)

    report = make_pipeline(session_factory, FakeEmbedder(model="another-model")).run(kb)

    assert sorted(report.updated) == ["KB-0001", "POL-0002", "POL-0005"]


def test_force_re_embeds_unchanged_files(session_factory, fake_embedder, kb):
    pipeline = make_pipeline(session_factory, fake_embedder)
    pipeline.run(kb)
    assert len(pipeline.run(kb, force=True).updated) == 3


def test_a_broken_file_does_not_stop_the_others(session_factory, fake_embedder, kb):
    (kb / "policies" / "broken.pdf").write_bytes(b"this is not a pdf")

    report = make_pipeline(session_factory, fake_embedder).run(kb)

    assert len(report.added) == 3
    assert len(report.failed) == 1
    assert report.failed[0][0].endswith("broken.pdf")
    assert report.pruning_skipped


def test_embedding_failure_leaves_previous_version_intact(session_factory, fake_embedder, kb):
    make_pipeline(session_factory, fake_embedder).run(kb)
    chunks_before = count(session_factory, Chunk)

    class FailingEmbedder(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("OpenAI is down")

    report = make_pipeline(session_factory, FailingEmbedder()).run(kb, force=True)

    assert len(report.failed) == 3
    assert count(session_factory, Chunk) == chunks_before


def test_duplicate_doc_ids_are_rejected(session_factory, fake_embedder, kb):
    shutil.copy(kb / "how-to" / "vpn-setup.md", kb / "how-to" / "vpn-copy.md")

    report = make_pipeline(session_factory, fake_embedder).run(kb)

    assert len(report.failed) == 1
    assert "Duplicate doc_id KB-0001" in report.failed[0][1]


def test_dimension_mismatch_is_caught_up_front(session_factory):
    with pytest.raises(ValueError, match="vector\\(1536\\)"):
        make_pipeline(session_factory, FakeEmbedder(dimensions=3072))
