from pathlib import Path

import pytest

from servicedesk.ingestion.chunker import Chunker
from servicedesk.ingestion.loaders import discover_documents, load_document
from servicedesk.ingestion.models import Section, SourceDocument

KB = Path(__file__).resolve().parent.parent / "data" / "kb"


def make_doc(*sections: Section) -> SourceDocument:
    return SourceDocument(
        doc_id="KB-9999",
        title="Test Doc",
        source_path="test.md",
        doc_type="markdown",
        category="how-to",
        service=None,
        sections=list(sections),
    )


def numbered_paragraphs(n: int) -> str:
    return "\n\n".join(
        f"Paragraph {i}: the VPN gateway rejects the connection when the device certificate "
        f"has expired, so the user must reconnect on the office network first."
        for i in range(n)
    )


def test_short_section_is_one_chunk_with_context_header():
    chunks = Chunker().split(make_doc(Section("Test Doc > Setup", "Install the app.")))
    assert len(chunks) == 1
    assert chunks[0].content == "Install the app."
    assert chunks[0].embedding_text.startswith("Document: Test Doc (KB-9999)\nSection: Test Doc")
    assert chunks[0].index == 0


def test_chunks_never_span_sections():
    doc = make_doc(Section("A", "First section text."), Section("B", "Second section text."))
    chunks = Chunker().split(doc)
    assert [c.section for c in chunks] == ["A", "B"]
    assert [c.index for c in chunks] == [0, 1]


def test_long_section_is_split_within_the_token_limit():
    chunker = Chunker(max_tokens=120, overlap_tokens=40)
    chunks = chunker.split(make_doc(Section("Long", numbered_paragraphs(12))))
    assert len(chunks) > 1
    assert all(c.token_count <= 120 for c in chunks)
    assert all(c.token_count == chunker.count_tokens(c.embedding_text) for c in chunks)
    # every paragraph survives somewhere
    for i in range(12):
        assert any(f"Paragraph {i}:" in c.content for c in chunks)


def test_consecutive_chunks_overlap():
    chunker = Chunker(max_tokens=120, overlap_tokens=40)
    chunks = chunker.split(make_doc(Section("Long", numbered_paragraphs(12))))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_paragraph_of_previous = previous.content.split("\n\n")[-1]
        assert current.content.startswith(last_paragraph_of_previous)


def test_a_single_huge_line_is_hard_split():
    chunker = Chunker(max_tokens=60, overlap_tokens=0)
    chunks = chunker.split(make_doc(Section("Wall", "word " * 500)))
    assert len(chunks) > 1
    assert all(c.token_count <= 60 for c in chunks)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        Chunker(max_tokens=100, overlap_tokens=100)


@pytest.mark.parametrize("path", discover_documents(KB), ids=lambda p: p.name)
def test_real_documents_respect_default_limit(path):
    chunks = Chunker().split(load_document(path))
    assert chunks
    assert all(c.token_count <= 500 for c in chunks)
