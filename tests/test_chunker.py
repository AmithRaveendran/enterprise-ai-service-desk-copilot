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


# ---------------------------------------------------------------------------
# Overlap with a visible input and the exact output.
#
# Input: one section of 5 short paragraphs. Token counts (tiktoken cl100k_base):
#   "Step 1: Open the VPN app."           9 tokens
#   "Step 2: Sign in with your email."   10 tokens
#   "Step 3: Approve the MFA prompt."    11 tokens
#   "Step 4: Pick the nearest gateway."   9 tokens
#   "Step 5: Click Connect."              7 tokens
# Every chunk also starts with a 13-token header,
# "Document: VPN (KB-9999)\nSection: Connect\n\n", so the body budget is
# max_tokens - 13. Each paragraph costs its tokens + 1 for the "\n\n" separator.
# ---------------------------------------------------------------------------
VPN_STEPS = (
    "Step 1: Open the VPN app.\n\n"
    "Step 2: Sign in with your email.\n\n"
    "Step 3: Approve the MFA prompt.\n\n"
    "Step 4: Pick the nearest gateway.\n\n"
    "Step 5: Click Connect."
)


def vpn_doc() -> SourceDocument:
    return SourceDocument(
        doc_id="KB-9999",
        title="VPN",
        source_path="vpn.md",
        doc_type="markdown",
        category="how-to",
        service=None,
        sections=[Section("Connect", VPN_STEPS)],
    )


def test_overlap_example_repeats_the_last_paragraph():
    # max 46 → body budget 33; overlap up to 12 tokens.
    chunks = Chunker(max_tokens=46, overlap_tokens=12).split(vpn_doc())

    # Chunk 0: steps 1-3 cost 10 + 11 + 12 = 33 = budget. Step 4 doesn't fit → close.
    # Chunk 1: starts with step 3 again (11 tokens ≤ 12 overlap), then steps 4-5.
    assert [c.content for c in chunks] == [
        (
            "Step 1: Open the VPN app.\n\n"
            "Step 2: Sign in with your email.\n\n"
            "Step 3: Approve the MFA prompt."
        ),
        (
            "Step 3: Approve the MFA prompt.\n\n"  # ← overlap: repeated from chunk 0
            "Step 4: Pick the nearest gateway.\n\n"
            "Step 5: Click Connect."
        ),
    ]
    # What actually gets embedded: the header + the body.
    assert chunks[1].embedding_text == (
        "Document: VPN (KB-9999)\nSection: Connect\n\n"
        "Step 3: Approve the MFA prompt.\n\n"
        "Step 4: Pick the nearest gateway.\n\n"
        "Step 5: Click Connect."
    )
    assert [c.token_count for c in chunks] == [43, 40]  # both ≤ 46


def test_overlap_example_skips_a_paragraph_bigger_than_the_overlap():
    # max 42 → body budget 29; overlap only up to 10 tokens.
    chunks = Chunker(max_tokens=42, overlap_tokens=10).split(vpn_doc())

    # Chunk 0: steps 1-2 (10 + 11 = 21); step 3 would make 33 > 29 → close.
    # Chunk 1: step 2 is repeated (10 tokens ≤ 10), then step 3; step 4 would overflow → close.
    # Chunk 2: step 3 is 11 tokens > 10, so it is NOT repeated (overlap is whole
    #          paragraphs, never half a sentence) → starts cleanly at step 4.
    assert [c.content for c in chunks] == [
        "Step 1: Open the VPN app.\n\nStep 2: Sign in with your email.",
        "Step 2: Sign in with your email.\n\nStep 3: Approve the MFA prompt.",  # step 2 repeated
        "Step 4: Pick the nearest gateway.\n\nStep 5: Click Connect.",  # no overlap
    ]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        Chunker(max_tokens=100, overlap_tokens=100)


@pytest.mark.parametrize("path", discover_documents(KB), ids=lambda p: p.name)
def test_real_documents_respect_default_limit(path):
    chunks = Chunker().split(load_document(path))
    assert chunks
    assert all(c.token_count <= 500 for c in chunks)
