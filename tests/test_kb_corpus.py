"""Data-quality checks on the sample knowledge base.

These run before any ingestion code exists: if a document is malformed or is missing
metadata, we want to find out here rather than as a confusing retrieval bug later.
"""

import re
from pathlib import Path

import pytest
import yaml

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
REQUIRED_MD_KEYS = {"id", "title", "category", "service", "owner", "last_updated"}
REQUIRED_HTML_META = {"kb-id", "category", "service", "owner", "last-updated"}
ID_PATTERN = re.compile(r"^(KB|KI|POL)-\d{4}$")

md_files = sorted(KB_DIR.rglob("*.md"))
html_files = sorted(KB_DIR.rglob("*.html"))
pdf_files = sorted(KB_DIR.rglob("*.pdf"))


def md_front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path.name} has no front matter"
    return yaml.safe_load(text.split("---", 2)[1])


def html_meta(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return dict(re.findall(r'<meta name="([^"]+)" content="([^"]+)"', text))


def test_corpus_has_mixed_formats():
    assert len(md_files) >= 10
    assert len(html_files) >= 3
    assert len(pdf_files) >= 3, "Run `uv run python scripts/build_pdfs.py` to generate PDFs"


@pytest.mark.parametrize("path", md_files, ids=lambda p: p.name)
def test_markdown_front_matter(path):
    meta = md_front_matter(path)
    assert meta.keys() >= REQUIRED_MD_KEYS
    assert ID_PATTERN.match(meta["id"])


@pytest.mark.parametrize("path", html_files, ids=lambda p: p.name)
def test_html_metadata(path):
    meta = html_meta(path)
    assert meta.keys() >= REQUIRED_HTML_META
    assert ID_PATTERN.match(meta["kb-id"])


def test_document_ids_are_unique():
    ids = [md_front_matter(p)["id"] for p in md_files]
    ids += [html_meta(p)["kb-id"] for p in html_files]
    ids += [p.stem for p in pdf_files]  # PDFs carry their id in PDF metadata; stems are unique
    assert len(ids) == len(set(ids))
