from pathlib import Path

import pytest

from servicedesk.ingestion.loaders import (
    UnsupportedFileTypeError,
    discover_documents,
    load_document,
)

KB = Path(__file__).resolve().parent.parent / "data" / "kb"
BOILERPLATE = ["INTERNAL", "Page 1/2", "Search the intranet", "Give feedback", "analytics.js"]


def all_text(doc) -> str:
    return "\n".join(s.text for s in doc.sections)


def section(doc, suffix: str):
    return next(s for s in doc.sections if s.heading.endswith(suffix))


def test_discovers_every_supported_file():
    paths = discover_documents(KB)
    assert {p.suffix for p in paths} == {".md", ".html", ".pdf"}
    assert len(paths) == 19


@pytest.mark.parametrize("path", discover_documents(KB), ids=lambda p: p.name)
def test_every_document_loads_cleanly(path):
    doc = load_document(path)
    assert doc.doc_id and doc.title and doc.sections
    assert doc.category in {"how-to", "policy", "known-issue"}
    assert not any(b in all_text(doc) for b in BOILERPLATE)


def test_markdown_front_matter_and_heading_paths():
    doc = load_document(KB / "how-to" / "vpn-setup.md")
    assert doc.doc_id == "KB-0001"
    assert doc.service == "VPN"
    assert doc.source_path.endswith("data/kb/how-to/vpn-setup.md")
    assert doc.metadata["last_updated"] == "2026-08-14"  # date converted to a JSON string
    assert "---" not in all_text(doc)
    troubleshooting = section(doc, "> Troubleshooting")
    assert troubleshooting.heading.startswith("Connecting to the Corporate VPN")


def test_markdown_tables_become_self_describing_rows():
    doc = load_document(KB / "how-to" / "vpn-setup.md")
    text = section(doc, "> Troubleshooting").text
    assert 'Symptom: "Gateway unreachable"; Likely cause:' in text
    assert "|---" not in text


def test_markdown_ignores_hash_lines_inside_code_blocks(tmp_path):
    md = tmp_path / "note.md"
    md.write_text("# Title\n\nIntro\n\n```bash\n# not a heading\n```\n\n## Next\n\nBody\n")
    doc = load_document(md)
    assert [s.heading for s in doc.sections] == ["Title", "Title > Next"]
    assert "# not a heading" in doc.sections[0].text


def test_html_strips_template_and_keeps_tables():
    doc = load_document(KB / "policies" / "incident-severity-and-sla.html")
    assert doc.doc_id == "POL-0002"
    assert doc.title == "Incident Severity Levels and Service Level Agreements"
    sla = section(doc, "> Service level targets").text
    assert "Priority: P1 – Critical; First response: 15 minutes; Resolution target: 4 hours" in sla


def test_html_numbers_ordered_list_items():
    doc = load_document(KB / "how-to" / "shared-drive-access.html")
    steps = section(doc, "> How to request access").text
    assert steps.startswith("1. Open the Service Desk portal")
    assert "\n5. The data owner approves" in steps


def test_pdf_metadata_sections_and_pages():
    doc = load_document(KB / "policies" / "information-security-policy.pdf")
    assert doc.doc_id == "POL-0005"
    assert doc.category == "policy"
    assert doc.service == "Security"
    headings = [s.heading.split(" > ")[-1] for s in doc.sections]
    assert headings[0] == "1. Purpose"
    assert "2. Classification levels" in headings
    assert section(doc, "6. Exceptions").page == 2


def test_pdf_rejoins_lines_wrapped_by_the_layout():
    doc = load_document(KB / "policies" / "information-security-policy.pdf")
    purpose = section(doc, "1. Purpose").text
    assert "\n" not in purpose  # one paragraph, even though the PDF wrapped it over 2 lines


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)
