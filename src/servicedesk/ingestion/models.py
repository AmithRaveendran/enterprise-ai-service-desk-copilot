"""In-memory representation of a parsed document, shared by every loader."""

from dataclasses import dataclass, field


@dataclass
class Section:
    """A run of text under one heading.

    heading is the full path of headings, e.g. "Connecting to the VPN > Troubleshooting",
    so a chunk taken from deep inside a document still says where it came from.
    """

    heading: str
    text: str
    page: int | None = None  # only known for PDFs


@dataclass
class SourceDocument:
    doc_id: str  # stable business id, e.g. "KB-0001"; survives file renames
    title: str
    source_path: str  # POSIX path relative to the project root
    doc_type: str  # "markdown" | "html" | "pdf"
    category: str | None
    service: str | None
    sections: list[Section]
    metadata: dict = field(default_factory=dict)
