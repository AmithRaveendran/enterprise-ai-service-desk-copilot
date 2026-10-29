"""Loaders turn a file into a SourceDocument: clean text split into titled sections, plus the
metadata needed for citations.

Each format has its own noise to remove:
- Markdown: YAML front matter (metadata, not content) and table syntax.
- HTML: navigation bars, footers and scripts from the intranet page template.
- PDF: running headers/footers repeated on every page, and lines wrapped by the layout.
"""

import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pypdf import PdfReader

from servicedesk.ingestion.models import Section, SourceDocument

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MD_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
# PDF section headings in our policies look like "2. Classification levels".
PDF_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.]{0,80}$")
WHITESPACE_RE = re.compile(r"\s+")

HTML_BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
HTML_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre"]
# Tags that separate words visually even with no whitespace around them in the source.
HTML_WORD_BREAK_TAGS = {*HTML_BLOCK_TAGS, "br", "div", "ul", "ol", "tr", "td", "th"}


class UnsupportedFileTypeError(ValueError):
    pass


def display_path(path: Path) -> str:
    """Path relative to the working directory when possible, so stored paths are the same on
    every machine. Always uses forward slashes."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def discover_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)


def load_document(path: Path) -> SourceDocument:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(f"No loader for '{path.suffix}' files: {path}")
    return loader(path)


# --- Markdown -----------------------------------------------------------------------------


def load_markdown(path: Path) -> SourceDocument:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    if match := FRONT_MATTER_RE.match(text):
        meta = yaml.safe_load(match.group(1)) or {}
        text = text[match.end() :]

    sections = _markdown_sections(text)
    title = meta.get("title") or (sections[0].heading if sections else path.stem)
    return _build_document(path, "markdown", meta.get("id"), title, meta, sections)


def _markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading) of the headings we are currently under
    buffer: list[str] = []
    in_code_block = False

    def flush() -> None:
        body = _markdown_body(buffer)
        if body:
            sections.append(Section(heading=" > ".join(h for _, h in stack), text=body))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        heading = None if in_code_block else MD_HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def _markdown_body(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        is_table = (
            line.startswith("|")
            and i + 1 < len(lines)
            and MD_TABLE_SEPARATOR_RE.match(lines[i + 1].strip())
        )
        if is_table:
            header = _split_markdown_row(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_markdown_row(lines[i].strip()))
                i += 1
            out.extend(_table_rows_to_text(header, rows))
            continue
        out.append(lines[i].rstrip())
        i += 1
    body = "\n".join(out).replace("**", "")  # bold markers are noise for embeddings
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


# --- HTML ---------------------------------------------------------------------------------


def load_html(path: Path) -> SourceDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    meta = {
        m["name"]: m["content"]
        for m in soup.find_all("meta", attrs={"name": True, "content": True})
    }
    page_title = _clean(soup.title.get_text()) if soup.title else None

    for tag in soup(HTML_BOILERPLATE_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.body or soup

    sections = _html_sections(root)
    title = page_title or (sections[0].heading if sections else path.stem)
    return _build_document(path, "html", meta.get("kb-id"), title, meta, sections)


def _html_sections(root: Tag) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    blocks: list[str] = []
    previous_was_list_item = False

    def flush() -> None:
        body = "\n\n".join(blocks).strip()
        if body:
            sections.append(Section(heading=" > ".join(h for _, h in stack), text=body))
        blocks.clear()

    for el in root.find_all(HTML_BLOCK_TAGS):
        if el.find_parent(["li", "table", "pre"]):
            continue  # its text is already included by the enclosing element
        is_list_item = el.name == "li"

        if el.name[0] == "h" and el.name[1:].isdigit():
            flush()
            level = int(el.name[1:])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _html_text(el)))
        elif el.name == "table":
            blocks.append(_html_table(el))
        elif is_list_item:
            item = _html_list_item(el)
            if previous_was_list_item:
                blocks[-1] += "\n" + item  # keep a list together as one paragraph
            else:
                blocks.append(item)
        elif el.name == "pre":
            blocks.append(el.get_text())
        else:
            blocks.append(_html_text(el))
        previous_was_list_item = is_list_item
    flush()
    return sections


def _html_text(el: Tag) -> str:
    """Visible text of an element. Unlike get_text(" "), inline tags (<strong>, <code>, <a>)
    get no extra spaces, so "<strong>This PC</strong>, choose" stays "This PC, choose"; block
    tags and <br> still separate words."""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif node.name in HTML_WORD_BREAK_TAGS:
            parts.append(" ")
    return _clean("".join(parts))


def _html_list_item(li: Tag) -> str:
    text = _html_text(li)
    parent = li.find_parent(["ol", "ul"])
    if parent is not None and parent.name == "ol":
        return f"{len(li.find_previous_siblings('li')) + 1}. {text}"
    return f"- {text}"


def _html_table(table: Tag) -> str:
    rows = [[_html_text(cell) for cell in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
    rows = [r for r in rows if r]
    if not rows:
        return ""
    first_row_is_header = table.find("tr").find("th") is not None
    if first_row_is_header:
        return "\n".join(_table_rows_to_text(rows[0], rows[1:]))
    return "\n".join(" | ".join(r) for r in rows)


# --- PDF ----------------------------------------------------------------------------------


def load_pdf(path: Path) -> SourceDocument:
    reader = PdfReader(path)
    info = reader.metadata
    meta: dict = {}
    if info:
        meta = {
            k: v
            for k, v in {
                "title": info.title,
                "id": info.subject,
                "owner": info.author,
            }.items()
            if v
        }
        # build_pdfs.py stores "category=policy; service=Security" in the Keywords field
        for pair in (info.get("/Keywords") or "").split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                meta[key.strip()] = value.strip()

    pages = [
        [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        for page in reader.pages
    ]
    title = meta.get("title") or path.stem
    sections = _pdf_sections(pages, title)
    return _build_document(path, "pdf", meta.get("id"), title, meta, sections)


def _pdf_sections(pages: list[list[str]], title: str) -> list[Section]:
    boilerplate = _repeated_lines(pages)
    sections: list[Section] = []
    heading = title
    start_page = 1
    paragraphs: list[str] = []

    def flush() -> None:
        body = "\n".join(paragraphs).strip()
        if body:
            full_heading = title if heading == title else f"{title} > {heading}"
            sections.append(Section(heading=full_heading, text=body, page=start_page))
        paragraphs.clear()

    for page_no, lines in enumerate(pages, start=1):
        for line in lines:
            if _normalise_digits(line) in boilerplate:
                continue
            if line == title and not sections and not paragraphs:
                continue  # the document title printed at the top of page 1
            if PDF_HEADING_RE.match(line):
                flush()
                heading, start_page = line, page_no
            else:
                _append_unwrapped(paragraphs, line)
    flush()
    return sections


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """Lines that appear on every page are running headers/footers, not content.

    Digits are normalised first so "Page 1/2" and "Page 2/2" count as the same line.
    Needs at least two pages to tell boilerplate from content.
    """
    if len(pages) < 2:
        return set()
    counts = Counter(
        normalised for lines in pages for normalised in {_normalise_digits(ln) for ln in lines}
    )
    return {line for line, n in counts.items() if n == len(pages)}


def _normalise_digits(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def _append_unwrapped(paragraphs: list[str], line: str) -> None:
    """PDF text extraction breaks lines where the page layout wrapped them. Re-join a line to
    the previous one unless it starts a bullet or the previous line ended a sentence."""
    if (
        paragraphs
        and not line.startswith("- ")
        and not paragraphs[-1].endswith((".", ":", "?", "!"))
    ):
        paragraphs[-1] += " " + line
    else:
        paragraphs.append(line)


# --- Shared helpers -----------------------------------------------------------------------


def _table_rows_to_text(header: list[str], rows: list[list[str]]) -> list[str]:
    """Render each table row as "Column: value; Column: value".

    A chunk may hold only part of a table, so every row carries its own column names instead
    of relying on a header row that might be in a different chunk.
    """
    return ["; ".join(f"{h}: {v}" for h, v in zip(header, row, strict=False) if v) for row in rows]


def _clean(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _jsonable(meta: dict) -> dict:
    """YAML parses dates into date objects; the JSONB metadata column needs plain strings."""
    return {k: v.isoformat() if isinstance(v, date | datetime) else v for k, v in meta.items()}


def _build_document(
    path: Path,
    doc_type: str,
    doc_id: str | None,
    title: str,
    meta: dict,
    sections: list[Section],
) -> SourceDocument:
    title = _clean(str(title))
    for section in sections:
        if not section.heading:  # text that appeared before the first heading
            section.heading = title
    return SourceDocument(
        doc_id=str(doc_id or path.stem),
        title=title,
        source_path=display_path(path),
        doc_type=doc_type,
        category=meta.get("category"),
        service=meta.get("service"),
        sections=sections,
        metadata=_jsonable(meta),
    )


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}
