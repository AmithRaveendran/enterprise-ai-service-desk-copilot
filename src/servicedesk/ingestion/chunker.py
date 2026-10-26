"""Split documents into token-bounded chunks for embedding.

Why chunk at all? An embedding is one vector for a whole piece of text. Embed a 3-page
policy as one vector and the specific answer gets diluted by everything else on those
pages. Small, focused chunks give sharper search results, and they also let us send the
LLM only the relevant passages instead of whole documents.

Strategy:
1. Chunk within a section, never across sections, so each chunk has one clear heading.
2. Prefer natural boundaries: paragraphs, then lines, then raw token windows as a last resort.
3. Overlap consecutive chunks of a long section slightly, so an idea that straddles a
   boundary can be found from either side.
4. Prepend a contextual header (document title + section) to the text that gets embedded.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

import tiktoken

from servicedesk.ingestion.models import SourceDocument

# text-embedding-3-* models use the cl100k_base tokenizer.
DEFAULT_ENCODING = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    index: int  # position within the document, 0-based
    section: str
    page: int | None
    content: str  # the passage itself, shown to users and the LLM
    embedding_text: str  # header + content: what we actually embed
    token_count: int  # tokens in embedding_text


class Chunker:
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be >= 0 and smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def split(self, doc: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in doc.sections:
            header = f"Document: {doc.title} ({doc.doc_id})\nSection: {section.heading}\n\n"
            budget = self.max_tokens - self.count_tokens(header)
            if budget <= 0:
                raise ValueError(f"Heading too long to fit in a chunk: {section.heading!r}")
            for body in self._split_text(section.text, budget):
                embedding_text = header + body
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        section=section.heading,
                        page=section.page,
                        content=body,
                        embedding_text=embedding_text,
                        token_count=self.count_tokens(embedding_text),
                    )
                )
        return chunks

    def _split_text(self, text: str, budget: int) -> list[str]:
        units = [(unit, self.count_tokens(unit)) for unit in self._units(text, budget)]
        return self._pack(units, budget)

    def _units(self, text: str, budget: int) -> Iterator[str]:
        """Break text into pieces that each fit the budget, using the largest natural unit
        that fits: a paragraph, else a line, else a raw window of tokens."""
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self.count_tokens(paragraph) <= budget:
                yield paragraph
                continue
            for line in paragraph.splitlines():
                line = line.strip()
                if not line:
                    continue
                if self.count_tokens(line) <= budget:
                    yield line
                else:
                    yield from self._hard_split(line, budget)

    def _hard_split(self, text: str, budget: int) -> Iterator[str]:
        tokens = self._encoding.encode(text)
        for start in range(0, len(tokens), budget):
            yield self._encoding.decode(tokens[start : start + budget])

    def _pack(self, units: list[tuple[str, int]], budget: int) -> list[str]:
        """Greedily fill chunks up to the budget. When a chunk is full, the next one starts
        with the trailing units of the previous one (up to overlap_tokens)."""
        chunks: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []
        size = 0
        for text, n in units:
            cost = n + 1  # +1 for the "\n\n" separator between units
            if current and size + cost > budget:
                chunks.append(current)
                current = self._overlap_tail(current)
                size = sum(c + 1 for _, c in current)
                if size + cost > budget:  # overlap would push us over: start clean
                    current, size = [], 0
            current.append((text, n))
            size += cost
        if current:
            chunks.append(current)
        return ["\n\n".join(text for text, _ in chunk) for chunk in chunks]

    def _overlap_tail(self, units: list[tuple[str, int]]) -> list[tuple[str, int]]:
        tail: list[tuple[str, int]] = []
        size = 0
        for text, n in reversed(units):
            if size + n > self.overlap_tokens:
                break
            tail.insert(0, (text, n))
            size += n
        return tail
