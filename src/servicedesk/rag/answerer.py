"""Answer a question from the knowledge base, with citations, or refuse.

    question -> retrieve top-k chunks -> evidence gate -> prompt the model -> check citations

Refusal happens at three points, cheapest first:
1. no_relevant_sources: the best chunk is not similar enough to the question. We refuse
   without calling the model at all ("what is the capital of France?").
2. not_in_sources: the sources look related but the model says they don't contain the
   answer ("what is the office wifi password?" finds the wifi article, which never says).
3. no_valid_citations: the model answered but cited nothing we gave it, so the answer
   can't be traced to a source and we don't show it.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from servicedesk.rag.llm import StructuredLLM
from servicedesk.rag.prompts import SYSTEM_PROMPT, DraftAnswer, build_user_prompt
from servicedesk.search.retriever import Retriever, SearchFilters, SearchHit

RefusalReason = Literal["no_relevant_sources", "not_in_sources", "no_valid_citations"]

REFUSAL_MESSAGE = (
    "I couldn't find this in the knowledge base, so I can't give you a reliable answer. "
    "Please contact the Service Desk on extension 4357 or raise a ticket."
)

# [1], [12] and also [1, 2] (models sometimes group them despite the instructions).
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    number: int  # the [n] used in the answer text (1 = first source cited)
    chunk_id: int
    doc_id: str
    title: str
    source_path: str
    section: str
    page: int | None
    content: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    citations: list[Citation] = field(default_factory=list)
    sources_considered: int = 0  # chunks retrieved for this question
    top_similarity: float | None = None  # best cosine similarity among them
    model: str | None = None  # None when we refused before calling the model
    input_tokens: int = 0
    output_tokens: int = 0


def check_citations(text: str, source_count: int) -> tuple[str, list[int]]:
    """Keep [n] markers that point at a real source, remove the rest, and renumber them in
    order of first appearance, so the answer reads [1], [2], [3] rather than [4], [5], [2].

    Returns the new text and the original source numbers in their new order: element 0 is
    the source now called [1]. A model can write "[7]" when we sent 5 sources; showing that
    would be a citation to nothing.
    """
    cited: list[int] = []

    def replace(match: re.Match) -> str:
        valid = [n for n in map(int, match.group(1).split(",")) if 1 <= n <= source_count]
        for n in valid:
            if n not in cited:
                cited.append(n)
        # [1, 2] -> [1][2]; invalid numbers vanish; duplicates within one marker collapse
        new_numbers = dict.fromkeys(cited.index(n) + 1 for n in valid)
        return "".join(f"[{n}]" for n in new_numbers)

    cleaned = CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # space left before punctuation
    return cleaned.strip(), cited


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: StructuredLLM,
        *,
        top_k: int = 5,
        min_similarity: float = 0.35,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity

    def ask(self, question: str, *, filters: SearchFilters | None = None) -> Answer:
        question = question.strip()
        hits = self.retriever.search(question, k=self.top_k, mode="hybrid", filters=filters)
        similarities = [h.similarity for h in hits if h.similarity is not None]
        top_similarity = max(similarities, default=None)
        context = {"question": question, "sources_considered": len(hits)}

        # 1. Evidence gate. RRF scores only say "better than the others", so the gate uses
        # cosine similarity, which says how close the best chunk actually is.
        if top_similarity is None or top_similarity < self.min_similarity:
            return _refusal("no_relevant_sources", **context, top_similarity=top_similarity)

        result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits), DraftAnswer)
        usage = {
            "top_similarity": top_similarity,
            "model": self.llm.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

        # 2. The model read the sources and says the answer isn't there.
        if not result.parsed.answerable:
            return _refusal("not_in_sources", **context, **usage)

        # 3. An answer must be traceable: at least one citation to a source we sent.
        text, cited = check_citations(result.parsed.answer, len(hits))
        if not cited:
            return _refusal("no_valid_citations", **context, **usage)

        return Answer(
            answer=text,
            refused=False,
            citations=[_citation(i, hits[n - 1]) for i, n in enumerate(cited, start=1)],
            **context,
            **usage,
        )


def _refusal(reason: RefusalReason, **fields) -> Answer:
    return Answer(answer=REFUSAL_MESSAGE, refused=True, refusal_reason=reason, **fields)


def _citation(number: int, hit: SearchHit) -> Citation:
    return Citation(
        number=number,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        title=hit.title,
        source_path=hit.source_path,
        section=hit.section,
        page=hit.page,
        content=hit.content,
    )
