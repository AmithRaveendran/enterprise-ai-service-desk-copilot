"""Search the knowledge base three ways: vector (semantic), keyword (full-text) and hybrid.

- Vector search embeds the question and finds chunks whose embeddings point the same way
  (cosine similarity). Good at meaning: "I can't sign in" finds "Resetting your password".
- Keyword search uses Postgres full-text search. Good at exact terms that embeddings blur:
  product names, error messages, IDs such as "KI-0007".
- Hybrid runs both and merges the two rankings with Reciprocal Rank Fusion (RRF), so a
  chunk found by both rises to the top.
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Select, Text, cast, func, select, text
from sqlalchemy.orm import Session, defer, sessionmaker

from servicedesk.db.models import Chunk, Document
from servicedesk.ingestion.embeddings import ConfigurationError, Embedder

SearchMode = Literal["vector", "keyword", "hybrid"]

# The RRF constant from the original paper (Cormack et al., 2009). Larger values flatten
# the difference between rank 1 and rank 10; 60 works well across many datasets.
RRF_K = 60


@dataclass(frozen=True)
class SearchFilters:
    category: str | None = None
    doc_type: str | None = None
    service: str | None = None


@dataclass
class SearchHit:
    chunk_id: int
    doc_id: str
    title: str
    source_path: str
    section: str
    page: int | None
    content: str
    score: float  # what results are sorted by: similarity, keyword rank or RRF score
    similarity: float | None = None  # cosine similarity, when vector search ran
    vector_rank: int | None = None  # position in the vector ranking (1 = best)
    keyword_rank: int | None = None  # position in the keyword ranking (1 = best)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Merge ranked lists of ids. Each list gives an id 1 / (k + rank), summed over lists.

    Only positions are used, never the raw scores, so a cosine similarity (0..1) and a
    keyword rank (unbounded) can be combined without guessing how to scale them.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: Embedder | None,
        candidates: int = 20,
    ):
        self.session_factory = session_factory
        self.embedder = embedder  # None is fine for keyword-only search (no API key needed)
        self.candidates = candidates  # how many results each method contributes to hybrid

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: SearchMode = "hybrid",
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        filters = filters or SearchFilters()
        limit = max(k, self.candidates)

        with self.session_factory() as session:
            vector = (
                self._vector_ranking(session, query, limit, filters) if mode != "keyword" else []
            )
            keyword = (
                self._keyword_ranking(session, query, limit, filters) if mode != "vector" else []
            )

            if mode == "vector":
                scores = dict(vector)
            elif mode == "keyword":
                scores = dict(keyword)
            else:
                scores = reciprocal_rank_fusion([[i for i, _ in vector], [i for i, _ in keyword]])

            top = sorted(scores, key=lambda i: (-scores[i], i))[:k]  # id breaks ties: stable
            return self._load_hits(session, top, scores, vector, keyword)

    def _vector_ranking(
        self, session: Session, query: str, limit: int, filters: SearchFilters
    ) -> list[tuple[int, float]]:
        """(chunk id, cosine similarity) pairs, most similar first."""
        if self.embedder is None:
            raise ConfigurationError("Vector search needs an embedding API key; use keyword mode.")
        query_vector = self.embedder.embed([query])[0]

        distance = Chunk.embedding.cosine_distance(query_vector)  # SQL: embedding <=> :vector
        stmt = select(Chunk.id, (1 - distance).label("similarity")).join(Document)
        stmt = _apply_filters(stmt, filters).order_by(distance).limit(limit)

        if filters != SearchFilters():
            # HNSW finds the nearest vectors first and applies WHERE afterwards, so a filter
            # can leave fewer than `limit` rows. Iterative scan keeps searching the graph until
            # enough rows pass the filter (pgvector 0.8+). SET LOCAL lasts for this transaction.
            session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
        return [(row.id, float(row.similarity)) for row in session.execute(stmt)]

    def _keyword_ranking(
        self, session: Session, query: str, limit: int, filters: SearchFilters
    ) -> list[tuple[int, float]]:
        """(chunk id, full-text rank) pairs, best match first."""
        # websearch_to_tsquery parses free text safely (never a syntax error) and normalises
        # words the same way the search_vector column was built: "Printers" -> 'printer'.
        # It ANDs the words, which is too strict for questions ("why is my vpn so slow" would
        # need every word), so '&' is swapped for '|': match ANY word, and ts_rank scores
        # chunks that contain more of them (and in headings, weight A) higher.
        parsed = cast(func.websearch_to_tsquery("english", query), Text)
        tsquery = func.to_tsquery("english", func.replace(parsed, "&", "|"))
        rank = func.ts_rank(Chunk.search_vector, tsquery)

        stmt = select(Chunk.id, rank.label("rank")).join(Document)
        stmt = _apply_filters(stmt, filters).where(Chunk.search_vector.op("@@")(tsquery))
        stmt = stmt.order_by(rank.desc(), Chunk.id).limit(limit)
        return [(row.id, float(row.rank)) for row in session.execute(stmt)]

    def _load_hits(
        self,
        session: Session,
        ids: list[int],
        scores: dict[int, float],
        vector: list[tuple[int, float]],
        keyword: list[tuple[int, float]],
    ) -> list[SearchHit]:
        if not ids:
            return []
        rows = session.execute(
            select(Chunk, Document)
            .join(Document)
            .where(Chunk.id.in_(ids))
            .options(defer(Chunk.embedding))  # 1536 floats per row we don't need here
        ).all()
        by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}
        similarity = dict(vector)
        vector_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector, start=1)}
        keyword_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword, start=1)}

        hits = []
        for chunk_id in ids:  # keep the ranked order, not the database's order
            chunk, doc = by_id[chunk_id]
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_path=doc.source_path,
                    section=chunk.section,
                    page=chunk.page,
                    content=chunk.content,
                    score=scores[chunk_id],
                    similarity=similarity.get(chunk_id),
                    vector_rank=vector_rank.get(chunk_id),
                    keyword_rank=keyword_rank.get(chunk_id),
                )
            )
        return hits


def _apply_filters(stmt: Select, filters: SearchFilters) -> Select:
    if filters.category:
        stmt = stmt.where(Document.category == filters.category)
    if filters.doc_type:
        stmt = stmt.where(Document.doc_type == filters.doc_type)
    if filters.service:
        stmt = stmt.where(Document.service == filters.service)
    return stmt
