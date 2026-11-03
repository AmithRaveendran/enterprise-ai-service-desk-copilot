"""Response models: the exact JSON shape each endpoint returns.

FastAPI validates every response against these and publishes them in the OpenAPI schema
(/docs), so API clients know the contract without reading the code.
"""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "unreachable"]
    documents: int | None = Field(None, description="Documents in the knowledge base.")
    chunks: int | None = Field(None, description="Searchable chunks.")


class SearchResult(BaseModel):
    rank: int = Field(description="1 = best match.")
    chunk_id: int
    doc_id: str = Field(examples=["KB-0001"])
    title: str
    source_path: str
    section: str = Field(description="Heading path, e.g. 'VPN Setup > Troubleshooting'.")
    page: int | None = Field(None, description="PDF page number, if any.")
    content: str
    score: float = Field(description="Sort key: similarity, keyword rank or RRF score.")
    similarity: float | None = Field(None, description="Cosine similarity, if vector search ran.")
    vector_rank: int | None = None
    keyword_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: Literal["vector", "keyword", "hybrid"]
    count: int
    took_ms: float
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How do I reset my password?"])
    category: str | None = Field(None, max_length=50, examples=["how-to"])
    doc_type: Literal["markdown", "html", "pdf"] | None = None
    service: str | None = Field(None, max_length=100, examples=["VPN"])


class CitationOut(BaseModel):
    number: int = Field(description="The [n] marker used in the answer text.")
    chunk_id: int
    doc_id: str = Field(examples=["KB-0002"])
    title: str
    source_path: str
    section: str
    page: int | None = None
    content: str = Field(description="The cited chunk, so a reader can verify the claim.")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    question: str
    answer: str = Field(description="The answer with [n] markers, or a refusal message.")
    refused: bool
    refusal_reason: Literal["no_relevant_sources", "not_in_sources", "no_valid_citations"] | None
    citations: list[CitationOut]
    sources_considered: int = Field(description="Chunks retrieved for the question.")
    top_similarity: float | None = Field(description="Best cosine similarity among them.")
    model: str | None = Field(description="Chat model used; null if refused before calling it.")
    usage: Usage
    took_ms: float
