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
