import time
from dataclasses import asdict
from typing import Annotated, Literal

import openai
from fastapi import APIRouter, Depends, HTTPException, Query, status

from servicedesk.api.dependencies import get_retriever
from servicedesk.api.schemas import SearchResponse, SearchResult
from servicedesk.ingestion.embeddings import ConfigurationError
from servicedesk.search.retriever import Retriever, SearchFilters, SearchMode

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    # Query(...) rules are checked before this function runs; a bad request gets a 422
    # response that says which parameter is wrong and why.
    q: Annotated[str, Query(min_length=2, max_length=500, description="The question.")],
    k: Annotated[int, Query(ge=1, le=20, description="Number of results.")] = 5,
    mode: SearchMode = "hybrid",
    category: Annotated[str | None, Query(max_length=50, examples=["how-to"])] = None,
    doc_type: Literal["markdown", "html", "pdf"] | None = None,
    service: Annotated[str | None, Query(max_length=100, examples=["VPN"])] = None,
) -> SearchResponse:
    """Find the knowledge-base chunks that best match a question."""
    filters = SearchFilters(category=category, doc_type=doc_type, service=service)
    started = time.perf_counter()
    try:
        hits = retriever.search(q, k=k, mode=mode, filters=filters)
    except ConfigurationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except openai.APIError as exc:
        # The embedding API failed even after retries: our dependency is down, not the caller's
        # fault, so 502 Bad Gateway rather than 500.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding service error.") from exc
    took_ms = (time.perf_counter() - started) * 1000

    results = [SearchResult(rank=i, **asdict(hit)) for i, hit in enumerate(hits, start=1)]
    return SearchResponse(
        query=q, mode=mode, count=len(results), took_ms=round(took_ms, 1), results=results
    )
