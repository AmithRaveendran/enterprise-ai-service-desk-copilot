import time
from dataclasses import asdict
from typing import Annotated

import openai
from fastapi import APIRouter, Depends, HTTPException, status

from servicedesk.api.dependencies import get_answerer
from servicedesk.api.schemas import AskRequest, AskResponse, CitationOut, Usage
from servicedesk.ingestion.embeddings import ConfigurationError
from servicedesk.rag.answerer import Answerer
from servicedesk.rag.llm import ModelRefusedError
from servicedesk.search.retriever import SearchFilters

router = APIRouter(tags=["answers"])


# POST, not GET: the question goes in a JSON body (it can be long, and questions shouldn't
# end up in URLs, browser history and access logs), and each call costs money to run.
@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,  # a Pydantic body: FastAPI validates the JSON and 422s if it's wrong
    answerer: Annotated[Answerer, Depends(get_answerer)],
) -> AskResponse:
    """Answer an IT question from the knowledge base, citing the sources used.

    Refuses (refused=true, with a reason) instead of guessing when the knowledge base doesn't
    cover the question.
    """
    filters = SearchFilters(
        category=request.category, doc_type=request.doc_type, service=request.service
    )
    started = time.perf_counter()
    try:
        result = answerer.ask(request.question, filters=filters)
    except ConfigurationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ModelRefusedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "The model declined.") from exc
    except openai.APIError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Language model service error.") from exc
    took_ms = (time.perf_counter() - started) * 1000

    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=[CitationOut(**asdict(c)) for c in result.citations],
        sources_considered=result.sources_considered,
        top_similarity=result.top_similarity,
        model=result.model,
        usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        took_ms=round(took_ms, 1),
    )
