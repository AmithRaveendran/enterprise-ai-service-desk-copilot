import time
from dataclasses import asdict
from typing import Annotated

import openai
from fastapi import APIRouter, Depends, HTTPException, Path, status

from servicedesk.agent.copilot import Copilot
from servicedesk.api.dependencies import get_copilot
from servicedesk.api.schemas import (
    THREAD_ID_PATTERN,
    ChatHistory,
    ChatRequest,
    ChatResponse,
    CitationOut,
    ToolUseOut,
    Usage,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, copilot: Annotated[Copilot, Depends(get_copilot)]) -> ChatResponse:
    """Talk to the assistant. Leave thread_id out to start a new conversation; send the
    returned thread_id back with the next message to continue it."""
    started = time.perf_counter()
    try:
        turn = copilot.chat(request.message, request.thread_id)
    except openai.APIError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Language model service error.") from exc
    return ChatResponse(
        thread_id=turn.thread_id,
        reply=turn.reply,
        citations=[CitationOut(**c) for c in turn.citations],
        tools_used=[ToolUseOut(**asdict(t)) for t in turn.tools_used],
        usage=Usage(input_tokens=turn.input_tokens, output_tokens=turn.output_tokens),
        took_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.get("/chat/{thread_id}", response_model=ChatHistory)
def chat_history(
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)],
    copilot: Annotated[Copilot, Depends(get_copilot)],
) -> ChatHistory:
    """The conversation so far: the person's messages and the assistant's replies."""
    messages = copilot.history(thread_id)
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No conversation with id {thread_id}.")
    return ChatHistory(thread_id=thread_id, messages=messages)
