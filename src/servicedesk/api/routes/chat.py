"""Chat with the agent, approve or reject its changes, and resume failed turns.

    POST /chat                          send a message (starts or continues a thread)
    POST /chat/{thread_id}/approve      run the change(s) the turn is waiting on
    POST /chat/{thread_id}/reject       don't run them; the assistant is told why
    POST /chat/{thread_id}/resume       finish a turn that failed part-way
    GET  /chat/{thread_id}              the conversation, its status and pending changes

A request the thread's state doesn't allow (a message while a change awaits approval,
approving when nothing is pending...) is a 409 with a machine-readable `code`; an unknown
thread is a 404 (handlers in app.py). If the language model fails mid-turn, the turn is
saved up to the failure and the 502 response says which thread to resume.
"""

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Annotated

import openai
from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.responses import JSONResponse

from servicedesk.agent.copilot import ChatTurn, Copilot, new_thread_id
from servicedesk.api.dependencies import get_copilot
from servicedesk.api.schemas import (
    THREAD_ID_PATTERN,
    ChatHistory,
    ChatRequest,
    ChatResponse,
    CitationOut,
    ErrorOut,
    PendingActionOut,
    RejectRequest,
    ToolUseOut,
    Usage,
)

router = APIRouter(tags=["chat"])
CopilotDep = Annotated[Copilot, Depends(get_copilot)]
ThreadPath = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]
ERRORS = {
    404: {"model": ErrorOut, "description": "No conversation with this id."},
    409: {"model": ErrorOut, "description": "The conversation's state doesn't allow this."},
    502: {"model": ErrorOut, "description": "Model error. The turn was saved: resume it."},
}


@router.post("/chat", response_model=ChatResponse, responses=ERRORS)
def chat(request: ChatRequest, copilot: CopilotDep) -> ChatResponse | JSONResponse:
    """Talk to the assistant. Leave thread_id out to start a new conversation; send the
    returned thread_id back with the next message to continue it.

    If the assistant wants to change something (create or update a ticket, add a comment),
    the response has status "awaiting_approval" and lists the pending_actions: nothing has
    changed yet."""
    # Choose the id here, not in Copilot, so even a failed first turn can be resumed.
    thread_id = request.thread_id or new_thread_id()
    return _respond(thread_id, lambda: copilot.chat(request.message, thread_id))


@router.post("/chat/{thread_id}/approve", response_model=ChatResponse, responses=ERRORS)
def approve(thread_id: ThreadPath, copilot: CopilotDep) -> ChatResponse | JSONResponse:
    """Approve the pending change(s): they run, and the assistant finishes its reply."""
    return _respond(thread_id, lambda: copilot.decide(thread_id, approved=True))


@router.post("/chat/{thread_id}/reject", response_model=ChatResponse, responses=ERRORS)
def reject(
    thread_id: ThreadPath, copilot: CopilotDep, request: RejectRequest | None = None
) -> ChatResponse | JSONResponse:
    """Reject the pending change(s): nothing changes, and the assistant is told (with your
    reason, if given) so it can ask what you'd like instead."""
    reason = request.reason if request else None
    return _respond(thread_id, lambda: copilot.decide(thread_id, approved=False, reason=reason))


@router.post("/chat/{thread_id}/resume", response_model=ChatResponse, responses=ERRORS)
def resume(thread_id: ThreadPath, copilot: CopilotDep) -> ChatResponse | JSONResponse:
    """Finish a turn that failed part-way (e.g. after a 502), from its last saved step."""
    return _respond(thread_id, lambda: copilot.resume(thread_id))


@router.get("/chat/{thread_id}", response_model=ChatHistory, responses={404: ERRORS[404]})
def chat_history(thread_id: ThreadPath, copilot: CopilotDep) -> ChatHistory:
    """The conversation so far (the person's messages and the assistant's replies), its
    status, and any changes waiting for approval."""
    thread_status = copilot.status(thread_id)
    if thread_status == "new":
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No conversation with id {thread_id}.")
    return ChatHistory(
        thread_id=thread_id,
        status=thread_status,
        pending_actions=copilot.pending_actions(thread_id),
        messages=copilot.history(thread_id),
    )


def _respond(thread_id: str, run: Callable[[], ChatTurn]) -> ChatResponse | JSONResponse:
    started = time.perf_counter()
    try:
        turn = run()
    except openai.APIError:
        # The SDK already retried with backoff. The checkpoint holds every step that
        # finished, so the client can retry the rest with /resume.
        error = ErrorOut(
            detail=(
                "Language model service error. The conversation was saved up to the failure: "
                f"retry with POST /chat/{thread_id}/resume."
            ),
            code="model_unavailable",
            thread_id=thread_id,
        )
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=error.model_dump())
    return ChatResponse(
        thread_id=turn.thread_id,
        status=turn.status,
        pending_actions=[PendingActionOut(**a) for a in turn.pending_actions],
        reply=turn.reply,
        citations=[CitationOut(**c) for c in turn.citations],
        tools_used=[ToolUseOut(**asdict(t)) for t in turn.tools_used],
        usage=Usage(input_tokens=turn.input_tokens, output_tokens=turn.output_tokens),
        took_ms=round((time.perf_counter() - started) * 1000, 1),
    )
