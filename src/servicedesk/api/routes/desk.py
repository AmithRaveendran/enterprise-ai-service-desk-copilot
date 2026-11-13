"""REST endpoints for service health and tickets.

Thin on purpose: request bodies are the same Pydantic classes the tools use (so validation
is identical), the work is done by ServiceDesk, and DeskErrors become 404/409 responses in
app.py's exception handlers.

The write endpoints accept an optional Idempotency-Key header: send the same key again
(e.g. retrying after a timeout) and you get the first result back instead of a second
ticket or comment. The first result is returned even if the body differs, so use a new key
for a new request.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, status

from servicedesk.api.dependencies import get_desk
from servicedesk.desk.schemas import (
    AddCommentInput,
    CommentInput,
    CreateTicketInput,
    GetServiceStatusInput,
    GetTicketInput,
    SearchTicketsInput,
    ServiceName,
    ServiceStatusList,
    ServiceStatusOut,
    TicketChanges,
    TicketId,
    TicketOut,
    TicketSearchResult,
    UpdateTicketInput,
)
from servicedesk.desk.service import ServiceDesk

router = APIRouter(tags=["service desk"])
Desk = Annotated[ServiceDesk, Depends(get_desk)]
TicketPath = Annotated[TicketId, Path(description="Ticket number, e.g. 1042 or INC-1042.")]
IdempotencyKey = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        pattern=r"^[A-Za-z0-9_.:-]{8,100}$",
        description="Optional. Repeating a request with the same key doesn't repeat the change.",
    ),
]


def _key(header: str | None) -> str | None:
    # Namespaced, so an API client's key can never collide with the agent's keys.
    return f"api:{header}" if header else None


@router.get("/services/status", response_model=ServiceStatusList)
def all_service_status(desk: Desk) -> ServiceStatusList:
    """Current health of every IT service."""
    return desk.get_service_status(GetServiceStatusInput())


@router.get("/services/{service}/status", response_model=ServiceStatusOut)
def service_status(desk: Desk, service: Annotated[ServiceName, Path()]) -> ServiceStatusOut:
    """Current health of one service (e.g. VPN, Email, Wi-Fi)."""
    return desk.get_service_status(GetServiceStatusInput(service=service)).services[0]


@router.get("/tickets", response_model=TicketSearchResult)
def search_tickets(desk: Desk, query: Annotated[SearchTicketsInput, Query()]) -> TicketSearchResult:
    """Find tickets. All filters are optional and combined with AND."""
    # Annotated[Model, Query()]: FastAPI reads the model's fields from the query string.
    return desk.search_tickets(query)


@router.post("/tickets", response_model=TicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(
    desk: Desk, ticket: CreateTicketInput, idempotency_key: IdempotencyKey = None
) -> TicketOut:
    """Raise a new ticket. Returns it with its number (INC-...) and SLA targets."""
    return desk.create_ticket(ticket, idempotency_key=_key(idempotency_key))


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
def get_ticket(desk: Desk, ticket_id: TicketPath) -> TicketOut:
    """One ticket with its comments."""
    return desk.get_ticket(GetTicketInput(ticket_id=ticket_id))


@router.patch("/tickets/{ticket_id}", response_model=TicketOut)
def update_ticket(
    desk: Desk,
    ticket_id: TicketPath,
    changes: TicketChanges,
    idempotency_key: IdempotencyKey = None,
) -> TicketOut:
    """Change status, priority, assignee or resolution. Only the fields sent are changed;
    a status change must follow the allowed transitions (409 otherwise)."""
    # exclude_unset: pass on only what the caller sent, so "not sent" stays different
    # from "set to null".
    return desk.update_ticket(
        UpdateTicketInput(ticket_id=ticket_id, **changes.model_dump(exclude_unset=True)),
        idempotency_key=_key(idempotency_key),
    )


@router.post(
    "/tickets/{ticket_id}/comments", response_model=TicketOut, status_code=status.HTTP_201_CREATED
)
def add_comment(
    desk: Desk,
    ticket_id: TicketPath,
    comment: CommentInput,
    idempotency_key: IdempotencyKey = None,
) -> TicketOut:
    """Add a comment. Returns the ticket, since a requester's reply can change its status."""
    return desk.add_comment(
        AddCommentInput(ticket_id=ticket_id, **comment.model_dump()),
        idempotency_key=_key(idempotency_key),
    )
