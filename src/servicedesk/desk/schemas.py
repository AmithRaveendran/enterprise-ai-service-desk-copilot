"""Inputs and outputs of the service desk tools, with all validation in one place.

Every way in (REST API, CLI, the LangGraph agent in Phase 5, the MCP server in Phase 7)
validates through these classes, so a rule written here is enforced everywhere. Input
from a language model is treated like input from the internet: untrusted until validated.

Field descriptions matter: they become the JSON schema an LLM reads to decide how to call
a tool, and the API docs at /docs.
"""

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

COMPANY_EMAIL_DOMAIN = "halcyon-dynamics.example"


class ServiceHealth(StrEnum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"


class Category(StrEnum):
    ACCESS = "access"
    COLLABORATION = "collaboration"
    EMAIL = "email"
    HARDWARE = "hardware"
    NETWORK = "network"
    SECURITY = "security"
    SOFTWARE = "software"
    OTHER = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class TicketStatus(StrEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_REQUESTER = "waiting_for_requester"  # the SLA clock pauses
    RESOLVED = "resolved"
    CLOSED = "closed"


# From the Incident Severity and SLA policy (data/kb/policies/incident-severity-and-sla.html).
SLA_TARGETS: dict[Priority, tuple[str, str]] = {
    Priority.P1: ("15 minutes", "4 hours"),
    Priority.P2: ("30 minutes", "8 hours"),
    Priority.P3: ("4 business hours", "3 business days"),
    Priority.P4: ("1 business day", "10 business days"),
}


# --- reusable validated types -------------------------------------------------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t \n \r
_EMAIL = re.compile(rf"^[a-z0-9._%+-]+@{re.escape(COMPANY_EMAIL_DOMAIN)}$")
_TICKET_NUMBER = re.compile(r"^\s*(?:INC[-\s]?)?(\d+)\s*$", re.IGNORECASE)


def _clean_text(value: object) -> object:
    """Strip invisible control characters and surrounding space; keep line breaks.
    Non-strings pass through untouched, so the normal type check reports them."""
    if not isinstance(value, str):
        return value
    value = _CONTROL_CHARS.sub("", value).replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", value)


def _single_line(value: object) -> object:
    value = _clean_text(value)
    return re.sub(r"\s+", " ", value) if isinstance(value, str) else value


def _company_email(value: str) -> str:
    value = value.strip().lower()
    if not _EMAIL.match(value):
        raise ValueError(f"must be a company email address (name@{COMPANY_EMAIL_DOMAIN})")
    return value


def _ticket_number(value: object) -> object:
    """Accept 1042, "1042", "INC-1042" or "inc 1042": people and LLMs write all of these."""
    if isinstance(value, str) and (match := _TICKET_NUMBER.match(value)):
        return int(match.group(1))
    return value  # anything else goes on to normal int validation (and fails clearly)


# Annotated = a type plus rules. Each item wraps the ones before it, so in
# Annotated[str, Field(...), BeforeValidator(f)] the call order is: f on the raw input →
# str check with the Field limits. Cleaning therefore runs *before* the length check
# ("   " is rejected as empty, with "at least 5 characters"). Field must come first: after
# a validator, Pydantic can only apply a generic length check with a vaguer message.
Title = Annotated[str, Field(min_length=5, max_length=200), BeforeValidator(_single_line)]
LongText = Annotated[str, Field(min_length=1, max_length=5000), BeforeValidator(_clean_text)]
Email = Annotated[
    str,
    AfterValidator(_company_email),
    Field(max_length=254, examples=[f"jane.doe@{COMPANY_EMAIL_DOMAIN}"]),
]
TicketId = Annotated[
    int,
    Field(ge=1, description="Ticket number, e.g. 1042 or INC-1042.", examples=[1042]),
    BeforeValidator(_ticket_number),
]
ServiceName = Annotated[
    str,
    Field(min_length=2, max_length=50, description="Service name, e.g. VPN, Email, Wi-Fi."),
    BeforeValidator(_single_line),
]


class StrictInput(BaseModel):
    # extra="forbid": an unknown field is an error, not silently ignored. If an LLM invents
    # an argument ("urgency": "high"), it hears about it and can correct the call.
    model_config = ConfigDict(extra="forbid")


# --- tool inputs --------------------------------------------------------------------------


class GetServiceStatusInput(StrictInput):
    service: ServiceName | None = Field(
        None, description="One service (e.g. VPN). Leave empty for all services."
    )


class CreateTicketInput(StrictInput):
    title: Title = Field(description="Short summary, e.g. 'VPN disconnects every few minutes'.")
    description: LongText = Field(
        description="What happened, error messages, what was already tried."
    )
    category: Category
    priority: Priority = Field(
        Priority.P3,
        description="P1 critical/many users, P2 high/no workaround, P3 one user with a "
        "workaround, P4 questions and requests.",
    )
    requester: Email = Field(description="The employee the ticket is for.")
    service: ServiceName | None = Field(None, description="Affected service, if known.")


class SearchTicketsInput(StrictInput):
    requester: Email | None = None
    status: TicketStatus | None = None
    open_only: bool = Field(False, description="Only tickets that are not resolved or closed.")
    priority: Priority | None = None
    service: ServiceName | None = None
    text: Annotated[str, Field(max_length=100), BeforeValidator(_single_line)] | None = Field(
        None, description="Words to look for in the title or description."
    )
    limit: int = Field(10, ge=1, le=50)


class GetTicketInput(StrictInput):
    ticket_id: TicketId


class TicketChanges(StrictInput):
    """The fields of a ticket that can be changed. Only the fields you send are changed."""

    status: TicketStatus | None = None
    priority: Priority | None = None
    assignee: Email | None = None
    resolution: LongText | None = Field(
        None, description="Required when setting status to resolved."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        # model_fields_set = the fields the caller actually sent (not defaults). ticket_id
        # (in UpdateTicketInput) doesn't count: it says which ticket, not what to change.
        if not self.model_fields_set & set(TicketChanges.model_fields):
            raise ValueError("send at least one field to change")
        if self.status == TicketStatus.RESOLVED and not self.resolution:
            raise ValueError("a resolution is required when resolving a ticket")
        return self


class UpdateTicketInput(TicketChanges):
    ticket_id: TicketId


class CommentInput(StrictInput):
    author: Email
    body: LongText


class AddCommentInput(CommentInput):
    ticket_id: TicketId


# --- outputs ------------------------------------------------------------------------------


class ServiceStatusOut(BaseModel):
    id: str
    name: str
    status: ServiceHealth
    message: str | None
    criticality: str
    owning_team: str
    updated_at: datetime


class ServiceStatusList(BaseModel):
    services: list[ServiceStatusOut]


class Sla(BaseModel):
    first_response: str
    resolution: str


class CommentOut(BaseModel):
    id: int
    author: str
    body: str
    created_at: datetime


class TicketSummary(BaseModel):
    id: int
    number: str = Field(examples=["INC-1042"])
    title: str
    category: Category
    priority: Priority
    status: TicketStatus
    requester: str
    service: str | None
    updated_at: datetime


class TicketOut(TicketSummary):
    description: str
    assignee: str | None
    resolution: str | None
    created_at: datetime
    resolved_at: datetime | None
    sla: Sla
    comments: list[CommentOut]
    notice: str | None = Field(
        None, description="Something the requester must do, e.g. phone in a P1."
    )


class TicketSearchResult(BaseModel):
    count: int
    tickets: list[TicketSummary]
