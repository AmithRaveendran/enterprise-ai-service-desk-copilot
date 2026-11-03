"""Service desk operations: service health and tickets.

The inputs are already validated (desk/schemas.py); this layer applies the business rules
from the IT policies and talks to the database. It knows nothing about HTTP, the CLI or
language models, so the API, the agent and the MCP server all share exactly this logic.

Business rules (from data/kb/policies/incident-severity-and-sla.html):
- A ticket moves through statuses along ALLOWED_TRANSITIONS only.
- Resolving needs a resolution note. A resolved ticket can be reopened; a closed one can't
  (a new ticket must reference it) and can't be changed or commented on.
- When the requester replies to a ticket that is waiting for them, it goes back to
  in_progress (the SLA clock restarts).
- P1 and P2 tickets must also be phoned in, so their output carries a notice.
"""

import re

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload, sessionmaker

from servicedesk.db.models import Service, Ticket, TicketComment
from servicedesk.desk.schemas import (
    SLA_TARGETS,
    AddCommentInput,
    CommentOut,
    CreateTicketInput,
    GetServiceStatusInput,
    GetTicketInput,
    Priority,
    SearchTicketsInput,
    ServiceHealth,
    ServiceStatusList,
    ServiceStatusOut,
    Sla,
    TicketOut,
    TicketSearchResult,
    TicketStatus,
    TicketSummary,
    UpdateTicketInput,
)

S = TicketStatus
ALLOWED_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    S.NEW: {S.IN_PROGRESS, S.WAITING_FOR_REQUESTER, S.RESOLVED},
    S.IN_PROGRESS: {S.WAITING_FOR_REQUESTER, S.RESOLVED},
    S.WAITING_FOR_REQUESTER: {S.IN_PROGRESS, S.RESOLVED},
    S.RESOLVED: {S.IN_PROGRESS, S.CLOSED},  # reopen, or close
    S.CLOSED: set(),
}

PHONE_NOTICE = (
    "P1 and P2 incidents must also be reported by phone on extension 4357 so the on-call "
    "engineer is paged immediately."
)


class DeskError(Exception):
    """A request that is valid in shape but can't be carried out. `code` is stable and
    machine-readable (API status codes and tool results are based on it)."""

    code = "error"


class NotFoundError(DeskError):
    code = "not_found"


class ConflictError(DeskError):
    """The request conflicts with the ticket's current state, e.g. reopening a closed one."""

    code = "conflict"


def ticket_number(ticket_id: int) -> str:
    return f"INC-{ticket_id}"


def _key(name: str) -> str:
    """ "Wi-Fi", "wifi" and "WIFI" all become "wifi"; "HR Portal" becomes "hrportal"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _like(text: str) -> str:
    """A LIKE pattern matching `text` anywhere. % and _ are wildcards in LIKE, so escape
    them: searching for "100%" should find "100%", not "100 anything"."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class ServiceDesk:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    # --- service health -------------------------------------------------------------------

    def get_service_status(self, args: GetServiceStatusInput) -> ServiceStatusList:
        with self.session_factory() as session:
            if args.service:
                services = [_find_service(session, args.service)]
            else:
                services = session.scalars(select(Service).order_by(Service.name)).all()
            return ServiceStatusList(services=[_status_out(s) for s in services])

    def set_service_status(
        self, service: str, status: ServiceHealth, message: str | None
    ) -> ServiceStatusOut:
        """Admin operation (CLI only). Deliberately not a tool: the assistant can read
        service health but must never change it."""
        with self.session_factory.begin() as session:
            row = _find_service(session, service)
            row.status, row.status_message = status, message
            row.status_updated_at = func.now()
            session.flush()
            session.refresh(row)
            return _status_out(row)

    # --- tickets --------------------------------------------------------------------------

    def create_ticket(self, args: CreateTicketInput) -> TicketOut:
        # sessionmaker.begin(): one transaction, committed when the block ends (or rolled
        # back if it raises), so a half-created ticket can never be saved.
        with self.session_factory.begin() as session:
            service = _find_service(session, args.service) if args.service else None
            ticket = Ticket(
                title=args.title,
                description=args.description,
                category=args.category,
                priority=args.priority,
                status=TicketStatus.NEW,
                requester=args.requester,
                service=service,
            )
            session.add(ticket)
            session.flush()  # INSERT now, so Postgres assigns the id (INC-1001...)
            session.refresh(ticket)  # load the values Postgres filled in (created_at...)
            return _ticket_out(ticket)

    def get_ticket(self, args: GetTicketInput) -> TicketOut:
        with self.session_factory() as session:
            return _ticket_out(_load_ticket(session, args.ticket_id))

    def search_tickets(self, args: SearchTicketsInput) -> TicketSearchResult:
        with self.session_factory() as session:
            stmt = select(Ticket).options(joinedload(Ticket.service))
            if args.requester:
                stmt = stmt.where(Ticket.requester == args.requester)
            if args.status:
                stmt = stmt.where(Ticket.status == args.status)
            if args.open_only:
                stmt = stmt.where(Ticket.status.not_in([S.RESOLVED, S.CLOSED]))
            if args.priority:
                stmt = stmt.where(Ticket.priority == args.priority)
            if args.service:
                stmt = stmt.where(Ticket.service_id == _find_service(session, args.service).id)
            if args.text:
                pattern = _like(args.text)
                stmt = stmt.where(
                    or_(
                        Ticket.title.ilike(pattern, escape="\\"),
                        Ticket.description.ilike(pattern, escape="\\"),
                    )
                )
            stmt = stmt.order_by(Ticket.updated_at.desc(), Ticket.id.desc()).limit(args.limit)
            tickets = [_summary(t) for t in session.scalars(stmt)]
            return TicketSearchResult(count=len(tickets), tickets=tickets)

    def update_ticket(self, args: UpdateTicketInput) -> TicketOut:
        changes = args.model_dump(exclude_unset=True, exclude={"ticket_id"})
        with self.session_factory.begin() as session:
            ticket = _load_ticket(session, args.ticket_id, lock=True)
            current = TicketStatus(ticket.status)
            if current == S.CLOSED:
                raise ConflictError(_closed_message(ticket.id))

            new = changes.get("status")
            if new and new != current:
                if new not in ALLOWED_TRANSITIONS[current]:
                    allowed = ", ".join(sorted(ALLOWED_TRANSITIONS[current])) or "none"
                    raise ConflictError(
                        f"{ticket_number(ticket.id)} can't move from {current} to {new} "
                        f"(allowed: {allowed})."
                    )
                if new == S.RESOLVED:
                    ticket.resolved_at = func.now()
                elif current == S.RESOLVED and new == S.IN_PROGRESS:
                    ticket.resolved_at = None  # reopened
            for field, value in changes.items():
                setattr(ticket, field, value)

            session.flush()
            session.refresh(ticket)
            return _ticket_out(ticket)

    def add_comment(self, args: AddCommentInput) -> TicketOut:
        with self.session_factory.begin() as session:
            ticket = _load_ticket(session, args.ticket_id, lock=True)
            if ticket.status == S.CLOSED:
                raise ConflictError(_closed_message(ticket.id))
            ticket.comments.append(TicketComment(author=args.author, body=args.body))
            if ticket.status == S.WAITING_FOR_REQUESTER and args.author == ticket.requester:
                ticket.status = S.IN_PROGRESS  # the requester answered: back to IT
            session.flush()
            session.refresh(ticket)
            return _ticket_out(ticket)


# --- helpers ------------------------------------------------------------------------------


def _find_service(session: Session, name: str) -> Service:
    # 10 rows: load them all and compare normalised names, so "wifi" finds "Wi-Fi".
    services = session.scalars(select(Service).order_by(Service.name)).all()
    wanted = _key(name)
    for service in services:
        if wanted in (_key(service.id), _key(service.name)):
            return service
    known = ", ".join(s.name for s in services)
    raise NotFoundError(f"Unknown service '{name}'. Known services: {known}.")


def _load_ticket(session: Session, ticket_id: int, lock: bool = False) -> Ticket:
    # with_for_update = SELECT ... FOR UPDATE: lock the row until the transaction ends, so
    # two simultaneous updates can't both pass the status check and overwrite each other.
    # selectinload (a second query) rather than joinedload (a LEFT JOIN): Postgres can't
    # lock rows on the nullable side of an outer join.
    ticket = session.get(
        Ticket,
        ticket_id,
        options=[selectinload(Ticket.comments), selectinload(Ticket.service)],
        with_for_update=lock,
    )
    if ticket is None:
        raise NotFoundError(f"Ticket {ticket_number(ticket_id)} does not exist.")
    return ticket


def _closed_message(ticket_id: int) -> str:
    return (
        f"{ticket_number(ticket_id)} is closed and can't be changed. Raise a new ticket that "
        f"references {ticket_number(ticket_id)}."
    )


def _status_out(service: Service) -> ServiceStatusOut:
    return ServiceStatusOut(
        id=service.id,
        name=service.name,
        status=service.status,
        message=service.status_message,
        criticality=service.criticality,
        owning_team=service.owning_team,
        updated_at=service.status_updated_at,
    )


def _summary(ticket: Ticket) -> TicketSummary:
    return TicketSummary(
        id=ticket.id,
        number=ticket_number(ticket.id),
        title=ticket.title,
        category=ticket.category,
        priority=ticket.priority,
        status=ticket.status,
        requester=ticket.requester,
        service=ticket.service.name if ticket.service else None,
        updated_at=ticket.updated_at,
    )


def _ticket_out(ticket: Ticket) -> TicketOut:
    priority = Priority(ticket.priority)
    first_response, resolution = SLA_TARGETS[priority]
    is_open = ticket.status not in (S.RESOLVED, S.CLOSED)
    return TicketOut(
        **_summary(ticket).model_dump(),
        description=ticket.description,
        assignee=ticket.assignee,
        resolution=ticket.resolution,
        created_at=ticket.created_at,
        resolved_at=ticket.resolved_at,
        sla=Sla(first_response=first_response, resolution=resolution),
        comments=[
            CommentOut(id=c.id, author=c.author, body=c.body, created_at=c.created_at)
            for c in ticket.comments
        ],
        notice=PHONE_NOTICE if is_open and priority in (Priority.P1, Priority.P2) else None,
    )
