"""Service desk business rules against the test database (seeded: INC-1001..INC-1005)."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from servicedesk.db.models import Ticket
from servicedesk.desk.schemas import (
    AddCommentInput,
    CreateTicketInput,
    GetServiceStatusInput,
    GetTicketInput,
    SearchTicketsInput,
    UpdateTicketInput,
)
from servicedesk.desk.seed import seed
from servicedesk.desk.service import ALLOWED_TRANSITIONS, ConflictError, NotFoundError
from tests.conftest import ROOT

pytestmark = pytest.mark.integration

PRIYA = "priya.sharma@halcyon-dynamics.example"
TOM = "tom.oconnor@halcyon-dynamics.example"
IT = "network.ops@halcyon-dynamics.example"


def new_ticket(desk, **overrides):
    fields = {
        "title": "Monitor flickers",
        "description": "The external monitor flickers every few seconds.",
        "category": "hardware",
        "requester": PRIYA,
    }
    return desk.create_ticket(CreateTicketInput(**fields | overrides))


def force_status(session_factory, ticket_id, status, resolution=None):
    """Put a ticket in a state directly, skipping the rules (test setup only)."""
    with session_factory.begin() as session:
        ticket = session.get(Ticket, ticket_id)
        ticket.status, ticket.resolution = status, resolution or ticket.resolution


def update(desk, ticket_id, **changes):
    return desk.update_ticket(UpdateTicketInput(ticket_id=ticket_id, **changes))


# --- service status -----------------------------------------------------------------------


def test_all_services_are_listed(desk):
    services = desk.get_service_status(GetServiceStatusInput()).services
    assert len(services) == 10
    vpn = next(s for s in services if s.id == "vpn")
    assert vpn.status == "degraded" and "KI-0007" in vpn.message


@pytest.mark.parametrize(
    ("asked", "found"),
    [("VPN", "vpn"), ("wifi", "wifi"), ("Wi-Fi", "wifi"), ("hr portal", "hr-portal")],
)
def test_service_names_are_matched_loosely(desk, asked, found):
    result = desk.get_service_status(GetServiceStatusInput(service=asked))
    assert [s.id for s in result.services] == [found]


def test_unknown_service_lists_the_known_ones(desk):
    with pytest.raises(NotFoundError, match="Known services: Email, File Shares"):
        desk.get_service_status(GetServiceStatusInput(service="printers"))


def test_reseeding_keeps_a_changed_status(desk, desk_session_factory):
    desk.set_service_status("email", "major_outage", "Mail delivery is delayed.")
    seed(desk_session_factory, ROOT / "data" / "seed")
    email = desk.get_service_status(GetServiceStatusInput(service="email")).services[0]
    assert email.status == "major_outage" and email.message == "Mail delivery is delayed."


# --- create and read ----------------------------------------------------------------------


def test_created_ticket_gets_a_number_sla_and_status_new(desk):
    ticket = new_ticket(desk, priority="P4", service="Hardware")
    assert ticket.id == 1006 and ticket.number == "INC-1006"  # after the 5 sample tickets
    assert ticket.status == "new" and ticket.service == "Hardware"
    assert ticket.sla.first_response == "1 business day"
    assert ticket.notice is None and ticket.comments == []


@pytest.mark.parametrize("priority", ["P1", "P2"])
def test_urgent_tickets_say_to_phone_in(desk, priority):
    assert "extension 4357" in new_ticket(desk, priority=priority).notice


def test_unknown_service_creates_nothing(desk, desk_session_factory):
    with pytest.raises(NotFoundError):
        new_ticket(desk, service="Fax machines")
    with desk_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Ticket)) == 5  # rolled back


def test_get_ticket_includes_comments(desk):
    ticket = desk.get_ticket(GetTicketInput(ticket_id="INC-1001"))
    assert ticket.title.startswith("VPN disconnects") and len(ticket.comments) == 1


def test_missing_ticket(desk):
    with pytest.raises(NotFoundError, match="INC-9999 does not exist"):
        desk.get_ticket(GetTicketInput(ticket_id=9999))


# --- search -------------------------------------------------------------------------------


def search(desk, **filters):
    return [t.number for t in desk.search_tickets(SearchTicketsInput(**filters)).tickets]


def test_search_by_requester(desk):
    assert sorted(search(desk, requester=PRIYA)) == ["INC-1001", "INC-1003"]


def test_search_open_only_skips_resolved_and_closed(desk):
    assert sorted(search(desk, open_only=True)) == ["INC-1001", "INC-1002", "INC-1003"]


def test_search_filters_combine(desk):
    assert search(desk, requester=PRIYA, service="vpn") == ["INC-1001"]
    assert search(desk, status="closed") == ["INC-1005"]
    assert search(desk, priority="P2") == ["INC-1005"]


def test_search_text_is_case_insensitive(desk):
    assert search(desk, text="OUTLOOK") == ["INC-1002"]
    assert search(desk, text="badge tap") == ["INC-1004"]  # found in the description


def test_search_text_wildcards_are_literal(desk):
    assert search(desk, text="%") == []  # not "match everything"
    new_ticket(desk, title="Disk 100% full", description="C: drive is full.")
    assert search(desk, text="100%") == ["INC-1006"]


def test_search_newest_first_and_limit(desk):
    new_ticket(desk)
    assert search(desk, limit=2)[0] == "INC-1006"
    assert len(search(desk, limit=2)) == 2


# --- status changes -----------------------------------------------------------------------

ALL_STATUSES = ["new", "in_progress", "waiting_for_requester", "resolved", "closed"]
PAIRS = [(a, b) for a in ALL_STATUSES for b in ALL_STATUSES if a != b and a != "closed"]


@pytest.mark.parametrize(("current", "target"), PAIRS)
def test_status_transitions_follow_the_policy(desk, desk_session_factory, current, target):
    ticket = new_ticket(desk)
    force_status(desk_session_factory, ticket.id, current, resolution="Fixed earlier.")
    resolution = {"resolution": "Fixed."} if target == "resolved" else {}
    if target in ALLOWED_TRANSITIONS[current]:
        assert update(desk, ticket.id, status=target, **resolution).status == target
    else:
        with pytest.raises(ConflictError, match=f"can't move from {current} to {target}"):
            update(desk, ticket.id, status=target, **resolution)


def test_resolving_and_reopening(desk):
    ticket = new_ticket(desk)
    resolved = update(desk, ticket.id, status="resolved", resolution="Replaced the cable.")
    assert resolved.resolved_at is not None and resolved.resolution == "Replaced the cable."
    reopened = update(desk, ticket.id, status="in_progress")
    assert reopened.status == "in_progress" and reopened.resolved_at is None


def test_closed_tickets_are_read_only(desk):
    with pytest.raises(ConflictError, match="Raise a new ticket that references INC-1005"):
        update(desk, 1005, priority="P4")
    with pytest.raises(ConflictError, match="closed"):
        desk.add_comment(AddCommentInput(ticket_id=1005, author=IT, body="Any update?"))


def test_other_fields_can_change_without_status(desk):
    ticket = update(desk, 1003, priority="P3", assignee="Access.Team@halcyon-dynamics.example")
    assert ticket.priority == "P3" and ticket.status == "new"
    assert ticket.assignee == "access.team@halcyon-dynamics.example"
    assert update(desk, 1003, assignee=None).assignee is None  # explicit null unassigns


def test_database_refuses_resolved_without_resolution(desk_session_factory):
    # Defence in depth: even code that skips the service layer can't store this.
    with pytest.raises(IntegrityError), desk_session_factory.begin() as session:
        session.get(Ticket, 1003).status = "resolved"


# --- comments -----------------------------------------------------------------------------


def test_requester_reply_moves_waiting_ticket_back_to_in_progress(desk):
    ticket = desk.add_comment(AddCommentInput(ticket_id=1002, author=TOM, body="Safe mode works."))
    assert ticket.status == "in_progress"
    assert [c.author for c in ticket.comments][-1] == TOM


def test_it_comment_leaves_waiting_ticket_waiting(desk):
    ticket = desk.add_comment(AddCommentInput(ticket_id=1002, author=IT, body="Any news?"))
    assert ticket.status == "waiting_for_requester" and len(ticket.comments) == 2
