"""Input validation rules (no database needed)."""

import pytest
from pydantic import ValidationError

from servicedesk.desk.schemas import (
    AddCommentInput,
    CreateTicketInput,
    GetTicketInput,
    SearchTicketsInput,
    TicketChanges,
    UpdateTicketInput,
)

VALID = {
    "title": "VPN keeps disconnecting",
    "description": "Drops every five minutes since this morning.",
    "category": "network",
    "requester": "jane.doe@halcyon-dynamics.example",
}


def errors(exc: pytest.ExceptionInfo) -> dict[str, str]:
    return {".".join(map(str, e["loc"])) or "(input)": e["msg"] for e in exc.value.errors()}


def test_valid_ticket_gets_defaults():
    ticket = CreateTicketInput(**VALID)
    assert ticket.priority == "P3" and ticket.service is None


@pytest.mark.parametrize(
    "email", ["Jane.Doe@Halcyon-Dynamics.example", "  jane.doe@halcyon-dynamics.example "]
)
def test_company_email_is_normalised(email):
    assert CreateTicketInput(**VALID | {"requester": email}).requester == (
        "jane.doe@halcyon-dynamics.example"
    )


@pytest.mark.parametrize(
    "email", ["jane@gmail.com", "jane", "jane@halcyon-dynamics.example.evil.com", ""]
)
def test_non_company_email_is_rejected(email):
    with pytest.raises(ValidationError) as exc:
        CreateTicketInput(**VALID | {"requester": email})
    assert "company email" in errors(exc)["requester"]


@pytest.mark.parametrize("value", [1042, "1042", "INC-1042", "inc-1042", "INC 1042", " INC1042 "])
def test_ticket_numbers_in_any_common_form(value):
    assert GetTicketInput(ticket_id=value).ticket_id == 1042


@pytest.mark.parametrize("value", ["abc", "INC-", 0, -5, "TASK-1042"])
def test_invalid_ticket_numbers(value):
    with pytest.raises(ValidationError):
        GetTicketInput(ticket_id=value)


def test_text_is_cleaned():
    ticket = CreateTicketInput(
        **VALID
        | {
            "title": "  VPN\n keeps\tdropping\x00  ",
            "description": "line 1\r\n\n\n\n\nline 2\x07",
        }
    )
    assert ticket.title == "VPN keeps dropping"  # one line, control chars gone
    assert ticket.description == "line 1\n\nline 2"  # line breaks kept, runs collapsed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("title", "   ab  ", "at least 5 characters"),
        ("title", "x" * 201, "at most 200 characters"),
        ("description", "   ", "at least 1 character"),
        ("category", "printers", "Input should be 'access'"),
        ("priority", "urgent", "Input should be 'P1'"),
    ],
)
def test_field_rules(field, value, message):
    with pytest.raises(ValidationError) as exc:
        CreateTicketInput(**VALID | {field: value})
    assert message in errors(exc)[field]


def test_unknown_fields_are_rejected():
    # An LLM inventing an argument hears about it instead of it being silently dropped.
    with pytest.raises(ValidationError) as exc:
        CreateTicketInput(**VALID | {"urgency": "high"})
    assert "urgency" in errors(exc)


def test_all_problems_are_reported_at_once():
    with pytest.raises(ValidationError) as exc:
        CreateTicketInput(title="x", description="", category="?", requester="bob@gmail.com")
    assert set(errors(exc)) == {"title", "description", "category", "requester"}


def test_changes_need_at_least_one_field():
    with pytest.raises(ValidationError, match="at least one field"):
        TicketChanges()
    with pytest.raises(ValidationError, match="at least one field"):
        UpdateTicketInput(ticket_id=1001)  # the id alone changes nothing


def test_resolving_needs_a_resolution():
    with pytest.raises(ValidationError, match="resolution is required"):
        TicketChanges(status="resolved")
    assert TicketChanges(status="resolved", resolution="Restarted the service.").resolution


def test_explicit_null_counts_as_a_change():
    # {"assignee": null} means "unassign", which is different from not sending assignee.
    changes = TicketChanges(assignee=None)
    assert changes.model_dump(exclude_unset=True) == {"assignee": None}


def test_search_limit_is_bounded():
    assert SearchTicketsInput().limit == 10
    with pytest.raises(ValidationError):
        SearchTicketsInput(limit=500)


def test_comment_needs_a_body_and_company_author():
    with pytest.raises(ValidationError) as exc:
        AddCommentInput(ticket_id=1001, author="x@y.com", body=" ")
    assert set(errors(exc)) == {"author", "body"}


def test_schema_shown_to_an_llm_carries_the_rules():
    schema = CreateTicketInput.model_json_schema()
    title = schema["properties"]["title"]
    assert title["minLength"] == 5 and title["maxLength"] == 200 and title["description"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"title", "description", "category", "requester"}
    assert "network" in schema["$defs"]["Category"]["enum"]
