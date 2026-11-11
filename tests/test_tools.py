"""The tool registry: what an agent or MCP client sees and gets back."""

import json

import pytest

from servicedesk.desk.tools import TOOLS, TOOLS_BY_NAME, run_tool

pytestmark = pytest.mark.integration


def test_the_six_tools_and_which_ones_write():
    assert set(TOOLS_BY_NAME) == {
        "get_service_status",
        "search_tickets",
        "get_ticket",
        "create_ticket",
        "update_ticket",
        "add_comment",
    }
    # These are the ones that will need human approval (Phase 6).
    assert {t.name for t in TOOLS if t.writes} == {"create_ticket", "update_ticket", "add_comment"}


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_every_tool_is_described_for_a_model(tool):
    schema = tool.input_schema()
    assert len(tool.description) > 30
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    json.dumps(schema)  # must be plain JSON to send to an LLM or MCP client


def test_run_tool_returns_json_ready_results(desk):
    result = run_tool(desk, "get_ticket", {"ticket_id": "INC-1001"})
    assert result["ok"] is True and result["result"]["number"] == "INC-1001"
    json.dumps(result)  # datetimes and enums already converted


def test_create_through_the_tool(desk):
    args = {
        "title": "Teams camera not detected",
        "description": "Teams says no camera found on my laptop.",
        "category": "collaboration",
        "requester": "li.wei@halcyon-dynamics.example",
        "service": "teams",
    }
    result = run_tool(desk, "create_ticket", args)
    assert result["ok"] and result["result"]["number"] == "INC-1006"


def test_invalid_arguments_come_back_as_a_list_of_problems(desk):
    result = run_tool(desk, "create_ticket", {"title": "x", "urgency": "high"})
    assert result["ok"] is False and result["error"]["code"] == "invalid_input"
    fields = {p["field"] for p in result["error"]["problems"]}
    assert fields == {"title", "description", "category", "requester", "urgency"}


@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("delete_ticket", {}, "unknown_tool"),
        ("get_ticket", {"ticket_id": 9999}, "not_found"),
        ("get_service_status", {"service": "fax"}, "not_found"),
        ("update_ticket", {"ticket_id": 1005, "priority": "P1"}, "conflict"),
    ],
)
def test_errors_are_results_not_exceptions(desk, name, args, code):
    result = run_tool(desk, name, args)
    assert result == {"ok": False, "error": result["error"]}
    assert result["error"]["code"] == code and result["error"]["message"]
