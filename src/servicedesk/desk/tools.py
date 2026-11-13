"""The service desk tools, defined once and shared by every caller.

A tool = a name, a description written for a language model, an input schema, and a
function. The LangGraph agent (Phase 5) and the MCP server (Phase 7) are built from TOOLS;
the `writes` flag is what Phase 6 uses to require human approval and Phase 7 to require
explicit confirmation.

run_tool() is the safe way to call one with untrusted arguments (e.g. JSON an LLM wrote):
it validates them, runs the tool and always returns a result dict, never raises for a bad
call. Errors come back as data ({"ok": false, "error": {...}}) so a model can read what
went wrong and fix its call instead of the whole conversation crashing.

Write tools also accept an idempotency key (passed on to ServiceDesk): the same key twice
means the write happens once. Read tools ignore it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from servicedesk.desk.schemas import (
    AddCommentInput,
    CreateTicketInput,
    GetServiceStatusInput,
    GetTicketInput,
    SearchTicketsInput,
    UpdateTicketInput,
)
from servicedesk.desk.service import DeskError, ServiceDesk


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., BaseModel]  # (desk, args) or, for writes, (desk, args, idempotency_key=)
    writes: bool  # changes data: needs human approval before an agent may run it

    def input_schema(self) -> dict:
        """JSON Schema of the arguments: what an LLM or MCP client is shown."""
        return self.input_model.model_json_schema()


TOOLS: list[Tool] = [
    Tool(
        name="get_service_status",
        description=(
            "Current health of IT services (operational, degraded, outage, maintenance) and "
            "any incident message. Check this first when someone reports that a service "
            "isn't working: if there is a known outage, a new ticket isn't needed."
        ),
        input_model=GetServiceStatusInput,
        handler=ServiceDesk.get_service_status,
        writes=False,
    ),
    Tool(
        name="search_tickets",
        description=(
            "Find existing tickets by requester, status, priority, service or words in the "
            "text. Use it to check for an existing ticket before creating a new one."
        ),
        input_model=SearchTicketsInput,
        handler=ServiceDesk.search_tickets,
        writes=False,
    ),
    Tool(
        name="get_ticket",
        description="Full details of one ticket, including its comments and SLA targets.",
        input_model=GetTicketInput,
        handler=ServiceDesk.get_ticket,
        writes=False,
    ),
    Tool(
        name="create_ticket",
        description=(
            "Raise a new ticket for an employee. Requires a title, description, category "
            "and the requester's company email. Priority defaults to P3."
        ),
        input_model=CreateTicketInput,
        handler=ServiceDesk.create_ticket,
        writes=True,
    ),
    Tool(
        name="update_ticket",
        description=(
            "Change a ticket's status, priority, assignee or resolution. Resolving requires "
            "a resolution note. Closed tickets can't be changed."
        ),
        input_model=UpdateTicketInput,
        handler=ServiceDesk.update_ticket,
        writes=True,
    ),
    Tool(
        name="add_comment",
        description=(
            "Add a comment to a ticket. A reply from the requester moves a ticket that is "
            "waiting for them back to in progress."
        ),
        input_model=AddCommentInput,
        handler=ServiceDesk.add_comment,
        writes=True,
    ),
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def run_tool(
    desk: ServiceDesk,
    name: str,
    arguments: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Validate `arguments`, run the tool, and return {"ok": True, "result": ...} or
    {"ok": False, "error": {"code", "message", ...}}. JSON-ready either way."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return _error("unknown_tool", f"No tool named '{name}'. Tools: {', '.join(TOOLS_BY_NAME)}.")
    try:
        args = tool.input_model.model_validate(arguments)
    except ValidationError as exc:
        return invalid_input(exc)
    try:
        if tool.writes:
            result = tool.handler(desk, args, idempotency_key=idempotency_key)
        else:
            result = tool.handler(desk, args)
    except DeskError as exc:
        return _error(exc.code, str(exc))
    return {"ok": True, "result": result.model_dump(mode="json")}


def invalid_input(exc: ValidationError) -> dict[str, Any]:
    """A validation error as a tool result: one entry per problem, naming the field, so
    the caller knows exactly what to fix."""
    problems = [
        {"field": ".".join(str(p) for p in err["loc"]) or "(input)", "problem": err["msg"]}
        for err in exc.errors()
    ]
    return _error("invalid_input", "The arguments are not valid.", problems=problems)


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, **details}}
