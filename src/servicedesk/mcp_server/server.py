"""The service desk as an MCP server: any MCP client (Claude Desktop, VS Code, MCP Inspector)
can use our tools, read our knowledge base and start from our prompt.

MCP (Model Context Protocol) is a standard way for an AI app to plug into outside tools
and data. A server offers three kinds of things:
- tools: actions the client's model can call (check a service, find or create a ticket);
- resources: things the client can read by URI (our KB articles: kb://articles/KB-0001);
- prompts: ready-made conversation starters a person can pick (report an IT issue).

Nothing about the service desk is redefined here. Each tool is the Phase 4 tool of the same
name, called through AgentTools (agent/tools.py), so the same validation, the same
"errors are data" results and the same database error handling apply as for our own agent.
The function signatures only describe the arguments to the client, reusing the validated
types from desk/schemas.py; the tool's input model still checks every call.

Writes (create_ticket, update_ticket, add_comment) need a confirmation token, see
confirmation.py: the first call only previews the change.
"""

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo
from sqlalchemy import select

from servicedesk.agent.tools import KB_DESCRIPTION, KB_TOOL, AgentTools, KnowledgeBaseInput
from servicedesk.db.models import Document
from servicedesk.desk.schemas import (
    Category,
    CreateTicketInput,
    Email,
    GetServiceStatusInput,
    LongText,
    Priority,
    SearchTicketsInput,
    ServiceName,
    TicketId,
    TicketStatus,
    Title,
    UpdateTicketInput,
)
from servicedesk.desk.service import ServiceDesk
from servicedesk.desk.tools import TOOLS_BY_NAME, invalid_input
from servicedesk.ingestion.loaders import load_document
from servicedesk.mcp_server.confirmation import Confirmations
from servicedesk.openai_client import ConfigurationError
from servicedesk.rag.answerer import Answer

INSTRUCTIONS = """\
IT service desk for Halcyon Dynamics employees.
- When someone reports a problem, call get_service_status first: a known outage needs no ticket.
- For how-to and policy questions, call search_knowledge_base and keep its [n] citations.
- Before creating a ticket, use search_tickets to check the person doesn't already have one.
- Changes (create_ticket, update_ticket, add_comment) take two calls. The first returns a
  preview and a confirmation_token and changes nothing. Show the person the preview, and only
  if they agree, call the same tool again with the same arguments plus confirmation_token.
- Write only what the person told you. Never invent an email address."""

# Hints for the client about each tool. Clients use them to decide e.g. whether to ask the
# person before calling it. readOnly = changes nothing; idempotent = repeating the same
# confirmed call has no extra effect (the token is its idempotency key); openWorld=False =
# it only touches our own systems.
READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
)

ARTICLE_URI = "kb://articles/{doc_id}"
CONFIRMATION_TOKEN_HELP = (
    "Leave empty on the first call: you get a preview to show the person, and a token. "
    "After they agree, call again with the same arguments and this token."
)

Token = Annotated[str | None, Field(description=CONFIRMATION_TOKEN_HELP)]


def about(model: type[BaseModel], field: str) -> FieldInfo:
    """The description a field already has in its input model, so it's written once."""
    return Field(description=model.model_fields[field].description)


def article_uri(doc_id: str) -> str:
    return ARTICLE_URI.format(doc_id=doc_id)


def build_mcp_server(
    desk: ServiceDesk,
    ask_knowledge_base: Callable[[str], Answer] | None,
    confirmations: Confirmations | None = None,
) -> MCPServer:
    """The MCP server, ready to run on any transport (stdio or HTTP, see cli.py).

    ask_knowledge_base is None without an OpenAI key: search_knowledge_base then answers
    "unavailable" and everything else still works.
    """
    tools = AgentTools(desk, ask_knowledge_base or _no_knowledge_base)
    confirmations = confirmations or Confirmations()
    server = MCPServer(
        name="servicedesk",
        title="Halcyon IT Service Desk",
        version=version("servicedesk-copilot"),
        instructions=INSTRUCTIONS,
    )

    def read(name: str, arguments: dict[str, Any]) -> CallToolResult:
        result, _ = tools.run(name, arguments)
        return as_tool_result(result)

    def write(name: str, arguments: dict[str, Any], token: str | None) -> CallToolResult:
        """Preview first; run only with a token issued for exactly these arguments."""
        preview = tools.preview(name, arguments)  # validated + normalised, or None
        if preview is None:
            return as_tool_result(validation_errors(name, arguments))  # nothing to confirm
        if token is None:
            return as_tool_result(confirmation_required(name, preview, confirmations))
        nonce = confirmations.check(token, name, preview)
        if nonce is None:
            # Made up, from an older server run, or the arguments changed since the
            # preview: the person hasn't seen this exact change.
            message = (
                "The confirmation_token doesn't match these arguments. Show the person this "
                "preview and use the new token if they agree."
            )
            return as_tool_result(
                _error(
                    "invalid_confirmation",
                    message,
                    **confirmation_required(name, preview, confirmations)["result"],
                )
            )
        result, _ = tools.run(name, arguments, idempotency_key=f"mcp:{nonce}")
        return as_tool_result(result)

    # --- tools: reading ------------------------------------------------------------------

    @server.tool(annotations=READ, description=_described("get_service_status"))
    def get_service_status(
        service: Annotated[ServiceName | None, about(GetServiceStatusInput, "service")] = None,
    ) -> CallToolResult:
        return read("get_service_status", sent(service=service))

    @server.tool(annotations=READ, description=_described("search_tickets"))
    def search_tickets(
        requester: Email | None = None,
        status: TicketStatus | None = None,
        open_only: Annotated[bool, about(SearchTicketsInput, "open_only")] = False,
        priority: Priority | None = None,
        service: ServiceName | None = None,
        text: Annotated[str | None, about(SearchTicketsInput, "text")] = None,
        limit: Annotated[int, Field(description="At most this many tickets (1-50).")] = 10,
    ) -> CallToolResult:
        return read(
            "search_tickets",
            sent(
                requester=requester,
                status=status,
                open_only=open_only,
                priority=priority,
                service=service,
                text=text,
                limit=limit,
            ),
        )

    @server.tool(annotations=READ, description=_described("get_ticket"))
    def get_ticket(ticket_id: TicketId) -> CallToolResult:
        return read("get_ticket", {"ticket_id": ticket_id})

    @server.tool(annotations=READ, description=KB_DESCRIPTION)
    def search_knowledge_base(
        question: Annotated[str, about(KnowledgeBaseInput, "question")],
    ) -> CallToolResult:
        result, _ = tools.run(KB_TOOL, {"question": question})
        if result["ok"] and result["result"]["found"]:
            # Point each source at its resource, so the client can open the whole article.
            for source in result["result"]["sources"]:
                source["uri"] = article_uri(source["doc_id"])
        return as_tool_result(result)

    # --- tools: changing (preview, then confirm) -----------------------------------------

    @server.tool(annotations=WRITE, description=_described("create_ticket"))
    def create_ticket(
        title: Annotated[Title, about(CreateTicketInput, "title")],
        description: Annotated[LongText, about(CreateTicketInput, "description")],
        category: Category,
        requester: Annotated[Email, about(CreateTicketInput, "requester")],
        priority: Annotated[Priority, about(CreateTicketInput, "priority")] = Priority.P3,
        service: Annotated[ServiceName | None, about(CreateTicketInput, "service")] = None,
        confirmation_token: Token = None,
    ) -> CallToolResult:
        arguments = sent(
            title=title,
            description=description,
            category=category,
            requester=requester,
            priority=priority,
            service=service,
        )
        return write("create_ticket", arguments, confirmation_token)

    @server.tool(annotations=WRITE, description=_described("update_ticket"))
    def update_ticket(
        ticket_id: TicketId,
        status: TicketStatus | None = None,
        priority: Priority | None = None,
        assignee: Email | None = None,
        resolution: Annotated[LongText | None, about(UpdateTicketInput, "resolution")] = None,
        confirmation_token: Token = None,
    ) -> CallToolResult:
        # Only the fields given are changed (unset = leave as is). A limit of this
        # signature: "clear the assignee" can't be expressed, as null means "not given".
        arguments = sent(
            ticket_id=ticket_id,
            status=status,
            priority=priority,
            assignee=assignee,
            resolution=resolution,
        )
        return write("update_ticket", arguments, confirmation_token)

    @server.tool(annotations=WRITE, description=_described("add_comment"))
    def add_comment(
        ticket_id: TicketId,
        author: Annotated[Email, Field(description="Who is writing the comment.")],
        body: Annotated[LongText, Field(description="The comment text.")],
        confirmation_token: Token = None,
    ) -> CallToolResult:
        arguments = {"ticket_id": ticket_id, "author": author, "body": body}
        return write("add_comment", arguments, confirmation_token)

    # --- resources: the knowledge base -----------------------------------------------------

    @server.resource(
        "kb://articles",
        name="kb_articles",
        title="Knowledge base articles",
        description="Every article in the IT knowledge base, with the URI to read each one.",
        mime_type="application/json",
    )
    def list_articles() -> str:
        with desk.session_factory() as session:
            documents = session.scalars(select(Document).order_by(Document.doc_id)).all()
            articles = [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "category": d.category,
                    "service": d.service,
                    "uri": article_uri(d.doc_id),
                }
                for d in documents
            ]
        return json.dumps({"count": len(articles), "articles": articles}, indent=2)

    @server.resource(
        ARTICLE_URI,
        name="kb_article",
        title="Knowledge base article",
        description="One article as Markdown, e.g. kb://articles/KB-0001.",
        mime_type="text/markdown",
    )
    def read_article(doc_id: str) -> str:
        with desk.session_factory() as session:
            document = session.scalars(
                select(Document).where(Document.doc_id == doc_id.upper())
            ).one_or_none()
            if document is None:
                raise ResourceError(f"No knowledge base article {doc_id}.")
            source_path = document.source_path
        # The text as it was ingested: re-read with the same loader, so a PDF comes back as
        # its text with headings, not as bytes. source_path comes from our database, never
        # from the client, so the client can't make us read any other file.
        path = Path(source_path)
        if not path.is_file():
            raise ResourceError(f"The source file of {doc_id} isn't available on this server.")
        return as_markdown(load_document(path))

    # --- prompts ---------------------------------------------------------------------------

    @server.prompt(
        title="Report an IT issue",
        description="Start helping an employee with a problem, the way the service desk does.",
    )
    def report_it_issue(
        problem: Annotated[str, Field(description="What isn't working, in their words.")],
        requester_email: Annotated[str, Field(description="Their company email address.")],
    ) -> str:
        return (
            f"An employee ({requester_email}) reports: {problem}\n\n"
            "Help them using the service desk tools: check the service status first; if "
            "there's no known outage, look for a fix in the knowledge base and cite it; check "
            "their open tickets; and if it still needs a ticket, preview one with the right "
            "category and priority and ask them before confirming it."
        )

    return server


# --- helpers ---------------------------------------------------------------------------


def sent(**arguments: Any) -> dict[str, Any]:
    """Only the arguments that were given: None means "not given" (the defaults of the
    input model then apply, and update_ticket changes only the fields that were sent)."""
    return {name: value for name, value in arguments.items() if value is not None}


def as_tool_result(result: dict[str, Any]) -> CallToolResult:
    """Our {"ok", "result" | "error"} dict as an MCP tool result: JSON text for any client,
    the same data as structured content, and is_error set when the call didn't succeed (so
    the client's model knows to read the error and fix its call)."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
        structured_content=result,
        is_error=not result["ok"],
    )


def confirmation_required(
    name: str, preview: dict[str, Any], confirmations: Confirmations
) -> dict[str, Any]:
    result = {
        "status": "confirmation_required",
        "tool": name,
        "preview": preview,
        "confirmation_token": confirmations.issue(name, preview),
        "next_step": (
            "Nothing has changed yet. Show the person this change and ask them to confirm. "
            "If they agree, call the tool again with the same arguments plus this "
            "confirmation_token. If they want something different, call it with the new "
            "arguments and no token, for a new preview."
        ),
    }
    return {"ok": True, "result": result}


def validation_errors(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        TOOLS_BY_NAME[name].input_model.model_validate(arguments)
    except ValidationError as exc:
        return invalid_input(exc)
    raise AssertionError("validation_errors() called with valid arguments")  # pragma: no cover


def as_markdown(document: Any) -> str:
    """A loaded document (ingestion/models.py SourceDocument) as one Markdown text."""
    parts = [f"# {document.title}", f"_{document.doc_id} ({document.source_path})_"]
    for section in document.sections:
        heading = section.heading.split(" > ")[-1]  # the last heading in the path
        if heading and heading != document.title:
            parts.append(f"## {heading}")
        parts.append(section.text)
    return "\n\n".join(parts) + "\n"


def _described(name: str) -> str:
    return TOOLS_BY_NAME[name].description


def _no_knowledge_base(question: str) -> Answer:
    raise ConfigurationError("The knowledge base needs an OpenAI API key (see .env).")


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, **details}}
