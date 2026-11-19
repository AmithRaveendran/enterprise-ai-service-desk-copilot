"""The MCP server: what a client sees (tools, schemas, hints), the preview-then-confirm rule
for writes, the knowledge base as resources, and the prompt.

Every test talks to the server through the SDK's real MCP client, connected in-process:
the same requests and results as over stdio or HTTP, without starting a process. The
client is async, so these tests are too (`pytestmark` below runs each one in an event loop).
"""

import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from servicedesk.agent.tools import KB_TOOL, KnowledgeBaseInput
from servicedesk.db.models import Ticket
from servicedesk.desk.service import ServiceDesk
from servicedesk.desk.tools import TOOLS, TOOLS_BY_NAME
from servicedesk.mcp_server.confirmation import Confirmations
from servicedesk.mcp_server.server import build_mcp_server
from tests.test_agent import FakeKnowledgeBase, kb_answer

pytestmark = pytest.mark.anyio

WRITE_TOOLS = {t.name for t in TOOLS if t.writes}
NEW_TICKET = {
    "title": "Laptop won't charge",
    "description": "The charger light stays off. Tried another socket.",
    "category": "hardware",
    "requester": "li.wei@halcyon-dynamics.example",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server(desk):
    return build_mcp_server(desk, FakeKnowledgeBase(kb_answer()))


def ticket_count(desk: ServiceDesk) -> int:
    with desk.session_factory() as session:
        return session.scalar(select(func.count()).select_from(Ticket))


async def call(server, tool: str, arguments: dict) -> tuple[bool, dict]:
    """Call a tool like a client would. Returns (is_error, our result dict)."""
    async with Client(server) as client:
        result = await client.call_tool(tool, arguments)
    return result.is_error, result.structured_content


# --- what a client sees -----------------------------------------------------------------


@pytest.mark.integration
async def test_every_registry_tool_is_offered_with_the_right_hints(server):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == {KB_TOOL} | set(TOOLS_BY_NAME)
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints.read_only_hint is (name not in WRITE_TOOLS)
        assert hints.open_world_hint is False
        if name in WRITE_TOOLS:
            assert hints.destructive_hint is False and hints.idempotent_hint is True
        if name != KB_TOOL:
            assert tool.description == TOOLS_BY_NAME[name].description  # written once


@pytest.mark.integration
async def test_tool_parameters_match_the_input_models(server):
    """The MCP signatures are written out by hand, so this guards against them drifting
    from the input models: same fields, same required ones (plus the confirmation token)."""
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    models = {name: tool.input_model for name, tool in TOOLS_BY_NAME.items()}
    models[KB_TOOL] = KnowledgeBaseInput
    for name, model in models.items():
        schema = tools[name].input_schema
        params = set(schema["properties"]) - {"confirmation_token"}
        assert params == set(model.model_fields), name
        assert set(schema.get("required", [])) == {
            field for field, info in model.model_fields.items() if info.is_required()
        }, name
        assert ("confirmation_token" in schema["properties"]) is (name in WRITE_TOOLS)


@pytest.mark.integration
async def test_the_schema_carries_the_validation_rules(server):
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    title = tools["create_ticket"].input_schema["properties"]["title"]
    assert (title["minLength"], title["maxLength"]) == (5, 200)
    assert title["description"].startswith("Short summary")


# --- reading ------------------------------------------------------------------------------


@pytest.mark.integration
async def test_a_read_tool_returns_its_result_as_structured_content(server):
    is_error, result = await call(server, "get_ticket", {"ticket_id": "INC-1002"})
    assert not is_error
    assert result["ok"] and result["result"]["number"] == "INC-1002"


@pytest.mark.integration
async def test_a_business_error_is_an_error_result_not_a_crash(server):
    is_error, result = await call(server, "get_ticket", {"ticket_id": 9999})
    assert is_error
    assert result["error"]["code"] == "not_found"


async def test_a_database_outage_is_reported_as_unavailable():
    def database_down():
        raise OperationalError("connect", {}, Exception("connection refused"))

    server = build_mcp_server(ServiceDesk(database_down), None)
    is_error, result = await call(server, "get_service_status", {"service": "vpn"})
    assert is_error and result["error"]["code"] == "unavailable"


@pytest.mark.integration
async def test_knowledge_base_sources_link_to_their_articles(server):
    is_error, result = await call(server, KB_TOOL, {"question": "vpn keeps dropping"})
    assert not is_error
    assert result["result"]["answer"] == "Restart the VPN client [1]."
    assert result["result"]["sources"][0]["uri"] == "kb://articles/KB-0007"


async def test_without_an_api_key_the_knowledge_base_is_unavailable():
    server = build_mcp_server(ServiceDesk(None), None)
    is_error, result = await call(server, KB_TOOL, {"question": "vpn keeps dropping"})
    assert is_error and result["error"]["code"] == "unavailable"


# --- writing: preview, then confirm -----------------------------------------------------------


@pytest.mark.integration
async def test_a_write_without_a_token_only_previews(server, desk):
    before = ticket_count(desk)
    is_error, result = await call(server, "create_ticket", NEW_TICKET)

    assert not is_error
    assert result["result"]["status"] == "confirmation_required"
    assert result["result"]["preview"]["priority"] == "P3"  # the default is shown too
    assert result["result"]["confirmation_token"]
    assert ticket_count(desk) == before  # nothing written


@pytest.mark.integration
async def test_the_confirmed_call_writes_once_even_if_sent_twice(server, desk):
    before = ticket_count(desk)
    _, preview = await call(server, "create_ticket", NEW_TICKET)
    token = preview["result"]["confirmation_token"]

    _, first = await call(server, "create_ticket", {**NEW_TICKET, "confirmation_token": token})
    _, again = await call(server, "create_ticket", {**NEW_TICKET, "confirmation_token": token})

    assert first["ok"] and first["result"]["title"] == "Laptop won't charge"
    assert again["result"]["id"] == first["result"]["id"]  # the retry is a no-op
    assert ticket_count(desk) == before + 1


@pytest.mark.integration
async def test_a_token_only_fits_the_arguments_it_was_issued_for(server, desk):
    _, preview = await call(server, "update_ticket", {"ticket_id": 1002, "priority": "P2"})
    token = preview["result"]["confirmation_token"]

    changed = {"ticket_id": 1002, "priority": "P1", "confirmation_token": token}
    is_error, result = await call(server, "update_ticket", changed)

    assert is_error and result["error"]["code"] == "invalid_confirmation"
    assert result["error"]["preview"] == {"ticket_id": 1002, "priority": "P1"}  # what to show
    _, ticket = await call(server, "get_ticket", {"ticket_id": 1002})
    assert ticket["result"]["priority"] == "P3"  # unchanged

    # The fresh token in the error is valid for the new arguments.
    fresh = {**changed, "confirmation_token": result["error"]["confirmation_token"]}
    _, done = await call(server, "update_ticket", fresh)
    assert done["result"]["priority"] == "P1"


@pytest.mark.integration
async def test_the_same_change_written_differently_keeps_its_token(server):
    """Tokens are checked against the normalised arguments: "INC-1002" and 1002 are the
    same ticket, so the person saw the same change."""
    _, preview = await call(server, "update_ticket", {"ticket_id": "INC-1002", "priority": "P2"})
    token = preview["result"]["confirmation_token"]
    confirmed = {"ticket_id": 1002, "priority": "P2", "confirmation_token": token}
    is_error, result = await call(server, "update_ticket", confirmed)
    assert not is_error and result["result"]["priority"] == "P2"


@pytest.mark.integration
async def test_a_made_up_or_foreign_token_is_rejected(server, desk):
    other_server = build_mcp_server(desk, None, Confirmations(b"another secret"))
    _, preview = await call(
        other_server,
        "add_comment",
        {"ticket_id": 1001, "author": "li.wei@halcyon-dynamics.example", "body": "Any news?"},
    )
    foreign = preview["result"]["confirmation_token"]

    for token in [foreign, "0123456789abcdef.00000000000000000000000000000000", "nonsense"]:
        comment = {
            "ticket_id": 1001,
            "author": "li.wei@halcyon-dynamics.example",
            "body": "Any news?",
            "confirmation_token": token,
        }
        is_error, result = await call(server, "add_comment", comment)
        assert is_error and result["error"]["code"] == "invalid_confirmation", token


@pytest.mark.integration
async def test_invalid_arguments_get_no_token_and_write_nothing(server, desk):
    before = ticket_count(desk)
    bad = {**NEW_TICKET, "category": "printers", "requester": "li@gmail.com"}
    async with Client(server) as client:
        result = await client.call_tool("create_ticket", bad)

    assert result.is_error  # rejected by the schema before our code runs
    assert "category" in result.content[0].text and "requester" in result.content[0].text
    assert ticket_count(desk) == before


@pytest.mark.integration
async def test_rules_across_fields_are_still_checked_by_the_input_model(server):
    """Each parameter can be valid on its own and the call still wrong: resolving needs a
    resolution. The MCP schema can't say that; the input model does."""
    is_error, result = await call(
        server, "update_ticket", {"ticket_id": 1001, "status": "resolved"}
    )
    assert is_error and result["error"]["code"] == "invalid_input"
    assert "confirmation_token" not in json.dumps(result)


@pytest.mark.integration
async def test_an_update_previews_only_the_fields_that_were_given(server):
    _, result = await call(server, "update_ticket", {"ticket_id": 1003, "status": "in_progress"})
    assert result["result"]["preview"] == {"ticket_id": 1003, "status": "in_progress"}


# --- resources and prompts -----------------------------------------------------------------


@pytest.fixture
def kb_server(kb_session_factory):
    return build_mcp_server(ServiceDesk(kb_session_factory), None)


@pytest.mark.integration
async def test_the_article_list_links_to_every_article(kb_server):
    async with Client(kb_server) as client:
        listed = await client.read_resource("kb://articles")
    articles = json.loads(listed.contents[0].text)["articles"]

    assert len(articles) == 19
    first = articles[0]
    assert first["doc_id"] == "KB-0001"
    assert first["uri"] == "kb://articles/KB-0001"


@pytest.mark.integration
async def test_an_article_reads_as_markdown_including_pdfs(kb_server):
    async with Client(kb_server) as client:
        markdown = await client.read_resource("kb://articles/KB-0001")
        pdf = await client.read_resource("kb://articles/POL-0006")

    assert markdown.contents[0].text.startswith("# Connecting to the Corporate VPN")
    assert "## Before you start" in markdown.contents[0].text
    assert pdf.contents[0].text.startswith("# Remote Working Equipment Policy")
    assert "## 1. Scope" in pdf.contents[0].text


@pytest.mark.integration
async def test_an_unknown_article_is_an_error(kb_server):
    async with Client(kb_server) as client:
        with pytest.raises(MCPError, match="No knowledge base article KB-9999"):
            await client.read_resource("kb://articles/KB-9999")


async def test_the_prompt_starts_with_the_persons_problem():
    server = build_mcp_server(ServiceDesk(None), None)
    async with Client(server) as client:
        prompt = await client.get_prompt(
            "report_it_issue",
            {"problem": "Outlook keeps asking for my password", "requester_email": "li@x"},
        )
    text = prompt.messages[0].content.text
    assert text.startswith("An employee (li@x) reports: Outlook keeps asking for my password")
    assert "check the service status first" in text


# --- confirmation tokens on their own --------------------------------------------------------


def test_a_token_checks_out_for_the_same_call_only():
    confirmations = Confirmations(b"secret")
    token = confirmations.issue("update_ticket", {"ticket_id": 1, "priority": "P2"})
    nonce = token.split(".")[0]

    assert confirmations.check(token, "update_ticket", {"priority": "P2", "ticket_id": 1}) == nonce
    assert confirmations.check(token, "update_ticket", {"ticket_id": 1, "priority": "P1"}) is None
    assert confirmations.check(token, "add_comment", {"ticket_id": 1, "priority": "P2"}) is None
    assert Confirmations(b"other").check(token, "update_ticket", {"ticket_id": 1}) is None
    assert confirmations.check("no-dot-here", "update_ticket", {}) is None


def test_every_preview_gets_a_new_token():
    confirmations = Confirmations()
    arguments = {"ticket_id": 1, "priority": "P2"}
    assert confirmations.issue("update_ticket", arguments) != confirmations.issue(
        "update_ticket", arguments
    )
