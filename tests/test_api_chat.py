"""POST /chat and GET /chat/{thread_id}, with a scripted model and in-memory memory."""

import openai
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from servicedesk.agent.copilot import Copilot
from servicedesk.agent.graph import build_graph
from servicedesk.agent.tools import KB_TOOL, AgentTools
from servicedesk.api.app import app
from servicedesk.api.dependencies import get_copilot
from tests.conftest import FakeChatModel, tool_call
from tests.test_agent import FakeKnowledgeBase, kb_answer


@pytest.fixture
def chat_client():
    """Returns a function: give it the model's script, get a TestClient using that model."""

    def make(*script, kb=None):
        tools = AgentTools(None, kb or FakeKnowledgeBase(kb_answer()))
        copilot = Copilot(build_graph(FakeChatModel(*script), tools, checkpointer=InMemorySaver()))
        app.dependency_overrides[get_copilot] = lambda: copilot
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_chat_returns_reply_citations_and_tools(chat_client):
    client = chat_client(tool_call(KB_TOOL, {"question": "vpn drops"}), "Restart the client [1].")
    response = client.post("/chat", json={"message": "My VPN keeps dropping"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Restart the client [1]."
    assert body["citations"][0]["doc_id"] == "KB-0007"
    assert body["tools_used"] == [{"name": KB_TOOL, "ok": True, "error_code": None}]
    assert body["thread_id"]  # a new conversation got an id


def test_chat_continues_a_thread_and_shows_history(chat_client):
    client = chat_client("Hi Priya.", "Your name is Priya.")
    thread = client.post("/chat", json={"message": "I'm Priya"}).json()["thread_id"]
    second = client.post("/chat", json={"message": "My name?", "thread_id": thread}).json()
    assert second["thread_id"] == thread

    history = client.get(f"/chat/{thread}").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant", "user", "assistant"]
    assert history["messages"][-1]["content"] == "Your name is Priya."


def test_unknown_thread_is_404(chat_client):
    assert chat_client().get("/chat/nope").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"message": ""},  # empty
        {"message": "x" * 2001},  # too long
        {"message": "hi", "thread_id": "../etc/passwd"},  # not a safe id
        {"message": "hi", "thread_id": "a" * 65},  # id too long
    ],
)
def test_bad_chat_requests_are_422(chat_client, body):
    assert chat_client().post("/chat", json=body).status_code == 422


def test_model_service_errors_are_502(chat_client):
    class BrokenModel(FakeChatModel):
        def _reply(self, messages, tool_choice):
            raise openai.APIConnectionError(request=None)

    tools = AgentTools(None, FakeKnowledgeBase())
    copilot = Copilot(build_graph(BrokenModel(), tools, checkpointer=InMemorySaver()))
    app.dependency_overrides[get_copilot] = lambda: copilot
    response = TestClient(app).post("/chat", json={"message": "hi"})
    assert response.status_code == 502
    body = response.json()
    assert body["code"] == "model_unavailable"
    assert body["thread_id"]  # chosen before the turn ran, so the client can resume it
    assert f"/chat/{body['thread_id']}/resume" in body["detail"]


# --- approval and recovery (Phase 6) --------------------------------------------------------

TICKET = {
    "title": "VPN drops every 10 minutes",
    "description": "It disconnects every 10 minutes since this morning.",
    "category": "network",
    "requester": "priya.sharma@halcyon-dynamics.example",
}


def test_a_write_waits_for_approval(chat_client):
    client = chat_client(tool_call("create_ticket", TICKET))
    body = client.post("/chat", json={"message": "raise a ticket"}).json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_actions"][0]["tool"] == "create_ticket"
    assert body["pending_actions"][0]["arguments"]["priority"] == "P3"

    history = client.get(f"/chat/{body['thread_id']}").json()
    assert history["status"] == "awaiting_approval"
    assert history["pending_actions"] == body["pending_actions"]


def test_approve_continues_the_turn(chat_client):
    client = chat_client(tool_call("create_ticket", TICKET), "Done.")
    thread = client.post("/chat", json={"message": "raise a ticket"}).json()["thread_id"]
    response = client.post(f"/chat/{thread}/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed" and body["reply"] == "Done."
    assert body["tools_used"][0]["name"] == "create_ticket"
    assert client.get(f"/chat/{thread}").json()["status"] == "idle"


@pytest.mark.parametrize("payload", [{"reason": "Wrong category"}, None])
def test_reject_with_or_without_a_reason(chat_client, payload):
    client = chat_client(tool_call("create_ticket", TICKET), "OK, not raised.")
    thread = client.post("/chat", json={"message": "raise a ticket"}).json()["thread_id"]
    response = client.post(f"/chat/{thread}/reject", json=payload)
    assert response.status_code == 200
    assert response.json()["tools_used"][0]["error_code"] == "rejected"


def test_state_errors_are_409_with_a_code(chat_client):
    client = chat_client(tool_call("create_ticket", TICKET))
    thread = client.post("/chat", json={"message": "raise a ticket"}).json()["thread_id"]

    response = client.post("/chat", json={"message": "hello?", "thread_id": thread})
    assert response.status_code == 409 and response.json()["code"] == "approval_pending"
    response = client.post(f"/chat/{thread}/resume")
    assert response.status_code == 409 and response.json()["code"] == "nothing_to_resume"


def test_approving_an_unknown_thread_is_404(chat_client):
    response = chat_client().post("/chat/nope/approve")
    assert response.status_code == 404 and response.json()["code"] == "thread_not_found"


def test_a_reject_reason_is_limited(chat_client):
    client = chat_client(tool_call("create_ticket", TICKET))
    thread = client.post("/chat", json={"message": "raise a ticket"}).json()["thread_id"]
    assert client.post(f"/chat/{thread}/reject", json={"reason": "x" * 501}).status_code == 422


def test_a_502_turn_can_be_resumed(chat_client):
    from tests.test_approval import FlakyModel

    tools = AgentTools(None, FakeKnowledgeBase(kb_answer()))
    model = FlakyModel(tool_call(KB_TOOL, {"question": "vpn"}), "Restart it [1].", fail_on={2})
    copilot = Copilot(build_graph(model, tools, checkpointer=InMemorySaver()))
    app.dependency_overrides[get_copilot] = lambda: copilot
    client = TestClient(app)

    failed = client.post("/chat", json={"message": "vpn"})
    assert failed.status_code == 502
    thread = failed.json()["thread_id"]
    assert client.get(f"/chat/{thread}").json()["status"] == "interrupted"

    resumed = client.post(f"/chat/{thread}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["reply"] == "Restart it [1]."
