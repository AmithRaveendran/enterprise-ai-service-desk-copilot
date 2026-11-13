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
    assert TestClient(app).post("/chat", json={"message": "hi"}).status_code == 502
