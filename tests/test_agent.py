"""The LangGraph agent: routing, tools, citations, memory and the safety limits.

The chat model is a scripted fake (conftest.FakeChatModel), so these tests are offline and
deterministic: they check what the graph does with the model's decisions, not the model.
Unit tests use LangGraph's in-memory checkpointer; the Postgres one is tested at the end.
"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from servicedesk.agent.checkpointer import checkpoint_conninfo, postgres_checkpointer
from servicedesk.agent.copilot import Copilot
from servicedesk.agent.graph import build_graph
from servicedesk.agent.prompts import ROUND_LIMIT_REPLY, SYSTEM_PROMPT
from servicedesk.agent.tools import KB_TOOL, AgentTools
from servicedesk.rag.answerer import REFUSAL_MESSAGE, Answer, Citation
from tests.conftest import FakeChatModel, tool_call


def kb_answer(text: str = "Restart the VPN client [1].", doc_id: str = "KB-0007") -> Answer:
    return Answer(
        question="q",
        answer=text,
        refused=False,
        citations=[
            Citation(
                number=1,
                chunk_id=42,
                doc_id=doc_id,
                title=f"Article {doc_id}",
                source_path=f"data/kb/{doc_id}.md",
                section="Troubleshooting",
                page=None,
                content="Restart the client, then reconnect.",
            )
        ],
    )


class FakeKnowledgeBase:
    """Returns the given answers in turn and records the questions it was asked."""

    def __init__(self, *answers: Answer):
        self.answers = list(answers)
        self.questions: list[str] = []

    def __call__(self, question: str) -> Answer:
        self.questions.append(question)
        return self.answers.pop(0)


def make_copilot(model, kb=None, desk=None, checkpointer=None, **options) -> Copilot:
    tools = AgentTools(desk, kb or FakeKnowledgeBase())
    graph = build_graph(model, tools, checkpointer=checkpointer or InMemorySaver(), **options)
    return Copilot(graph)


def tool_results(copilot: Copilot, thread_id: str) -> list[dict]:
    state = copilot.graph.get_state({"configurable": {"thread_id": thread_id}})
    return [json.loads(m.text) for m in state.values["messages"] if isinstance(m, ToolMessage)]


# --- tools offered to the model ---------------------------------------------------------


def test_the_agent_is_offered_the_knowledge_base_and_the_desk_tools():
    tools = AgentTools(None, FakeKnowledgeBase())
    names = {spec["function"]["name"] for spec in tools.specs}
    reads = {KB_TOOL, "get_service_status", "search_tickets", "get_ticket"}
    writes = {"create_ticket", "update_ticket", "add_comment"}
    assert names == reads | writes
    assert {n for n in names if tools.needs_approval(n)} == writes
    assert "set_service_status" not in names  # admin-only: never a tool


def test_tool_schemas_forbid_unknown_arguments():
    for spec in AgentTools(None, FakeKnowledgeBase()).specs:
        assert spec["function"]["parameters"]["additionalProperties"] is False


# --- one turn ---------------------------------------------------------------------------


def test_a_plain_reply_uses_no_tools():
    model = FakeChatModel("Hello! How can I help?")
    turn = make_copilot(model).chat("hi")
    assert turn.reply == "Hello! How can I help?"
    assert turn.tools_used == [] and turn.citations == []
    # The system prompt always goes first.
    first = model.calls[0]["messages"][0]
    assert isinstance(first, SystemMessage) and first.content == SYSTEM_PROMPT


def test_knowledge_base_answer_keeps_its_citation():
    kb = FakeKnowledgeBase(kb_answer())
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "vpn keeps dropping"}),
        "Try this: restart the VPN client [1].",
    )
    turn = make_copilot(model, kb).chat("My VPN keeps dropping")

    assert kb.questions == ["vpn keeps dropping"]
    assert turn.reply == "Try this: restart the VPN client [1]."
    assert [(c["number"], c["doc_id"]) for c in turn.citations] == [(1, "KB-0007")]
    assert [(t.name, t.ok) for t in turn.tools_used] == [(KB_TOOL, True)]
    # The model saw the tool result: the grounded answer and its sources, not chunk texts.
    result = json.loads(model.calls[1]["messages"][-1].text)
    assert result["result"]["found"] is True
    assert result["result"]["sources"][0] == {
        "number": 1,
        "doc_id": "KB-0007",
        "title": "Article KB-0007",
        "section": "Troubleshooting",
    }


def test_two_knowledge_base_calls_get_distinct_numbers():
    kb = FakeKnowledgeBase(
        kb_answer("Restart it [1].", "KB-0007"), kb_answer("Use MFA [1].", "KB-0003")
    )
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "vpn"}, "c1"),
        tool_call(KB_TOOL, {"question": "mfa"}, "c2"),
        lambda messages: AIMessage("MFA first [2], then restart [1]."),
    )
    copilot = make_copilot(model, kb)
    turn = copilot.chat("vpn and mfa")

    # The second answer's [1] was shifted to [2] before the model saw it...
    second = tool_results(copilot, turn.thread_id)[1]
    assert second["result"]["answer"] == "Use MFA [2]."
    # ...and the reply is renumbered in order of appearance, like Phase 3 answers.
    assert turn.reply == "MFA first [1], then restart [2]."
    assert [c["doc_id"] for c in turn.citations] == ["KB-0003", "KB-0007"]


def test_invented_citations_are_removed():
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "vpn"}), "Restart the client [1]. Also reboot [3]."
    )
    turn = make_copilot(model, FakeKnowledgeBase(kb_answer())).chat("vpn")
    assert turn.reply == "Restart the client [1]. Also reboot."
    assert len(turn.citations) == 1


def test_citations_without_a_knowledge_base_call_are_removed():
    turn = make_copilot(FakeChatModel("The VPN is degraded [1].")).chat("vpn?")
    assert turn.reply == "The VPN is degraded." and turn.citations == []


def test_knowledge_base_refusal_reaches_the_model_as_not_found():
    refusal = Answer(
        question="q", answer=REFUSAL_MESSAGE, refused=True, refusal_reason="no_relevant_sources"
    )
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "capital of France"}), "I couldn't find that."
    )
    copilot = make_copilot(model, FakeKnowledgeBase(refusal))
    turn = copilot.chat("capital of France?")
    assert tool_results(copilot, turn.thread_id)[0] == {
        "ok": True,
        "result": {"found": False, "reason": "no_relevant_sources"},
    }
    assert turn.citations == []


# --- safety -----------------------------------------------------------------------------


def test_a_tool_that_isnt_offered_is_refused_even_if_the_model_calls_one():
    model = FakeChatModel(
        tool_call("set_service_status", {"service": "vpn", "status": "operational"}),
        "I can't change service status.",
    )
    copilot = make_copilot(model)  # desk=None: running the tool would crash, proving it didn't
    turn = copilot.chat("mark the VPN as fixed")
    assert turn.tools_used[0].error_code == "not_allowed"


def test_invalid_arguments_come_back_as_data_and_the_model_can_retry():
    kb = FakeKnowledgeBase(kb_answer())
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "x", "urgency": "high"}, "c1"),
        tool_call(KB_TOOL, {"question": "vpn drops"}, "c2"),
        "Restart the client [1].",
    )
    copilot = make_copilot(model, kb)
    turn = copilot.chat("vpn")

    first = tool_results(copilot, turn.thread_id)[0]
    assert first["error"]["code"] == "invalid_input"
    assert {p["field"] for p in first["error"]["problems"]} == {"question", "urgency"}
    assert [t.ok for t in turn.tools_used] == [False, True]
    assert turn.reply == "Restart the client [1]."


def test_unparseable_tool_arguments_get_an_error_result():
    broken = AIMessage(
        "", invalid_tool_calls=[{"name": KB_TOOL, "args": "{oops", "id": "c1", "error": "bad"}]
    )
    turn = make_copilot(FakeChatModel(broken, "Sorry.")).chat("vpn")
    assert turn.tools_used[0].error_code == "invalid_input"
    assert turn.reply == "Sorry."


def test_tool_loop_is_cut_off_after_the_round_limit():
    rounds = iter(range(100))

    def keeps_calling_tools(messages):  # a new message each time, with a unique call id
        return tool_call(KB_TOOL, {"question": "vpn again"}, f"c{next(rounds)}")

    kb = FakeKnowledgeBase(*[kb_answer() for _ in range(10)])
    model = FakeChatModel(*[keeps_calling_tools] * 4)
    turn = make_copilot(model, kb, max_tool_rounds=3).chat("vpn")

    assert len(kb.questions) == 3  # exactly max_tool_rounds rounds of tools ran
    assert [c["tool_choice"] for c in model.calls] == [None, None, None, "none"]
    # The model still asked for a tool on the last call; it was not run and the reply is
    # the fallback message, with no unanswered tool call left in the history.
    assert turn.reply == ROUND_LIMIT_REPLY


def test_a_knowledge_base_outage_is_reported_as_a_tool_error():
    import openai

    def broken_kb(question):
        raise openai.APIConnectionError(request=None)

    model = FakeChatModel(tool_call(KB_TOOL, {"question": "vpn"}), "The KB is down, sorry.")
    turn = make_copilot(model, broken_kb).chat("vpn")
    assert turn.tools_used[0].error_code == "unavailable"


# --- memory -----------------------------------------------------------------------------


def test_the_same_thread_remembers_earlier_messages():
    model = FakeChatModel("Hi Priya.", "Your name is Priya.")
    copilot = make_copilot(model)
    first = copilot.chat("I'm Priya")
    second = copilot.chat("What's my name?", first.thread_id)

    assert second.thread_id == first.thread_id
    sent = [m.text for m in model.calls[1]["messages"][1:]]
    assert sent == ["I'm Priya", "Hi Priya.", "What's my name?"]
    assert copilot.history(first.thread_id) == [
        {"role": "user", "content": "I'm Priya"},
        {"role": "assistant", "content": "Hi Priya."},
        {"role": "user", "content": "What's my name?"},
        {"role": "assistant", "content": "Your name is Priya."},
    ]


def test_a_new_thread_starts_empty():
    model = FakeChatModel("Hi Priya.", "I don't know your name.")
    copilot = make_copilot(model)
    copilot.chat("I'm Priya")
    copilot.chat("What's my name?")  # no thread id: a new conversation
    assert [m.text for m in model.calls[1]["messages"][1:]] == ["What's my name?"]


def test_unknown_thread_has_no_history():
    assert make_copilot(FakeChatModel()).history("no-such-thread") == []


def test_only_recent_messages_are_sent_and_never_a_half_tool_exchange():
    model = FakeChatModel(
        "one", "two", tool_call(KB_TOOL, {"question": "vpn"}), "three [1]", "four"
    )
    copilot = make_copilot(model, FakeKnowledgeBase(kb_answer()), history_messages=4)
    thread = copilot.chat("first").thread_id
    copilot.chat("second", thread)
    copilot.chat("third", thread)
    copilot.chat("fourth", thread)

    sent = model.calls[-1]["messages"]
    assert isinstance(sent[1], HumanMessage)  # the window starts on a person's message
    assert len(sent) - 1 <= 4  # system prompt + at most 4 recent messages
    assert sent[-1].text == "fourth"


def test_citations_are_reset_for_every_turn():
    model = FakeChatModel(
        tool_call(KB_TOOL, {"question": "vpn"}), "Restart [1].", "You're welcome."
    )
    copilot = make_copilot(model, FakeKnowledgeBase(kb_answer()))
    first = copilot.chat("vpn")
    second = copilot.chat("thanks", first.thread_id)
    assert len(first.citations) == 1 and second.citations == []


# --- with the real service desk and Postgres ---------------------------------------------


@pytest.mark.integration
def test_status_check_runs_against_the_database(desk):
    model = FakeChatModel(
        tool_call("get_service_status", {"service": "vpn"}), "The VPN is degraded right now."
    )
    copilot = make_copilot(model, desk=desk)
    turn = copilot.chat("VPN is broken")
    result = tool_results(copilot, turn.thread_id)[0]
    assert result["result"]["services"][0]["name"] == "VPN"
    assert result["result"]["services"][0]["status"] == "degraded"


def test_checkpoint_conninfo_drops_the_sqlalchemy_driver():
    url = "postgresql+psycopg://u:p@localhost:5432/db"
    assert checkpoint_conninfo(url) == "postgresql://u:p@localhost:5432/db"


@pytest.mark.integration
def test_a_conversation_survives_a_restart(test_database_url):
    thread = f"test-{uuid.uuid4()}"
    with postgres_checkpointer(test_database_url) as saver:
        make_copilot(FakeChatModel("Hi Priya."), checkpointer=saver).chat("I'm Priya", thread)

    # A new pool and a new graph: as if the API had restarted.
    with postgres_checkpointer(test_database_url) as saver:
        model = FakeChatModel("Your name is Priya.")
        copilot = make_copilot(model, checkpointer=saver)
        copilot.chat("What's my name?", thread)
        assert [m.text for m in model.calls[0]["messages"][1:]] == [
            "I'm Priya",
            "Hi Priya.",
            "What's my name?",
        ]
        assert len(copilot.history(thread)) == 4
