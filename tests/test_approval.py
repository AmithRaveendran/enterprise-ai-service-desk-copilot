"""Phase 6: human approval for writes, and recovering a turn that failed part-way.

Same approach as test_agent.py: a scripted fake chat model and an in-memory checkpointer,
plus the real service desk and Postgres where the point is what gets written.
"""

import uuid

import openai
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy import func, select

from servicedesk.agent.checkpointer import postgres_checkpointer
from servicedesk.agent.copilot import Copilot, ThreadNotFoundError, ThreadStateError
from servicedesk.agent.graph import build_graph
from servicedesk.agent.prompts import APPROVAL_REPLY
from servicedesk.agent.tools import KB_TOOL, AgentTools
from servicedesk.db.models import Ticket
from tests.conftest import FakeChatModel, tool_call
from tests.test_agent import FakeKnowledgeBase, kb_answer, make_copilot, tool_results

PRIYA = "priya.sharma@halcyon-dynamics.example"
NEW_TICKET = {
    "title": "VPN drops every 10 minutes",
    "description": "Since this morning the VPN disconnects every 10 minutes.",
    "category": "network",
    "requester": PRIYA,
}


def create_call(args=None, call_id="call_1"):
    return tool_call("create_ticket", args or NEW_TICKET, call_id)


def ticket_count(session_factory) -> int:
    with session_factory() as session:
        return session.scalar(select(func.count()).select_from(Ticket))


# --- pausing for approval -----------------------------------------------------------------


def test_a_write_call_pauses_the_turn_before_anything_runs():
    model = FakeChatModel(create_call())
    copilot = make_copilot(model)  # desk=None: running the write would fail, proving it didn't
    turn = copilot.chat("Please raise a ticket for my VPN")

    assert turn.status == "awaiting_approval"
    assert turn.reply == APPROVAL_REPLY  # the model wrote no text alongside its call
    assert turn.pending_actions == [
        {"id": "call_1", "tool": "create_ticket", "arguments": NEW_TICKET | {"priority": "P3"}}
    ]
    assert copilot.status(turn.thread_id) == "awaiting_approval"
    assert tool_results(copilot, turn.thread_id) == []  # no tool has run


def test_the_person_sees_normalised_arguments():
    model = FakeChatModel(
        tool_call("update_ticket", {"ticket_id": "inc 1003", "status": "in_progress"})
    )
    turn = make_copilot(model).chat("Mark INC-1003 as in progress")
    # "inc 1003" is shown as validated (1003); fields the model didn't send are left out.
    assert turn.pending_actions[0]["arguments"] == {"ticket_id": 1003, "status": "in_progress"}


def test_read_tools_run_without_approval():
    model = FakeChatModel(tool_call(KB_TOOL, {"question": "vpn drops"}), "Restart it [1].")
    turn = make_copilot(model, FakeKnowledgeBase(kb_answer())).chat("vpn")
    assert turn.status == "completed" and turn.pending_actions == []


def test_a_write_with_invalid_arguments_is_not_shown_for_approval():
    model = FakeChatModel(
        create_call({"title": "VPN"}),  # missing fields: nothing worth approving
        "I need a few more details: what happens, and your email?",
    )
    copilot = make_copilot(model)
    turn = copilot.chat("ticket please")
    assert turn.status == "completed"
    assert turn.tools_used[0].error_code == "invalid_input"


def test_rejecting_runs_nothing_and_tells_the_model_why():
    model = FakeChatModel(create_call(), "OK, I won't raise it. Want me to change anything?")
    copilot = make_copilot(model)
    thread = copilot.chat("raise a ticket").thread_id

    turn = copilot.decide(thread, approved=False, reason="Make it P2 instead")
    assert turn.status == "completed"
    assert turn.reply == "OK, I won't raise it. Want me to change anything?"
    [result] = tool_results(copilot, thread)
    assert result["error"]["code"] == "rejected"
    assert result["error"]["reason"] == "Make it P2 instead"
    assert copilot.status(thread) == "idle"


def test_a_malformed_resume_value_counts_as_a_rejection():
    model = FakeChatModel(create_call(), "Not done.")
    copilot = make_copilot(model)
    thread = copilot.chat("raise a ticket").thread_id
    # Bypass Copilot.decide and resume with something that isn't {"approved": True}.
    copilot.graph.invoke(Command(resume="yes please"), {"configurable": {"thread_id": thread}})
    assert tool_results(copilot, thread)[0]["error"]["code"] == "rejected"


def test_one_decision_covers_every_write_in_the_message():
    both = AIMessage(
        "",
        tool_calls=[
            {"name": "create_ticket", "args": NEW_TICKET, "id": "c1"},
            {
                "name": "add_comment",
                "args": {"ticket_id": 1001, "author": PRIYA, "body": "Still broken."},
                "id": "c2",
            },
        ],
    )
    copilot = make_copilot(FakeChatModel(both, "Nothing changed."))
    turn = copilot.chat("ticket + comment")
    assert [a["tool"] for a in turn.pending_actions] == ["create_ticket", "add_comment"]
    copilot.decide(turn.thread_id, approved=False)
    codes = [r["error"]["code"] for r in tool_results(copilot, turn.thread_id)]
    assert codes == ["rejected", "rejected"]


# --- what each thread state allows ---------------------------------------------------------


def test_a_new_message_is_refused_while_a_change_awaits_approval():
    copilot = make_copilot(FakeChatModel(create_call()))
    thread = copilot.chat("raise a ticket").thread_id
    with pytest.raises(ThreadStateError) as error:
        copilot.chat("hello?", thread)
    assert error.value.code == "approval_pending"


def test_approving_when_nothing_is_pending_is_refused():
    copilot = make_copilot(FakeChatModel("Hi."))
    thread = copilot.chat("hi").thread_id
    with pytest.raises(ThreadStateError) as error:
        copilot.decide(thread, approved=True)
    assert error.value.code == "no_pending_action"
    with pytest.raises(ThreadNotFoundError):
        copilot.decide("no-such-thread", approved=True)
    with pytest.raises(ThreadStateError) as error:
        copilot.resume(thread)
    assert error.value.code == "nothing_to_resume"


# --- recovery ------------------------------------------------------------------------------


class FlakyModel(FakeChatModel):
    """Raises a model API error on the given call numbers (1-based), like an outage."""

    def __init__(self, *script, fail_on=(2,)):
        super().__init__(*script)
        self.fail_on, self.attempts = set(fail_on), 0

    def _reply(self, messages, tool_choice):
        self.attempts += 1
        if self.attempts in self.fail_on:
            raise openai.APIConnectionError(request=None)
        return super()._reply(messages, tool_choice)


def test_a_turn_that_failed_part_way_resumes_without_repeating_finished_steps():
    kb = FakeKnowledgeBase(kb_answer())
    model = FlakyModel(tool_call(KB_TOOL, {"question": "vpn"}), "Restart it [1].", fail_on={2})
    copilot = make_copilot(model, kb)
    thread = "t-recover"

    with pytest.raises(openai.APIConnectionError):
        copilot.chat("vpn", thread)  # the KB ran, then the model call after it failed
    assert copilot.status(thread) == "interrupted"
    with pytest.raises(ThreadStateError):
        copilot.chat("are you there?", thread)  # must resume first

    turn = copilot.resume(thread)
    assert turn.reply == "Restart it [1]." and len(turn.citations) == 1
    assert kb.questions == ["vpn"]  # the finished tool step was not run again
    assert copilot.status(thread) == "idle"


def test_a_failed_first_step_can_be_resumed_too():
    copilot = make_copilot(FlakyModel("Hello!", fail_on={1}))
    with pytest.raises(openai.APIConnectionError):
        copilot.chat("hi", "t-first")
    assert copilot.status("t-first") == "interrupted"
    assert copilot.resume("t-first").reply == "Hello!"


# --- with the real service desk ------------------------------------------------------------


@pytest.mark.integration
def test_an_approved_ticket_is_created(desk, desk_session_factory):
    model = FakeChatModel(create_call(), "Done: INC-1006 is raised for you.")
    copilot = make_copilot(model, desk=desk)
    thread = copilot.chat("raise a ticket").thread_id
    assert ticket_count(desk_session_factory) == 5  # nothing yet

    turn = copilot.decide(thread, approved=True)
    assert turn.status == "completed" and turn.reply == "Done: INC-1006 is raised for you."
    [result] = tool_results(copilot, thread)
    assert result["ok"] and result["result"]["number"] == "INC-1006"
    assert ticket_count(desk_session_factory) == 6


class CrashAfterWrites(AgentTools):
    """Runs the tool, then 'crashes' once: as if the process died after the database
    committed the write but before LangGraph saved the checkpoint."""

    crashed = False

    def run(self, name, arguments, citations_so_far=0, idempotency_key=None):
        result = super().run(name, arguments, citations_so_far, idempotency_key)
        if self.needs_approval(name) and not self.crashed:
            CrashAfterWrites.crashed = True
            raise RuntimeError("simulated crash after the write")
        return result


@pytest.mark.integration
def test_a_crash_after_an_approved_write_does_not_write_twice(desk, desk_session_factory):
    CrashAfterWrites.crashed = False
    tools = CrashAfterWrites(desk, FakeKnowledgeBase())
    model = FakeChatModel(create_call(), "Raised INC-1006.")
    copilot = Copilot(build_graph(model, tools, checkpointer=InMemorySaver()))
    thread = copilot.chat("raise a ticket").thread_id

    with pytest.raises(RuntimeError, match="simulated crash"):
        copilot.decide(thread, approved=True)
    assert ticket_count(desk_session_factory) == 6  # the write had committed
    assert copilot.status(thread) == "interrupted"  # but the tools step wasn't saved

    turn = copilot.resume(thread)  # re-runs the tools step: same call id, same key
    assert ticket_count(desk_session_factory) == 6  # still one ticket
    assert tool_results(copilot, thread)[0]["result"]["number"] == "INC-1006"
    assert turn.reply == "Raised INC-1006."


@pytest.mark.integration
def test_an_approval_can_come_after_a_restart(test_database_url, desk, desk_session_factory):
    thread = f"test-{uuid.uuid4()}"
    with postgres_checkpointer(test_database_url) as saver:
        copilot = make_copilot(FakeChatModel(create_call()), desk=desk, checkpointer=saver)
        assert copilot.chat("raise a ticket", thread).status == "awaiting_approval"

    # A new pool and a new graph, as if the API restarted while waiting for the person.
    with postgres_checkpointer(test_database_url) as saver:
        model = FakeChatModel("Raised INC-1006.")
        copilot = make_copilot(model, desk=desk, checkpointer=saver)
        assert copilot.pending_actions(thread)[0]["tool"] == "create_ticket"
        assert copilot.decide(thread, approved=True).reply == "Raised INC-1006."
    assert ticket_count(desk_session_factory) == 6


def test_a_tool_database_error_reaches_the_model_as_data():
    from sqlalchemy.exc import OperationalError

    from servicedesk.desk.service import ServiceDesk

    def database_down():  # a session factory that can't connect
        raise OperationalError("connect", {}, Exception("connection refused"))

    model = FakeChatModel(
        tool_call("get_service_status", {"service": "vpn"}), "I can't check right now."
    )
    copilot = make_copilot(model, desk=ServiceDesk(database_down))
    turn = copilot.chat("is the VPN down?")
    assert turn.tools_used[0].error_code == "unavailable"
    assert turn.reply == "I can't check right now."


def test_a_tool_bug_does_not_crash_the_conversation():
    def broken_kb(question):
        raise KeyError("oops")

    model = FakeChatModel(tool_call(KB_TOOL, {"question": "vpn"}), "Something went wrong.")
    turn = make_copilot(model, broken_kb).chat("vpn")
    assert turn.tools_used[0].error_code == "tool_failed"
    assert turn.reply == "Something went wrong."
