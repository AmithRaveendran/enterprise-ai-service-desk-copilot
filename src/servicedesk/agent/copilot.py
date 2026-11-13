"""A small, framework-free front for the graph. The API and the CLI both use it, so neither
needs to know about LangGraph configs, message classes, interrupts or checkpoints.

A conversation (thread) is always in one of these states, read from its last checkpoint:

    new                nothing saved yet
    idle               the last turn finished           → chat()
    awaiting_approval  paused before a write            → decide(approved=True/False)
    interrupted        the last turn failed part-way    → resume()
                       (e.g. the model API was down, or the process crashed)

Each operation is only allowed in its state, so a thread can't, say, receive a new message
while a change is waiting for approval: LangGraph would start a fresh run and leave the
paused tool call unanswered.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from servicedesk.agent.graph import build_graph
from servicedesk.agent.prompts import APPROVAL_REPLY
from servicedesk.agent.tools import AgentTools
from servicedesk.config import Settings
from servicedesk.desk.service import ServiceDesk
from servicedesk.rag.answerer import Answerer

ThreadStatus = Literal["new", "idle", "awaiting_approval", "interrupted"]


class ChatError(Exception):
    """A request the thread's current state doesn't allow. `code` is machine-readable."""

    code = "chat_error"


class ThreadNotFoundError(ChatError):
    code = "thread_not_found"


class ThreadStateError(ChatError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ToolUse:
    name: str
    ok: bool
    error_code: str | None = None


@dataclass
class ChatTurn:
    thread_id: str
    reply: str
    # "awaiting_approval": the turn is paused; pending_actions lists the changes to decide on.
    status: Literal["completed", "awaiting_approval"] = "completed"
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[ToolUse] = field(default_factory=list)
    # The agent's own model calls in this turn so far (knowledge-base answers not included).
    input_tokens: int = 0
    output_tokens: int = 0


def new_thread_id() -> str:
    return str(uuid4())


class Copilot:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    # --- the three ways to move a conversation forward ------------------------------------

    def chat(self, message: str, thread_id: str | None = None) -> ChatTurn:
        """Send a message. Starts a new thread if thread_id is None or unknown."""
        thread_id = thread_id or new_thread_id()
        status = self.status(thread_id)
        if status == "awaiting_approval":
            raise ThreadStateError(
                "approval_pending", "A change is waiting for approval: approve or reject it first."
            )
        if status == "interrupted":
            raise ThreadStateError(
                "turn_incomplete", "The last turn didn't finish: resume it first."
            )
        # The new message is appended to the saved conversation (add_messages); citations,
        # the tool-round counter and approvals start again for this turn.
        turn_input = {
            "messages": [HumanMessage(message.strip())],
            "citations": [],
            "tool_rounds": 0,
            "approvals": {},
        }
        self.graph.invoke(turn_input, _config(thread_id))
        return self._turn(thread_id)

    def decide(self, thread_id: str, approved: bool, reason: str | None = None) -> ChatTurn:
        """Approve or reject the change(s) the paused turn is waiting on, and continue it."""
        self._require(thread_id, "awaiting_approval", "no_pending_action", "Nothing to approve.")
        # Command(resume=...) continues the paused run: the approve node runs again and its
        # interrupt() call returns this value.
        answer = {"approved": approved, "reason": reason or ""}
        self.graph.invoke(Command(resume=answer), _config(thread_id))
        return self._turn(thread_id)

    def resume(self, thread_id: str) -> ChatTurn:
        """Finish a turn that failed part-way, from its last saved step. Steps that had
        finished (tool calls included) are not run again; an approved write that may have
        run just before a crash is protected by its idempotency key."""
        self._require(thread_id, "interrupted", "nothing_to_resume", "Nothing to resume.")
        self.graph.invoke(None, _config(thread_id))  # None = no new input: carry on
        return self._turn(thread_id)

    # --- reading a thread -----------------------------------------------------------------

    def status(self, thread_id: str) -> ThreadStatus:
        snapshot = self.graph.get_state(_config(thread_id))
        if not snapshot.values:
            return "new"
        if snapshot.interrupts:
            return "awaiting_approval"
        if snapshot.next:  # steps still to run, but nothing is waiting: it stopped on an error
            return "interrupted"
        return "idle"

    def pending_actions(self, thread_id: str) -> list[dict[str, Any]]:
        """The changes waiting for approval (empty unless awaiting_approval)."""
        snapshot = self.graph.get_state(_config(thread_id))
        return [action for i in snapshot.interrupts for action in i.value["actions"]]

    def history(self, thread_id: str) -> list[dict[str, str]]:
        """The conversation as the person saw it: their messages and the final replies
        (tool calls and tool results are left out). Empty if the thread doesn't exist."""
        snapshot = self.graph.get_state(_config(thread_id))
        messages = snapshot.values.get("messages", [])
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.text}
            for m in messages
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]

    # --- helpers --------------------------------------------------------------------------

    def _require(self, thread_id: str, wanted: ThreadStatus, code: str, message: str) -> None:
        status = self.status(thread_id)
        if status == "new":
            raise ThreadNotFoundError(f"No conversation with id {thread_id}.")
        if status != wanted:
            raise ThreadStateError(code, f"{message} The conversation is {status}.")

    def _turn(self, thread_id: str) -> ChatTurn:
        """Summarise the current turn from the saved state."""
        snapshot = self.graph.get_state(_config(thread_id))
        messages = snapshot.values["messages"]
        turn = _current_turn(messages)
        usage = [m.usage_metadata for m in turn if isinstance(m, AIMessage) and m.usage_metadata]
        common = {
            "thread_id": thread_id,
            "tools_used": [_tool_use(m) for m in turn if isinstance(m, ToolMessage)],
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
        }
        if snapshot.interrupts:
            return ChatTurn(
                reply=messages[-1].text or APPROVAL_REPLY,
                status="awaiting_approval",
                pending_actions=self.pending_actions(thread_id),
                **common,
            )
        return ChatTurn(reply=messages[-1].text, citations=snapshot.values["citations"], **common)


def build_copilot(
    model: BaseChatModel,
    desk: ServiceDesk,
    answerer: Answerer,
    checkpointer: BaseCheckpointSaver,
    settings: Settings,
) -> Copilot:
    """Wire the pieces together: Phase 3's Answerer and Phase 4's ServiceDesk become tools."""
    tools = AgentTools(desk, answerer.ask)
    graph = build_graph(
        model,
        tools,
        checkpointer=checkpointer,
        max_tool_rounds=settings.agent_max_tool_rounds,
        history_messages=settings.agent_history_messages,
    )
    return Copilot(graph)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _current_turn(messages: list) -> list:
    """The messages after the last HumanMessage: everything this turn produced."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return messages


def _tool_use(message: ToolMessage) -> ToolUse:
    try:
        result = json.loads(message.text)
    except ValueError:
        return ToolUse(message.name or "unknown", ok=False, error_code="bad_result")
    ok = bool(result.get("ok"))
    return ToolUse(
        message.name or "unknown", ok=ok, error_code=None if ok else result["error"]["code"]
    )
