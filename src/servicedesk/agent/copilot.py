"""A small, framework-free front for the graph: chat(message, thread_id) → ChatTurn.

The API and the CLI both use this, so neither needs to know about LangGraph configs,
message classes or checkpoints.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from servicedesk.agent.graph import build_graph
from servicedesk.agent.tools import AgentTools
from servicedesk.config import Settings
from servicedesk.desk.service import ServiceDesk
from servicedesk.rag.answerer import Answerer


@dataclass
class ToolUse:
    name: str
    ok: bool
    error_code: str | None = None


@dataclass
class ChatTurn:
    thread_id: str
    reply: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[ToolUse] = field(default_factory=list)
    input_tokens: int = 0  # the agent's own model calls (knowledge-base answers not included)
    output_tokens: int = 0


class Copilot:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    def chat(self, message: str, thread_id: str | None = None) -> ChatTurn:
        thread_id = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        # The new message is appended to the saved conversation (add_messages); citations
        # and the tool-round counter start again for this turn.
        state = self.graph.invoke(
            {"messages": [HumanMessage(message.strip())], "citations": [], "tool_rounds": 0},
            config,
        )
        turn = _current_turn(state["messages"])
        tools_used = [_tool_use(m) for m in turn if isinstance(m, ToolMessage)]
        usage = [m.usage_metadata for m in turn if isinstance(m, AIMessage) and m.usage_metadata]
        return ChatTurn(
            thread_id=thread_id,
            reply=state["messages"][-1].text,
            citations=state["citations"],
            tools_used=tools_used,
            input_tokens=sum(u["input_tokens"] for u in usage),
            output_tokens=sum(u["output_tokens"] for u in usage),
        )

    def history(self, thread_id: str) -> list[dict[str, str]]:
        """The conversation as the person saw it: their messages and the final replies
        (tool calls and tool results are left out). Empty if the thread doesn't exist."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = snapshot.values.get("messages", [])
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.text}
            for m in messages
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]


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
