"""The agent as a LangGraph state machine.

        START
          │
          ▼
       ┌───────┐  wants tools (and under the round limit)  ┌───────┐
       │ agent │ ─────────────────────────────────────────▶ │ tools │
       └───────┘ ◀───────────────────────────────────────── └───────┘
          │ final reply                  tool results
          ▼
      ┌──────────┐
      │ finalize │  check [n] citations against this turn's sources
      └──────────┘
          │
          ▼
         END

- agent: the model reads the system prompt + recent conversation and either replies or asks
  for one or more tool calls (it decides the route: no hard-coded intent classifier).
- tools: runs every requested call through AgentTools and appends one ToolMessage per call.
- finalize: like Phase 3, keeps only [n] markers that point at a real source found in this
  turn, renumbers them [1], [2]..., and returns only the sources actually cited.

The loop is bounded: after `max_tool_rounds` the model is called with tool_choice="none",
so it must answer with what it has. A checkpointer (Postgres in production) saves the
state after every step, keyed by thread_id.
"""

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from servicedesk.agent.prompts import ROUND_LIMIT_REPLY, SYSTEM_PROMPT
from servicedesk.agent.state import AgentState
from servicedesk.agent.tools import AgentTools
from servicedesk.rag.answerer import check_citations


def build_graph(
    model: BaseChatModel,
    tools: AgentTools,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    max_tool_rounds: int = 5,
    history_messages: int = 40,
) -> CompiledStateGraph:
    model_with_tools = model.bind_tools(tools.specs)
    # Same tools visible (the history mentions them), but the model may not call any.
    model_must_answer = model.bind_tools(tools.specs, tool_choice="none")

    def agent(state: AgentState) -> dict[str, Any]:
        # Only the most recent messages go to the model, so a long conversation can't
        # overflow the context window. start_on="human" never cuts a tool exchange in half.
        recent = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,  # counts messages, not tokens: simple and predictable
            max_tokens=history_messages,
            start_on="human",
        )
        llm = model_with_tools if state["tool_rounds"] < max_tool_rounds else model_must_answer
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), *recent])
        return {"messages": [response]}

    def run_tools(state: AgentState) -> dict[str, Any]:
        request: AIMessage = state["messages"][-1]
        citations = list(state["citations"])
        results: list[ToolMessage] = []
        for call in request.tool_calls:
            result, new_citations = tools.run(call["name"], call["args"], len(citations))
            citations.extend(new_citations)
            results.append(_tool_message(call["id"], call["name"], result))
        # Arguments that weren't even valid JSON. Every call must get an answer, or the
        # next request to the model is rejected, so these get an error result too.
        for bad in request.invalid_tool_calls:
            error = {"code": "invalid_input", "message": "The arguments were not valid JSON."}
            result = {"ok": False, "error": error}
            results.append(_tool_message(bad["id"], bad["name"] or "unknown", result))
        rounds = state["tool_rounds"] + 1
        return {"messages": results, "citations": citations, "tool_rounds": rounds}

    def finalize(state: AgentState) -> dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        text, cited = check_citations(last.text, len(state["citations"]))
        citations = [
            {**state["citations"][n - 1], "number": i} for i, n in enumerate(cited, start=1)
        ]
        # Same id → add_messages replaces the model's message with the checked one. The new
        # message has no tool calls, so a cut-off loop can't leave unanswered calls behind.
        reply = AIMessage(
            content=text or ROUND_LIMIT_REPLY,
            id=last.id,
            usage_metadata=last.usage_metadata,
            response_metadata=last.response_metadata,
        )
        return {"messages": [reply], "citations": citations}

    def route(state: AgentState) -> Literal["tools", "finalize"]:
        last: AIMessage = state["messages"][-1]
        wants_tools = bool(last.tool_calls or last.invalid_tool_calls)
        return "tools" if wants_tools and state["tool_rounds"] < max_tool_rounds else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _tool_message(call_id: str, name: str, result: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False),
        tool_call_id=call_id,
        name=name,
        status="success" if result.get("ok") else "error",
    )
