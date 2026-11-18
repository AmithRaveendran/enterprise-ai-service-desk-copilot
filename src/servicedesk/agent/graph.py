"""The agent as a LangGraph state machine.

        START
          │
          ▼
       ┌───────┐  wants read tools only                    ┌───────┐
       │ agent │ ─────────────────────────────────────────▶ │ tools │
       └───────┘ ◀───────────────────────────────────────── └───────┘
        │     │          tool results                          ▲
        │     │ wants a write tool   ┌─────────┐  decision     │
        │     └────────────────────▶ │ approve │ ──────────────┘
        │                            └─────────┘
        │ final reply               pauses with interrupt() until
        ▼                           a person approves or rejects
    ┌──────────┐
    │ finalize │  check [n] citations against this turn's sources
    └──────────┘
          │
          ▼
         END

- agent: the model reads the system prompt + recent conversation and either replies or asks
  for one or more tool calls (it decides the route: no hard-coded intent classifier).
- approve (Phase 6): if any requested call would change data, the graph stops here with
  interrupt(). The state is saved by the checkpointer, the request returns, and the run
  continues only when Copilot.decide() resumes it with the person's answer, maybe hours
  later or in another process.
- tools: runs every requested call. A write runs only if it was approved; a rejected one
  gets a "rejected" result, so the model can tell the person and ask what to change.
- finalize: like Phase 3, keeps only [n] markers that point at a real source found in this
  turn, renumbers them [1], [2]..., and returns only the sources actually cited.

The loop is bounded: after `max_tool_rounds` the model is called with tool_choice="none",
so it must answer with what it has. A checkpointer (Postgres in production) saves the
state after every step, keyed by thread_id, which is also what makes a crashed run
resumable from its last completed step (Copilot.resume).
"""

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from servicedesk.agent.prompts import ROUND_LIMIT_REPLY, SYSTEM_PROMPT
from servicedesk.agent.state import AgentState
from servicedesk.agent.tools import AgentTools
from servicedesk.rag.answerer import check_citations

REASON_MAX_CHARS = 500


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

    def approve(state: AgentState) -> dict[str, Any]:
        actions = pending_actions(state["messages"][-1], tools)
        # The first time through, interrupt() saves the state and stops the run; `actions`
        # is what the caller is shown. When the run is resumed with Command(resume=answer),
        # this node starts again from the top and interrupt() returns `answer` instead.
        # Nothing before interrupt() has side effects, so running it twice is harmless.
        answer = interrupt({"actions": actions})
        decision = _decision(answer)
        approvals = {**state["approvals"], **{a["id"]: decision for a in actions}}
        return {"approvals": approvals}

    def run_call(
        call: dict[str, Any], state: AgentState, thread_id: str, citations_so_far: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one call, applying the approval rule. The one place a write can happen."""
        name, args = call["name"], call["args"]
        if tools.needs_approval(name) and tools.preview(name, args) is not None:
            decision = state["approvals"].get(call["id"])
            if decision is None:  # can't happen via route(); refuse rather than trust it
                error = {"code": "approval_required", "message": "This change wasn't approved."}
                return {"ok": False, "error": error}, []
            if not decision["approved"]:
                error = {
                    "code": "rejected",
                    "message": (
                        "The person you are helping declined this change, so nothing was "
                        "changed. `reason` is what they said, if anything."
                    ),
                    "reason": decision["reason"],  # their words: data, not instructions
                }
                return {"ok": False, "error": error}, []
            # Approved. The key makes a re-run of this same call (after a crash) a no-op.
            key = f"agent:{thread_id}:{call['id']}"
            return tools.run(name, args, citations_so_far, idempotency_key=key)
        # Read tools, and write calls with invalid arguments (validation fails before
        # anything runs, and the errors go back to the model).
        return tools.run(name, args, citations_so_far)

    def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        request: AIMessage = state["messages"][-1]
        thread_id = config["configurable"]["thread_id"]
        citations = list(state["citations"])
        results: list[ToolMessage] = []
        for call in request.tool_calls:
            result, new_citations = run_call(call, state, thread_id, len(citations))
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

    def route(state: AgentState) -> Literal["approve", "tools", "finalize"]:
        last: AIMessage = state["messages"][-1]
        wants_tools = bool(last.tool_calls or last.invalid_tool_calls)
        if not wants_tools or state["tool_rounds"] >= max_tool_rounds:
            return "finalize"
        return "approve" if pending_actions(last, tools) else "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("approve", approve)
    graph.add_node("tools", run_tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("approve", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def pending_actions(message: AIMessage, tools: AgentTools) -> list[dict[str, Any]]:
    """The write calls in a model message that need a person's approval, with their
    validated arguments. A write call with invalid arguments isn't listed: there is nothing
    to approve, and the tools node returns the validation errors to the model."""
    actions = []
    for call in message.tool_calls:
        if not tools.needs_approval(call["name"]):
            continue
        arguments = tools.preview(call["name"], call["args"])
        if arguments is not None:
            actions.append({"id": call["id"], "tool": call["name"], "arguments": arguments})
    return actions


def _decision(answer: Any) -> dict[str, Any]:
    """The resume value as a decision. Anything other than an explicit approval counts as a
    rejection: a malformed answer must never approve a write."""
    if not isinstance(answer, dict):
        return {"approved": False, "reason": ""}
    reason = answer.get("reason")
    return {
        "approved": answer.get("approved") is True,
        "reason": str(reason)[:REASON_MAX_CHARS] if reason else "",
    }


def _tool_message(call_id: str, name: str, result: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(result, ensure_ascii=False),
        tool_call_id=call_id,
        name=name,
        status="success" if result.get("ok") else "error",
    )
