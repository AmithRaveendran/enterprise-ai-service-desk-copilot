"""What the graph remembers: one AgentState per conversation (thread).

LangGraph passes the state to every node; a node returns only the keys it changes, and
LangGraph merges them in. How a key is merged is its *reducer*:

- messages uses `add_messages`: new messages are appended (and a message with an id that
  already exists replaces it). This list is the conversation memory.
- citations and tool_rounds have no reducer, so a returned value simply replaces the old
  one. Both are reset at the start of every user turn (see Copilot.chat).

The checkpointer saves this state after every node, keyed by thread_id. That is what makes
the agent *stateful*: the next message on the same thread starts from the saved state.
Plain JSON-friendly values only (dicts, not our dataclasses), so checkpoints serialise safely.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Knowledge-base sources found during this turn, numbered [1], [2]... across the turn.
    citations: list[dict[str, Any]]
    # How many times the model has asked for tools during this turn (a runaway-loop guard).
    tool_rounds: int
