"""The tools the agent may call, and the code that runs them safely.

Built from the Phase 4 registry (desk/tools.py) plus one knowledge-base tool that wraps the
Phase 3 Answerer, so every guarantee from those phases still holds:
- desk tools go through run_tool(): strict validation, errors returned as data;
- knowledge-base answers go through the evidence gate and the citation check.

Phase 6 gives the agent the write tools too (create_ticket, update_ticket, add_comment),
but this class never decides on its own to run one: the graph pauses for a person's
approval first (graph.py, the `approve` node). `preview()` is what that person is shown.

Nothing here raises for a bad call. Every result is a JSON-ready dict:
{"ok": True, "result": ...} or {"ok": False, "error": {"code", "message", ...}}.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Annotated, Any

import openai
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from servicedesk.desk.schemas import StrictInput
from servicedesk.desk.service import ServiceDesk
from servicedesk.desk.tools import TOOLS, TOOLS_BY_NAME, invalid_input, run_tool
from servicedesk.openai_client import ConfigurationError
from servicedesk.rag.answerer import Answer
from servicedesk.rag.llm import ModelRefusedError

logger = logging.getLogger(__name__)

KB_TOOL = "search_knowledge_base"
KB_DESCRIPTION = (
    "Answer a question from the company IT knowledge base (how-to guides, policies, known "
    "issues). Returns an answer with [n] citation markers and its sources, or found=false "
    "when the knowledge base doesn't cover it. Ask one focused question per call."
)

# Tools that change data. The graph never runs one without a person's approval.
WRITE_TOOLS: frozenset[str] = frozenset(tool.name for tool in TOOLS if tool.writes)


class KnowledgeBaseInput(StrictInput):
    question: Annotated[
        str,
        Field(min_length=3, max_length=500, description="The IT question, in plain words."),
    ]


def tool_spec(name: str, description: str, input_model: type[BaseModel]) -> dict[str, Any]:
    """The OpenAI function-calling format: what the model is shown about one tool."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_model.model_json_schema(),
        },
    }


class AgentTools:
    def __init__(self, desk: ServiceDesk | None, ask_knowledge_base: Callable[[str], Answer]):
        self.desk = desk
        self.ask_knowledge_base = ask_knowledge_base
        # The Phase 4 registry, minus nothing: tools with writes=True are offered too, and
        # the approval step in the graph is what guards them.
        self.specs = [tool_spec(KB_TOOL, KB_DESCRIPTION, KnowledgeBaseInput)] + [
            tool_spec(t.name, t.description, t.input_model) for t in TOOLS
        ]
        self.allowed = {spec["function"]["name"] for spec in self.specs}

    @staticmethod
    def needs_approval(name: str) -> bool:
        return name in WRITE_TOOLS

    @staticmethod
    def preview(name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """The validated, normalised arguments of a write call, as the person approving it
        will see them (e.g. ticket "1003" shown as "INC-1003", the default priority filled
        in). None if the arguments are invalid: there is nothing to approve, and running the
        call returns the validation errors to the model instead."""
        try:
            args = TOOLS_BY_NAME[name].input_model.model_validate(arguments)
        except ValidationError:
            return None
        shown = args.model_dump(mode="json")
        # Leave out empty optional fields the model didn't send, but keep anything it did
        # send (even null) and any default with a value (priority P3).
        return {k: v for k, v in shown.items() if v is not None or k in args.model_fields_set}

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
        citations_so_far: int = 0,
        idempotency_key: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one tool call. Returns (result for the model, new citations for the state).

        Only call this for a write tool after it was approved. The idempotency key makes a
        repeated run of the same approved call (e.g. when resuming after a crash) a no-op.
        """
        if name not in self.allowed:
            # The model only sees the allowed tools, but models can still invent a call.
            return _error("not_allowed", f"'{name}' is not available to this assistant."), []
        try:
            if name == KB_TOOL:
                return self._knowledge_base(arguments, citations_so_far)
            return run_tool(self.desk, name, arguments, idempotency_key), []
        except SQLAlchemyError:
            # The database is down or timed out. A write's transaction was rolled back.
            logger.exception("Tool %s failed: database error", name)
            return _error("unavailable", "The service desk database is unavailable."), []
        except Exception:
            # A bug in a tool must not crash the whole conversation: the model is told it
            # failed and can say so. The traceback is in the logs.
            logger.exception("Tool %s failed unexpectedly", name)
            return _error("tool_failed", f"'{name}' failed unexpectedly."), []

    def _knowledge_base(
        self, arguments: dict[str, Any], offset: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            args = KnowledgeBaseInput.model_validate(arguments)
        except ValidationError as exc:
            return invalid_input(exc), []
        try:
            answer = self.ask_knowledge_base(args.question)
        except (ConfigurationError, ModelRefusedError, openai.APIError) as exc:
            message = f"The knowledge base is unavailable ({type(exc).__name__})."
            return _error("unavailable", message), []

        if answer.refused:
            result = {"found": False, "reason": answer.refusal_reason}
            return {"ok": True, "result": result}, []

        # The Answerer numbers its sources [1], [2]... If this is the second knowledge-base
        # call in the turn, shift them (to [3], [4]...) so numbers stay unique in the reply.
        text = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + offset}]", answer.answer)
        citations = [{**asdict(c), "number": c.number + offset} for c in answer.citations]
        # The model gets the answer and where it came from, not the full chunk texts: the
        # answer is already grounded, and that keeps the agent's context small.
        sources = [
            {key: c[key] for key in ("number", "doc_id", "title", "section")} for c in citations
        ]
        result = {"found": True, "answer": text, "sources": sources}
        return {"ok": True, "result": result}, citations


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, **details}}
