"""The tools the agent may call, and the code that runs them safely.

Built from the Phase 4 registry (desk/tools.py) plus one knowledge-base tool that wraps the
Phase 3 Answerer, so every guarantee from those phases still holds:
- desk tools go through run_tool(): strict validation, errors returned as data;
- knowledge-base answers go through the evidence gate and the citation check.

Phase 5 gives the agent READ-ONLY tools. The write tools (create_ticket, update_ticket,
add_comment) arrive in Phase 6 together with human approval, so the agent never has a way
to change data without a person saying yes, not even temporarily.
"""

import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Annotated, Any

import openai
from pydantic import BaseModel, Field, ValidationError

from servicedesk.desk.schemas import StrictInput
from servicedesk.desk.service import ServiceDesk
from servicedesk.desk.tools import TOOLS, Tool, run_tool
from servicedesk.openai_client import ConfigurationError
from servicedesk.rag.answerer import Answer
from servicedesk.rag.llm import ModelRefusedError

KB_TOOL = "search_knowledge_base"
KB_DESCRIPTION = (
    "Answer a question from the company IT knowledge base (how-to guides, policies, known "
    "issues). Returns an answer with [n] citation markers and its sources, or found=false "
    "when the knowledge base doesn't cover it. Ask one focused question per call."
)

# The desk tools the agent gets in Phase 5: the ones that only read.
READ_ONLY_TOOLS: list[Tool] = [tool for tool in TOOLS if not tool.writes]


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
    """Runs the agent's tool calls. Every call returns a JSON-ready dict, never raises:
    {"ok": True, "result": ...} or {"ok": False, "error": {"code", "message", ...}}."""

    def __init__(self, desk: ServiceDesk | None, ask_knowledge_base: Callable[[str], Answer]):
        self.desk = desk
        self.ask_knowledge_base = ask_knowledge_base
        self.specs = [tool_spec(KB_TOOL, KB_DESCRIPTION, KnowledgeBaseInput)] + [
            tool_spec(t.name, t.description, t.input_model) for t in READ_ONLY_TOOLS
        ]
        self.allowed = {spec["function"]["name"] for spec in self.specs}

    def run(
        self, name: str, arguments: dict[str, Any], citations_so_far: int = 0
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one tool call. Returns (result for the model, new citations for the state)."""
        if name not in self.allowed:
            # The model only sees the allowed tools, but models can still invent a call.
            # Write tools are refused here too, which is the real enforcement point.
            return _error("not_allowed", f"'{name}' is not available to this assistant."), []
        if name == KB_TOOL:
            return self._knowledge_base(arguments, citations_so_far)
        return run_tool(self.desk, name, arguments), []

    def _knowledge_base(
        self, arguments: dict[str, Any], offset: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            args = KnowledgeBaseInput.model_validate(arguments)
        except ValidationError as exc:
            problems = [
                {"field": ".".join(str(p) for p in e["loc"]) or "(input)", "problem": e["msg"]}
                for e in exc.errors()
            ]
            return _error("invalid_input", "The arguments are not valid.", problems=problems), []
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
