"""The evaluation dataset: what we ask, and what a good result looks like.

evals/dataset.jsonl has one JSON object per line (JSON Lines), of two kinds:

- "question": a knowledge-base question with the documents that contain the answer
  (`expected_docs`, any of them counts) and a short reference answer written by hand.
  `expected_docs: []` means the knowledge base does NOT cover it: the right result is a
  refusal. These measure retrieval, grounding and answer correctness.
- "agent": one message to the agent and what it should do: which tools it must call, which
  change it should propose (it is never approved during an eval, so nothing is written),
  and words its reply should contain. These measure task completion.

Each line is validated with the models below when loaded, so a typo in the file fails
loudly instead of quietly skewing the numbers.
"""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

DATASET_PATH = Path("evals/dataset.jsonl")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt key is an error, not ignored

    id: str = Field(pattern=r"^[a-z]\d{2}$")
    tags: list[str] = []


class QuestionCase(Case):
    kind: Literal["question"]
    question: str = Field(min_length=5)
    expected_docs: list[str]  # any of these counts as relevant; [] = should be refused
    reference_answer: str | None

    @property
    def answerable(self) -> bool:
        return bool(self.expected_docs)


class ExpectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = {}  # each of these must equal the proposed argument


class AgentCase(Case):
    kind: Literal["agent"]
    message: str = Field(min_length=3)
    expect_tools: list[str] = []  # must all be called
    forbid_tools: list[str] = []  # must not be called (or proposed)
    expect_status: Literal["completed", "awaiting_approval"] | None = None  # None = either
    expect_action: ExpectedAction | None = None  # a change it should propose
    forbid_action: str | None = None  # a change it must not propose (e.g. a duplicate)
    reply_includes_any: list[str] = []  # the reply contains at least one (any case)


# `kind` decides which model a line is validated with (a "discriminated union").
AnyCase = Annotated[QuestionCase | AgentCase, Field(discriminator="kind")]
_adapter: TypeAdapter[AnyCase] = TypeAdapter(AnyCase)


def load_dataset(path: Path = DATASET_PATH) -> list[QuestionCase | AgentCase]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(_adapter.validate_python(json.loads(line)))
        except ValueError as exc:  # JSON errors and ValidationError are both ValueErrors
            raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [case.id for case in cases]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"{path}: duplicate case ids {duplicates}")
    return cases


def questions(cases: list) -> list[QuestionCase]:
    return [c for c in cases if isinstance(c, QuestionCase)]


def agent_tasks(cases: list) -> list[AgentCase]:
    return [c for c in cases if isinstance(c, AgentCase)]
