"""The evaluation code: the dataset file, the metrics, the judge plumbing, the agent checks
and the report. All offline: the retriever, answerer, judge and agent are fakes, so these
tests check the scoring, not the models (the models are what `servicedesk eval` measures).
"""

import json

import pytest

from servicedesk.agent.copilot import ChatTurn, ToolUse
from servicedesk.agent.tools import KB_TOOL
from servicedesk.desk.tools import TOOLS_BY_NAME
from servicedesk.evals.dataset import (
    AgentCase,
    QuestionCase,
    agent_tasks,
    load_dataset,
    questions,
)
from servicedesk.evals.judge import JudgeVerdict, judge_answer, judge_prompt
from servicedesk.evals.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean_reciprocal_rank,
    percentile,
    rate,
)
from servicedesk.evals.report import render_report
from servicedesk.evals.runner import (
    check_agent_turn,
    evaluate_agent,
    evaluate_answers,
    evaluate_retrieval,
)
from servicedesk.ingestion.loaders import discover_documents, load_document
from servicedesk.rag.answerer import Answer, Citation
from servicedesk.rag.llm import LLMResult
from tests.conftest import KB_DIR

# --- the dataset file --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_the_dataset_has_enough_cases_of_each_kind(dataset):
    qs = questions(dataset)
    assert len(qs) >= 40
    assert sum(not q.answerable for q in qs) >= 5  # some must be refused
    assert len(agent_tasks(dataset)) >= 10


def test_every_expected_document_exists_in_the_knowledge_base(dataset):
    """A typo in a doc id would make a question unanswerable by construction."""
    kb_ids = {load_document(path).doc_id for path in discover_documents(KB_DIR)}
    for case in questions(dataset):
        assert set(case.expected_docs) <= kb_ids, case.id
    covered = {doc for case in questions(dataset) for doc in case.expected_docs}
    assert covered == kb_ids  # and every document is asked about at least once


def test_every_answerable_question_has_a_reference_answer(dataset):
    for case in questions(dataset):
        assert bool(case.reference_answer) is case.answerable, case.id


def test_agent_tasks_only_name_real_tools(dataset):
    known = set(TOOLS_BY_NAME) | {KB_TOOL}
    for task in agent_tasks(dataset):
        named = set(task.expect_tools) | set(task.forbid_tools)
        if task.expect_action:
            named.add(task.expect_action.tool)
            assert TOOLS_BY_NAME[task.expect_action.tool].writes, task.id
        if task.forbid_action:
            named.add(task.forbid_action)
        assert named <= known, task.id


def test_a_bad_line_is_reported_with_its_line_number(tmp_path):
    path = tmp_path / "dataset.jsonl"
    good = {
        "kind": "question",
        "id": "q01",
        "question": "Why is it slow?",
        "expected_docs": [],
        "reference_answer": None,
    }
    typo = {**good, "id": "q02", "expected_doc": ["KB-0001"]}  # misspelt key
    path.write_text(json.dumps(good) + "\n" + json.dumps(typo) + "\n")
    with pytest.raises(ValueError, match="dataset.jsonl:2"):
        load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path):
    line = {
        "kind": "question",
        "id": "q01",
        "question": "Why not?",
        "expected_docs": [],
        "reference_answer": None,
    }
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(line) + "\n" + json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_dataset(path)


# --- metrics ---------------------------------------------------------------------------------


def test_first_relevant_rank_is_one_based():
    assert first_relevant_rank(["KB-0005", "KB-0002", "KB-0002"], ["KB-0002", "POL-0001"]) == 2
    assert first_relevant_rank(["KB-0005"], ["KB-0002"]) is None


def test_hit_at_k_and_mrr():
    ranks = [1, 2, None, 4]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.5
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_a_rate_of_nothing_is_none_not_zero():
    assert rate([]) is None
    assert rate([True, False, True, True]) == 0.75


def test_percentile_uses_the_nearest_rank():
    values = [float(v) for v in range(1, 21)]  # 1..20
    assert percentile(values, 95) == 19.0
    assert percentile(values, 50) == 10.0
    assert percentile([], 95) is None


# --- retrieval -------------------------------------------------------------------------------


class Hit:
    def __init__(self, doc_id):
        self.doc_id = doc_id


class FakeRetriever:
    """Returns fixed doc ids per mode, whatever the question."""

    def __init__(self, by_mode):
        self.by_mode = by_mode

    def search(self, query, *, k, mode):
        return [Hit(d) for d in self.by_mode[mode]][:k]


def question(case_id="q01", expected=("KB-0002",), reference="At least 14 characters."):
    return QuestionCase(
        kind="question",
        id=case_id,
        question="How long must my password be?",
        expected_docs=list(expected),
        reference_answer=reference if expected else None,
    )


def test_retrieval_is_scored_per_mode_and_skips_out_of_scope_questions():
    retriever = FakeRetriever(
        {"vector": ["KB-0002"], "keyword": ["KB-0005", "KB-0002"], "hybrid": ["KB-0005"]}
    )
    cases = [question("q01"), question("u01", expected=())]
    result = evaluate_retrieval(retriever, cases, k=5, modes=("vector", "keyword", "hybrid"))

    modes = result["summary"]["modes"]
    assert result["summary"]["questions"] == 1  # u01 has nothing to find
    assert (modes["vector"]["hit@1"], modes["vector"]["mrr"]) == (1.0, 1.0)
    assert (modes["keyword"]["hit@1"], modes["keyword"]["hit@3"]) == (0.0, 1.0)
    assert modes["hybrid"]["misses"] == ["q01"]


# --- answers ---------------------------------------------------------------------------------


def citation(doc_id, number=1):
    return Citation(
        number=number,
        chunk_id=number,
        doc_id=doc_id,
        title="T",
        source_path="p",
        section="S",
        page=None,
        content=f"Text of {doc_id}.",
    )


class FakeAnswerer:
    def __init__(self, answers):
        self.answers = answers  # question id → Answer (or an exception to raise)
        self.calls = []

    def ask(self, text):
        case_id = text.split(":")[0]
        self.calls.append(case_id)
        result = self.answers[case_id]
        if isinstance(result, Exception):
            raise result
        return result


class FakeJudge:
    model = "fake-judge"

    def __init__(self, grounding="supported", correctness="correct"):
        self.verdict = JudgeVerdict(
            grounding=grounding,
            unsupported_claims=[] if grounding == "supported" else ["an invented step"],
            correctness=correctness,
            missing_points=[],
            explanation="Fine.",
        )
        self.prompts = []

    def generate(self, system, user, schema):
        self.prompts.append(user)
        return LLMResult(parsed=self.verdict, input_tokens=300, output_tokens=40)


def answered(*docs):
    return Answer(
        question="q",
        answer="Use 14 characters [1].",
        refused=False,
        citations=[citation(d, i) for i, d in enumerate(docs, start=1)],
        input_tokens=900,
        output_tokens=100,
    )


def refused(reason="no_relevant_sources"):
    return Answer(question="q", answer="I don't know.", refused=True, refusal_reason=reason)


def ask_case(case_id, expected=("KB-0002",)):
    case = question(case_id, expected)
    return case.model_copy(update={"question": f"{case_id}: a question"})


def test_answers_are_scored_for_refusals_citations_and_the_judges_verdict():
    answerer = FakeAnswerer(
        {
            "q01": answered("KB-0002"),  # right document
            "q02": answered("KB-0005", "KB-0002"),  # one of two citations is expected
            "q03": refused(),  # should have answered
            "u01": refused(),  # right to refuse
            "u02": answered("KB-0001"),  # should have refused
        }
    )
    cases = [
        ask_case("q01"),
        ask_case("q02"),
        ask_case("q03"),
        ask_case("u01", ()),
        ask_case("u02", ()),
    ]
    result = evaluate_answers(answerer, cases, judge=FakeJudge(), workers=1)
    s = result["summary"]

    assert s["answered_when_answerable"] == pytest.approx(2 / 3)
    assert s["refused_when_out_of_scope"] == 0.5
    assert (s["false_refusals"], s["false_answers"]) == (["q03"], ["u02"])
    assert s["citation_hit_rate"] == 1.0
    assert s["citation_precision"] == 0.75
    assert s["judged"] == 2  # only answered, answerable questions are judged
    assert s["answer_tokens"] == 3 * 1000
    assert s["judge_tokens"] == 2 * 340


def test_the_judge_sees_the_cited_sources_and_the_reference():
    judge = FakeJudge(grounding="partially_supported", correctness="partially_correct")
    answerer = FakeAnswerer({"q01": answered("KB-0002")})
    result = evaluate_answers(answerer, [ask_case("q01")], judge=judge, workers=1)

    prompt = judge.prompts[0]
    assert '<source number="1" doc="KB-0002">\nText of KB-0002.\n</source>' in prompt
    assert "<reference>\nAt least 14 characters.\n</reference>" in prompt
    s = result["summary"]
    assert (s["grounded_rate"], s["correct_rate"]) == (0.0, 0.0)
    assert s["not_fully_grounded"] == ["q01"]


def test_a_crashing_case_is_recorded_and_the_run_goes_on():
    answerer = FakeAnswerer({"q01": RuntimeError("API down"), "q02": answered("KB-0002")})
    result = evaluate_answers(answerer, [ask_case("q01"), ask_case("q02")], workers=2)
    assert result["summary"]["errors"] == ["q01"]
    assert "RuntimeError: API down" in result["cases"][0]["error"]
    assert result["summary"]["answered_when_answerable"] == 1.0  # counted without q01


def test_judge_answer_returns_the_verdict_and_tokens():
    judged = judge_answer(FakeJudge(), "Q?", "A [1].", [(1, "KB-0002", "text")], "ref")
    assert judged.verdict.grounding == "supported"
    assert judged.input_tokens + judged.output_tokens == 340
    assert "<question>\nQ?\n</question>" in judge_prompt("Q?", "A", [], "ref")


# --- agent -----------------------------------------------------------------------------------


def task(**fields):
    return AgentCase(kind="agent", id="a01", message="My VPN is broken", **fields)


def turn(tools=(), status="completed", actions=(), reply="All good."):
    return ChatTurn(
        thread_id="t",
        reply=reply,
        status=status,
        tools_used=[ToolUse(name=t, ok=True) for t in tools],
        pending_actions=[{"id": "c1", "tool": t, "arguments": a} for t, a in actions],
    )


def test_a_turn_that_meets_every_expectation_passes():
    case = task(
        expect_tools=["search_tickets"],
        expect_status="awaiting_approval",
        expect_action={"tool": "update_ticket", "arguments": {"ticket_id": 1003}},
        reply_includes_any=["INC-1003"],
    )
    result = turn(
        tools=["search_tickets"],
        status="awaiting_approval",
        actions=[("update_ticket", {"ticket_id": 1003, "status": "in_progress"})],
        reply="I'd like to update INC-1003.",
    )
    assert check_agent_turn(case, result) == []


@pytest.mark.parametrize(
    ("fields", "result", "failure"),
    [
        ({"expect_tools": ["get_service_status"]}, turn(), "didn't call get_service_status"),
        (
            {"forbid_tools": ["add_comment"]},
            turn(actions=[("add_comment", {})]),
            "used add_comment",
        ),
        ({"expect_status": "awaiting_approval"}, turn(), "status completed"),
        (
            {"expect_action": {"tool": "create_ticket"}},
            turn(),
            "didn't propose create_ticket",
        ),
        (
            {"expect_action": {"tool": "create_ticket", "arguments": {"category": "hardware"}}},
            turn(actions=[("create_ticket", {"category": "network"})]),
            "create_ticket arguments differ: {'category': 'network'}",
        ),
        (
            {"forbid_action": "create_ticket"},
            turn(actions=[("create_ticket", {})]),
            "proposed create_ticket",
        ),
        (
            {"reply_includes_any": ["KI-0007", "TCP"]},
            turn(reply="Restart it."),
            "reply mentions none",
        ),
    ],
)
def test_each_missed_expectation_is_named(fields, result, failure):
    failures = check_agent_turn(task(**fields), result)
    assert len(failures) == 1 and failures[0].startswith(failure)


def test_reply_words_are_matched_case_insensitively():
    case = task(reply_includes_any=["maintenance"])
    assert check_agent_turn(case, turn(reply="Planned MAINTENANCE on Saturday.")) == []


class FakeCopilot:
    def __init__(self, turns):
        self.turns = turns  # message → ChatTurn or exception

    def chat(self, message, thread_id):
        result = self.turns[message]
        if isinstance(result, Exception):
            raise result
        return result


def test_agent_tasks_are_summarised_as_task_completion():
    cases = [
        AgentCase(kind="agent", id="a01", message="vpn?", expect_tools=["get_service_status"]),
        AgentCase(kind="agent", id="a02", message="email?", expect_tools=["get_service_status"]),
        AgentCase(kind="agent", id="a03", message="crash", expect_tools=[]),
    ]
    copilot = FakeCopilot(
        {
            "vpn?": turn(tools=["get_service_status"]),
            "email?": turn(tools=["search_knowledge_base"]),
            "crash": RuntimeError("model API down"),
        }
    )
    result = evaluate_agent(copilot, cases, workers=1)
    assert result["summary"]["task_completion"] == pytest.approx(1 / 3)
    assert result["summary"]["failed"] == ["a02", "a03"]
    assert result["cases"][2]["failures"] == ["error"]


# --- report ----------------------------------------------------------------------------------


def test_the_report_shows_every_suite_and_the_flagged_answers():
    retriever = FakeRetriever({m: ["KB-0002"] for m in ("vector", "keyword", "hybrid")})
    judge = FakeJudge(grounding="partially_supported")
    answerer = FakeAnswerer({"q01": answered("KB-0002")})
    copilot = FakeCopilot({"vpn?": turn(tools=["get_service_status"])})
    agent_case = AgentCase(
        kind="agent", id="a01", message="vpn?", expect_tools=["get_service_status"]
    )
    results = {
        "run": {
            "date": "today",
            "chat_model": "m",
            "questions": 1,
            "answerable": 1,
            "agent_tasks": 1,
        },
        "retrieval": evaluate_retrieval(retriever, [question()], k=5),
        "answers": evaluate_answers(answerer, [ask_case("q01")], judge=judge, workers=1),
        "agent": evaluate_agent(copilot, [agent_case], workers=1),
    }
    report = render_report(results)

    assert "| hybrid | 100% | 100% | 100% | 1.00 | none |" in report
    assert "| Grounded (judge: fully supported by its sources) | 0% of 1 |" in report
    assert "**q01** q01: a question: partially_supported, correct" in report
    assert "**Task completion: 100%** (1/1)" in report


def test_a_report_with_one_suite_still_renders():
    retriever = FakeRetriever({m: [] for m in ("vector", "keyword", "hybrid")})
    report = render_report({"retrieval": evaluate_retrieval(retriever, [question()], k=5)})
    assert "## Retrieval" in report and "## Answers" not in report


# --- the scratch database for agent tasks ---------------------------------------------------


@pytest.mark.integration
def test_the_eval_database_is_reset_to_the_sample_data(test_database_url):
    from sqlalchemy import func, select

    from servicedesk.db.models import Ticket
    from servicedesk.evals.database import fresh_desk_database

    factory = fresh_desk_database(test_database_url, name="servicedesk_eval_test")
    with factory.begin() as session:
        session.get(Ticket, 1001).title = "Changed by an earlier run"
    factory = fresh_desk_database(test_database_url, name="servicedesk_eval_test")
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Ticket)) == 5
        assert session.get(Ticket, 1001).title == "VPN disconnects every few minutes on MacBook"
