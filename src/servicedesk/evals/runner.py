"""Run the three evaluation suites and summarise them.

- retrieval: search each answerable question in vector, keyword and hybrid mode; where is
  the first chunk from an expected document? → hit@1/3/k and MRR per mode. Cheap: only
  query embeddings, no chat model.
- answers: the full Phase 3 pipeline (Answerer.ask) on every question. Answerable ones
  should be answered, citing an expected document; out-of-scope ones should be refused.
  Each answer is then graded by the judge (judge.py) for grounding and correctness.
- agent: one message per task to the LangGraph agent; check the tools it called and the
  change it proposed. Writes are never approved here, so nothing is changed.

Each function returns {"summary": {...}, "cases": [...]}: the summary for the report, the
cases so any number can be traced back to the questions behind it. A case that crashes is
recorded with its error and counts as a failure; it doesn't stop the run.
"""

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from servicedesk.agent.copilot import ChatTurn, Copilot, new_thread_id
from servicedesk.evals.dataset import AgentCase, QuestionCase
from servicedesk.evals.judge import judge_answer
from servicedesk.evals.metrics import (
    distribution,
    first_relevant_rank,
    hit_at_k,
    mean,
    mean_reciprocal_rank,
    percentile,
    rate,
)
from servicedesk.rag.answerer import Answerer
from servicedesk.rag.llm import StructuredLLM
from servicedesk.search.retriever import Retriever, SearchMode

Progress = Callable[[str], None]


def _parallel[C, R](run: Callable[[C], R], cases: list[C], workers: int) -> list[R]:
    """Run one case per call, `workers` at a time; results in the same order as cases.
    The work is mostly waiting for the model API, so threads are enough."""
    if workers <= 1:
        return [run(case) for case in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, cases))


# --- retrieval -------------------------------------------------------------------------------


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[QuestionCase],
    *,
    k: int = 5,
    modes: tuple[SearchMode, ...] = ("vector", "keyword", "hybrid"),
) -> dict[str, Any]:
    rows = []
    for case in (c for c in cases if c.answerable):
        row: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "expected": case.expected_docs,
        }
        for mode in modes:
            docs = [hit.doc_id for hit in retriever.search(case.question, k=k, mode=mode)]
            row[mode] = {"rank": first_relevant_rank(docs, case.expected_docs), "docs": docs}
        rows.append(row)

    summary = {"k": k, "questions": len(rows), "modes": {}}
    for mode in modes:
        ranks = [row[mode]["rank"] for row in rows]
        summary["modes"][mode] = {
            "hit@1": hit_at_k(ranks, 1),
            "hit@3": hit_at_k(ranks, 3),
            f"hit@{k}": hit_at_k(ranks, k),
            "mrr": mean_reciprocal_rank(ranks),
            "misses": [row["id"] for row in rows if row[mode]["rank"] is None],
        }
    return {"summary": summary, "cases": rows}


# --- answers ---------------------------------------------------------------------------------


def evaluate_answers(
    answerer: Answerer,
    cases: list[QuestionCase],
    *,
    judge: StructuredLLM | None = None,
    workers: int = 4,
    progress: Progress = lambda _: None,
) -> dict[str, Any]:
    def run(case: QuestionCase) -> dict[str, Any]:
        row = answer_case(answerer, case, judge)
        progress(f"{case.id} {_answer_label(row)}")
        return row

    rows = _parallel(run, cases, workers)
    return {"summary": summarise_answers(rows), "cases": rows}


def answer_case(
    answerer: Answerer, case: QuestionCase, judge: StructuredLLM | None
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case.id, "question": case.question, "answerable": case.answerable}
    started = time.perf_counter()
    try:
        answer = answerer.ask(case.question)
    except Exception as exc:
        return {**row, "error": _describe(exc)}
    row["latency_s"] = round(time.perf_counter() - started, 2)
    cited = [c.doc_id for c in answer.citations]
    row |= {
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "answer": answer.answer,
        "cited_docs": cited,
        "top_similarity": answer.top_similarity,
        "tokens": answer.input_tokens + answer.output_tokens,
    }
    if not case.answerable or answer.refused:
        return row

    expected = set(case.expected_docs)
    row["citation_hit"] = any(doc in expected for doc in cited)
    row["citation_precision"] = rate(doc in expected for doc in cited)
    if judge is not None and case.reference_answer:
        sources = [(c.number, c.doc_id, c.content) for c in answer.citations]
        try:
            judged = judge_answer(
                judge, case.question, answer.answer, sources, case.reference_answer
            )
        except Exception as exc:
            row["judge_error"] = _describe(exc)
        else:
            row["judge"] = judged.verdict.model_dump()
            row["judge_tokens"] = judged.input_tokens + judged.output_tokens
    return row


def summarise_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]
    judged = [r for r in answered if "judge" in r]
    latencies = [r["latency_s"] for r in ok]
    return {
        "questions": len(rows),
        "errors": [r["id"] for r in rows if "error" in r],
        # Should answer / should refuse:
        "answered_when_answerable": rate(not r["refused"] for r in answerable),
        "refused_when_out_of_scope": rate(r["refused"] for r in unanswerable),
        "false_refusals": [r["id"] for r in answerable if r["refused"]],
        "false_answers": [r["id"] for r in unanswerable if not r["refused"]],
        "refusal_reasons": distribution(r["refusal_reason"] for r in ok if r["refused"]),
        # Citations (answered, answerable questions):
        "citation_hit_rate": rate(r["citation_hit"] for r in answered),
        "citation_precision": mean(r["citation_precision"] for r in answered),
        # The judge:
        "judged": len(judged),
        "grounding": distribution(r["judge"]["grounding"] for r in judged),
        "correctness": distribution(r["judge"]["correctness"] for r in judged),
        "grounded_rate": rate(r["judge"]["grounding"] == "supported" for r in judged),
        "correct_rate": rate(r["judge"]["correctness"] == "correct" for r in judged),
        "not_fully_grounded": [r["id"] for r in judged if r["judge"]["grounding"] != "supported"],
        "not_correct": [r["id"] for r in judged if r["judge"]["correctness"] != "correct"],
        # Cost and speed:
        "latency_s_mean": mean(latencies),
        "latency_s_p95": percentile(latencies, 95),
        "answer_tokens": sum(r["tokens"] for r in ok),
        "judge_tokens": sum(r.get("judge_tokens", 0) for r in ok),
    }


def _answer_label(row: dict[str, Any]) -> str:
    if "error" in row:
        return "ERROR"
    if row["refused"]:
        return f"refused ({row['refusal_reason']})"
    judge = row.get("judge")
    return f"answered, {judge['grounding']}, {judge['correctness']}" if judge else "answered"


# --- agent -----------------------------------------------------------------------------------


def evaluate_agent(
    copilot: Copilot,
    cases: list[AgentCase],
    *,
    workers: int = 4,
    progress: Progress = lambda _: None,
) -> dict[str, Any]:
    def run(case: AgentCase) -> dict[str, Any]:
        row = agent_case(copilot, case)
        progress(f"{case.id} {'pass' if row['passed'] else 'FAIL ' + '; '.join(row['failures'])}")
        return row

    rows = _parallel(run, cases, workers)
    latencies = [r["latency_s"] for r in rows if "latency_s" in r]
    summary = {
        "tasks": len(rows),
        "task_completion": rate(r["passed"] for r in rows),
        "failed": [r["id"] for r in rows if not r["passed"]],
        "latency_s_mean": mean(latencies),
        "latency_s_p95": percentile(latencies, 95),
        "tokens": sum(r.get("tokens", 0) for r in rows),
    }
    return {"summary": summary, "cases": rows}


def agent_case(copilot: Copilot, case: AgentCase) -> dict[str, Any]:
    row: dict[str, Any] = {"id": case.id, "message": case.message}
    started = time.perf_counter()
    try:
        turn = copilot.chat(case.message, new_thread_id())
    except Exception as exc:
        return {**row, "passed": False, "failures": ["error"], "error": _describe(exc)}
    failures = check_agent_turn(case, turn)
    return row | {
        "passed": not failures,
        "failures": failures,
        "latency_s": round(time.perf_counter() - started, 2),
        "status": turn.status,
        "tools_called": [t.name for t in turn.tools_used],
        "proposed": [
            {"tool": a["tool"], "arguments": a["arguments"]} for a in turn.pending_actions
        ],
        "reply": turn.reply,
        "tokens": turn.input_tokens + turn.output_tokens,
    }


def check_agent_turn(case: AgentCase, turn: ChatTurn) -> list[str]:
    """Every expectation the turn missed, in words. Empty = the task passed."""
    failures = []
    called = {t.name for t in turn.tools_used}
    proposed = {a["tool"]: a["arguments"] for a in turn.pending_actions}

    if missing := [t for t in case.expect_tools if t not in called]:
        failures.append(f"didn't call {', '.join(missing)}")
    if forbidden := [t for t in case.forbid_tools if t in called or t in proposed]:
        failures.append(f"used {', '.join(forbidden)}")
    if case.expect_status and turn.status != case.expect_status:
        failures.append(f"status {turn.status}, expected {case.expect_status}")
    if case.expect_action:
        action = case.expect_action
        if action.tool not in proposed:
            failures.append(f"didn't propose {action.tool}")
        else:
            wrong = {
                name: proposed[action.tool].get(name)
                for name, value in action.arguments.items()
                if proposed[action.tool].get(name) != value
            }
            if wrong:
                failures.append(f"{action.tool} arguments differ: {wrong}")
    if case.forbid_action and case.forbid_action in proposed:
        failures.append(f"proposed {case.forbid_action}")
    reply = turn.reply.lower()
    if case.reply_includes_any and not any(w.lower() in reply for w in case.reply_includes_any):
        failures.append(f"reply mentions none of {case.reply_includes_any}")
    return failures


def _describe(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(exc)).strip()
