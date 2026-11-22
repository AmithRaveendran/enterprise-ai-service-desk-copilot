# Phase 8: Tests + evaluation dataset

## What was built
```
 evals/dataset.jsonl   40 questions (35 answerable + 5 out of scope) · 12 agent tasks
        │  load_dataset(): validated per line (Pydantic, discriminated by "kind")
        ▼
 servicedesk eval  ─┬─ retrieval: Retriever.search × (vector, keyword, hybrid) → hit@1/3/5, MRR
                    ├─ answers:   Answerer.ask → refused? cites expected docs?
                    │             → LLM judge (rubric) → grounding + correctness
                    └─ agent:     Copilot.chat on a scratch DB → check_agent_turn() → task completion
        ▼
 evals/results.json (every case)  →  evals/report.md (tables + every flagged case)
```
- `src/servicedesk/evals/`: `dataset.py` (case models, loader), `metrics.py` (hit@k, MRR,
  rates, p95), `judge.py` (rubric + `JudgeVerdict` schema), `runner.py` (the three suites,
  4 cases at a time in threads), `database.py` (`servicedesk_eval`, reset to the seed data),
  `report.py` (Markdown).
- **The dataset** is hand-written from the knowledge base: every one of the 19 documents is
  the expected source of at least one question; each answerable question has a short
  reference answer; 5 questions are outside the KB and must be refused; 12 agent tasks cover
  status checks, KB questions, ticket lookups, proposed creates/updates/comments, a duplicate
  that must not be created, an unknown email that must not be invented, and an off-topic
  request.
- `servicedesk eval [--suite retrieval|answers|agent] [--limit N] [--no-judge]
  [--judge-model M]`: real models, ~110 model calls for a full run (~4 minutes). Running one
  suite keeps the others' last results so the report stays complete.
- **Tests consolidated:** `pytest-cov` with branch coverage (**89%**, was 86% before this
  phase); new offline tests for the eval code (the metrics, the checks, the report, the
  dataset file itself) and for the CLI commands against the test database (cli.py 25% →
  42%). 383 tests.
- **Two fixes found by the evals** (see Results).

## Results (gpt-5-mini, effort low; judge gpt-5-mini; see evals/report.md)
| Suite | Metric | Result |
|---|---|---|
| Retrieval (35 q, top 5) | hit@1 vector / keyword / hybrid | 100% / 83% / 97% |
| | hit@5, all modes | 100% |
| | MRR vector / keyword / hybrid | 1.00 / 0.91 / 0.99 |
| Answers (40 q) | answered when answerable / refused when out of scope | 100% / 100% |
| | cites an expected document / citation precision | 100% / 97% |
| | grounded (judge) / correct (judge) | 97% / 89% |
| | latency mean / p95 | 3.8 s / 5.7 s |
| Agent (12 tasks) | task completion | **100%** (was 92% before the fixes) |

What the failures taught (read case by case, not just the numbers):
1. **a03** "Can I use the HR portal on Saturday morning?": the agent answered from the KB
   ("open 06:00–22:00") and missed the planned Saturday maintenance on the status board.
   Failed in 2 of 2 runs. Fix: rule 2 of the agent prompt now also covers "is it working or
   available, now or at a given time" and says status includes planned maintenance.
2. **a08** "change INC-1003 to in progress": in one run the agent used the *text* search for
   "INC-1003", found nothing and said the ticket didn't exist. Fix: the `search_tickets`
   description now says to use `get_ticket` for a ticket number (the registry is shared, so
   the MCP server got the same fix). After both fixes: 12/12 in three consecutive runs.
3. **Hybrid isn't always best:** on this small, clean corpus vector search alone puts the
   right document first every time; RRF lets a keyword ranking push it to #2 once. Hybrid
   still helps on exact identifiers (KI-0007, error codes); worth re-checking on a bigger
   corpus before changing the default.
4. **q19** (stolen laptop) misses "report it to the police", which lives in a second document
   (the remote-working policy). Multi-document answers are the weak spot of top-5 retrieval.
5. **Judge false alarms:** q06 was marked "partially supported" for facts that *are* in KB-0001
   but not in the chunk the answer cited: the judge only sees cited chunks. And "partially
   correct" was often reference detail beyond the question (q18, q28).
6. **Variance:** grounded was 91% in one full run and 97% in the next with no change to the
   answer pipeline. With 35 judged answers one case is ~3 points, so differences of a few
   points aren't meaningful.

## Key concepts
- **Tests vs evals:** tests check our code with fakes (deterministic, free, every change);
  evals check the system with the real models against a dataset (quality, costs money,
  varies). Both are needed.
- **hit@k / MRR:** rank of the first result from an expected document. hit@k = share with
  rank ≤ k; MRR = mean of 1/rank. Measures retrieval apart from generation.
- **Grounding (faithfulness) vs correctness:** "is every claim in the cited sources?" vs
  "does it have the reference's key facts?". A grounded answer can still miss the point; a
  complete answer can still contain an invented step.
- **LLM-as-judge:** a model grades with a rubric. Cheap and scalable, but it's a model:
  fixed labels, structured output, required reasons, and humans reading the flagged cases.
- **Task completion:** for agents, check behaviour (tools, proposed change, what not to do),
  not exact wording.
- **Eval-driven development:** measure → read failures → general fix → re-measure → watch
  for variance and for overfitting to the dataset.

## Design decisions
- **JSON Lines, validated:** one case per line (easy to add, clean diffs); Pydantic models
  with `extra="forbid"` so a misspelt key fails loudly with its line number; a test checks
  every expected doc id exists and every document is covered.
- **Expected documents, not expected chunks:** chunk ids change when documents are
  re-chunked; doc ids are stable. "Any of" lists allow facts that live in two documents.
- **Judge sees only the cited chunks:** that's what "grounded in its citations" means, and it
  keeps the prompt small. Cost: false alarms when the cited chunk is shorter than the document.
- **Agent tasks never approve changes**, and run on a scratch database reset to the seed data:
  safe to run anytime, and ticket numbers are known.
- **Two separate labels with fixed values** (not a 1–10 score): easier to agree on, to count,
  and to check by reading the case.
- **Threads, not async:** cases wait on the model API; `ThreadPoolExecutor(4)` cut the run
  from ~10 to ~4 minutes with no change to the synchronous code.
- **`rate([])` is None:** shown as n/a, so a `--limit` run without out-of-scope questions doesn't
  report "0% refused".

## Interview talking points
- "I built a 52-case evaluation set by hand from the knowledge base and one command that scores
  retrieval, grounding, correctness and agent task completion with the real models, and writes
  a report with every failing case."
- "Retrieval is 100% hit@5 and 0.99 MRR; answers are 100% right about when to refuse, 97%
  grounded and 89% correct by an LLM judge; agent task completion went from 92% to 100%."
- "The evals found two real agent bugs I'd never have caught by hand: it ignored planned
  maintenance, and once it searched for a ticket number as text. Both were general fixes in the
  prompt and the tool description, verified over three runs."
- "I don't trust the judge blindly: it must list unsupported claims and missing facts, and I
  read every flag. Some were false alarms because it only sees the cited chunk."
- "Small dataset, so I report variance honestly: one case is ~3 points."

## Known limitations (to revisit)
- **Small, author-written dataset** (52 cases, same author as the KB): optimistic; real user
  questions are messier. Next: collect real questions, add paraphrases and adversarial cases.
- **Same model judges itself** (gpt-5-mini); use `--judge-model` with a stronger model, and
  spot-check judge agreement with human labels.
- **Judge sees only cited chunks** → some false "unsupported" flags.
- **One run per number:** no confidence intervals; repeated runs would show the spread.
- **Agent tasks are single-turn** and stop at the proposed change (approval flows and
  multi-turn conversations are covered by tests with a scripted model, not by evals).
- Cost/latency aren't gated: the report shows them but nothing fails on a regression. CI could
  run the cheap retrieval suite on every PR (Phase 9).

## Commands
```bash
uv run servicedesk eval                         # all suites → evals/report.md
uv run servicedesk eval --suite retrieval       # cheap: embeddings only
uv run servicedesk eval --suite agent           # agent tasks only
uv run servicedesk eval --limit 3 --no-judge    # smoke check
uv run pytest --cov                             # 383 tests, 89% coverage
```
