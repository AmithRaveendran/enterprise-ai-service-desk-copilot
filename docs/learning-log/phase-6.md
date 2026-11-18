# Phase 6: Human approval, error handling, workflow recovery

## What was built
```
 POST /chat · /approve · /reject · /resume        servicedesk chat (asks "Approve? [y/N]")
        │
   agent/copilot.py   chat() · decide(approved, reason) · resume() · status()
        │
   agent/graph.py     START → agent ─┬─(reads only)──────────────▶ tools ─▶ agent ...
                                     ├─(a write)─▶ approve ──────▶ tools
                                     │            interrupt(): the run stops here,
                                     │            saved in Postgres, until a person decides
                                     └─(reply)──▶ finalize → END
        │
   desk/service.py    writes take an idempotency key, recorded in the same transaction
   (tool_executions)  as the write → a resumed run can't create the same ticket twice
```
- **The agent gets the write tools** (`create_ticket`, `update_ticket`, `add_comment`), and
  none of them can run without a person's approval.
- `agent/graph.py`: a new **approve** node. `route()` sends a model message with any valid
  write call to it; it calls LangGraph's `interrupt()` with the pending actions, which saves
  the state and ends the request. `Command(resume={"approved": ..., "reason": ...})`
  continues the same run later, even from another process.
- **tools** node: `run_call()` is the single place a write can happen. Approved → run with an
  idempotency key; rejected → a `rejected` result with the person's reason; no decision →
  `approval_required` (defence in depth: route() should make this impossible).
- `agent/copilot.py`: thread states (`new`, `idle`, `awaiting_approval`, `interrupted`) and
  the operation each allows: `chat`, `decide`, `resume`. Wrong state → `ThreadStateError`
  with a code (409 in the API), unknown thread → 404.
- **Recovery:** `resume()` = `graph.invoke(None, config)`: continue from the last checkpoint.
  Finished steps are not repeated.
- **Idempotency:** migration `0004` adds `tool_executions(key, tool, result)`. The service
  claims the key with `INSERT ... ON CONFLICT DO NOTHING` inside the write's transaction and
  stores the result before commit. Same key again → the saved result, no second write.
  The REST write endpoints accept an `Idempotency-Key` header with the same guarantee.
- **Error handling:** tool database errors → `unavailable`, unexpected tool bugs →
  `tool_failed` (logged with traceback), both as data so the conversation goes on; a model
  API failure (after the SDK's retries with backoff) → 502 with `code` and `thread_id`, and
  the turn can be resumed. All chat errors have a JSON body `{detail, code}`.
- CLI: pending changes are shown as a table; approve with `y`, or reject with an optional
  reason. Re-opening a paused or failed thread (`chat -t <id>`) offers to finish it first.
- 32 new tests (321 in total).

## How approval runs
"Please add a comment to INC-1009: 'It also happens in Zoom.'" (a real API run):
1. **agent**: the model asks for `add_comment {ticket_id: 1009, author: li.wei@…, body: …}`.
2. **route**: it's a write with valid arguments → **approve**. (Invalid arguments skip
   approval: the validation errors go straight back to the model, nothing to approve.)
3. **approve**: `interrupt({"actions": [...]})`. The checkpoint is saved with the pending
   interrupt; `POST /chat` returns `status: "awaiting_approval"` and the validated arguments
   (ticket shown as `1009`, defaults like priority P3 filled in).
4. Meanwhile `POST /chat` on that thread → **409 `approval_pending`**.
5. `POST /chat/{id}/approve` → `Command(resume={"approved": true})`. The **approve** node runs
   again from the top; this time `interrupt()` returns the decision, which is stored in
   `approvals[call_id]`.
6. **tools**: approved → `add_comment` runs with key `agent:<thread>:<call_id>` → **agent**
   writes "Done, I added your comment to INC-1009." → **finalize**.
7. A second approve → **409 `no_pending_action`**.

## How recovery runs
- **The model fails mid-turn** (tested with a fake that raises `APIConnectionError` on its
  2nd call): the KB tool had already run and been checkpointed. The thread is `interrupted`
  (`snapshot.next` is non-empty, no interrupt pending). `resume()` re-runs only the failed
  **agent** step: the KB was asked once, not twice.
- **The process dies right after a write committed but before the checkpoint was saved**
  (tested by a tool runner that raises after the write): the tools step is lost, so resume
  re-runs it, with the same `call_id` → the same idempotency key → the saved result is
  returned. One ticket, not two.
- **The approval arrives after a restart** (tested with two separate Postgres pools): the
  interrupt and its pending actions are in the checkpoint, not in memory.

## Key concepts

**`interrupt()` and `Command(resume=...)`.** A node calls `interrupt(value)`: the first time,
LangGraph saves the state and stops, returning the value to the caller (in
`snapshot.interrupts`). Resuming with `Command(resume=answer)` re-runs **the whole node from
the start**, and `interrupt()` now returns `answer`. So the code before `interrupt()` must
have no side effects. That's why approval is its own node, not a line inside the tools
node: re-running the tools node would re-run tools that already ran.

**Why approval needs a checkpointer.** Without one there's nowhere to keep the paused run.
With Postgres, the person can approve hours later, from another API worker.

**Validate before asking.** Only write calls whose arguments pass the Pydantic schema are
shown for approval, and they are shown *normalised*. A person should approve exactly what
will run, never a call that would fail anyway.

**Anything but an explicit yes is a no.** The resume value is checked: only
`{"approved": True}` approves. A malformed value is a rejection (tested).

**Exactly-once writes in an at-least-once system.** Checkpoint-based resume means a step can
run more than once (it's saved *after* it finishes). Reads don't care; writes need
idempotency. The key is the tool call id, which is stable because the model's message
(with its tool calls) is already in the checkpoint. The claim (`ON CONFLICT DO NOTHING`)
and the result are in the write's own transaction, so there is no window where the write
committed but the key didn't; and a concurrent duplicate waits for the first transaction
instead of writing too (tested with two threads).

**A thread is a state machine.** `new → idle ⇄ awaiting_approval`, and any step can end in
`interrupted` if it fails. Each API call is only valid in one state. LangGraph alone would
happily accept a new message on a paused thread and start a fresh run (checked with a
probe), leaving the paused tool call unanswered.

## Design decisions
- **One decision per model message.** If the model asks for two writes at once, approve or
  reject covers both (both are listed). Simpler for the person and the API; per-action
  decisions can come later.
- **No "edit and approve".** To change a proposal, reject with a reason ("make it P2"); the
  model proposes a corrected change, which is asked again. Editing arguments in the
  approval would need re-validation and makes "what was approved" less clear.
- **The rejection reason goes into the tool result as data**, capped at 500 chars, and
  labelled as the person's words.
- **No new message while a change is pending** (409). Treating it as an implicit rejection
  was possible, but explicit is easier to reason about and to test.
- **Errors from tools become data; errors from the model become a 502 + resumable thread.**
  The model can't fix a model outage, but it can explain a tool outage.
- **API keys are namespaced** (`api:` / `agent:`), so a client key can't collide with an
  agent call.
- **The approval is not an authorisation system.** Anyone holding the thread id can approve
  (there's no auth yet). It's the user confirming the assistant's action; who may approve
  what (roles, requester only) belongs with authentication (Phase 9 security doc).

## Results with the real model (gpt-5-mini, reasoning effort low)
| Conversation | What happened |
|---|---|
| "Outlook keeps crashing… raise a ticket. I'm tom.oconnor@…" | `get_service_status` + `search_tickets` found his open INC-1002 → offered to comment or change it instead of a duplicate. "Raise it as P2" → `update_ticket {priority: P2}` paused → approved → done, with the phone-in notice |
| "Webcam black in Teams… raise a ticket. I'm li.wei@…" | `create_ticket` paused. Rejected: "mention it is a ThinkPad T14 and make it P4" → the model proposed the corrected ticket → approved → INC-1009 |
| API: "add a comment to INC-1009" | 200 `awaiting_approval` → new message 409 → approve 200 → approve again 409 |

Two prompt fixes came from these runs:
- The first rejection was described as "the system rejected it". The `rejected` result now
  says the *person* declined, and the prompt says to propose a corrected change if the
  reason says what to change.
- A draft ticket included details the person never said ("tried rebooting, external mouse
  works"). The prompt now says to write only what the person said. Approval is what caught
  it: the person sees the exact text before it's saved.

## Interview talking points
- "How do you keep a human in the loop?" Every write tool pauses the LangGraph run with
  `interrupt()` in a dedicated approve node. The state is checkpointed in Postgres; approve
  or reject resumes it with `Command(resume=...)`. Writes can only run in one function, and
  only with an explicit approval for that tool call id.
- "What if the process crashes between the write and the checkpoint?" Idempotency key =
  thread + tool call id, recorded in the write's own transaction. Resuming re-runs the step
  and gets the saved result. Tested by crashing right after the commit.
- "What if the LLM API is down?" The SDK retries with backoff; if it still fails, the API
  returns 502 with the thread id, and the thread is `interrupted`. `/resume` continues from
  the last finished step, without repeating tool calls.
- "Why not let the model call write tools and just log them?" Because an LLM can be
  prompt-injected (a ticket description is untrusted text) or simply wrong. Live runs
  showed it adding details the person never gave; approval stopped them being saved.
- "How do you test approval without a real model?" A scripted fake model plus an in-memory
  checkpointer for the graph logic; Postgres for the restart and crash tests.

## Known limitations (to revisit)
- No authentication or roles: anyone with the thread id can approve (Phase 9 security).
- One decision covers all writes in a model message; no editing arguments in place.
- A paused change has no expiry, and the ticket may change before it's approved (then
  the service rules still apply when it runs, e.g. a `conflict` result if it was closed).
- Two simultaneous requests on the same thread aren't serialised (writes are still
  protected by idempotency keys, but both runs would reply). A per-thread lock (Postgres
  advisory lock) would fix that.
- The stored result of an idempotent replay is the ticket *as it was* after the first write.
- The REST `Idempotency-Key` doesn't check that a repeated request has the same body.
- Token usage of a paused turn is reported again, cumulatively, after the decision.

## Commands
```bash
uv run alembic upgrade head                               # adds tool_executions (0004)
uv run servicedesk chat                                   # approve/reject at the prompt
uv run servicedesk chat -t <thread-id>                    # finish a paused or failed turn
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Raise a ticket: my webcam is black in Teams. I am li.wei@halcyon-dynamics.example"}'
curl -X POST http://127.0.0.1:8000/chat/<thread-id>/approve
curl -X POST http://127.0.0.1:8000/chat/<thread-id>/reject -H "Content-Type: application/json" \
     -d '{"reason": "Make it P4"}'
curl -X POST http://127.0.0.1:8000/chat/<thread-id>/resume
curl -X POST http://127.0.0.1:8000/tickets -H "Idempotency-Key: my-request-0001" ...
uv run pytest tests/test_approval.py tests/test_api_chat.py -v
```
