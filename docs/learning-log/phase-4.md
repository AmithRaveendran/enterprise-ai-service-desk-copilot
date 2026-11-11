# Phase 4: Service-status and ticket tools with input validation

## What was built
```
                 ┌── REST API (api/routes/desk.py)
callers ─────────┼── CLI (`servicedesk tool …`, `status`, `set-status`, `seed`)
                 └── (Phase 5 agent, Phase 7 MCP server)
                          │
              desk/tools.py   TOOLS registry + run_tool(): validate → run → {"ok", result|error}
                          │
              desk/schemas.py Pydantic inputs/outputs: every validation rule, in one place
                          │
              desk/service.py ServiceDesk: business rules (status transitions, closed = read-only…)
                          │
              Postgres        services, tickets, ticket_comments (+ CHECK constraints)
```
- Migration `0003`: `services` (the IT service catalogue with each service's current health),
  `tickets` (numbers start at 1001, shown as INC-1001) and `ticket_comments`.
- Six tools: `get_service_status`, `search_tickets`, `get_ticket` (read) and
  `create_ticket`, `update_ticket`, `add_comment` (write, flagged for approval in Phase 6).
- REST: `GET /services/status`, `GET /services/{service}/status`, `GET /tickets`,
  `POST /tickets`, `GET /tickets/{id}`, `PATCH /tickets/{id}`, `POST /tickets/{id}/comments`.
- CLI: `seed`, `status`, `set-status` (admin only, not a tool), `tools`, `tool NAME '{json}'`.
- Seed data (`data/seed/`): the 10 services from the KB's service catalogue (VPN degraded to
  match known issue KI-0007, HR Portal in maintenance) and 5 sample tickets.
- 100 new tests (258 in total).

## Rules taken from the knowledge base
The IT policies in `data/kb/` define the rules, so the tools and the answers agree:

| Policy says | Enforced by |
|---|---|
| Priorities P1–P4 with SLA targets | `Priority` enum; every ticket returns its SLA targets |
| P1/P2 must also be phoned in to 4357 | `notice` on open P1/P2 tickets |
| "Waiting for Requester" pauses the SLA | status `waiting_for_requester`; a requester reply moves it back to `in_progress` |
| Resolved tickets can be reopened; closed ones can't | `ALLOWED_TRANSITIONS`; closed = read-only, with a message to raise a new ticket |
| Services in the catalogue | `services` table seeded from the catalogue; loose name matching ("wifi" → Wi-Fi) |

## Key concepts

**One definition, many callers.** A tool = name + description (written for an LLM) + input
schema + function + `writes` flag. The REST API uses the same input classes as request
bodies, so a rule added once is enforced for HTTP, CLI, agent and MCP.

**Validation layers (defence in depth).**
1. *Pydantic schemas*: types, enums, lengths, company email, ticket number formats,
   unknown fields forbidden (`extra="forbid"`), cross-field rules (resolving needs a
   resolution; an update must change something).
2. *Service layer*: rules that depend on current data (status transitions, closed tickets,
   unknown service names).
3. *Database*: CHECK constraints on every enum-like column and "resolved/closed needs a
   resolution", so even a bug or a manual SQL insert can't store an invalid ticket.

**Normalise, then validate.** Text is cleaned *before* the length check: control characters
removed, titles collapsed to one line, emails lower-cased, "INC-1042" / "inc 1042" / "1042"
all accepted. Being liberal in what you accept and strict in what you store helps both
people and LLMs.

**Errors as data for tools.** `run_tool` never raises for a bad call. It returns
`{"ok": false, "error": {"code", "message", "problems": [{field, problem}…]}}`, reporting
every problem at once, so a model can fix its arguments in one retry. Codes are stable:
`invalid_input`, `not_found`, `conflict`, `unknown_tool`.

**HTTP status codes.** 201 created, 404 not found, **409 conflict** (valid request, but the
ticket's current state doesn't allow it), 422 invalid input. Business errors map to status
codes in one place (an exception handler in `app.py`), not in every route.

**Transactions and locking.** Every write runs in `sessionmaker.begin()` (all or nothing).
Updates load the ticket with `SELECT … FOR UPDATE`, so two simultaneous changes can't both
pass the status check. (`joinedload` would add a LEFT JOIN, and Postgres can't lock the
nullable side of an outer join, so related rows are loaded with `selectinload`.)

**PATCH semantics.** Only the fields sent are changed: `model_dump(exclude_unset=True)`
tells "not sent" apart from "sent as null" (`{"assignee": null}` unassigns).

**CHECK constraints instead of Postgres ENUM types.** Adding a value to an ENUM can't be
undone easily and is awkward in migrations; a CHECK constraint is replaced in one line.
Allowed values are written out in the migration, not imported, so an old migration always
means what it meant when it was written.

**Least privilege.** The assistant can read service health but not change it:
`set_service_status` exists for admins (CLI) and is deliberately not a tool.

## Design decisions
- **Ticket numbers start at 1001** (`Identity(start=1001)`), displayed as INC-1001.
- **Search uses ILIKE with escaped wildcards**: searching "100%" finds "100%" literally.
  Fine for thousands of tickets; full-text search (as for the KB) would be the next step.
- **Seeding is idempotent**: services are upserted (`INSERT … ON CONFLICT DO UPDATE`) without
  overwriting a status changed since; sample tickets only go into an empty table.
- **Outputs are typed too** (`TicketOut`, `ServiceStatusOut`), so the agent and API clients
  get a stable contract, documented in `/docs`.

## Interview talking points
- "How do you validate tool input from an LLM?" Treat it as untrusted: strict Pydantic
  schemas with `extra="forbid"`, normalisation, enums; errors returned as structured data
  listing every field so the model can self-correct.
- "Where do business rules live?" In a service layer with no HTTP/LLM knowledge, reused by
  every interface; the database repeats the critical ones as constraints.
- "How do you prevent race conditions on updates?" Row lock (`SELECT … FOR UPDATE`) inside
  a transaction.
- "Why 409 vs 422?" 422 = the request itself is malformed; 409 = well-formed but conflicts
  with the resource's current state.

## Known limitations (to revisit)
- No authentication: the requester is whatever email is sent. Phase 9 documents how
  identity would come from SSO; Phase 6 adds human approval before any write by the agent.
- No audit history of field changes (only `updated_at` and comments).
- SLA times are reported, not tracked (no breach timers or escalations).
- Ticket search is substring matching, not ranked full-text search.

## Commands
```bash
uv run alembic upgrade head                      # adds the service desk tables
uv run servicedesk seed                          # catalogue + sample tickets (safe to re-run)
uv run servicedesk status                        # all services; `status vpn` for one
uv run servicedesk set-status vpn major_outage --message "SecureLink is down"
uv run servicedesk tools
uv run servicedesk tool search_tickets '{"requester": "priya.sharma@halcyon-dynamics.example"}'
uv run servicedesk tool update_ticket '{"ticket_id": "INC-1003", "status": "in_progress"}'
curl http://127.0.0.1:8000/services/status
curl "http://127.0.0.1:8000/tickets?open_only=true"
```
