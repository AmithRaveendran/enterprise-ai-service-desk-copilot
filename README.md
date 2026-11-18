# Enterprise AI Service Desk Copilot

An agentic AI support assistant for an IT service desk, built with **Python, FastAPI,
LangGraph, PostgreSQL/pgvector and MCP**.

> Personal portfolio project, developed independently. It is being built in phases.
> See the roadmap below.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations, local Postgres + pgvector, sample knowledge base | ✅ |
| 1 | Document ingestion (parse → chunk → embed → store) | ✅ |
| 2 | Semantic + hybrid search, first FastAPI endpoints | ✅ |
| 3 | RAG answers with source citations | ✅ |
| 4 | Service-status and ticket-management tools with input validation | ✅ |
| 5 | Stateful LangGraph workflow | ✅ |
| 6 | Human approval for writes, error handling, workflow recovery | ✅ |
| 7 | MCP server | ⏳ |
| 8 | Automated tests + evaluation dataset | |
| 9 | Containerisation, CI, architecture/security/deployment docs | |

## Sample knowledge base

`data/kb/` holds 19 fictional IT documents for a fictional company, *Halcyon Dynamics*.
They come in three formats (Markdown with front matter, intranet HTML exports, and PDFs) so
that ingestion has to deal with real-world variety. The PDFs are generated from
`scripts/pdf_sources/` by `scripts/build_pdfs.py`.

## Getting started

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker Desktop, Git.

```bash
uv sync                                  # create .venv and install dependencies
cp .env.example .env                     # then add your OpenAI key
docker compose up -d                     # start Postgres + pgvector
uv run alembic upgrade head              # create the database tables
uv run servicedesk seed                  # service catalogue + sample tickets
uv run pytest                            # run the test suite
```

## Ingesting the knowledge base

```bash
uv run servicedesk ingest --dry-run      # free preview: documents, chunks, token count
uv run servicedesk ingest                # embed with OpenAI and store in pgvector
uv run servicedesk chunks data/kb/how-to/vpn-setup.md   # see how one file is chunked
```

Ingestion is idempotent. Unchanged files are skipped by content hash, changed files are
re-embedded, and deleted files are removed. See
[docs/learning-log/phase-1.md](docs/learning-log/phase-1.md) for how it works.

## Searching

```bash
uv run servicedesk search "vpn keeps dropping on my macbook"   # hybrid (vector + keyword)
uv run servicedesk search "secure print" --mode keyword        # works without an API key
uv run servicedesk serve                                       # HTTP API on port 8000
```

Then open <http://127.0.0.1:8000/docs> to try `GET /search` and `GET /health` in the browser.
Search combines pgvector cosine similarity with Postgres full-text search using Reciprocal
Rank Fusion. See [docs/learning-log/phase-2.md](docs/learning-log/phase-2.md).

## Asking questions

```bash
uv run servicedesk ask "My VPN keeps dropping on my MacBook, what should I do?"
uv run servicedesk ask "what is the capital of France"     # refused: not in the knowledge base
```

Or `POST /ask` with `{"question": "..."}` (try it at `/docs`). Answers cite their sources
with `[n]` markers, and each citation names the document, section and chunk it came from.
The copilot refuses instead of guessing when the evidence is weak: off-topic questions are
stopped before the language model is called, the model can decline when the sources don't
cover the question, and an answer without a valid citation is never shown. See
[docs/learning-log/phase-3.md](docs/learning-log/phase-3.md).

## Service status and tickets

```bash
uv run servicedesk status                     # health of every IT service
uv run servicedesk tools                      # the tools the assistant can use
uv run servicedesk tool search_tickets '{"open_only": true}'
uv run servicedesk tool get_ticket '{"ticket_id": "INC-1001"}'
```

The API has the same operations: `GET /services/status`, `GET /tickets`, `POST /tickets`,
`GET /tickets/{id}`, `PATCH /tickets/{id}` and `POST /tickets/{id}/comments` (the writes accept an
`Idempotency-Key` header, so a retried request doesn't create a second ticket). Each tool is
defined once (`desk/tools.py`), with a strict input schema, and is shared by the API, the
CLI and the AI agent. Inputs are validated (company emails, ticket numbers like
`INC-1042`, enums, lengths, unknown fields rejected), and ticket status changes follow the
IT policy (for example, closed tickets can't be reopened). See
[docs/learning-log/phase-4.md](docs/learning-log/phase-4.md).

## Chatting with the assistant

```bash
uv run servicedesk chat                                   # interactive conversation
uv run servicedesk chat -m "VPN is broken on my MacBook"  # one message
uv run servicedesk chat -t <thread-id> -m "Do I already have a ticket for it?"
```

Or `POST /chat` with `{"message": "...", "thread_id": "..."}` (leave `thread_id` out to start
a conversation) and `GET /chat/{thread_id}` for the history. The assistant is a LangGraph
agent: the model decides which tools to call (service status, ticket lookups, and the
knowledge base, which answers with validated citations), the graph runs them and loops
until it can reply, with a limit on tool rounds. Conversations are saved in Postgres by
LangGraph's checkpointer, so a thread continues after a restart. See
[docs/learning-log/phase-5.md](docs/learning-log/phase-5.md).

### Approving changes

The assistant can create tickets, update them and add comments, but **never without your
approval**. When it wants to change something, the turn pauses and shows the exact change:

- CLI: the change is printed as a table; answer `y` to approve, or `n` and an optional
  reason ("make it P4"), and the assistant proposes a corrected change.
- API: the response has `"status": "awaiting_approval"` and `pending_actions`. Continue with
  `POST /chat/{thread_id}/approve` or `POST /chat/{thread_id}/reject` (`{"reason": "..."}`).

If a turn fails part-way (for example the language model API is down), the API answers 502
with the `thread_id`; `POST /chat/{thread_id}/resume` (or `servicedesk chat -t <id>`) finishes
it from the last saved step without repeating finished tool calls, and a write that already
happened is never repeated (idempotency keys). See
[docs/learning-log/phase-6.md](docs/learning-log/phase-6.md).

## License

MIT
