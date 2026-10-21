# Enterprise AI Service Desk Copilot

An agentic AI support assistant for an IT service desk, built with **Python, FastAPI,
LangGraph, PostgreSQL/pgvector and MCP**.

> Personal portfolio project, developed independently. It is being built in phases.
> See the roadmap below.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations, local Postgres + pgvector, sample knowledge base | ✅ |
| 1 | Document ingestion (parse → chunk → embed → store) | ⏳ |
| 2 | Semantic + hybrid search, first FastAPI endpoints | |
| 3 | RAG answers with source citations | |
| 4 | Service-status and ticket-management tools with input validation | |
| 5 | Stateful LangGraph workflow | |
| 6 | Human approval for writes, error handling, workflow recovery | |
| 7 | MCP server | |
| 8 | Automated tests + evaluation dataset | |
| 9 | Containerisation, CI, architecture/security/deployment docs | |

## Sample knowledge base

`data/kb/` holds about 20 fictional IT documents for a fictional company, *Halcyon Dynamics*.
They come in three formats (Markdown with front matter, intranet HTML exports, and PDFs) so
that ingestion has to deal with real-world variety. The PDFs are generated from
`scripts/pdf_sources/` by `scripts/build_pdfs.py`.

## Getting started

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker Desktop, Git.

```bash
uv sync                                  # create .venv and install dependencies
cp .env.example .env                     # then add your OpenAI key
docker compose up -d                     # start Postgres + pgvector
uv run pytest                            # run the test suite
```

Check that pgvector is enabled:

```bash
docker compose exec db psql -U servicedesk -c "\dx"
```

## License

MIT
