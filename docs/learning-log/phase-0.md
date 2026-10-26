# Phase 0: Foundations and sample documents

## What was built
- Project scaffold using the `src/` layout, managed by **uv** (`pyproject.toml` + `uv.lock`).
- `servicedesk.config.Settings`: typed configuration from env vars / `.env` via
  **pydantic-settings**, with API keys wrapped in `SecretStr`.
- `docker-compose.yml`: PostgreSQL 17 with the **pgvector** extension enabled by an init script.
- A sample knowledge base of 19 docs for the fictional company *Halcyon Dynamics*:
  12 Markdown, 3 HTML, and 4 PDFs (generated from `scripts/pdf_sources/`).
- Tests: config behaviour plus data-quality checks on the corpus.
- Tooling: ruff (lint + format), pytest, pre-commit hooks.

## Key concepts
- **src layout:** the package lives in `src/servicedesk`, so tests import the *installed*
  package rather than whatever happens to be in the current directory. This catches
  packaging mistakes early.
- **uv:** a fast replacement for pip + venv + pip-tools. `uv sync` creates `.venv` from the
  lockfile, so every machine gets the exact same versions.
- **12-factor config:** settings come from the environment, never from hard-coded values.
  `.env` is git-ignored and `.env.example` documents what is needed.
- **SecretStr:** prevents API keys from appearing in logs, tracebacks and `repr()`.
- **pgvector:** a Postgres extension that adds a `vector` column type and similarity
  operators. It lets one database hold relational data (tickets) *and* embeddings.
- **Docker init scripts:** files in `/docker-entrypoint-initdb.d` run only when the data
  volume is empty. To re-run them: `docker compose down -v`.

## Design decisions
- **Mixed document formats on purpose.** Each format has its own messiness: Markdown front
  matter, HTML nav/footer boilerplate, and PDF running headers. Ingestion has to normalise
  all three.
- **Consistent facts across documents** (for example, P1 = 15-minute response; passwords need 14
  characters). This lets the Phase 8 evaluation check answers objectively.
- **Data-quality tests on the corpus.** Bad metadata breaks citations later, so it is
  caught early.

## Interview talking points
- "Why pgvector rather than a dedicated vector DB?" One system to run, transactional
  consistency between documents and their chunks, SQL filtering, and good performance at
  this scale with HNSW indexes.
- "How do you manage secrets?" Secrets live in a git-ignored `.env`, are wrapped in
  SecretStr so they don't leak into logs, and a pre-commit hook blocks committed private
  keys. (That hook does not detect API keys; full secret scanning, e.g. gitleaks, is
  planned for Phase 9.)

## Commands
```bash
uv sync
docker compose up -d
docker compose exec db psql -U servicedesk -c "\dx"   # shows the 'vector' extension
uv run pytest
uv run python scripts/build_pdfs.py                    # regenerate PDFs
```
