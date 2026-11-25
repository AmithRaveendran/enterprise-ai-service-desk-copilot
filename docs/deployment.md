# Deployment

The application ships as one Docker image (`Dockerfile`). The same image runs the HTTP API
(default command), the MCP server, migrations, ingestion and evaluations: only the command
changes.

## Run the whole stack locally

```bash
cp .env.example .env            # add OPENAI_API_KEY; APP_ENV=dev keeps auth optional
docker compose --profile app up -d --build
docker compose run --rm api servicedesk seed      # first time: services + sample tickets
docker compose run --rm api servicedesk ingest    # first time: embed the knowledge base
```

| Service | What it does | Reach it at |
|---|---|---|
| `db` | PostgreSQL 17 + pgvector, data in the `pgdata` volume | `localhost:5432` |
| `migrate` | `alembic upgrade head`, then exits | — |
| `api` | `servicedesk serve` (FastAPI), health-checked | <http://localhost:8000/docs> |
| `mcp` | `servicedesk mcp --transport http` | `http://localhost:8001/mcp` |

Order is enforced by `depends_on`: `db` healthy → `migrate` completed successfully → `api`
and `mcp`. A failed migration stops the rollout instead of starting the app on an old
schema. `docker compose up -d` without the profile still starts only the database, for
working with `uv run` on the host.

Useful commands:

```bash
docker compose --profile app ps                 # health status of each service
docker compose logs -f api                      # one line per request, with its request ID
docker compose run --rm api servicedesk eval --suite retrieval
docker compose --profile app down               # stop (add -v to delete the database too)
```

## Configuration

Everything is an environment variable (`src/servicedesk/config.py`, documented in
`.env.example`). The ones that matter for a deployment:

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `prod` in the image | `prod` requires `API_KEY` |
| `API_KEY` | — | 16+ characters; callers send it as `X-API-Key` |
| `DATABASE_URL` | localhost | `postgresql+psycopg://user:pass@host:5432/db` |
| `LLM_PROVIDER` | `openai` | or `azure` (+ the `AZURE_OPENAI_*` variables) |
| `OPENAI_API_KEY` | — | without it the API still serves search (keyword), status and tickets; `/ask` and `/chat` answer 503 |
| `LLM_TIMEOUT_SECONDS` | 60 | upper bound for one model call |
| `RAG_MIN_SIMILARITY` | 0.35 | re-measure if the embedding model changes |

## Image

- Multi-stage build: dependencies are installed by uv from `uv.lock` (`--frozen`) in a
  build stage; the runtime stage gets only the virtual environment, `alembic.ini`,
  `migrations/`, `data/` and the eval dataset.
- Runs as the unprivileged user `app` (uid 10001); listens on `0.0.0.0:8000`.
- `HEALTHCHECK` calls `/health`, which answers 503 when the database is unreachable.
- tiktoken's token tables are downloaded at build time, so the container needs no internet
  access except to the LLM provider.
- Dependency layers are cached separately from the code: a code-only change rebuilds in
  seconds.

## Continuous integration (`.github/workflows/`)

| Workflow | When | What |
|---|---|---|
| `ci.yml` → `lint` | every push to main, every PR | `ruff check`, `ruff format --check` |
| `ci.yml` → `test` | same | full test suite against a pgvector service container, coverage ≥ 85% enforced, `coverage.xml` uploaded |
| `ci.yml` → `docker` | after lint | builds the image (layer cache in GitHub Actions), starts `db → migrate → api` with compose and waits for the API's health check |
| `eval.yml` | manual (Run workflow) | ingests the KB with real embeddings and runs the retrieval eval; fails below a hit@5 threshold; uploads the report. Needs the `OPENAI_API_KEY` repository secret |

CI needs no secrets: the tests use offline fakes for OpenAI. Integration tests fail rather
than skip in CI, so a broken database setup can't produce a green run.

## Deploying to the cloud (reference layout)

The image runs anywhere containers run. A typical managed layout on Azure (AWS equivalents
in brackets):

```
            Internet
               │ HTTPS
     ┌─────────▼──────────┐
     │ Front Door / APIM  │  TLS, WAF, rate limits, auth (OIDC)   [CloudFront + API Gateway]
     └─────────┬──────────┘
               │ private network
     ┌─────────▼──────────┐      ┌───────────────────────────┐
     │ Container Apps     │─────▶│ Azure OpenAI              │  [Bedrock / OpenAI]
     │  api  (2+ replicas)│      └───────────────────────────┘
     │  migrate (job)     │      ┌───────────────────────────┐
     └─────────┬──────────┘      │ Key Vault (managed id.)   │  [Secrets Manager]
               │                 └───────────────────────────┘
     ┌─────────▼──────────┐
     │ PostgreSQL Flexible│  pgvector extension, backups, HA     [RDS / Aurora PostgreSQL]
     │ Server             │
     └────────────────────┘
```

Steps:

1. **Database:** managed PostgreSQL with the `vector` extension allowed (Azure: add
   `VECTOR` to `azure.extensions`). Private networking only.
2. **Image:** push to a registry (ACR/ECR/GHCR). Tag with the git SHA, never deploy `latest`.
3. **Migrations:** run the image as a one-off job with `alembic upgrade head` before each
   rollout (the cloud equivalent of the `migrate` service). Migrations are written to be
   additive so the old version keeps working while the new one rolls out.
4. **API:** at least 2 replicas behind the gateway; liveness and readiness probes on
   `/health`; secrets from the vault as environment variables.
5. **Ingestion:** a scheduled or on-demand job (`servicedesk ingest`). It is idempotent, so
   it can simply run whenever the KB source changes.
6. **Observability:** ship stdout logs to the platform (Log Analytics/CloudWatch); search
   by `request_id`. Alert on 5xx rate, p95 latency and LLM error rate.

### Scaling notes
- The API is stateless: conversation state is in Postgres (LangGraph checkpointer), so any
  replica can continue any thread, including a pending approval.
- One uvicorn process per container; FastAPI runs the synchronous endpoints in a thread
  pool. Scale out with replicas rather than workers per container.
- Each replica holds a SQLAlchemy pool and, after the first chat, a checkpointer pool:
  size Postgres `max_connections` for replicas × pools (or add PgBouncer).
- Most time is spent waiting on the LLM API (see [performance.md](performance.md)), so
  the provider's rate limits are the first ceiling, not CPU.

### The MCP server in production
Keep stdio for desktop clients (the client starts `servicedesk mcp` locally). The HTTP
transport has no authentication of its own: expose it only behind a gateway that
authenticates, and add the public hostname with `--allowed-host`.

## Rollback
The image is versioned by tag, so rolling back means redeploying the previous tag. Alembic
can downgrade (`alembic downgrade -1`), but prefer forward fixes for schema changes that
have already stored data.
