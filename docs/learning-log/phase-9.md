# Phase 9: Containerisation, CI, and production docs

## What was built
```
 Dockerfile (multi-stage, uv, non-root, HEALTHCHECK)
        │  one image: `servicedesk serve` by default, any `servicedesk` command on demand
        ▼
 docker compose --profile app:  db ──healthy──▶ migrate ──completed──▶ api :8000 + mcp :8001
        │                        (ports on 127.0.0.1 / ::1 only)
        ▼
 GitHub Actions ci.yml:  lint ─▶ docker (build + start stack + health + auth check)
                         test (pgvector service, coverage ≥ 85%, fail-not-skip)
                 eval.yml (manual): ingest with real embeddings → retrieval eval → threshold
```
- **Dockerfile:**
  - Dependencies install in their own cached layer (`uv sync --frozen --no-install-project`),
    and the package installs afterwards (`--no-editable`), so the runtime stage copies only
    `.venv`.
  - tiktoken's tables are fetched at build time.
  - Runs as uid 10001 with `APP_ENV=prod` and a `HEALTHCHECK` on `/health`.
  - `.dockerignore` keeps `.env`, tests and notes out of the build.
- **docker-compose.yml:** an `app` profile with `migrate` (one-shot `alembic upgrade head`),
  `api` and `mcp` (HTTP). `docker compose up -d` still starts only the database.
- **API security** (`api/security.py`):
  - The `X-API-Key` header is required on every router except `/health`, and checked with
    `hmac.compare_digest`.
  - The app refuses to start when `APP_ENV=prod` has no key (or one shorter than 16
    characters).
  - `API_KEY=` (empty) means no key.
- **Request IDs** (`api/middleware.py`):
  - Every response has `X-Request-ID`. An incoming ID is reused only if it matches
    `[A-Za-z0-9._-]{1,64}`.
  - One `key=value` log line per request, with no bodies; `/health` is logged at DEBUG.
  - Uvicorn's access log is off, since it would repeat the line.
- **MCP over HTTP in Docker:** DNS-rebinding protection is now always on. The SDK enables it
  only for host 127.0.0.1, and in a container we bind 0.0.0.0. `--allowed-host` adds a real
  hostname.
- **Docs:** [architecture](../architecture.md) (Mermaid diagrams),
  [security](../security.md) (threats → controls → gaps), [deployment](../deployment.md)
  (compose, CI, a cloud reference layout), and [performance](../performance.md) (measured).
- **Tests:** 404 (up from 383); 21 new tests for auth, the prod check and request IDs. Coverage 89%.
- Version **1.0.0**.

## Found while verifying (each would have broken something real)
1. **`API_KEY=` loaded as an empty key, not as no key.** Anyone who copied `.env.example`
   would have got 401 on every request in dev. Found by starting the container in prod mode.
   Fixed with a `field_validator`; I didn't use pydantic-settings' global `env_ignore_empty`,
   because `LLM_REASONING_EFFORT=` deliberately means "none".
2. **IPv6 `localhost` hang.** I published Postgres on `127.0.0.1` only. On Windows,
   `localhost` resolves to `::1` first, and a connection to the unpublished `::1` port hung
   until it timed out. The test suite stalled on engines with no connect timeout. Fixed by
   publishing on both loopbacks (`127.0.0.1` and `[::1]`).
3. **Four tests passed only because my machine has an OpenAI key.** Without one, `/ask`
   answers 503 before validating the body. Found by running the CI steps in a clean copy with
   no `.env` and no key. The tests now use the fake model.
4. **The CI smoke test would have crash-looped.** With no `.env`, the image's `APP_ENV=prod`
   means no key, so the API refuses to start. CI now writes a throwaway key, and the smoke
   test also proves the key is enforced (401 without it, 200 with it).

## Measured (details in performance.md)
| | |
|---|---|
| keyword / hybrid / vector search, server p50 | 9 ms / 191 ms / 211 ms |
| vector search split | OpenAI embed 206 ms (p95 818) · pgvector 10 ms |
| `/ask` mean / p95 | 3.8 s / 5.7 s (~1,200 tokens) |
| agent turn mean / p95 | 7.1 s / 15.9 s (~4,800 tokens) |
| API restart → healthy · image | 8.8 s · 465 MB |

The database accounts for well under 10% of any request. Latency is model calls.

## Key concepts
- **Multi-stage build:** build tools in one stage, only the artefacts in the final one. The
  image is smaller, and an attacker who gets in finds fewer tools.
- **Layer caching:** order the Dockerfile steps from least to most often changed
  (lock file, then code), so a code change doesn't reinstall the dependencies.
- **Health check vs readiness:** `/health` checks the database. A 503 tells Docker or a load
  balancer to stop sending traffic. `depends_on: condition` orders start-up.
- **Migrations as a one-shot job:** run once per release, before the app, never by each
  replica at start (replicas would race each other).
- **Secure by default / fail closed:** a missing security setting stops start-up instead of
  serving openly.
- **Timing-safe comparison:** `hmac.compare_digest` doesn't leak how many characters of a
  guessed key were right.
- **CI service containers:** a real Postgres per job. Fail, don't skip, when it's missing.
- **Correlation IDs:** one ID per request, in the response and in the log, to trace a
  request across systems.

## Design decisions
- **One image, many commands:** the API, migrations, ingestion, MCP and evals stay in step,
  with one thing to build, scan and version.
- **An API key, not OAuth:** this is the simplest real control for service-to-service calls.
  Per-user identity belongs at a gateway or with OIDC, and the dependency is the one place to
  swap it in.
- **Compose profiles:** the everyday dev loop (`uv run` against the database) is unchanged.
- **Ports on loopback only:** the dev database's password is "servicedesk", so it must not be
  reachable from the LAN.
- **The retrieval eval is manual, not on every PR:** it costs money and needs a secret.
  Forked PRs can't see secrets anyway.
- **Logs without bodies:** questions and tickets contain personal data. The request ID is
  enough to correlate.

## Interview talking points
- "One multi-stage Docker image, non-root, with a health check, built from the uv lock file.
  Compose runs migrations as a one-shot step before the API and MCP server start."
- "CI lints, runs 404 tests against a real pgvector database with a coverage floor, builds
  the image, and boots the stack in production mode to prove it gets healthy and enforces
  auth."
- "In prod the API refuses to start without an API key. I'd rather fail a deploy than serve an
  open API."
- "I measured before writing the performance doc. The database is about 10 ms; the time is
  model calls. So the next wins are streaming, embedding caching and fewer model calls per
  turn, not database tuning."
- "Verifying the containers found four real bugs: an empty-key config trap, an IPv6
  localhost hang, tests that silently depended on my API key, and a CI smoke test that
  would have crash-looped."

## Known limitations
- A shared API key: no per-user identity and no rate limiting (security.md lists the gateway
  and OIDC plan).
- No image scanning or dependency bot yet, and actions are pinned by version, not by SHA.
- The MCP HTTP transport has no auth, so it should stay on stdio or localhost.
- One process per container with synchronous endpoints, which is fine while the waits are
  on the LLM.
- The cloud layout is a reference design: it has not been deployed.

## Commands
```bash
docker compose --profile app up -d --build
docker compose run --rm api servicedesk ingest
docker compose logs -f api
docker compose --profile app down
uv run pytest --cov
```
