# Security

What this service protects, the threats that matter for an AI assistant that can change
data, what is in place against each, and what a production deployment still needs. It is
written to be honest about gaps: this is a portfolio project, not a hardened product.

## What there is to protect

| Asset | Why it matters |
|---|---|
| Tickets and comments | employee names, emails and problem descriptions (personal data) |
| The ability to write | creating, changing and commenting on tickets on someone's behalf |
| Knowledge base and answers | wrong or injected guidance could make employees do something unsafe |
| OpenAI / Azure keys | billable, and a way to send data to the provider |
| Database credentials | full access to all of the above |

## Threats and controls

### 1. Unauthenticated access to the API
- **Control:** every route except `/health` requires `X-API-Key` when `API_KEY` is set
  (`api/security.py`), compared with `hmac.compare_digest` (constant time).
- **Secure by default:** the Docker image sets `APP_ENV=prod`, and in prod the API **refuses
  to start** without a key of 16+ characters. Forgetting the variable fails loudly instead of
  serving an open API.
- `/health` stays open for load balancers; it returns only status and two counts.
- **Gap:** one shared key means no per-user identity. Production should put the API behind an
  identity-aware gateway or validate OAuth2/OIDC tokens (e.g. Entra ID) in the same
  dependency, and take the requester's email from the token rather than from the request.

### 2. Prompt injection (direct and indirect)
Someone types "ignore your instructions…", or a KB document / ticket comment contains
instructions that the model reads as tool output.
- **Controls:**
  - The system prompt says tool results are data, not instructions.
  - Prompts mark sources with XML-style tags and numbers, so the model can tell which text
    is a source.
  - **The model can't act alone.** Every write pauses for a human (`interrupt()` in the
    graph, or the MCP confirmation token), and the person sees the exact arguments first.
    Injection can at worst propose a change, and a person sees it.
  - Answers must cite valid sources, and an answer without a valid citation is never shown.
- **Gap:** an approver can still approve a malicious change without reading it. Mitigations
  are UI and process, such as highlighting fields the user didn't type.

### 3. Tool misuse and excessive agency
- **Least privilege:** the agent has 6 desk tools and the KB, with no SQL, shell or HTTP tools.
- **Strict input schemas:**
  - Pydantic models with `extra="forbid"`.
  - Enums for status, priority and category.
  - Length limits.
  - Company-domain emails only.
  - The `INC-1234` ticket number format.
- **Business rules in the service layer:** a closed ticket can't be reopened, and resolving
  needs a note. These hold whoever the caller is.
- **Bounded loops:** `AGENT_MAX_TOOL_ROUNDS` limits tool rounds per message.
- **Idempotency keys:** a retried or resumed write happens once.

### 4. MCP-specific
- **Writes take two calls.** The first returns a preview and an HMAC-SHA256 token bound to
  the tool name and the exact arguments. A changed argument, a forged token or a token for
  another tool is rejected. The secret is per process, so tokens die with the server.
- **DNS rebinding:** the HTTP transport accepts only `localhost`/`127.0.0.1` Host headers,
  enforced even when it listens on `0.0.0.0` in Docker. A foreign Host header gets 421.
  `--allowed-host` adds a real hostname.
- **Gap:** the MCP HTTP transport has no authentication. Keep it on stdio or localhost (the
  compose file publishes it on 127.0.0.1 only), or add OAuth, which the MCP spec defines,
  before exposing it.

### 5. Injection into the database
- All queries go through SQLAlchemy with bound parameters. The only SQL built as text uses
  constants, plus one database name in the eval/test helpers, which comes from code, not
  from users.
- Full-text queries use `websearch_to_tsquery`, which parses user text safely.

### 6. Secrets
- Settings come from environment variables (`pydantic-settings`), and keys are `SecretStr`,
  so they print as `**********` in logs and errors.
- `.env` is git-ignored and excluded from the Docker build context (`.dockerignore`), so a
  key is never baked into an image layer.
- CI needs no secrets. The manual eval workflow reads `OPENAI_API_KEY` from GitHub
  encrypted secrets.
- **Production:** use the platform's secret store (Azure Key Vault, AWS Secrets Manager)
  with managed identity instead of env files, and rotate keys.

### 7. Container and supply chain
- **Container:**
  - Multi-stage build, so no compilers or uv in the runtime image.
  - Runs as an unprivileged user (uid 10001).
  - All compose ports are published on 127.0.0.1 only.
- **Locked dependencies:** `uv.lock` with hashes, installed with `--frozen`.
- **Pinned versions:** the uv build image and the GitHub Actions are pinned to versions.
- **Least privilege in CI:** `permissions: contents: read`.
- **Gaps:**
  - Image scanning (Trivy or Grype in CI).
  - Dependabot or Renovate for updates.
  - Pinning actions by commit SHA.
  - A read-only root filesystem.

### 8. Request logging and privacy
- Each request gets an `X-Request-ID`. The log line has method, path, status and duration,
  and **no bodies, queries or headers**, so questions and personal data stay out of the logs.
- An incoming request ID is reused only if it looks like an ID (`[A-Za-z0-9._-]{1,64}`), which
  prevents log forging.
- **Data sent to the LLM provider:** questions, retrieved KB text and ticket fields the agent
  reads. Use Azure OpenAI (data stays in your tenant, not used for training) where that matters.

### 9. Abuse and cost
- **Gap:** no rate limiting. Each `/ask` or `/chat` costs model tokens, so put limits at the
  gateway (per key or user) and set a spending cap with the provider.
  `LLM_TIMEOUT_SECONDS` stops slow calls from tying up workers.
- **Inputs are size-limited:** questions ≤ 1,000 characters, chat messages ≤ 2,000, search
  queries ≤ 500, ticket text ≤ 5,000.

## Production checklist
- [ ] TLS in front of the API (gateway, ingress or App Service). The app itself speaks HTTP.
- [ ] Real identity (OIDC) instead of, or alongside, the shared API key.
- [ ] Secrets from a vault, keys rotated, Postgres on a private network with a strong password.
- [ ] Rate limits and a spending cap.
- [ ] Image scanning and dependency updates in CI.
- [ ] Database backups, encryption at rest (managed Postgres does both), and a retention
      policy for conversation checkpoints.
- [ ] Disable or protect `/docs` if the API is public.
