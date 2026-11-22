# syntax=docker/dockerfile:1
# The application image: the HTTP API by default; the same image also runs migrations,
# ingestion, the MCP server and every other `servicedesk` command.
#
#   docker build -t servicedesk-copilot .
#   docker compose --profile app up -d --build      (the whole stack, see docker-compose.yml)
#
# Two stages: "build" has uv and installs the dependencies; "runtime" copies only the
# finished virtual environment, so the final image has no build tools or caches in it.

ARG PYTHON_VERSION=3.13

# ---------- build stage ----------
FROM python:${PYTHON_VERSION}-slim AS build

# uv as a single static binary, pinned to the version the lock file was made with.
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

# Compile .py files to bytecode now (faster container start); copy files out of uv's cache
# instead of hard-linking (the cache is a separate mount); use the image's Python only.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Dependencies only. This layer depends on nothing but the lock file, so Docker reuses
#    it from cache when only our code changes: rebuilds take seconds, not minutes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# 2) Our package. --no-editable installs it into the venv like any other library, so the
#    runtime stage doesn't need the source tree. --frozen: fail if uv.lock is out of date.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# tiktoken downloads its token tables the first time they are used. Fetch them now, so the
# container doesn't need internet access for that (or fail where egress is blocked).
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---------- runtime stage ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Never run as root: if the app were compromised, the attacker would get an unprivileged
# user with no write access to the code.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/.tiktoken /app/.tiktoken
# Files the commands read at run time, relative to /app: migrations, the sample knowledge
# base and seed data, and the evaluation dataset.
COPY alembic.ini ./
COPY migrations ./migrations
COPY data ./data
COPY evals/dataset.jsonl ./evals/dataset.jsonl

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    APP_ENV=prod

USER app
EXPOSE 8000

# Docker marks the container unhealthy when /health fails (503 = database unreachable).
# Python's standard library does the request, so the image needs no curl.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# 0.0.0.0: listen on the container's network interface, not just its own loopback.
CMD ["servicedesk", "serve", "--host", "0.0.0.0", "--port", "8000"]
