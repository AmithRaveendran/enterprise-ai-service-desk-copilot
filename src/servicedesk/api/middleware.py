"""Request IDs and one log line per request.

Every response carries an `X-Request-ID` header. A caller (or a gateway in front of us) can
send its own ID and we reuse it, so one request can be followed across systems; otherwise
we make one. The same ID is in the log line, so a user reporting "request abc123 failed"
leads straight to it in the logs.

The log line uses key=value pairs (`request_id=... status=200 duration_ms=41`): easy to
read, and easy for log tools to search and graph.
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response

logger = logging.getLogger("servicedesk.api")

# Only reuse IDs that look like IDs: a header is user input, and putting arbitrary text in
# logs would let someone forge log lines (e.g. by sending a newline).
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Health checks arrive every few seconds; logging them at INFO would drown everything else.
_QUIET_PATHS = {"/health"}


async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _VALID_ID.match(incoming) else uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _log(logging.ERROR, request, request_id, 500, start)
        raise
    response.headers["X-Request-ID"] = request_id
    level = logging.DEBUG if request.url.path in _QUIET_PATHS else logging.INFO
    _log(level, request, request_id, response.status_code, start)
    return response


def _log(level: int, request: Request, request_id: str, status: int, start: float) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    logger.log(
        level,
        "request_id=%s method=%s path=%s status=%d duration_ms=%.0f",
        request_id,
        request.method,
        request.url.path,
        status,
        duration_ms,
    )
