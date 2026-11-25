"""The FastAPI application. Run it with `servicedesk serve`, then open /docs."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from servicedesk.agent.copilot import ChatError, ThreadNotFoundError
from servicedesk.api.dependencies import close_checkpointer
from servicedesk.api.middleware import request_context
from servicedesk.api.routes import ask, chat, desk, health, search
from servicedesk.api.security import check_production_settings, require_api_key
from servicedesk.config import get_settings
from servicedesk.desk.service import ConflictError, DeskError, NotFoundError

# Business-rule errors from the service layer → HTTP status codes. Registered once here,
# so routes don't need try/except: 404 = it doesn't exist, 409 = it exists but its current
# state doesn't allow this (e.g. reopening a closed ticket).
ERROR_STATUS = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
}


async def desk_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DeskError)
    code = ERROR_STATUS.get(type(exc), status.HTTP_400_BAD_REQUEST)
    return JSONResponse(status_code=code, content={"detail": str(exc), "code": exc.code})


async def chat_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Conversation-state errors: 404 for an unknown thread, otherwise 409 (e.g. a new
    message while a change is waiting for approval). `code` says which."""
    assert isinstance(exc, ChatError)
    if isinstance(exc, ThreadNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    else:
        code = status.HTTP_409_CONFLICT
    return JSONResponse(status_code=code, content={"detail": str(exc), "code": exc.code})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: refuse to serve with unsafe production settings (e.g. no API key). Raising
    # here stops uvicorn before it accepts a single request.
    check_production_settings(get_settings())
    yield  # the app serves requests here
    close_checkpointer()  # shutdown: close the conversation store's connection pool


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise AI Service Desk Copilot",
        version=version("servicedesk-copilot"),
        description=(
            "IT support answers with cited sources, service status and tickets. "
            "Interactive docs at /docs."
        ),
        lifespan=lifespan,
    )
    app.include_router(health.router)  # open: health checks send no credentials
    protected = [Depends(require_api_key)]  # 401 without a valid key, when one is set
    for router in (search.router, ask.router, desk.router, chat.router):
        app.include_router(router, dependencies=protected)
    app.add_exception_handler(DeskError, desk_error_handler)
    app.add_exception_handler(ChatError, chat_error_handler)
    app.middleware("http")(request_context)  # X-Request-ID + one log line per request
    return app


app = create_app()
