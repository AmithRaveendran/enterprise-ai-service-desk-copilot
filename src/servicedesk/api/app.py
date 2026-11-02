"""The FastAPI application. Run it with `servicedesk serve`, then open /docs."""

from importlib.metadata import version

from fastapi import FastAPI

from servicedesk.api.routes import health, search


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise AI Service Desk Copilot",
        version=version("servicedesk-copilot"),
        description="Knowledge-base search for IT support. Interactive docs at /docs.",
    )
    app.include_router(health.router)
    app.include_router(search.router)
    return app


app = create_app()
