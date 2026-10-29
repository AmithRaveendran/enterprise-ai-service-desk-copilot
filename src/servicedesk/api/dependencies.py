"""Objects the endpoints need, provided through FastAPI's dependency injection.

An endpoint declares `retriever: Annotated[Retriever, Depends(get_retriever)]` and FastAPI
calls get_retriever() for it. Tests swap these functions out with
`app.dependency_overrides[get_retriever] = ...` to use a test database and a fake embedder.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.db.session import get_sessionmaker
from servicedesk.ingestion.embeddings import ConfigurationError, Embedder, get_embedder
from servicedesk.search.retriever import Retriever


def get_session_factory() -> sessionmaker[Session]:
    return get_sessionmaker()


@lru_cache
def _embedder() -> Embedder | None:
    """Built once. None without an API key: the API still starts, and keyword search works."""
    try:
        return get_embedder()
    except ConfigurationError:
        return None


def get_retriever(
    session_factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> Retriever:
    return Retriever(session_factory, _embedder())
