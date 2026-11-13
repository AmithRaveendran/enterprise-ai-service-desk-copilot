"""Objects the endpoints need, provided through FastAPI's dependency injection.

An endpoint declares `retriever: Annotated[Retriever, Depends(get_retriever)]` and FastAPI
calls get_retriever() for it. Tests swap these functions out with
`app.dependency_overrides[get_retriever] = ...` to use a test database and a fake embedder.
"""

import threading
from contextlib import ExitStack
from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import Depends, HTTPException, status
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.agent.checkpointer import postgres_checkpointer
from servicedesk.agent.copilot import Copilot, build_copilot
from servicedesk.agent.llm import get_chat_model
from servicedesk.config import get_settings
from servicedesk.db.session import get_sessionmaker
from servicedesk.desk.service import ServiceDesk
from servicedesk.ingestion.embeddings import ConfigurationError, Embedder, get_embedder
from servicedesk.rag.answerer import Answerer
from servicedesk.rag.llm import StructuredLLM, get_llm
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


@lru_cache
def _llm() -> StructuredLLM | None:
    """Built once. None without an API key: /ask then answers 503, everything else works."""
    try:
        return get_llm()
    except ConfigurationError:
        return None


def get_answerer(retriever: Annotated[Retriever, Depends(get_retriever)]) -> Answerer:
    llm = _llm()
    if llm is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Answers need a chat model API key (see .env)."
        )
    settings = get_settings()
    return Answerer(
        retriever, llm, top_k=settings.rag_top_k, min_similarity=settings.rag_min_similarity
    )


def get_desk(
    session_factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> ServiceDesk:
    return ServiceDesk(session_factory)


@lru_cache
def _chat_model() -> BaseChatModel | None:
    """The agent's chat model, built once. None without an API key (/chat then answers 503)."""
    try:
        return get_chat_model()
    except ConfigurationError:
        return None


# The checkpointer owns a connection pool: opened on the first /chat request (so the API
# still starts when only search is needed), shared by all requests, closed at shutdown.
_checkpointer_lock = threading.Lock()
_checkpointer_stack = ExitStack()
_checkpointer: BaseCheckpointSaver | None = None


def _get_checkpointer() -> BaseCheckpointSaver:
    global _checkpointer
    with _checkpointer_lock:  # two first requests at once must not open two pools
        if _checkpointer is None:
            _checkpointer = _checkpointer_stack.enter_context(
                postgres_checkpointer(get_settings().database_url)
            )
        return _checkpointer


def close_checkpointer() -> None:
    """Called when the app shuts down."""
    global _checkpointer
    with _checkpointer_lock:
        _checkpointer_stack.close()
        _checkpointer = None


def get_copilot(
    desk: Annotated[ServiceDesk, Depends(get_desk)],
    answerer: Annotated[Answerer, Depends(get_answerer)],  # 503 if there is no API key
) -> Copilot:
    model = _chat_model()
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Chat needs a chat model API key (see .env)."
        )
    try:
        checkpointer = _get_checkpointer()
    except psycopg.OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The conversation store is unreachable."
        ) from exc
    # Building the graph per request is cheap (no I/O); the model, the checkpointer and
    # its pool are shared.
    return build_copilot(model, desk, answerer, checkpointer, get_settings())
