"""Shared test fixtures.

Integration tests run against a separate `servicedesk_test` database on the Docker Postgres
server, so they never touch the data you ingest for real. If Postgres isn't running they
are skipped rather than failed.
"""

import hashlib
import math
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from servicedesk.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"
TEST_DATABASE = "servicedesk_test"


class FakeEmbedder:
    """Deterministic, offline stand-in for OpenAI.

    Hashes each word into one of 1536 buckets ("feature hashing"), so texts that share
    words get similar vectors. Good enough to exercise storage and, later, search.
    """

    def __init__(self, model: str = "fake-hashing-embedder", dimensions: int = 1536):
        self.model = model
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)


class FakeLLM:
    """Offline stand-in for the chat model: returns a canned DraftAnswer (or raises) and
    records every prompt it was given, so tests can check what the model would have seen."""

    model = "fake-llm"

    def __init__(self, answer: str = "Do the thing [1].", answerable: bool = True, error=None):
        self.answer = answer
        self.answerable = answerable
        self.error = error
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system, user, schema):
        from servicedesk.rag.llm import LLMResult

        self.prompts.append((system, user))
        if self.error:
            raise self.error
        parsed = schema(answerable=self.answerable, answer=self.answer)
        return LLMResult(parsed=parsed, input_tokens=100, output_tokens=20)


class FakeChatModel:
    """Offline stand-in for the agent's LangChain chat model.

    Plays back a script: each call returns the next reply. A reply is an AIMessage, a plain
    string (a final answer), or a callable(messages) -> AIMessage for replies that depend on
    what the model was sent. Records every call: the messages and the tool_choice used.
    The graph only needs bind_tools() and invoke(), so no LangChain base class is needed.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.tools: list[dict] = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        return _BoundFakeModel(self, kwargs.get("tool_choice"))

    def _reply(self, messages, tool_choice):
        from langchain_core.messages import AIMessage

        self.calls.append({"messages": list(messages), "tool_choice": tool_choice})
        if not self.script:
            raise AssertionError("FakeChatModel ran out of scripted replies")
        reply = self.script.pop(0)
        if callable(reply):
            reply = reply(messages)
        return AIMessage(reply) if isinstance(reply, str) else reply


class _BoundFakeModel:
    def __init__(self, model: FakeChatModel, tool_choice):
        self.model, self.tool_choice = model, tool_choice

    def invoke(self, messages):
        return self.model._reply(messages, self.tool_choice)


def tool_call(name: str, args: dict | None = None, call_id: str = "call_1"):
    """An AIMessage in which the model asks for one tool call."""
    from langchain_core.messages import AIMessage

    return AIMessage("", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Create the test database if needed and migrate it to the latest schema."""
    url = make_url(get_settings().database_url)
    admin = create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",  # CREATE DATABASE can't run inside a transaction
        connect_args={"connect_timeout": 3},
    )
    try:
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DATABASE}
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DATABASE}"'))
    except OperationalError:
        pytest.skip("Postgres is not running (start it with `docker compose up -d`)")
    finally:
        admin.dispose()

    test_url = url.set(database=TEST_DATABASE).render_as_string(hide_password=False)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_url.replace("%", "%%"))
    command.upgrade(config, "head")  # also proves the migration works on an empty database
    return test_url


@pytest.fixture
def session_factory(test_database_url):
    engine = create_engine(test_database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents RESTART IDENTITY CASCADE"))
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture(scope="module")
def kb_session_factory(test_database_url):
    """The test database with the whole sample KB ingested (fake embeddings), shared by all
    tests in a module. Search tests only read, so one ingest per module is enough."""
    from servicedesk.ingestion.chunker import Chunker
    from servicedesk.ingestion.pipeline import IngestionPipeline

    engine = create_engine(test_database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents RESTART IDENTITY CASCADE"))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    report = IngestionPipeline(factory, FakeEmbedder(), Chunker()).run(KB_DIR)
    assert not report.failed
    yield factory
    engine.dispose()


@pytest.fixture
def desk_session_factory(test_database_url):
    """The test database with fresh service desk data: the 10 catalogue services and the 5
    sample tickets (INC-1001..INC-1005). Emptied and re-seeded for every test, because these
    tests change tickets."""
    from servicedesk.desk.seed import seed

    engine = create_engine(test_database_url)
    with engine.begin() as conn:
        # RESTART IDENTITY: ticket numbers start again at 1001.
        conn.execute(text("TRUNCATE ticket_comments, tickets, services RESTART IDENTITY CASCADE"))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    seed(factory, ROOT / "data" / "seed")
    yield factory
    engine.dispose()


@pytest.fixture
def desk(desk_session_factory):
    from servicedesk.desk.service import ServiceDesk

    return ServiceDesk(desk_session_factory)
