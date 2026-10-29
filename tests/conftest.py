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
