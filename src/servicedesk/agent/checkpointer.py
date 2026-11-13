"""Conversation memory in Postgres: LangGraph's PostgresSaver.

After every step of the graph, the saver writes a *checkpoint* (the whole AgentState) to
tables in our database, keyed by thread_id. A later message on the same thread loads the
latest checkpoint, so the conversation survives restarts, and any number of API workers can
serve the same conversation.

The saver creates its own tables (checkpoints, checkpoint_blobs, checkpoint_writes,
checkpoint_migrations) with setup(), which is idempotent and versioned by LangGraph itself.
They are not in our Alembic migrations, because LangGraph owns and upgrades that schema.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url


def checkpoint_conninfo(database_url: str) -> str:
    """Our SQLAlchemy URL (postgresql+psycopg://...) as a plain libpq one (postgresql://...)."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@contextmanager
def postgres_checkpointer(database_url: str, max_connections: int = 5) -> Iterator[PostgresSaver]:
    """A PostgresSaver on a small connection pool, with its tables created if needed.

    A pool rather than one connection: API requests run in parallel threads, and a psycopg
    connection must not be shared between threads at the same time. The connection settings
    are the ones PostgresSaver requires (autocommit, dict rows, no prepared statements).
    """
    pool = ConnectionPool(
        checkpoint_conninfo(database_url),
        min_size=1,
        max_size=max_connections,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    try:
        saver = PostgresSaver(pool)
        saver.setup()
        yield saver
    finally:
        pool.close()
