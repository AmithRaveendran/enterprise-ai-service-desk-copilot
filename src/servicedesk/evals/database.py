"""A scratch database for the agent tasks, reset to the sample data before every run.

The agent tasks talk about specific tickets ("INC-1002 is waiting for the requester"), so
they need known data. The dev database changes as you use it, so the agent suite gets its
own database, `servicedesk_eval`, on the same Postgres server: created and migrated if
needed, then emptied and seeded with data/seed (services + INC-1001..INC-1005).

Only the desk tables live here. Knowledge-base answers still come from the main database,
where the real embeddings are.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.desk.seed import seed

EVAL_DATABASE = "servicedesk_eval"


def fresh_desk_database(database_url: str, name: str = EVAL_DATABASE) -> sessionmaker[Session]:
    url = make_url(database_url)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:  # CREATE DATABASE can't run inside a transaction
            if not conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}):
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()

    eval_url = url.set(database=name).render_as_string(hide_password=False)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", eval_url.replace("%", "%%"))
    command.upgrade(config, "head")

    engine = create_engine(eval_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE ticket_comments, tickets, services, tool_executions "
                "RESTART IDENTITY CASCADE"  # ticket numbers start again at 1001
            )
        )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    seed(factory, Path("data/seed"))
    return factory
