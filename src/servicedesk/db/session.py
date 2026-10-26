"""Engine and session factory.

The engine owns a pool of database connections; a session is a unit of work that borrows
one. Create the engine once per process and a short-lived session per task.
"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(
        url or get_settings().database_url,
        pool_pre_ping=True,  # replace connections that died, e.g. after a DB restart
        connect_args={"connect_timeout": 5},
    )


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)
