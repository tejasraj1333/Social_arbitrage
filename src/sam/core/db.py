"""Database engine & session management (SQLAlchemy 2.0).

Provides a declarative Base for ORM models and a FastAPI-friendly session
dependency. Engine is created lazily and cached so importing this module
never opens a connection (keeps tests and CLI fast).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sam.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


@lru_cache
def get_engine() -> Engine:
    """Return the cached SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(
        settings.db.url,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache
def _get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error.

    Usable both as a FastAPI dependency and a context-managed generator.
    """
    session = _get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
