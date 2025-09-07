"""
Shared pytest fixtures for FinExtract-Bench tests.

Key design decisions:
- All tests use an in-memory SQLite DB (no file created on disk).
- The DB is created fresh for each test function (function-scoped fixture).
- All imports are done inside fixtures where needed to avoid import errors
  when optional dependencies (docling, etc.) are not installed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from finextract.storage.models import Base


@pytest.fixture()
def db_engine():
    """
    Create an in-memory SQLite engine with all tables.

    Dropped at the end of each test to ensure isolation.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """
    Yield a SQLAlchemy Session bound to the in-memory engine.

    Rolls back after each test so tests don't affect each other.
    """
    Session = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = Session()
    yield session
    session.rollback()
    session.close()
