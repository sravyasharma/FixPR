"""
storage/db.py

Async SQLAlchemy 2.x engine and session factory.

Uses asyncpg under the hood for non-blocking PostgreSQL access.
The worker runs fully async — no sync SQLAlchemy calls allowed.

Provides:
  - get_engine()         → singleton AsyncEngine
  - get_session_factory() → singleton async_sessionmaker
  - get_db_session()     → async context manager for unit-of-work
  - create_all_tables()  → used in tests and first-run setup

Architecture role:
  - Imported by storage/repositories.py ONLY
  - FastAPI endpoints use get_db_session() via dependency injection
  - Worker uses get_db_session() directly in job handler
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from shared.config import get_settings
from shared.logger import get_logger
from storage.models import Base

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """
    Return the singleton AsyncEngine.

    Connection pool settings are tuned for a production worker:
    - pool_size: baseline concurrent connections
    - max_overflow: burst headroom above pool_size
    - pool_timeout: fail fast rather than queue indefinitely
    - pool_pre_ping: detect stale connections before using them
    """
    settings = get_settings()
    database_url = str(settings.database_url)
    if database_url.startswith("sqlite://"):
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if database_url.startswith("sqlite+aiosqlite:///:memory:"):
        database_url = "sqlite+aiosqlite:///:memory:"

    engine_kwargs: dict[str, object] = {
        "echo": settings.db_echo_sql,
        "future": True,
    }
    if database_url.startswith("sqlite"):
        engine_kwargs["pool_pre_ping"] = True
    else:
        engine_kwargs.update(
            {
                "pool_size": settings.db_pool_size,
                "max_overflow": settings.db_max_overflow,
                "pool_timeout": settings.db_pool_timeout,
                "pool_pre_ping": True,
            }
        )

    engine = create_async_engine(database_url, **engine_kwargs)
    logger.info(
        "Database engine created",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return the singleton session factory.

    expire_on_commit=False is critical for async: without it, accessing
    model attributes after commit raises MissingGreenlet errors.
    """
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


@asynccontextmanager
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    Async context manager that provides a transactional database session.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(select(ReviewModel))

    Commits on clean exit, rolls back on exception.
    Always closes session (returns connection to pool).
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_all_tables() -> None:
    """
    Create all tables defined in ORM models.

    Used by:
    - docker-compose healthcheck / startup script
    - Test fixtures (with a test database URL)

    In production, prefer Alembic migrations instead.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created (or already exist)")


async def drop_all_tables() -> None:
    """
    Drop all tables. TEST USE ONLY.

    Never call this in production code.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped")


async def close_engine() -> None:
    """
    Gracefully dispose the engine connection pool.

    Call during application shutdown (lifespan event).
    """
    engine = get_engine()
    await engine.dispose()
    logger.info("Database engine disposed")