"""
queue/redis_client.py

Async Redis client singleton using redis-py's async support.

Architecture role:
  - Imported only by queue/task_queue.py
  - Provides a connection pool shared across the process lifetime
  - Webhook API and Worker each maintain their own pool (separate processes)
"""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis
from redis.asyncio import Redis

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """
    Return a singleton async Redis client.

    Uses a connection pool sized to match worker concurrency.
    decode_responses=False because we store JSON bytes.
    """
    settings = get_settings()

    client = aioredis.from_url(
        str(settings.redis_url),
        encoding="utf-8",
        decode_responses=False,
        max_connections=settings.worker_concurrency * 2 + 5,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )
    logger.info("Redis client initialized", url=str(settings.redis_url))
    return client


async def close_redis() -> None:
    """Graceful shutdown — call during application lifespan teardown."""
    client = get_redis_client()
    await client.aclose()
    logger.info("Redis client closed")


async def ping_redis() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        client = get_redis_client()
        return await client.ping()
    except Exception as exc:
        logger.error("Redis ping failed", error=str(exc))
        return False