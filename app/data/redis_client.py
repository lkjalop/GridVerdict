"""Shared async Redis connection pool.

Returns None when Redis is not configured or unavailable, allowing all
callers to fall back gracefully to in-process alternatives.

Usage:
    redis = await get_redis()
    if redis is not None:
        await redis.set("key", "value")
    else:
        # in-process fallback
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None


async def get_redis() -> Any | None:
    """Return a connected async Redis client, or None if not configured/available."""
    global _client
    if _client is not None:
        return _client

    from config.settings import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return None

    try:
        import redis.asyncio as aioredis  # type: ignore[import]
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await client.ping()
        _client = client
        logger.info("Redis connected: %s", settings.redis_url)
        try:
            from app.api.metrics_registry import redis_connected
            redis_connected.set(1)
        except Exception:
            pass
        return _client
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — in-process fallback active", exc)
        try:
            from app.api.metrics_registry import redis_connected
            redis_connected.set(0)
        except Exception:
            pass
        return None


async def close_redis() -> None:
    """Close the connection pool on shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None


def reset_redis_client() -> None:
    """Reset the cached client — for testing only."""
    global _client
    _client = None
