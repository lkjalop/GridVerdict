"""Async pub-sub event bus for live event streams.

In-process mode (default / no Redis configured): events fan out directly to
local asyncio.Queue objects held by SSE endpoint generators.

Redis mode (when redis_url is set in settings): publish() writes to a Redis
pub/sub channel. A per-worker background listener receives channel messages
and fans them out to local queues, enabling cross-worker delivery.

The subscribe/unsubscribe interface is unchanged — SSE consumers always pull
from a local asyncio.Queue regardless of backend.

Call start_redis_listener() from the app lifespan to enable cross-worker mode.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_CHANNEL = "gv:events"


@dataclass
class BusEvent:
    type: str
    payload: dict[str, Any]
    region: str | None = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Local subscriber registry: sub_id → (queue, region_filter, type_filter)
_subs: dict[int, tuple[asyncio.Queue, str | None, frozenset[str] | None]] = {}
_counter = 0
_listener_task: asyncio.Task | None = None


def subscribe(
    region: str | None = None,
    types: set[str] | frozenset[str] | None = None,
) -> asyncio.Queue[BusEvent]:
    """Register a subscriber and return a queue that receives matching BusEvents."""
    global _counter
    q: asyncio.Queue[BusEvent] = asyncio.Queue(maxsize=100)
    _counter += 1
    _subs[_counter] = (q, region, frozenset(types) if types else None)
    logger.debug("SSE subscriber %d registered (region=%s types=%s)", _counter, region, types)
    return q


def unsubscribe(queue: asyncio.Queue) -> None:
    """Remove a subscriber by its queue reference."""
    stale = [k for k, (q, *_) in _subs.items() if q is queue]
    for k in stale:
        del _subs[k]
    if stale:
        logger.debug("SSE subscriber(s) %s unregistered", stale)


def _dispatch_local(event: BusEvent) -> None:
    """Fan out a BusEvent to all matching local subscribers (non-blocking)."""
    for sub_id, (q, sub_region, sub_types) in list(_subs.items()):
        if sub_region and event.region and sub_region != event.region:
            continue
        if sub_types and event.type not in sub_types:
            continue
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "SSE subscriber %d queue full — dropping %s (region=%s)",
                sub_id, event.type, event.region,
            )


async def publish(
    event_type: str,
    payload: dict[str, Any],
    region: str | None = None,
) -> None:
    """Broadcast an event to all matching subscribers.

    Routes through Redis when available so events reach subscribers in other
    worker processes. Falls back to direct in-process dispatch on Redis failure.
    """
    from app.data.redis_client import get_redis
    redis = await get_redis()

    if redis is not None:
        try:
            await redis.publish(_REDIS_CHANNEL, json.dumps({
                "type": event_type,
                "payload": payload,
                "region": region,
                "ts": datetime.now(timezone.utc).isoformat(),
            }))
            return
        except Exception as exc:
            logger.warning("Redis publish failed, falling back to in-process: %s", exc)

    # In-process fallback (also used when Redis is not configured)
    _dispatch_local(BusEvent(type=event_type, payload=payload, region=region))


async def start_redis_listener() -> None:
    """Start the Redis channel listener (called from app lifespan).

    Subscribes to the Redis event channel and dispatches incoming messages to
    local asyncio queues. Silently no-ops when Redis is not configured.
    """
    global _listener_task
    from app.data.redis_client import get_redis
    redis = await get_redis()
    if redis is None:
        logger.debug("Redis not configured — event bus running in in-process mode")
        return

    async def _listen() -> None:
        try:
            pubsub = redis.pubsub()
            await pubsub.subscribe(_REDIS_CHANNEL)
            logger.info("Redis event bus listener started on channel %s", _REDIS_CHANNEL)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    _dispatch_local(BusEvent(
                        type=data["type"],
                        payload=data.get("payload", {}),
                        region=data.get("region"),
                        ts=data.get("ts", datetime.now(timezone.utc).isoformat()),
                    ))
                except Exception as exc:
                    logger.debug("Redis event parse error: %s", exc)
        except asyncio.CancelledError:
            logger.debug("Redis event bus listener cancelled")
        except Exception as exc:
            logger.warning("Redis event bus listener failed: %s", exc)

    _listener_task = asyncio.create_task(_listen())


async def stop_redis_listener() -> None:
    """Cancel the Redis listener task on app shutdown."""
    global _listener_task
    if _listener_task is not None and not _listener_task.done():
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
        _listener_task = None


def subscriber_count() -> int:
    """Return the number of active subscribers (for health checks and metrics)."""
    return len(_subs)
