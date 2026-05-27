"""Rate limiting middleware — per-tenant sliding-window token bucket.

Protects expensive endpoints from runaway requests, accidental cost burn,
and brute-force auth attempts. All limits are configurable and can be
disabled per-tenant in future.

Limits (requests per window):
  /api/sessions/*/query  — 30 req / 60 s  (LLM + model training path)
  /api/auth/token        — 10 req / 60 s  (brute-force protection)
  /api/market/forecast   — 20 req / 60 s  (model fitting path)

Tenant identity is read from the Bearer JWT sub claim. If auth is bypassed
(dev_no_auth mode) the client IP is used as the identity key so development
tools don't accidentally trigger limits during automated test runs.

Returns HTTP 429 with Retry-After header when limit exceeded.

Backend selection (automatic, no config required):
  - Redis available: sorted-set sliding window with atomic Lua script;
    enforces limits across all worker processes.
  - Redis unavailable / not configured: pure asyncio in-memory sliding
    window (original behaviour, single-process only).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# (route_prefix, window_seconds, max_requests)
_LIMITS: list[tuple[str, int, int]] = [
    ("/api/sessions/", 60, 30),      # query path
    ("/api/auth/token", 60, 10),     # login brute-force
    ("/api/market/forecast", 60, 20),  # model fitting
]

# In-process fallback: request_timestamps[identity_key] = sorted list of monotonic timestamps
_windows: dict[str, list[float]] = defaultdict(list)
_lock = asyncio.Lock()

# Redis Lua script: atomic sliding-window check-and-record.
# Returns [1, 0] (allowed) or [0, retry_after_seconds] (rejected).
_RL_SCRIPT = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_r  = tonumber(ARGV[3])
local cutoff = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
local count = redis.call('ZCARD', key)

if count >= max_r then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry  = math.ceil(window - (now - tonumber(oldest[2]))) + 1
    return {0, retry}
end

-- Unique member: timestamp + random suffix avoids ZADD collision on same-ms burst
redis.call('ZADD', key, now, tostring(now) .. ':' .. tostring(math.random(1000000)))
redis.call('EXPIRE', key, window + 1)
return {1, 0}
"""
_redis_script: object | None = None


def _find_limit(path: str) -> tuple[int, int] | None:
    """Return (window_s, max_requests) for the matching prefix, or None."""
    for prefix, window_s, max_req in _LIMITS:
        if path.startswith(prefix):
            return window_s, max_req
    return None


def _extract_identity(request: Request) -> str:
    """Extract a per-tenant identity key from the JWT sub or client IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:]
        try:
            from app.api.auth import decode_token
            payload = decode_token(token)
            return f"tenant:{payload.tenant_id}"
        except Exception:
            pass
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def _make_429(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


async def _check_redis(
    redis: object,
    key: str,
    now: float,
    window_s: int,
    max_req: int,
) -> tuple[bool | None, int]:
    """Attempt Redis rate limit check. Returns (allowed, retry_after) or (None, 0) on error."""
    global _redis_script
    try:
        if _redis_script is None:
            _redis_script = redis.register_script(_RL_SCRIPT)  # type: ignore[attr-defined]
        result = await _redis_script(keys=[key], args=[now, window_s, max_req])
        return bool(result[0]), int(result[1])
    except Exception as exc:
        logger.warning("Redis rate-limit script failed (%s) — falling back to in-process", exc)
        return None, 0  # None signals: caller should fall back


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter. Uses Redis when available, in-process otherwise."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        limit = _find_limit(request.url.path)
        if limit is None:
            return await call_next(request)

        # Skip rate limiting in dev_no_auth mode for automated tests
        if _settings.gridverdict_dev_no_auth:
            return await call_next(request)

        window_s, max_req = limit
        identity = _extract_identity(request)
        route_seg = request.url.path.split("/")[3] if request.url.path.count("/") >= 3 else "root"
        key = f"{identity}:{route_seg}"
        now = time.monotonic()

        # ── Redis backend ──────────────────────────────────────────────────
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is not None:
            redis_key = f"gv:rl:{key}"
            allowed, retry_after = await _check_redis(redis, redis_key, now, window_s, max_req)
            if allowed is not None:
                if not allowed:
                    return _make_429(retry_after)
                return await call_next(request)
            # allowed is None → Redis errored, fall through to in-process

        # ── In-process fallback ────────────────────────────────────────────
        async with _lock:
            timestamps = _windows[key]
            cutoff = now - window_s
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)

            if len(timestamps) >= max_req:
                oldest = timestamps[0]
                retry_after = int(window_s - (now - oldest)) + 1
                return _make_429(retry_after)

            timestamps.append(now)

        return await call_next(request)


def reset_rate_limits() -> None:
    """Clear all in-process rate limit windows — for testing only."""
    _windows.clear()
