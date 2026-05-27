"""Two-layer cache for AEMO market data.

Layer 1: in-memory (asyncio.Lock protected), TTL per data type.
Layer 2: filesystem JSON under NEMWEB_CACHE_DIR — survives process restarts.

Cache keys:
  dispatch_snapshot        — LiveMarketSnapshot, TTL = live_dispatch_max_age_s
  notices                  — list[MarketNotice], TTL = notices_max_age_s
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()


class _CacheEntry:
    __slots__ = ("value", "written_at", "expires_at")

    def __init__(self, value: Any, ttl_s: int) -> None:
        self.value = value
        self.written_at = time.monotonic()
        self.expires_at = self.written_at + ttl_s

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    def age_seconds(self) -> float:
        return time.monotonic() - self.written_at


class MarketCache:
    """Thread-safe async cache for market snapshots and notices."""

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cache_dir = Path(_settings.nemweb_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _ttl(self, key: str) -> int:
        if key.startswith("dispatch"):
            return _settings.live_dispatch_max_age_s
        if key.startswith("notices"):
            return _settings.notices_max_age_s
        if key.startswith("nem_news"):
            return _settings.nem_news_max_age_s
        if key.startswith("predispatch"):
            return _settings.predispatch_max_age_s
        if key.startswith("calibration"):
            return 6 * 3600
        return 300

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry and not entry.is_expired():
                return entry.value
        # Try filesystem fallback
        return self._read_fs(key)

    async def set(self, key: str, value: Any) -> None:
        ttl = self._ttl(key)
        async with self._lock:
            self._store[key] = _CacheEntry(value, ttl)
        self._write_fs(key, value)

    async def age_seconds(self, key: str) -> float | None:
        """Return how many seconds ago this key was last written, or None if absent/expired."""
        async with self._lock:
            entry = self._store.get(key)
            if entry and not entry.is_expired():
                return entry.age_seconds()
        # Try filesystem
        try:
            path = self._fs_path(key)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                age = time.time() - raw["ts"]
                return age if age < raw["ttl"] else None
        except Exception:
            pass
        return None

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
        fs_path = self._fs_path(key)
        if fs_path.exists():
            fs_path.unlink(missing_ok=True)

    def _fs_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe}.json"

    def _write_fs(self, key: str, value: Any) -> None:
        try:
            path = self._fs_path(key)
            serializable = value if isinstance(value, (dict, list, str, int, float, bool, type(None))) else _try_to_dict(value)
            path.write_text(
                json.dumps({"ts": time.time(), "ttl": self._ttl(key), "data": serializable}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Cache FS write failed for %s: %s", key, exc)

    def _read_fs(self, key: str) -> Any | None:
        try:
            path = self._fs_path(key)
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            if time.time() > raw["ts"] + raw["ttl"]:
                return None
            return raw["data"]
        except Exception as exc:
            logger.debug("Cache FS read failed for %s: %s", key, exc)
            return None


def _try_to_dict(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# Module-level singleton
_cache: MarketCache | None = None


def get_cache() -> MarketCache:
    global _cache
    if _cache is None:
        _cache = MarketCache()
    return _cache
