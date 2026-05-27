"""RegionSnapshot — per-region market state memory for the commentary engine.

Snapshots are persisted in Redis so they survive scheduler restarts.
The ChangeDetector compares successive snapshots to find material changes.
If Redis is unavailable, returns None (ChangeDetector handles gracefully).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

_KEY = "gv:commentary:snapshot:{region}"


@dataclass
class RegionSnapshot:
    region: str
    price_rrp: float
    demand_mw: float
    headroom_mw: float
    regime: str
    valid_time: datetime
    spike_prob_300: float = 0.0
    spike_prob_1000: float = 0.0
    notice_ids: list[str] = field(default_factory=list)
    forecast_p90: float | None = None
    # Sprint R: extended snapshot fields
    staleness_seconds: int = 0
    binding_constraint_ids: list[str] = field(default_factory=list)
    weather_pressure: float = 0.0   # 0–1 composite pressure score


async def load_snapshot(region: str) -> RegionSnapshot | None:
    """Load the last saved snapshot for a region from Redis. Returns None on miss."""
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return None
        raw = await redis.get(_KEY.format(region=region))
        if raw is None:
            return None
        d = json.loads(raw)
        return RegionSnapshot(
            region=d["region"],
            price_rrp=float(d["price_rrp"]),
            demand_mw=float(d["demand_mw"]),
            headroom_mw=float(d["headroom_mw"]),
            regime=str(d["regime"]),
            valid_time=datetime.fromisoformat(d["valid_time"]),
            spike_prob_300=float(d.get("spike_prob_300") or 0.0),
            spike_prob_1000=float(d.get("spike_prob_1000") or 0.0),
            notice_ids=list(d.get("notice_ids") or []),
            forecast_p90=d.get("forecast_p90"),
            staleness_seconds=int(d.get("staleness_seconds") or 0),
            binding_constraint_ids=list(d.get("binding_constraint_ids") or []),
            weather_pressure=float(d.get("weather_pressure") or 0.0),
        )
    except Exception as exc:
        logger.debug("Snapshot load failed for %s: %s", region, exc)
        return None


async def save_snapshot(snapshot: RegionSnapshot) -> None:
    """Persist a snapshot to Redis. Non-fatal on failure."""
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return
        payload = json.dumps({
            "region": snapshot.region,
            "price_rrp": snapshot.price_rrp,
            "demand_mw": snapshot.demand_mw,
            "headroom_mw": snapshot.headroom_mw,
            "regime": snapshot.regime,
            "valid_time": snapshot.valid_time.isoformat(),
            "spike_prob_300": snapshot.spike_prob_300,
            "spike_prob_1000": snapshot.spike_prob_1000,
            "notice_ids": snapshot.notice_ids,
            "forecast_p90": snapshot.forecast_p90,
            "staleness_seconds": snapshot.staleness_seconds,
            "binding_constraint_ids": snapshot.binding_constraint_ids,
            "weather_pressure": snapshot.weather_pressure,
        })
        await redis.set(_KEY.format(region=snapshot.region), payload)
    except Exception as exc:
        logger.debug("Snapshot save failed for %s: %s", snapshot.region, exc)
