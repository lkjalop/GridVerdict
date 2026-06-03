"""CommentaryEngine — automated Why Engine runs triggered by material market changes.

Singleton pattern matches the existing _spike_detector in the scheduler.
Each process_tick() call:
  1. Loads the previous RegionSnapshot from Redis
  2. Runs ChangeDetector to find material changes
  3. For each non-cooldown change, runs scatter_gather + WhyBuilder
  4. Stores the resulting CommentaryEvent to DB
  5. Returns events for the scheduler to publish on the event bus

All operations are non-fatal — a failed commentary event never blocks
the scheduler's dispatch_refresh job.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_COOLDOWN_KEY = "gv:commentary:cooldown:{region}:{change_type}"


@dataclass
class CommentaryEvent:
    """Output type — one per material market change with full Why Engine analysis."""
    id: str
    region: str
    valid_time: datetime
    system_time: datetime
    event_type: str
    severity: str
    headline: str
    contributing_factors: list[dict[str, Any]]
    missing_data: list[str]
    evidence_refs: list[dict[str, Any]]
    confidence: float
    corroborations: dict[str, bool]
    next_watch: list[str]
    counterargument: str | None
    snapshot_before: dict[str, Any] | None
    snapshot_after: dict[str, Any]
    trace_id: str | None = None
    claim_map: list[dict[str, Any]] = None  # Sprint R: typed claim evidence map

    def __post_init__(self):
        if self.claim_map is None:
            self.claim_map = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "region": self.region,
            "valid_time": self.valid_time.isoformat(),
            "system_time": self.system_time.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "headline": self.headline,
            "contributing_factors": self.contributing_factors,
            "claim_map": self.claim_map,
            "missing_data": self.missing_data,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
            "corroborations": self.corroborations,
            "next_watch": self.next_watch,
            "counterargument": self.counterargument,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "trace_id": self.trace_id,
        }


class CommentaryEngine:
    """Stateful singleton that detects material changes and runs the Why Engine.

    All state (previous snapshots, cooldowns) is stored in Redis so the
    engine survives scheduler restarts. Holds no in-memory state itself.
    """

    async def process_tick(
        self,
        region: str,
        price_rrp: float,
        demand_mw: float,
        headroom_mw: float,
        regime: str,
        valid_time: datetime,
        cache: Any,
    ) -> list[CommentaryEvent]:
        """Evaluate one dispatch tick for commentary-worthy changes.

        Returns a (possibly empty) list of CommentaryEvents. The scheduler
        publishes each event on the bus and they are stored in the DB.
        """
        from app.engines.commentary.snapshot import RegionSnapshot, load_snapshot, save_snapshot
        from app.engines.commentary.detector import detect
        from app.engines.commentary.store import write_event

        prev = await load_snapshot(region)
        notices = await cache.get(f"notices_{region}") or []

        curr = RegionSnapshot(
            region=region,
            price_rrp=price_rrp,
            demand_mw=demand_mw,
            headroom_mw=headroom_mw,
            regime=regime,
            valid_time=valid_time,
        )

        # Enrich with latest forecast data
        forecast_cache = await cache.get(f"forecast_{region}")
        if isinstance(forecast_cache, dict):
            curr.spike_prob_300 = float(forecast_cache.get("spike_prob_300") or 0.0)
            curr.spike_prob_1000 = float(forecast_cache.get("spike_prob_1000") or 0.0)
            curr.forecast_p90 = forecast_cache.get("p90")
        curr.notice_ids = [n.get("notice_id", "") for n in notices if n.get("notice_id")]

        # Sprint R: populate extended snapshot fields
        dispatch_snap = await cache.get("dispatch_snapshot")
        if isinstance(dispatch_snap, dict):
            curr.staleness_seconds = int(dispatch_snap.get("staleness_seconds") or 0)
        constraints_cache = await cache.get(f"constraints_{region}")
        if isinstance(constraints_cache, list):
            curr.binding_constraint_ids = [
                c.get("constraint_id", "") for c in constraints_cache
                if c.get("constraint_id")
            ]
        weather_cache = await cache.get(f"weather_{region}")
        if isinstance(weather_cache, dict):
            curr.weather_pressure = _compute_weather_pressure(weather_cache)

        changes = detect(prev, curr, notices)

        # Sprint Z: generator trip detection — compare unit dispatch prev/curr
        try:
            from app.engines.commentary.detector import detect_generator_trips
            _prev_dispatch: dict[str, float] = await cache.get(f"unit_dispatch_prev_{region}") or {}
            _curr_dispatch: dict[str, float] = await cache.get(f"unit_dispatch_curr_{region}") or {}
            if _prev_dispatch and _curr_dispatch:
                trip_changes = detect_generator_trips(
                    _prev_dispatch, _curr_dispatch, region, valid_time
                )
                changes.extend(trip_changes)
        except Exception as _trip_exc:
            logger.debug("Generator trip detection failed (non-fatal): %s", _trip_exc)

        events: list[CommentaryEvent] = []

        baseline_enabled = isinstance(await cache.get("dispatch_snapshot"), dict)
        if prev is None and baseline_enabled and not await _baseline_on_cooldown(region):
            try:
                from app.engines.commentary.store import has_recent_baseline
                if await has_recent_baseline(region, curr.valid_time):
                    await _set_baseline_cooldown(region)
                    await save_snapshot(curr)
                    return events
                evt = _build_baseline_event(curr, notices)
                await write_event(evt)
                await _set_baseline_cooldown(region)
                events.append(evt)
            except Exception as exc:
                logger.debug("Baseline commentary event failed for %s: %s", region, exc)

        for change in changes:
            if await _is_on_cooldown(change):
                logger.debug(
                    "Commentary cooldown active: %s %s",
                    region, change.change_type.value,
                )
                continue

            try:
                evt = await _build_event(change, curr, prev, cache)
                await write_event(evt)
                await _set_cooldown(change)
                events.append(evt)
                logger.info(
                    "Commentary event: %s %s %s conf=%.2f",
                    evt.region, evt.event_type, evt.severity, evt.confidence,
                )
            except Exception as exc:
                logger.debug(
                    "Commentary event build failed for %s %s: %s",
                    region, change.change_type.value, exc,
                )

        await save_snapshot(curr)
        return events


async def _build_event(
    change: "MaterialChange",
    curr: "RegionSnapshot",
    prev: "RegionSnapshot | None",
    cache: Any,
) -> CommentaryEvent:
    """Run the full Why Engine pipeline for one material change."""
    from app.engines.commentary.detector import _CHANGE_DECOMPOSITION, ChangeType
    from app.engines.commentary.prose import format_factors, format_headline
    from app.core.schema import IntentLabel, QueryDecomposition
    from app.agents.scatter_gather import scatter_gather
    from app.agents.why_sources import assemble_why_sources
    from app.agents.why_builder import build_why
    from app.data.aemo_live_client import get_aemo_client

    decomp_overrides = _CHANGE_DECOMPOSITION.get(
        change.change_type,
        {"requires_why": True, "causal_targets": ["demand"]},
    )

    decomp = QueryDecomposition(
        raw_query=change.description,
        intent=IntentLabel.EXPLANATION,
        entities={"regions": [curr.region]},
        **decomp_overrides,
    )

    client = get_aemo_client()
    gather = await scatter_gather(curr.region, client, cache, include_weather=True)
    sources = assemble_why_sources(decomp, gather, curr.region)
    why = build_why(sources)

    headline = format_headline(change, why)
    factors = format_factors(why)

    corroborations = {
        "weather": bool(sources.weather.relevant and sources.weather.available),
        "news": bool(sources.news.commentary_items or sources.news.notices),
        "notices": bool(sources.news.notices),
    }

    return CommentaryEvent(
        id=str(uuid.uuid4()),
        region=curr.region,
        valid_time=curr.valid_time,
        system_time=datetime.now(timezone.utc),
        event_type=change.change_type.value,
        severity=change.severity,
        headline=headline,
        contributing_factors=factors,
        claim_map=[item.model_dump() for item in why.claim_map],
        missing_data=why.missing_data,
        evidence_refs=[ref.model_dump() for ref in why.evidence_refs],
        confidence=why.confidence,
        corroborations=corroborations,
        next_watch=why.next_watch,
        counterargument=why.counterargument or None,
        snapshot_before=_snap_to_dict(prev) if prev else None,
        snapshot_after=_snap_to_dict(curr),
    )


def _build_baseline_event(curr: "RegionSnapshot", notices: list[dict[str, Any]]) -> CommentaryEvent:
    """Create a first-snapshot event so Live Feed is visibly alive on startup."""
    evidence_refs = [
        {
            "source": "AEMO_DISPATCH_PRICE",
            "region": curr.region,
            "interval": curr.valid_time.isoformat(),
            "field": "price_rrp",
            "value": curr.price_rrp,
            "raw_ref": f"dispatch_{curr.valid_time.isoformat()}",
        },
        {
            "source": "AEMO_DISPATCH_PRICE",
            "region": curr.region,
            "interval": curr.valid_time.isoformat(),
            "field": "headroom_mw",
            "value": curr.headroom_mw,
            "raw_ref": f"dispatch_{curr.valid_time.isoformat()}",
        },
    ]
    return CommentaryEvent(
        id=str(uuid.uuid4()),
        region=curr.region,
        valid_time=curr.valid_time,
        system_time=datetime.now(timezone.utc),
        event_type="market_baseline",
        severity="LOW",
        headline=(
            f"{curr.region} baseline captured: ${curr.price_rrp:.0f}/MWh, "
            f"{curr.headroom_mw:.0f} MW headroom"
        ),
        contributing_factors=[
            {"label": "dispatch_price", "tier": "confirmed", "present": True},
            {"label": "headroom", "tier": "confirmed", "present": True},
        ],
        claim_map=[
            {
                "claim_type": "price_assertion",
                "label": "Baseline dispatch price captured",
                "tier": "confirmed",
                "present": True,
                "confidence": 0.9,
                "evidence_ref_ids": [],
                "note": "First snapshot after startup; not a material market-change alert.",
            }
        ],
        missing_data=[],
        evidence_refs=evidence_refs,
        confidence=0.6,
        corroborations={"weather": False, "news": False, "notices": bool(notices)},
        next_watch=[
            "Watch the next dispatch interval for price/headroom movement.",
            "Ask about this event to inspect drivers, analogs, and forecast support.",
        ],
        counterargument="This is a baseline snapshot, not evidence of a new market incident.",
        snapshot_before=None,
        snapshot_after=_snap_to_dict(curr),
    )


def _compute_weather_pressure(weather_cache: dict) -> float:
    """Derive a 0–1 pressure score from cached weather consensus.

    High temperature and wind drought are the two primary NEM weather stress signals.
    Score 0.9 = extreme (≥38°C), 0.7 = elevated (35–38°C or wind < 3 m/s), 0 otherwise.
    """
    consensus = weather_cache.get("consensus") or {}
    temp = consensus.get("temperature_c") or weather_cache.get("temperature_c")
    wind_kmh = consensus.get("wind_speed_kmh") or weather_cache.get("wind_speed_kmh")
    wind_ms = (float(wind_kmh) / 3.6) if wind_kmh is not None else None
    score = 0.0
    if temp is not None:
        t = float(temp)
        if t >= 38:
            score = max(score, 0.9)
        elif t >= 35:
            score = max(score, 0.7)
    if wind_ms is not None and wind_ms < 3.0:
        score = max(score, 0.7)
    return score


def _snap_to_dict(s: "RegionSnapshot") -> dict[str, Any]:
    return {
        "region": s.region,
        "price_rrp": s.price_rrp,
        "demand_mw": s.demand_mw,
        "headroom_mw": s.headroom_mw,
        "regime": s.regime,
        "valid_time": s.valid_time.isoformat(),
        "spike_prob_300": s.spike_prob_300,
        "forecast_p90": s.forecast_p90,
    }


async def _is_on_cooldown(change: "MaterialChange") -> bool:
    from app.engines.commentary.detector import _COOLDOWN_SECONDS
    cooldown = _COOLDOWN_SECONDS.get(change.change_type)
    if not cooldown:
        return False
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return False
        key = _COOLDOWN_KEY.format(region=change.region, change_type=change.change_type.value)
        return await redis.get(key) is not None
    except Exception:
        return False


async def _set_cooldown(change: "MaterialChange") -> None:
    from app.engines.commentary.detector import _COOLDOWN_SECONDS
    cooldown = _COOLDOWN_SECONDS.get(change.change_type)
    if not cooldown:
        return
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return
        key = _COOLDOWN_KEY.format(region=change.region, change_type=change.change_type.value)
        await redis.set(key, "1", ex=cooldown)
    except Exception:
        pass


async def _baseline_on_cooldown(region: str) -> bool:
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return False
        return await redis.get(f"gv:commentary:baseline:{region}") is not None
    except Exception:
        return False


async def _set_baseline_cooldown(region: str) -> None:
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is None:
            return
        await redis.set(f"gv:commentary:baseline:{region}", "1", ex=3600)
    except Exception:
        pass
