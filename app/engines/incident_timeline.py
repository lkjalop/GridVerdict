"""Market Incident Timeline engine.

Reconstructs the causal sequence of events around a dispatch price move.

Answers: what changed, when, in what order, and what evidence tier supports
each driver?

Output is ordered newest → oldest (anchor event first) so the rendering
layer can present it as a top-down incident report without resorting.

Evidence tiers follow the same four-level system as DriverConfidenceTier:
  CONFIRMED  — direct AEMO telemetry present and fresh
  SUPPORTED  — archived MMSDM evidence corroborates the claim
  PLAUSIBLE  — pattern/model/weather signal (indirect)
  UNCONFIRMED — mechanistic expectation without hard data (or data missing)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_HEADROOM_WARN_MW = 500.0
_PRICE_SPIKE_THRESHOLD = 300.0
_SIGNIFICANT_UNIT_DELTA_MW = 50.0
_REBID_LOOKBACK_HOURS = 3
_FCAS_TIGHT_THRESHOLD = 100.0    # $/MWh — FCAS price above this flags a tight market
_OUTAGE_MIN_DROP_MW = 100.0      # MW drop in availability to flag as an outage event


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    event_id: str
    interval: datetime
    category: str       # price | headroom | constraint | interconnector |
                        # unit_dispatch | rebid | fcas | outage | weather | missing_data
    description: str
    tier: str           # confirmed | supported | plausible | unconfirmed
    evidence_summary: str = ""
    evidence_ref_ids: list[str] = field(default_factory=list)
    delta: dict[str, Any] | None = None   # {"from": float, "to": float} for price moves
    missing: bool = False                 # True = expected but not found


@dataclass
class IncidentTimeline:
    region: str
    anchor_interval: datetime
    anchor_price: float | None
    anchor_regime: str | None
    lookback_minutes: int
    events: list[TimelineEvent]
    verdict: dict[str, list[str]]   # {confirmed, supported, plausible, unconfirmed, missing_coverage}
    coverage_grade: str             # full | partial | minimal
    as_of: datetime


def _as_utc(dt: datetime) -> datetime:
    """Normalize DB/cache datetimes so timeline sorting never mixes tz states."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── Public entry point ────────────────────────────────────────────────────────

async def build_incident_timeline(
    session,
    region: str,
    anchor_time: datetime,
    lookback_minutes: int = 30,
) -> IncidentTimeline:
    """Build a chronological incident timeline for a dispatch price move.

    Queries all available evidence tables within the lookback window and
    emits a TimelineEvent per evidence type — including explicit UNCONFIRMED
    events when expected data is absent (so gaps are visible).
    """
    anchor_time = _as_utc(anchor_time)
    window_start = anchor_time - timedelta(minutes=lookback_minutes)
    events: list[TimelineEvent] = []
    covered: set[str] = set()
    missing: set[str] = set()

    anchor_price, anchor_regime = await _add_price_events(
        session, region, anchor_time, window_start, events, covered, missing
    )
    await _add_constraint_events(session, region, anchor_time, window_start, events, covered, missing)
    await _add_interconnector_events(session, region, anchor_time, window_start, events, covered, missing)
    await _add_unit_dispatch_events(session, region, anchor_time, window_start, events, covered, missing)
    await _add_rebid_events(session, region, anchor_time, events, covered, missing)
    await _add_fcas_events(session, region, anchor_time, window_start, events, covered, missing)
    await _add_outage_events(session, region, anchor_time, window_start, events, covered, missing)
    await _add_weather_event(region, anchor_time, events, covered, missing)

    # Newest first
    events.sort(key=lambda e: _as_utc(e.interval), reverse=True)

    verdict = _build_verdict(events, missing)
    coverage_grade = _coverage_grade(verdict)

    return IncidentTimeline(
        region=region,
        anchor_interval=anchor_time,
        anchor_price=anchor_price,
        anchor_regime=anchor_regime,
        lookback_minutes=lookback_minutes,
        events=events,
        verdict=verdict,
        coverage_grade=coverage_grade,
        as_of=datetime.now(timezone.utc),
    )


# ── Evidence collectors ───────────────────────────────────────────────────────

async def _add_price_events(
    session, region, anchor_time, window_start,
    events, covered, missing
) -> tuple[float | None, str | None]:
    from sqlalchemy import select
    from app.db.models import MarketEvent

    anchor_price: float | None = None
    anchor_regime: str | None = None

    try:
        result = await session.execute(
            select(MarketEvent)
            .where(MarketEvent.source == "AEMO_DISPATCH_PRICE")
            .where(MarketEvent.region == region)
            .where(MarketEvent.valid_time >= window_start)
            .where(MarketEvent.valid_time <= anchor_time)
            .order_by(MarketEvent.valid_time.asc())
        )
        price_rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("Price trajectory query failed: %s", exc)
        price_rows = []

    if price_rows:
        covered.add("price")
        latest = price_rows[-1]
        anchor_price = float(latest.price_rrp or 0.0)
        anchor_demand = float(latest.demand_mw or 0.0)
        anchor_avail = float(latest.availability_mw or 0.0)
        anchor_headroom = anchor_avail - anchor_demand
        anchor_regime = _classify_regime(anchor_price)

        # Price anchor event
        price_ev_id = _ev_id()
        delta: dict[str, Any] = {"to": round(anchor_price, 2)}
        if len(price_rows) >= 2:
            prev = float(price_rows[-2].price_rrp or 0.0)
            delta["from"] = round(prev, 2)
            desc = f"Dispatch price rises from ${prev:.0f} → ${anchor_price:.0f}/MWh"
        else:
            desc = f"Dispatch price at ${anchor_price:.0f}/MWh (regime: {anchor_regime})"

        events.append(TimelineEvent(
            event_id=price_ev_id,
            interval=latest.valid_time,
            category="price",
            description=desc,
            tier="confirmed",
            evidence_summary=f"AEMO_DISPATCH_PRICE {latest.raw_ref}",
            evidence_ref_ids=[price_ev_id],
            delta=delta,
        ))

        # Headroom event when compressed
        if anchor_headroom < _HEADROOM_WARN_MW:
            covered.add("headroom")
            hdroom_id = _ev_id()
            demand_id = _ev_id()
            events.append(TimelineEvent(
                event_id=hdroom_id,
                interval=latest.valid_time,
                category="headroom",
                description=(
                    f"Headroom compressed to {anchor_headroom:.0f} MW "
                    f"({anchor_demand:.0f} MW demand vs {anchor_avail:.0f} MW available)"
                ),
                tier="confirmed",
                evidence_summary=f"demand + availability {latest.raw_ref}",
                evidence_ref_ids=[hdroom_id, demand_id],
            ))
    else:
        missing.add("price")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time,
            category="missing_data",
            description="Dispatch price data not available for this interval (not yet ingested or stale)",
            tier="unconfirmed",
            missing=True,
        ))

    return anchor_price, anchor_regime


async def _add_constraint_events(
    session, region, anchor_time, window_start,
    events, covered, missing
):
    from sqlalchemy import select
    from app.db.models import MarketDriverEvent

    try:
        result = await session.execute(
            select(MarketDriverEvent)
            .where(MarketDriverEvent.driver_type == "constraint")
            .where(MarketDriverEvent.region == region)
            .where(MarketDriverEvent.valid_time >= window_start)
            .where(MarketDriverEvent.valid_time <= anchor_time)
            .order_by(MarketDriverEvent.valid_time.desc())
            .limit(20)
        )
        rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("Constraint query failed: %s", exc)
        rows = []

    binding = [
        r for r in rows
        if abs(float((r.values or {}).get("marginal_value") or 0.0)) > 0
        or abs(float((r.values or {}).get("violation_degree") or 0.0)) > 0
    ]

    if binding:
        covered.add("constraint")
        top = binding[0]
        mv = float((top.values or {}).get("marginal_value") or 0.0)
        ev_id = _ev_id()
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=top.valid_time,
            category="constraint",
            description=f"Constraint {top.element_id} binding (shadow price ${mv:.0f}/MWh)",
            tier="supported",
            evidence_summary=f"{top.source} {top.raw_ref}",
            evidence_ref_ids=[ev_id],
            delta={"marginal_value": round(mv, 2)},
        ))
        # Surface additional binding constraints if present
        for extra in binding[1:3]:
            mv2 = float((extra.values or {}).get("marginal_value") or 0.0)
            if abs(mv2) > 0:
                eid2 = _ev_id()
                events.append(TimelineEvent(
                    event_id=eid2,
                    interval=extra.valid_time,
                    category="constraint",
                    description=f"Also binding: {extra.element_id} (${mv2:.0f}/MWh)",
                    tier="supported",
                    evidence_summary=f"{extra.source} {extra.raw_ref}",
                    evidence_ref_ids=[eid2],
                ))
    elif rows:
        # Data present but none binding — still note it
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=rows[0].valid_time,
            category="constraint",
            description=f"{len(rows)} constraint rows present — none binding at this window",
            tier="unconfirmed",
            evidence_summary="DISPATCHCONSTRAINT rows ingested, marginal_value = 0",
        ))
    else:
        missing.add("constraint")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time - timedelta(minutes=10),
            category="missing_data",
            description="Constraint data (DISPATCHCONSTRAINT) not ingested — binding status unknown",
            tier="unconfirmed",
            missing=True,
        ))


async def _add_interconnector_events(
    session, region, anchor_time, window_start,
    events, covered, missing
):
    from sqlalchemy import select
    from app.db.models import MarketDriverEvent

    try:
        result = await session.execute(
            select(MarketDriverEvent)
            .where(MarketDriverEvent.driver_type == "interconnector")
            .where(MarketDriverEvent.valid_time >= window_start)
            .where(MarketDriverEvent.valid_time <= anchor_time)
            .order_by(MarketDriverEvent.valid_time.desc())
            .limit(20)
        )
        rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("Interconnector query failed: %s", exc)
        rows = []

    tight = [r for r in rows if _ic_is_tight(r.values or {})]

    if tight:
        covered.add("interconnector")
        top = tight[0]
        vals = top.values or {}
        flow = float(vals.get("mw_flow") or vals.get("metered_mw_flow") or 0.0)
        ev_id = _ev_id()
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=top.valid_time,
            category="interconnector",
            description=(
                f"Interconnector {top.element_id} near export/import limit "
                f"({flow:.0f} MW flow)"
            ),
            tier="supported",
            evidence_summary=f"{top.source} {top.raw_ref}",
            evidence_ref_ids=[ev_id],
            delta={"mw_flow": round(flow, 1)},
        ))
    elif rows:
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=rows[0].valid_time,
            category="interconnector",
            description=f"{len(rows)} interconnector rows — none near limit",
            tier="unconfirmed",
            evidence_summary="DISPATCHINTERCONNECTORRES rows found, none near limit",
        ))
    else:
        missing.add("interconnector")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time - timedelta(minutes=15),
            category="missing_data",
            description="Interconnector flow data (DISPATCHINTERCONNECTORRES) not ingested",
            tier="unconfirmed",
            missing=True,
        ))


async def _add_unit_dispatch_events(
    session, region, anchor_time, window_start,
    events, covered, missing
):
    from sqlalchemy import select
    from app.db.models import UnitDispatchEvent

    try:
        result = await session.execute(
            select(UnitDispatchEvent)
            .where(UnitDispatchEvent.region == region)
            .where(UnitDispatchEvent.valid_time >= window_start)
            .where(UnitDispatchEvent.valid_time <= anchor_time)
            .order_by(UnitDispatchEvent.valid_time.desc())
            .limit(50)
        )
        rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("Unit dispatch query failed: %s", exc)
        rows = []

    significant = [
        r for r in rows
        if abs(float(r.total_cleared_mw or 0.0) - float(r.initial_mw or 0.0))
           >= _SIGNIFICANT_UNIT_DELTA_MW
    ]

    if significant:
        covered.add("unit_dispatch")
        by_fuel: dict[str, list] = {}
        for r in significant:
            ft = r.fuel_type or "unknown"
            by_fuel.setdefault(ft, []).append(r)

        fuel_parts = [
            f"{ft} ({len(rs)})"
            for ft, rs in sorted(by_fuel.items(), key=lambda x: -len(x[1]))
            if ft != "unknown"
        ]
        fuel_summary = ", ".join(fuel_parts) if fuel_parts else f"{len(significant)} DUIDs"
        ev_id = _ev_id()
        first = significant[0]
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=first.valid_time,
            category="unit_dispatch",
            description=(
                f"Significant unit dispatch change: {len(significant)} DUIDs "
                f"(≥{_SIGNIFICANT_UNIT_DELTA_MW:.0f} MW delta) — {fuel_summary}"
            ),
            tier="supported",
            evidence_summary=f"DISPATCH_UNIT_SOLUTION {first.raw_ref}",
            evidence_ref_ids=[ev_id],
        ))
    elif rows:
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=rows[0].valid_time,
            category="unit_dispatch",
            description=(
                f"{len(rows)} unit dispatch rows — no movements ≥ "
                f"{_SIGNIFICANT_UNIT_DELTA_MW:.0f} MW delta"
            ),
            tier="unconfirmed",
            evidence_summary="DISPATCH_UNIT_SOLUTION rows present, no large deltas",
        ))
    else:
        missing.add("unit_dispatch")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time - timedelta(minutes=20),
            category="missing_data",
            description="Unit dispatch data (DISPATCH_UNIT_SOLUTION) not ingested for this interval",
            tier="unconfirmed",
            missing=True,
        ))


async def _add_rebid_events(
    session, region, anchor_time,
    events, covered, missing
):
    from app.engines.rebid_engine import detect_rebids

    settlement_dt = anchor_time.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        rebids = await detect_rebids(session, region, settlement_dt, threshold_mw=50.0, min_price=100.0)
    except Exception as exc:
        logger.debug("Rebid engine failed: %s", exc)
        rebids = []

    # Filter to rebids whose period falls within the lookback window
    lookback_start = anchor_time - timedelta(hours=_REBID_LOOKBACK_HOURS)
    relevant = [
        r for r in rebids
        if r.period_start_utc and lookback_start <= r.period_start_utc <= anchor_time
    ]

    if relevant:
        covered.add("rebid")
        top = relevant[0]
        ev_id = _ev_id()
        strategic_count = sum(1 for r in relevant if r.rebid_flag == "strategic")
        total_mw = round(sum(r.withdrawal_mw for r in relevant), 1)
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=top.period_start_utc,
            category="rebid",
            description=(
                f"Intraday rebid: {len(relevant)} withdrawal(s) totalling {total_mw} MW "
                f"({strategic_count} strategic) in {_REBID_LOOKBACK_HOURS}h before anchor"
            ),
            tier="supported",
            evidence_summary=(
                f"largest: {top.duid} −{top.withdrawal_mw} MW "
                f"({top.rebid_flag}, {top.severity})"
            ),
            evidence_ref_ids=[ev_id],
        ))
    elif rebids:
        # Data present but no rebids within the lookback window
        covered.add("rebid")
        ev_id = _ev_id()
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=anchor_time - timedelta(minutes=25),
            category="rebid",
            description="No significant availability withdrawals detected in the anchor window",
            tier="supported",
            evidence_summary="BIDDAYOFFER vs BIDPEROFFER comparison performed",
            evidence_ref_ids=[ev_id],
        ))
    else:
        missing.add("rebid")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time - timedelta(minutes=25),
            category="missing_data",
            description=(
                "Rebid / bid availability change — no BIDPEROFFER rows ingested. "
                "Strategic rebidding cannot be confirmed."
            ),
            tier="unconfirmed",
            missing=True,
        ))


async def _add_fcas_events(
    session, region, anchor_time, window_start,
    events, covered, missing
):
    from sqlalchemy import select
    from app.db.models import FcasPriceEvent

    try:
        result = await session.execute(
            select(FcasPriceEvent)
            .where(FcasPriceEvent.region == region)
            .where(FcasPriceEvent.valid_time >= window_start)
            .where(FcasPriceEvent.valid_time <= anchor_time)
            .order_by(FcasPriceEvent.valid_time.desc())
            .limit(12)
        )
        rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("FCAS price query failed: %s", exc)
        rows = []

    if not rows:
        missing.add("fcas")
        events.append(TimelineEvent(
            event_id=_ev_id(),
            interval=anchor_time - timedelta(minutes=10),
            category="missing_data",
            description="FCAS ancillary service prices not yet ingested for this interval",
            tier="unconfirmed",
            missing=True,
        ))
        return

    covered.add("fcas")
    latest = rows[0]
    tight_services: list[str] = []
    service_map = {
        "Raise 6s": latest.raise_6sec_rrp,
        "Raise 60s": latest.raise_60sec_rrp,
        "Raise 5min": latest.raise_5min_rrp,
        "Raise Reg": latest.raise_reg_rrp,
        "Lower 6s": latest.lower_6sec_rrp,
        "Lower 60s": latest.lower_60sec_rrp,
        "Lower 5min": latest.lower_5min_rrp,
        "Lower Reg": latest.lower_reg_rrp,
    }
    for svc, price in service_map.items():
        if price is not None and price > _FCAS_TIGHT_THRESHOLD:
            tight_services.append(f"{svc} ${price:.0f}")

    ev_id = _ev_id()
    if tight_services:
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=latest.valid_time,
            category="fcas",
            description=(
                f"Tight FCAS market: {', '.join(tight_services[:4])} above "
                f"${_FCAS_TIGHT_THRESHOLD:.0f}/MWh threshold"
            ),
            tier="confirmed",
            evidence_summary=f"AEMO_DISPATCH_PRICE FCAS columns {latest.raw_ref}",
            evidence_ref_ids=[ev_id],
        ))
    else:
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=latest.valid_time,
            category="fcas",
            description="FCAS prices within normal range — no contingency reserve pressure detected",
            tier="confirmed",
            evidence_summary=f"AEMO_DISPATCH_PRICE FCAS columns {latest.raw_ref}",
            evidence_ref_ids=[ev_id],
        ))


async def _add_outage_events(
    session, region, anchor_time, window_start,
    events, covered, missing
):
    """Detect unit availability drops ≥ 100 MW as outage signals."""
    from sqlalchemy import select
    from app.db.models import UnitDispatchEvent

    try:
        result = await session.execute(
            select(UnitDispatchEvent)
            .where(UnitDispatchEvent.region == region)
            .where(UnitDispatchEvent.valid_time >= window_start - timedelta(minutes=10))
            .where(UnitDispatchEvent.valid_time <= anchor_time)
            .order_by(UnitDispatchEvent.duid, UnitDispatchEvent.valid_time.asc())
        )
        rows = list(result.scalars().all())
    except Exception as exc:
        logger.debug("Outage detection query failed: %s", exc)
        rows = []

    if not rows:
        missing.add("outage")
        return

    # Group by DUID, find drops in availability_mw between consecutive intervals
    by_duid: dict[str, list] = {}
    for r in rows:
        by_duid.setdefault(r.duid, []).append(r)

    outage_events: list[dict] = []
    for duid, duid_rows in by_duid.items():
        duid_rows.sort(key=lambda r: r.valid_time)
        for prev, curr in zip(duid_rows, duid_rows[1:]):
            prev_avail = prev.availability_mw
            curr_avail = curr.availability_mw
            if prev_avail is None or curr_avail is None:
                continue
            drop = float(prev_avail) - float(curr_avail)
            if drop >= _OUTAGE_MIN_DROP_MW:
                outage_events.append({
                    "duid": duid,
                    "station_name": curr.station_name or duid,
                    "fuel_type": curr.fuel_type,
                    "valid_time": curr.valid_time,
                    "prev_avail": round(float(prev_avail), 1),
                    "curr_avail": round(float(curr_avail), 1),
                    "drop_mw": round(drop, 1),
                    "raw_ref": curr.raw_ref,
                })

    if outage_events:
        covered.add("outage")
        outage_events.sort(key=lambda e: e["drop_mw"], reverse=True)
        top = outage_events[0]
        ev_id = _ev_id()
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=top["valid_time"],
            category="outage",
            description=(
                f"Unit availability drop: {top['station_name']} ({top['duid']}) "
                f"fell {top['prev_avail']} → {top['curr_avail']} MW "
                f"(−{top['drop_mw']} MW)"
            ),
            tier="confirmed",
            evidence_summary=f"DISPATCH_UNIT_SOLUTION {top['raw_ref']}",
            evidence_ref_ids=[ev_id],
            delta={"from": top["prev_avail"], "to": top["curr_avail"]},
        ))
        if len(outage_events) > 1:
            ev_id2 = _ev_id()
            total_drop = round(sum(e["drop_mw"] for e in outage_events), 1)
            events.append(TimelineEvent(
                event_id=ev_id2,
                interval=outage_events[-1]["valid_time"],
                category="outage",
                description=(
                    f"{len(outage_events)} unit(s) lost availability totalling "
                    f"{total_drop} MW in the anchor window"
                ),
                tier="supported",
                evidence_summary="DISPATCH_UNIT_SOLUTION — multiple DUIDs",
                evidence_ref_ids=[ev_id2],
            ))
    else:
        # Data present but no large availability drops — note absence
        covered.add("outage")
        ev_id = _ev_id()
        events.append(TimelineEvent(
            event_id=ev_id,
            interval=anchor_time - timedelta(minutes=5),
            category="outage",
            description="No unit availability drops ≥ 100 MW detected in the anchor window",
            tier="confirmed",
            evidence_summary="DISPATCH_UNIT_SOLUTION — no forced outages detected",
            evidence_ref_ids=[ev_id],
        ))


async def _add_weather_event(
    region, anchor_time,
    events, covered, missing
):
    try:
        from app.data.cache import get_cache
        weather_data = await get_cache().get(f"weather_{region.upper()}")
        if weather_data:
            covered.add("weather")
            consensus = weather_data.get("consensus", {})
            temp = consensus.get("temperature_c")
            wind = consensus.get("wind_speed_kmh")
            parts = []
            if temp is not None:
                parts.append(f"temperature {temp:.1f}°C")
            if wind is not None:
                parts.append(f"wind {wind:.1f} km/h")
            desc = "Weather demand pressure: " + (
                ", ".join(parts) if parts else "weather consensus available"
            )
            ev_id = _ev_id()
            events.append(TimelineEvent(
                event_id=ev_id,
                interval=anchor_time - timedelta(minutes=30),
                category="weather",
                description=desc,
                tier="plausible",
                evidence_summary=(
                    f"weather consensus ({weather_data.get('source_count', 1)} sources, "
                    f"confidence {weather_data.get('confidence', 0):.0%})"
                ),
                evidence_ref_ids=[ev_id],
            ))
        else:
            missing.add("weather")
            events.append(TimelineEvent(
                event_id=_ev_id(),
                interval=anchor_time - timedelta(minutes=30),
                category="weather",
                description="Weather consensus unavailable — demand pressure via weather not assessable",
                tier="unconfirmed",
                missing=True,
            ))
    except Exception as exc:
        logger.debug("Weather cache read failed: %s", exc)
        missing.add("weather")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ev_id() -> str:
    return f"ev-{uuid.uuid4().hex[:8]}"


def _classify_regime(price: float) -> str:
    if price >= 5000:
        return "extreme"
    if price >= _PRICE_SPIKE_THRESHOLD:
        return "spike"
    if price >= 100:
        return "elevated"
    return "normal"


def _ic_is_tight(values: dict[str, Any]) -> bool:
    flow = values.get("mw_flow") or values.get("metered_mw_flow")
    if flow is None:
        return False
    flow_f = float(flow)
    for key in ("export_limit", "import_limit"):
        limit = values.get(key)
        if limit is None:
            continue
        limit_f = float(limit)
        if abs(limit_f) > 0 and abs(abs(flow_f) - abs(limit_f)) <= max(10.0, abs(limit_f) * 0.05):
            return True
    return False


def _build_verdict(events: list[TimelineEvent], missing_cats: set[str]) -> dict[str, list[str]]:
    verdict: dict[str, list[str]] = {
        "confirmed": [],
        "supported": [],
        "plausible": [],
        "unconfirmed": [],
        "missing_coverage": sorted(missing_cats),
    }
    seen: set[str] = set()
    for e in events:
        if e.missing:
            continue
        key = e.tier
        if key not in verdict:
            continue
        label = e.description[:100]
        if label not in seen:
            seen.add(label)
            verdict[key].append(label)
    return verdict


def _coverage_grade(verdict: dict[str, list[str]]) -> str:
    n_confirmed = len(verdict["confirmed"])
    n_supported = len(verdict["supported"])
    if n_confirmed >= 1 and n_supported >= 2:
        return "full"
    if n_confirmed >= 1 or n_supported >= 1:
        return "partial"
    return "minimal"


def timeline_to_dict(tl: IncidentTimeline) -> dict:
    return {
        "region": tl.region,
        "anchor_interval": _as_utc(tl.anchor_interval).isoformat(),
        "anchor_price": tl.anchor_price,
        "anchor_regime": tl.anchor_regime,
        "lookback_minutes": tl.lookback_minutes,
        "coverage_grade": tl.coverage_grade,
        "verdict": tl.verdict,
        "as_of": tl.as_of.isoformat(),
        "events": [
            {
                "event_id": e.event_id,
                "interval": _as_utc(e.interval).isoformat(),
                "category": e.category,
                "description": e.description,
                "tier": e.tier,
                "evidence_summary": e.evidence_summary,
                "evidence_ref_ids": e.evidence_ref_ids,
                "delta": e.delta,
                "missing": e.missing,
            }
            for e in tl.events
        ],
    }
