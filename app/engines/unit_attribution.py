"""Technology attribution over DUID-level dispatch evidence.

This module is deterministic: it summarises observed unit dispatch and emits
caveats for sources that are not ingested yet. It does not infer fuel scarcity,
water value, outage cause, or trading intent from dispatch rows alone.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


async def retrieve_unit_dispatch(
    session,
    region: str,
    valid_time: datetime,
    window_minutes: int = 5,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return unit dispatch rows around an interval for a region."""
    from sqlalchemy import select
    from app.db.models import UnitDispatchEvent

    start = valid_time - timedelta(minutes=window_minutes)
    end = valid_time + timedelta(minutes=window_minutes)
    result = await session.execute(
        select(UnitDispatchEvent)
        .where(UnitDispatchEvent.valid_time >= start)
        .where(UnitDispatchEvent.valid_time <= end)
        .where((UnitDispatchEvent.region == region) | (UnitDispatchEvent.region.is_(None)))
        .order_by(UnitDispatchEvent.valid_time.desc(), UnitDispatchEvent.total_cleared_mw.desc().nullslast())
        .limit(limit)
    )
    return [_row_to_dict(row) for row in result.scalars().all()]


def summarise_unit_dispatch(unit_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate DUID dispatch rows into per-fuel technology evidence."""
    by_fuel: dict[str, dict[str, Any]] = {}
    for event in unit_events:
        fuel = (event.get("fuel_type") or "unknown").lower()
        bucket = by_fuel.setdefault(
            fuel,
            {
                "fuel_type": fuel,
                "unit_count": 0,
                "total_cleared_mw": 0.0,
                "initial_mw": 0.0,
                "availability_mw": 0.0,
                "semi_dispatch_cap_count": 0,
                "top_units": [],
                "raw_refs": set(),
            },
        )
        cleared = _num(event.get("total_cleared_mw"))
        initial = _num(event.get("initial_mw"))
        availability = _num(event.get("availability_mw"))
        bucket["unit_count"] += 1
        bucket["total_cleared_mw"] += cleared
        bucket["initial_mw"] += initial
        bucket["availability_mw"] += availability
        if event.get("semi_dispatch_cap") not in (None, "", 0, 0.0):
            bucket["semi_dispatch_cap_count"] += 1
        if event.get("raw_ref"):
            bucket["raw_refs"].add(event["raw_ref"])
        bucket["top_units"].append({
            "duid": event.get("duid"),
            "station_name": event.get("station_name"),
            "total_cleared_mw": cleared,
            "delta_mw": cleared - initial,
        })

    for bucket in by_fuel.values():
        bucket["delta_mw"] = bucket["total_cleared_mw"] - bucket["initial_mw"]
        bucket["top_units"] = sorted(
            bucket["top_units"],
            key=lambda row: abs(row.get("delta_mw") or row.get("total_cleared_mw") or 0.0),
            reverse=True,
        )[:5]
        bucket["raw_refs"] = sorted(bucket["raw_refs"])

    caveats = technology_caveats(by_fuel)
    return {
        "by_fuel": dict(sorted(by_fuel.items())),
        "has_unit_evidence": bool(unit_events),
        "caveats": caveats,
    }


def technology_caveats(by_fuel: dict[str, dict[str, Any]]) -> list[str]:
    caveats: list[str] = []
    if not by_fuel:
        return ["unit_dispatch_events"]
    if "hydro" in by_fuel:
        caveats.append("hydro_water_storage")
    if "coal" in by_fuel:
        caveats.append("coal_outage_commitment")
    if "gas" in by_fuel:
        caveats.append("fuel_costs")
    renewable = {"wind", "solar"} & set(by_fuel)
    if renewable:
        caveats.append("renewable_forecast_actual")
    if "unknown" in by_fuel:
        caveats.append("generator_metadata")
    return caveats


def _row_to_dict(row) -> dict[str, Any]:
    return {
        "source": row.source,
        "duid": row.duid,
        "station_name": row.station_name,
        "participant": row.participant,
        "region": row.region,
        "fuel_type": row.fuel_type,
        "valid_time": row.valid_time,
        "initial_mw": row.initial_mw,
        "total_cleared_mw": row.total_cleared_mw,
        "availability_mw": row.availability_mw,
        "target_mw": row.target_mw,
        "ramp_rate": row.ramp_rate,
        "semi_dispatch_cap": row.semi_dispatch_cap,
        "data": row.data or {},
        "raw_ref": row.raw_ref,
    }


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
