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
    """Return unit dispatch rows around an interval.

    Primary: DISPATCH_UNIT_SOLUTION rows from UnitDispatchEvent (participant-only,
    requires data pipeline feeding that table).
    Fallback: merit-order reconstruction from BidOffer + GeneratorUnit when no
    UnitDispatchEvent rows are present.
    """
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
    rows = [_row_to_dict(row) for row in result.scalars().all()]
    if rows:
        return rows

    # Fallback: reconstruct approximate dispatch from bid data + generator registry
    return await _reconstruct_from_bids(session, region, valid_time, limit)


async def _reconstruct_from_bids(
    session,
    region: str,
    valid_time: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Approximate unit dispatch via merit-order from BidOffer + GeneratorUnit.

    Uses the latest ENERGY bid for each DUID in the region (BIDDAYOFFER or most
    recent BIDPEROFFER), maps to fuel_type via GeneratorUnit, then runs a simple
    economic dispatch simulation to meet observed demand.

    Data tier: "bid_reconstruction" — better than static priors, not as accurate
    as DISPATCH_UNIT_SOLUTION. Demand value comes from caller context (not available
    here), so we use total offered capacity as a proxy and dispatch all offered units.
    """
    try:
        from sqlalchemy import select, and_
        from app.db.models import BidOffer, GeneratorUnit

        # Get generator registry for the region
        gen_result = await session.execute(
            select(GeneratorUnit).where(GeneratorUnit.region == region.upper())
        )
        generators: dict[str, "GeneratorUnit"] = {g.duid: g for g in gen_result.scalars().all()}
        if not generators:
            return []

        # Get the most recent ENERGY bids per DUID on or before valid_time
        day_start = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
        bid_result = await session.execute(
            select(BidOffer)
            .where(
                and_(
                    BidOffer.region == region.upper(),
                    BidOffer.bid_type == "ENERGY",
                    BidOffer.settlement_date >= day_start,
                    BidOffer.settlement_date <= valid_time,
                )
            )
            .order_by(BidOffer.settlement_date.desc(), BidOffer.offer_date.desc())
            .limit(limit * 2)
        )
        bids = bid_result.scalars().all()
        if not bids:
            return []

        # Keep latest bid per DUID
        seen: set[str] = set()
        best_bids: list = []
        for b in bids:
            if b.duid not in seen:
                seen.add(b.duid)
                best_bids.append(b)

        # Build units with cheapest price band and max offered capacity
        units = []
        for bid in best_bids:
            gen = generators.get(bid.duid)
            if gen is None:
                continue
            price_bands = bid.price_bands or {}
            avail_bands = bid.avail_bands or {}
            if not price_bands:
                continue
            # Find cheapest non-market-cap price
            _MARKET_PRICE_CAP = 15500.0
            min_price = min(
                (float(v) for v in price_bands.values() if float(v) < _MARKET_PRICE_CAP),
                default=_MARKET_PRICE_CAP,
            )
            total_avail = sum(float(v) for v in avail_bands.values())
            max_cap = float(gen.max_capacity_mw or total_avail or 0.0)
            offered_mw = min(total_avail, max_cap)
            if offered_mw <= 0:
                continue
            units.append({
                "duid": bid.duid,
                "station_name": gen.station_name,
                "fuel_type": (gen.fuel_type or "unknown").lower(),
                "region": region,
                "valid_time": valid_time,
                "min_price_band": min_price,
                "max_avail_mw": offered_mw,
            })

        # Sort by merit order (cheapest first)
        units.sort(key=lambda u: u["min_price_band"])

        # Dispatch all offered units (we don't have demand here; caller's summarise handles it)
        dispatched = []
        for unit in units[:limit]:
            dispatched.append({
                "source": "BID_RECONSTRUCTION",
                "duid": unit["duid"],
                "station_name": unit["station_name"],
                "participant": None,
                "region": region,
                "fuel_type": unit["fuel_type"],
                "valid_time": unit["valid_time"],
                "initial_mw": unit["max_avail_mw"],
                "total_cleared_mw": unit["max_avail_mw"],
                "availability_mw": unit["max_avail_mw"],
                "target_mw": unit["max_avail_mw"],
                "ramp_rate": None,
                "semi_dispatch_cap": None,
                "data": {"min_price_band": unit["min_price_band"], "reconstruction": True},
                "raw_ref": "bid_reconstruction",
            })
        return dispatched

    except Exception:
        return []


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
