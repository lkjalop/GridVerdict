"""Fuel-type mix engine — per-source breakdown and "best buy" recommendation.

Answers: "Which energy source should I buy from right now, and why?"

Data sources (in order of preference):
  1. unit_dispatch_events — DISPATCH_UNIT_SOLUTION rows if ingested
  2. generator_units + market price — infer fuel-type exposure from
     capacity-weighted availability and the current spot price
  3. Static fuel-type cost-curve priors — fallback when neither source has data

Output is a dict that the API and frontend can render as:
  - A donut/bar chart of capacity by fuel type
  - A "best buy" recommendation with reasoning
  - Per-fuel-type price bands and trend signals
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

logger = logging.getLogger(__name__)

# Fuel type grouping — maps raw DUDETAILSUMMARY values to display labels
_FUEL_GROUPS: dict[str, str] = {
    "coal": "coal",
    "Black Coal": "coal",
    "Brown Coal": "coal",
    "Coal": "coal",
    "gas": "gas",
    "Gas": "gas",
    "Gas/Diesel": "gas",
    "CCGT": "gas",
    "OCGT": "gas",
    "Gas - Natural": "gas",
    "Kerosene": "gas",
    "Liquid Fuel": "gas",
    "hydro": "hydro",
    "Hydro": "hydro",
    "Pump Storage Hydro": "hydro",
    "Pumped Hydro": "hydro",
    "wind": "wind",
    "Wind": "wind",
    "Wind Farm": "wind",
    "solar": "solar",
    "Solar": "solar",
    "battery": "battery",
    "Large Format Battery Storage": "battery",
    "Battery Storage": "battery",
    "Battery": "battery",
    "Biomass": "other",
    "Landfill Gas": "other",
    "Other": "other",
    "demand_response": "other",
    "unknown": "other",
}

# Typical marginal cost bands $/MWh (used when no bid data available)
_COST_PRIORS: dict[str, dict[str, float]] = {
    "wind":    {"low": 0.0,   "typical": 5.0,   "high": 20.0,  "volatile": 0.2},
    "solar":   {"low": -10.0, "typical": 0.0,   "high": 10.0,  "volatile": 0.3},
    "hydro":   {"low": 0.0,   "typical": 30.0,  "high": 300.0, "volatile": 0.6},
    "coal":    {"low": 30.0,  "typical": 55.0,  "high": 80.0,  "volatile": 0.1},
    "gas":     {"low": 50.0,  "typical": 90.0,  "high": 200.0, "volatile": 0.5},
    "battery": {"low": 0.0,   "typical": 120.0, "high": 15000.0, "volatile": 0.9},
    "other":   {"low": 20.0,  "typical": 60.0,  "high": 150.0, "volatile": 0.4},
}

_FUEL_ORDER = ["wind", "solar", "coal", "hydro", "gas", "battery", "other"]


async def get_fuel_mix(
    region: str,
    session: Any = None,
    weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return per-fuel-type capacity mix, price context, and a source recommendation.

    Falls back gracefully through three data tiers:
      tier 1 — unit dispatch events (DISPATCH_UNIT_SOLUTION)
      tier 2 — generator unit metadata + current spot price
      tier 3 — static priors only
    """
    region = region.upper()
    now = datetime.now(timezone.utc)

    # Try tier 1: real dispatch output by fuel type
    unit_mix = await _from_unit_dispatch(region, session, now)
    # Try tier 2: generator capacity metadata
    capacity_mix = await _from_generator_units(region, session)
    # Current spot price for context
    spot_price = await _current_spot_price(region, session)

    # Merge: prefer real dispatch output where available
    fuel_types = set(_FUEL_ORDER)
    if unit_mix:
        fuel_types = set(unit_mix.keys()) | set(_FUEL_ORDER)

    sources: list[dict[str, Any]] = []
    for fuel in _FUEL_ORDER:
        cap = capacity_mix.get(fuel, {})
        dispatch = unit_mix.get(fuel, {})
        prior = _COST_PRIORS.get(fuel, _COST_PRIORS["other"])

        mw_dispatched = dispatch.get("mw", None)
        mw_capacity = cap.get("mw", None)
        unit_count = cap.get("count", 0)

        # Estimate effective buy price for this fuel type
        if mw_dispatched is not None and spot_price is not None:
            effective_price = spot_price     # all fuel types price at spot
        else:
            effective_price = None

        sources.append({
            "fuel_type": fuel,
            "mw_dispatched": round(mw_dispatched, 1) if mw_dispatched is not None else None,
            "mw_capacity": round(mw_capacity, 1) if mw_capacity is not None else None,
            "unit_count": unit_count,
            "effective_price_mwh": round(effective_price, 2) if effective_price is not None else None,
            "marginal_cost_low": prior["low"],
            "marginal_cost_typical": prior["typical"],
            "marginal_cost_high": prior["high"],
            "volatility_index": prior["volatile"],
            "data_tier": "dispatch" if mw_dispatched is not None else ("capacity" if mw_capacity else "prior"),
        })

    has_dispatch = any(s["mw_dispatched"] is not None for s in sources)
    has_capacity = any((s["mw_capacity"] or 0) > 0 for s in sources)
    recommendation = _recommend_source(sources, spot_price, region, weather=weather, as_of=now)

    return {
        "region": region,
        "as_of": now.isoformat(),
        "spot_price_rrp": round(spot_price, 2) if spot_price is not None else None,
        "data_tier": "dispatch" if has_dispatch else ("capacity" if has_capacity else "prior"),
        "sources": sources,
        "recommendation": recommendation,
        "caveat": (
            "Source recommendation is decision-support only. "
            "Fuel-type prices all clear at the NEM spot price; the recommendation "
            "reflects availability, volatility, and opportunity-cost signals, "
            "not a guarantee of lower cost. Not financial advice."
        ),
    }


# ── Data tier 1: unit dispatch events ────────────────────────────────────────

async def _from_unit_dispatch(
    region: str,
    session: Any,
    now: datetime,
) -> dict[str, dict]:
    if session is None:
        return {}
    try:
        from sqlalchemy import text
        cutoff = (now - timedelta(minutes=30)).isoformat()
        result = await session.execute(text("""
            SELECT u.fuel_type, SUM(u.total_cleared_mw) AS mw,
                   COUNT(DISTINCT u.duid) AS units
            FROM unit_dispatch_events u
            WHERE u.region = :region
              AND u.valid_time >= :cutoff
            GROUP BY u.fuel_type
        """), {"region": region, "cutoff": cutoff})
        rows = result.fetchall()
        if not rows:
            return {}
        mix: dict[str, dict] = {}
        for raw_fuel, mw, units in rows:
            label = _fuel_group(raw_fuel)
            if label not in mix:
                mix[label] = {"mw": 0.0, "count": 0}
            mix[label]["mw"] += float(mw or 0)
            mix[label]["count"] += int(units or 0)
        return mix
    except Exception as exc:
        logger.debug("Unit dispatch fuel mix failed: %s", exc)
        return {}


# ── Data tier 2: generator unit capacity metadata ────────────────────────────

async def _from_generator_units(
    region: str,
    session: Any,
) -> dict[str, dict]:
    if session is None:
        return {}
    try:
        from sqlalchemy import text
        result = await session.execute(text("""
            SELECT fuel_type,
                   SUM(max_capacity_mw) AS mw,
                   COUNT(*) AS cnt
            FROM generator_units
            WHERE region = :region
              AND dispatch_type = 'GENERATOR'
            GROUP BY fuel_type
        """), {"region": region})
        rows = result.fetchall()
        if not rows:
            return {}
        mix: dict[str, dict] = {}
        for raw_fuel, mw, cnt in rows:
            label = _fuel_group(raw_fuel)
            if label not in mix:
                mix[label] = {"mw": 0.0, "count": 0}
            mix[label]["mw"] += float(mw or 0)
            mix[label]["count"] += int(cnt or 0)
        return mix
    except Exception as exc:
        logger.debug("Generator unit capacity mix failed: %s", exc)
        return {}


# ── Current spot price ────────────────────────────────────────────────────────

async def _current_spot_price(region: str, session: Any) -> float | None:
    if session is None:
        return None
    try:
        from sqlalchemy import text
        result = await session.execute(text("""
            SELECT price_rrp
            FROM market_events
            WHERE region = :region
              AND source = 'AEMO_DISPATCH_PRICE'
            ORDER BY valid_time DESC
            LIMIT 1
        """), {"region": region})
        row = result.fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception as exc:
        logger.debug("Spot price lookup failed: %s", exc)
        return None


# ── Source recommendation logic ───────────────────────────────────────────────

def _recommend_source(
    sources: list[dict],
    spot_price: float | None,
    region: str,
    *,
    weather: dict[str, Any] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Determine the best energy source to buy from with a clear reason.

    Heuristic (deterministic, no LLM):
      - At very low spot (≤ $50): renewables/baseload — cheapest opportunity cost
      - At moderate spot ($50-$150): stable baseload (coal/hydro)
      - At high spot ($150-$500): gas covering; avoid; wind if available
      - At spike (> $500): battery arbitrage / rebid risk; monitor only
      - When spot is unknown: rank by lowest typical marginal cost
    """
    if spot_price is None:
        ranked = sorted(sources, key=lambda s: s["marginal_cost_typical"])
        best = ranked[0] if ranked else None
        return {
            "fuel_type": best["fuel_type"] if best else "unknown",
            "reason": "No live price available — recommending lowest marginal cost source as fallback.",
            "confidence": "low",
            "action": "monitor",
        }

    # Find which sources have actual capacity/dispatch
    available = [s for s in sources if (s["mw_capacity"] or 0) > 0 or (s["mw_dispatched"] or 0) > 0]
    if not available:
        available = sources  # use all as fallback
    availability = _renewable_availability(region, weather, as_of or datetime.now(timezone.utc))

    if spot_price <= 0:
        # Negative price — absorb from solar/wind
        preferred = ["solar", "wind"]
        reason = f"Negative spot price (${spot_price:.0f}/MWh): solar and wind are dispatching below cost — best to absorb renewable surplus."
        action = "buy"
    elif spot_price <= 50:
        preferred = ["wind", "solar", "coal"]
        reason = f"Low spot price (${spot_price:.0f}/MWh): renewable and baseload coal are the cheapest sources with lowest opportunity cost."
        action = "buy"
    elif spot_price <= 150:
        preferred = ["coal", "hydro", "wind"]
        reason = f"Moderate spot price (${spot_price:.0f}/MWh): coal baseload and hydro offer stable procurement with predictable cost."
        action = "buy_cautious"
    elif spot_price <= 500:
        preferred = ["hydro", "wind", "gas"]
        if not availability["wind_available"] and not availability["solar_available"]:
            preferred = ["hydro", "coal", "gas"]
        elif not availability["wind_available"]:
            preferred = ["hydro", "coal", "gas"]
        reason = f"Elevated spot price (${spot_price:.0f}/MWh): hydro and wind are preferred; gas is marginal setter — consider deferring non-essential load."
        if preferred == ["hydro", "coal", "gas"]:
            reason = (
                f"Elevated spot price (${spot_price:.0f}/MWh): current weather/time signals "
                "reduce wind and solar confidence, so dispatchable hydro/coal/gas move up the order."
            )
        elif not availability["solar_available"]:
            reason += " Solar is weak in this interval, so it is not preferred despite low marginal cost."
        action = "monitor"
    else:
        preferred = ["wind", "solar"]
        reason = f"Spike conditions (${spot_price:.0f}/MWh): battery and gas are price-setting. Renewables are cheapest if available; consider demand response."
        action = "defer"

    # Pick best available from preferred list
    best_fuel = None
    for pref in preferred:
        match = next((s for s in available if s["fuel_type"] == pref), None)
        if match:
            best_fuel = pref
            break
    if best_fuel is None:
        best_fuel = available[0]["fuel_type"] if available else "unknown"

    notes = list(availability["notes"])
    if "hydro" in preferred:
        notes.append("Hydro ranking is conditional because live water storage/drought opportunity-cost data is not ingested.")

    return {
        "fuel_type": best_fuel,
        "reason": reason,
        "preferred_order": preferred,
        "action": action,
        "confidence": "medium" if any(s["data_tier"] == "dispatch" for s in sources) else "low",
        "spot_price_context": spot_price,
        "availability_signals": availability,
        "notes": notes,
    }


def _fuel_group(raw_fuel: Any) -> str:
    if raw_fuel is None:
        return "other"
    text = str(raw_fuel).strip()
    return _FUEL_GROUPS.get(text) or _FUEL_GROUPS.get(text.lower()) or "other"


def _renewable_availability(
    region: str,
    weather: dict[str, Any] | None,
    as_of: datetime,
) -> dict[str, Any]:
    local = as_of.astimezone(ZoneInfo("Australia/Sydney"))
    daylight = 6 <= local.hour < 18
    consensus = (weather or {}).get("consensus") or {}
    tags = set((weather or {}).get("relevance_tags") or (weather or {}).get("tags") or [])
    wind = consensus.get("wind_speed_kmh")
    cloud = consensus.get("cloud_cover_pct")
    rain = consensus.get("precipitation_mm")

    low_wind = "low_wind_risk" in tags or (wind is not None and float(wind) <= 10)
    low_solar = (
        "low_solar_risk" in tags
        or not daylight
        or (cloud is not None and float(cloud) >= 75)
        or (rain is not None and float(rain) > 0)
    )
    notes: list[str] = []
    if not daylight:
        notes.append(f"Solar de-rated: local time {local:%H:%M} is outside the daylight window.")
    elif cloud is not None and float(cloud) >= 75:
        notes.append(f"Solar de-rated: cloud cover is {float(cloud):.0f}%.")
    if wind is not None and float(wind) <= 10:
        notes.append(f"Wind de-rated: wind speed is {float(wind):.1f} km/h.")
    if rain is not None and float(rain) > 0:
        notes.append(f"Storm/rain context present: rain is {float(rain):.1f} mm.")

    return {
        "region": region,
        "local_time": local.isoformat(),
        "daylight": daylight,
        "solar_available": not low_solar,
        "wind_available": not low_wind,
        "wind_speed_kmh": wind,
        "cloud_cover_pct": cloud,
        "weather_tags": sorted(tags),
        "notes": notes,
    }
