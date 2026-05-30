"""Marginal price-setter identification for NEM causal attribution.

The marginal setter is the most expensive dispatched generator in a given
dispatch interval — its bid price approximately equals the regional spot price.
Identifying it answers the core 'why' question: which generator is setting
the price right now, and what type of fuel is it?

NEM dispatch follows merit order: generators bid in price bands, NEMDE dispatches
cheapest first until demand is met. The last (most expensive) dispatched unit
is the marginal setter.

Data sources (in decreasing quality order):
  DISPATCH   UnitDispatchEvent rows with actual MW output (written by scheduler
             from DISPATCHLOAD in live zip and from archive gap-fill).
             Data tier: DISPATCH (highest confidence).

  BID_RECON  Reconstruction from BidOffer + GeneratorUnit merit order.
             Approximates dispatch when DISPATCHLOAD rows are absent.
             Data tier: SUPPORTED (medium confidence).

  PRIOR      Static marginal-cost priors by fuel type.
             Used when no DB data is available at all.
             Data tier: PRIOR (lowest confidence).

Framework boundary: no imports from data/, mcp/, or domain/nem/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_PRIOR_MARGINAL_COST: dict[str, float] = {
    "solar":   0.0,
    "wind":    5.0,
    "hydro":  30.0,
    "battery": 40.0,
    "coal":   55.0,
    "gas":    85.0,
    "diesel": 200.0,
    "liquid": 200.0,
    "unknown": 80.0,
}

_FUEL_DISPLAY: dict[str, str] = {
    "solar": "Solar", "wind": "Wind", "hydro": "Hydro",
    "coal": "Coal", "gas": "Gas", "battery": "Battery",
    "diesel": "Diesel", "liquid": "Liquid fuel", "unknown": "Unknown",
}

_RENEWABLE_FUELS = {"solar", "wind"}
_DISPATCHABLE_FUELS = {"coal", "gas", "hydro", "battery", "diesel", "liquid"}


@dataclass
class FuelDispatchSummary:
    fuel_type: str
    dispatched_mw: float
    unit_count: int
    headroom_mw: float          # availability - totalcleared (unused capacity)
    is_renewable: bool
    top_unit: str | None = None # DUID of the highest-output unit
    data_tier: str = "prior"    # dispatch | bid_reconstruction | prior


@dataclass
class MarginalSetterResult:
    """Result of marginal price-setter identification."""
    fuel_type: str                         # "gas" | "coal" | "hydro" | etc.
    display_name: str                      # "Gas peaker" | "Black coal" etc.
    duid: str | None                       # generator ID (None when using priors)
    station_name: str | None              # human name of the station
    estimated_price_mwh: float            # bid price that set the market
    certainty_tier: str                   # "dispatch" | "bid_reconstruction" | "prior"
    explanation: str                       # one-sentence causal explanation
    by_fuel: list[FuelDispatchSummary]    # full dispatch breakdown by fuel
    constraint_narrative: str = ""         # binding constraints if any
    interconnector_narrative: str = ""    # interconnector context if any
    data_tier: str = "prior"


def identify_marginal_setter(
    unit_events: list[dict[str, Any]],
    driver_events: list[dict[str, Any]],
    spot_price: float,
    region: str,
) -> MarginalSetterResult:
    """Identify the marginal price-setter from dispatch + driver evidence.

    Called by why_builder.py for EXPLANATION and ACTION_RECOMMENDATION intents.
    Upgrades the fuel-mix answer from "prior cost model" to dispatch-backed evidence.

    Args:
        unit_events:   rows from UnitDispatchEvent (T9 scatter result)
        driver_events: rows from MarketDriverEvent (T10 scatter result)
        spot_price:    regional RRP from T1 ($/MWh)
        region:        NEM region code

    Returns MarginalSetterResult with data_tier indicating evidence quality.
    """
    if unit_events:
        return _from_dispatch(unit_events, driver_events, spot_price, region)
    return _from_priors(spot_price, region)


def _from_dispatch(
    unit_events: list[dict[str, Any]],
    driver_events: list[dict[str, Any]],
    spot_price: float,
    region: str,
) -> MarginalSetterResult:
    """Build marginal setter result from actual DUID dispatch rows."""
    from app.engines.unit_attribution import summarise_unit_dispatch

    summary = summarise_unit_dispatch(unit_events)
    by_fuel_raw: dict[str, dict[str, Any]] = summary.get("by_fuel", {})
    data_tier = "bid_reconstruction" if any(
        e.get("source") == "BID_RECONSTRUCTION" for e in unit_events
    ) else "dispatch"

    # Build FuelDispatchSummary per fuel type
    by_fuel_list: list[FuelDispatchSummary] = []
    total_dispatch = sum(b.get("total_cleared_mw", 0.0) for b in by_fuel_raw.values())

    for fuel, bucket in sorted(by_fuel_raw.items(), key=lambda kv: -kv[1].get("total_cleared_mw", 0)):
        top = bucket.get("top_units", [{}])[0]
        avail = bucket.get("availability_mw", 0.0)
        cleared = bucket.get("total_cleared_mw", 0.0)
        by_fuel_list.append(FuelDispatchSummary(
            fuel_type=fuel,
            dispatched_mw=round(cleared, 1),
            unit_count=bucket.get("unit_count", 0),
            headroom_mw=round(max(avail - cleared, 0.0), 1),
            is_renewable=fuel in _RENEWABLE_FUELS,
            top_unit=top.get("duid"),
            data_tier=data_tier,
        ))

    # Identify marginal setter: most expensive dispatched non-renewable fuel
    # (renewables bid near zero and are never the marginal setter)
    marginal_fuel = _identify_marginal_fuel(by_fuel_raw, spot_price, data_tier)
    fuel_bucket = by_fuel_raw.get(marginal_fuel, {})
    top_units = fuel_bucket.get("top_units", [])
    top_duid = top_units[0].get("duid") if top_units else None
    top_station = None
    if top_duid:
        for ev in unit_events:
            if ev.get("duid") == top_duid:
                top_station = ev.get("station_name")
                break

    # Constraint narrative
    constraint_narrative = _build_constraint_narrative(driver_events)
    interconnector_narrative = _build_interconnector_narrative(driver_events, region)

    explanation = _build_explanation(
        marginal_fuel, spot_price, fuel_bucket, total_dispatch,
        constraint_narrative, data_tier
    )

    return MarginalSetterResult(
        fuel_type=marginal_fuel,
        display_name=_FUEL_DISPLAY.get(marginal_fuel, marginal_fuel.capitalize()),
        duid=top_duid,
        station_name=top_station,
        estimated_price_mwh=_PRIOR_MARGINAL_COST.get(marginal_fuel, spot_price),
        certainty_tier=data_tier,
        explanation=explanation,
        by_fuel=by_fuel_list,
        constraint_narrative=constraint_narrative,
        interconnector_narrative=interconnector_narrative,
        data_tier=data_tier,
    )


def _from_priors(spot_price: float, region: str) -> MarginalSetterResult:
    """Fall back to marginal-cost priors when no dispatch data is available."""
    # Infer most likely marginal fuel from spot price bracket
    if spot_price < 0:
        fuel = "solar"          # solar flood + negative prices
        explanation = (
            f"{region} spot is ${spot_price:.0f}/MWh (negative). "
            "Excess renewable generation typical — solar or wind surplus forcing price below zero."
        )
    elif spot_price < 30:
        fuel = "wind"
        explanation = (
            f"{region} at ${spot_price:.0f}/MWh suggests wind or solar is setting price "
            "(prior cost model — no live unit dispatch data)."
        )
    elif spot_price < 80:
        fuel = "coal"
        explanation = (
            f"{region} at ${spot_price:.0f}/MWh is consistent with coal dispatch "
            "(marginal cost ~$55/MWh prior — no live unit dispatch data)."
        )
    elif spot_price < 200:
        fuel = "gas"
        explanation = (
            f"{region} at ${spot_price:.0f}/MWh is consistent with gas peaker dispatch "
            "(marginal cost ~$85/MWh prior — no live unit dispatch data)."
        )
    else:
        fuel = "gas"
        explanation = (
            f"{region} at ${spot_price:.0f}/MWh suggests gas peakers bidding at high prices "
            "or constraint-driven dispatch (prior model — no live data to confirm)."
        )

    return MarginalSetterResult(
        fuel_type=fuel,
        display_name=_FUEL_DISPLAY.get(fuel, fuel.capitalize()),
        duid=None,
        station_name=None,
        estimated_price_mwh=_PRIOR_MARGINAL_COST.get(fuel, spot_price),
        certainty_tier="prior",
        explanation=explanation,
        by_fuel=[],
        data_tier="prior",
    )


def _identify_marginal_fuel(
    by_fuel: dict[str, dict[str, Any]],
    spot_price: float,
    data_tier: str,
) -> str:
    """Find the most expensive dispatched fuel type that plausibly set the price.

    Logic: Sort non-renewable fuels by their prior marginal cost. The most expensive
    dispatched non-renewable fuel is the likely price setter.
    """
    dispatchable_fuels = {
        f: b for f, b in by_fuel.items()
        if f not in _RENEWABLE_FUELS and b.get("total_cleared_mw", 0) > 0
    }
    if not dispatchable_fuels:
        # Only renewables dispatched → check for price anomaly
        if spot_price < 10:
            return "solar" if "solar" in by_fuel else "wind"
        return "unknown"

    # Sort by prior marginal cost descending — most expensive first
    ranked = sorted(
        dispatchable_fuels.keys(),
        key=lambda f: _PRIOR_MARGINAL_COST.get(f, 80.0),
        reverse=True,
    )
    return ranked[0]


def _build_explanation(
    fuel: str,
    spot_price: float,
    fuel_bucket: dict[str, Any],
    total_dispatch: float,
    constraint_narrative: str,
    data_tier: str,
) -> str:
    mw = fuel_bucket.get("total_cleared_mw", 0.0)
    units = fuel_bucket.get("unit_count", 0)
    display = _FUEL_DISPLAY.get(fuel, fuel.capitalize())
    tier_label = {
        "dispatch": "unit dispatch data",
        "bid_reconstruction": "bid reconstruction (archive-backed)",
        "prior": "prior cost model",
    }.get(data_tier, data_tier)

    parts = [
        f"{display} is the likely marginal setter at ${spot_price:.0f}/MWh "
        f"({mw:.0f} MW across {units} unit{'s' if units != 1 else ''}, "
        f"from {tier_label})."
    ]
    if constraint_narrative:
        parts.append(constraint_narrative)
    return " ".join(parts)


def _build_constraint_narrative(driver_events: list[dict[str, Any]]) -> str:
    """Summarise binding constraints from driver_events."""
    def _mv(ev: dict) -> float:
        # MarketDriverEvent stores marginal_value in values JSON dict
        return float(ev.get("values", {}).get("marginal_value") or ev.get("marginal_value") or 0)

    binding = [ev for ev in driver_events if ev.get("driver_type") == "constraint" and _mv(ev) > 0]
    if not binding:
        return ""
    top = sorted(binding, key=lambda e: -_mv(e))[:3]
    names = [e.get("element_id") or e.get("constraint_id", "unknown") for e in top]
    total_mv = sum(_mv(e) for e in top[:3])
    return (
        f"{len(binding)} constraint{'s' if len(binding) > 1 else ''} binding "
        f"({', '.join(names[:2])}{'...' if len(binding) > 2 else ''}); "
        f"combined shadow price ~${total_mv:.0f}/MWh adds to effective dispatch cost."
    )


def _build_interconnector_narrative(
    driver_events: list[dict[str, Any]], region: str
) -> str:
    """Summarise interconnector flows relevant to the query region."""
    ic_events = [
        ev for ev in driver_events
        if ev.get("driver_type") == "interconnector"
    ]
    if not ic_events:
        return ""
    at_limit = [ev for ev in ic_events if ev.get("at_export_limit") or ev.get("at_import_limit")]
    if not at_limit:
        flows = [(ev.get("interconnector_id", "?"), ev.get("metered_mw_flow", 0)) for ev in ic_events[:3]]
        parts = [f"{ic}: {flow:.0f} MW" for ic, flow in flows if flow is not None]
        return f"Interconnector flows: {'; '.join(parts)}." if parts else ""
    constrained = [ev.get("interconnector_id", "?") for ev in at_limit]
    return (
        f"Interconnector{'s' if len(constrained) > 1 else ''} "
        f"{', '.join(constrained[:2])} at limit — "
        "import/export capacity is constraining regional supply."
    )


def fuel_mix_from_dispatch(
    unit_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a fuel-mix breakdown from dispatch rows for the answer planner.

    Returns a dict matching the shape expected by plan_fuel_source():
      {
        "data_tier": "dispatch" | "bid_reconstruction" | "prior",
        "by_fuel": { "coal": {"dispatched_mw": 3200, ...}, ... },
        "spot_price_rrp": ...,
        "recommendation": { "fuel_type": ..., "action": ..., ... },
        "sources": [...]
      }

    This replaces the static prior-model recommendation when live dispatch data exists.
    """
    if not unit_events:
        return {}

    from app.engines.unit_attribution import summarise_unit_dispatch
    summary = summarise_unit_dispatch(unit_events)
    by_fuel_raw = summary.get("by_fuel", {})
    data_tier = "bid_reconstruction" if any(
        e.get("source") == "BID_RECONSTRUCTION" for e in unit_events
    ) else "dispatch"

    # Sort fuels by MW dispatched to find what's actually running
    ranked = sorted(by_fuel_raw.items(), key=lambda kv: -kv[1].get("total_cleared_mw", 0))

    sources = []
    for fuel, bucket in ranked:
        prior_cost = _PRIOR_MARGINAL_COST.get(fuel, 80.0)
        sources.append({
            "fuel_type": fuel,
            "mw_dispatched": round(bucket.get("total_cleared_mw", 0), 1),
            "mw_capacity": round(bucket.get("availability_mw", 0), 1),
            "unit_count": bucket.get("unit_count", 0),
            "marginal_cost_typical": prior_cost,
            "marginal_cost_low": max(prior_cost * 0.8, 0),
            "marginal_cost_high": prior_cost * 1.3,
            "data_tier": data_tier,
        })

    return {
        "data_tier": data_tier,
        "by_fuel": dict(ranked),
        "sources": sources,
    }
