"""Deterministic answer planner for concise user-facing responses.

The WhyEngine builds the full audit narrative. This planner builds the short
visible answer. It does not fetch data, call an LLM, or invent facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.agents.why_sources import WhySources
from app.core.schema import FactualVerdict, IntentLabel


@dataclass
class PlannedAnswer:
    headline: str
    direct_answer: list[str] = field(default_factory=list)
    key_evidence: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    continuation: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def sections(self) -> list[dict[str, Any]]:
        sections = [
            ("Answer", self.direct_answer),
            ("Evidence", self.key_evidence),
            ("Drivers", self.drivers),
            ("Continuation", self.continuation),
            ("Missing", self.missing),
        ]
        return [
            {"title": title, "items": _cap(items)}
            for title, items in sections
            if items
        ]

    def details_payload(self) -> dict[str, Any]:
        return {"headline": self.headline, **self.details}


def plan_answer(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    analogs: list[dict[str, Any]] | None = None,
    evidence_quality: dict[str, Any] | None = None,
    temporal_evidence: list[dict[str, Any]] | None = None,
    provenance: list[dict[str, Any]] | None = None,
    fuel_mix: dict[str, Any] | None = None,
) -> PlannedAnswer:
    """Build concise visible answer sections from approved evidence."""
    requested = (sources.decomp.requested_output or "").lower()
    query = (sources.decomp.raw_query or "").lower()

    if requested == "data_freshness_status" or "stale" in query:
        return _plan_data_status(sources, factual, evidence_quality, provenance)
    if requested == "regional_comparison":
        return _plan_comparison(sources, factual)
    if requested == "historical_analog_outcome":
        return _plan_retrospective(sources, factual, analogs)
    if requested == "weather_notice_news_correlation":
        return _plan_weather_news(sources, factual)
    if requested == "price_fluctuation_attribution":
        return _plan_price_fluctuation(sources, factual, fuel_mix)
    if requested == "fuel_source_recommendation":
        return _plan_fuel_source(sources, factual, fuel_mix)
    if requested == "portfolio_action":
        return _plan_portfolio_or_action(sources, factual)
    # requires_forecast is a data signal, not an intent override — keep this one.
    if requested == "causal_explanation_with_forecast" or sources.decomp.requires_forecast:
        return _plan_explanation(sources, factual, include_forecast=True)
    if requested == "causal_explanation":
        return _plan_explanation(sources, factual, include_forecast=False)
    return _plan_lookup(sources, factual)


def apply_plan_to_verdict(factual: FactualVerdict, plan: PlannedAnswer) -> FactualVerdict:
    """Return a new verdict with planner sections and detail payload attached."""
    return factual.model_copy(update={
        "answer_sections": plan.sections(),
        "answer_details": plan.details_payload(),
    })


def _plan_explanation(sources: WhySources, factual: FactualVerdict, *, include_forecast: bool) -> PlannedAnswer:
    c = sources.current
    headline = f"{c.region} price is {c.regime}, but the primary driver is not confirmed."
    direct = [
        f"{c.region} is ${c.price_rrp:.2f}/MWh with {c.headroom_mw:.0f} MW headroom.",
    ]
    trend = _trend_line(sources)
    if trend:
        direct.append(trend)
    analog_line = _analog_outcome_line(sources)
    if analog_line:
        direct.append(analog_line)

    evidence = [
        f"Live dispatch: demand {c.demand_mw:.0f} MW, available generation {c.availability_mw:.0f} MW.",
    ]
    if sources.news.explained:
        evidence.append(f"AEMO notice present: {sources.news.top_notice_type}.")
    elif sources.news.notices_stale:
        evidence.append("AEMO notice context is stale or unavailable.")
    if sources.weather.relevant and sources.weather.available:
        evidence.append(_weather_line(sources))

    drivers = _driver_lines(sources)
    continuation = _continuation_lines(sources) if include_forecast else []
    missing = _missing_lines(factual)
    if trend:
        missing = [m for m in missing if m != "recent price trend"]

    return PlannedAnswer(
        headline=headline,
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=continuation,
        missing=missing,
        details=_details(sources, factual),
    )


def _plan_retrospective(
    sources: WhySources,
    factual: FactualVerdict,
    analogs: list[dict[str, Any]] | None,
) -> PlannedAnswer:
    a = sources.analogs
    headline = (
        f"Yes: {a.count} comparable historical states were found."
        if a.count else "No close historical analog set was found."
    )
    direct = [
        f"HippoGraph found {a.count} comparable states."
        if a.count else "HippoGraph did not return enough comparable states.",
    ]
    if a.outcome_summary:
        direct.append(f"Outcome split: {a.outcome_summary}.")
    examples = _top_analog_lines(analogs or a.top_items)
    direct.extend(examples[:1])

    evidence = examples[1:3] or [
        "Analog retrieval compares price, demand, headroom, and regime state."
    ]
    drivers = [
        "This is a base-rate comparison, not proof that the same driver is active now."
    ]
    missing = _missing_lines(factual)
    return PlannedAnswer(
        headline=headline,
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=[],
        missing=missing,
        details=_details(sources, factual, analogs=analogs),
    )


def _plan_weather_news(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    headline = "Weather/news context is present, but causality is limited."
    direct: list[str] = []
    if sources.weather.available:
        direct.append(_weather_support_line(sources))
    else:
        direct.append("Weather evidence is unavailable for this query.")
    if sources.news.explained:
        direct.append(f"AEMO notice context is present: {sources.news.top_notice_type}.")
    else:
        direct.append("No relevant AEMO notice confirms the price move.")
    if sources.news.commentary_items:
        direct.append("RSS/news matched NEM keywords, but is contextual rather than confirmed driver evidence.")

    evidence = []
    if sources.weather.available:
        evidence.append(_weather_line(sources))
    if sources.news.commentary_items:
        evidence.append(f"Top RSS item: {sources.news.commentary_items[0].get('title', 'untitled')}.")

    return PlannedAnswer(
        headline=headline,
        direct_answer=direct,
        key_evidence=evidence,
        drivers=_driver_lines(sources),
        continuation=[],
        missing=_missing_lines(factual),
        details=_details(sources, factual),
    )


def _plan_price_fluctuation(
    sources: WhySources,
    factual: FactualVerdict,
    fuel_mix: dict[str, Any] | None,
) -> PlannedAnswer:
    c = sources.current
    trend = _trend_line(sources)
    movement = _price_movement_line(sources)

    # Extract prices the user described in the query when DB history is unavailable
    query_prices = _extract_query_price_path(sources.decomp.raw_query or "")

    if movement:
        direct = [movement]
    elif query_prices:
        path_str = " → ".join(f"${p:.0f}" for p in query_prices)
        swing = max(query_prices) - min(query_prices)
        direct = [
            f"You described a price path of {path_str}/MWh; live reading is ${c.price_rrp:.2f}/MWh.",
            f"The observed swing of ${swing:.0f}/MWh is consistent with a {_swing_label(swing)} event.",
        ]
    else:
        direct = [
            f"{c.region} is now ${c.price_rrp:.2f}/MWh; recent price history is not yet in the database.",
        ]

    if trend:
        direct.append(trend)
    direct.append(
        "Confirmed driver and fuel-source attribution requires unit dispatch, bids/rebids, and constraint evidence."
    )

    evidence = [
        f"Current dispatch: demand {c.demand_mw:.0f} MW, available generation {c.availability_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW.",
    ]
    if sources.weather.available:
        evidence.append(_weather_line(sources))
    fuel_line = _fuel_driver_context_line(fuel_mix)
    if fuel_line:
        evidence.append(fuel_line)

    drivers = _driver_lines(sources)
    fuel_reason = _fuel_reason_line(fuel_mix)
    if fuel_reason:
        drivers.insert(0, fuel_reason)

    missing = _missing_lines(factual)
    # Surface the dispatch history gap explicitly so the user understands why
    # the price path cannot be verified from the DB yet.
    if not movement and not sources.recent_dispatch:
        if not any("price trend" in m or "dispatch" in m for m in missing):
            missing = ["recent price trend from database"] + missing

    return PlannedAnswer(
        headline=f"{c.region} price moved recently, but the driver is not confirmed.",
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=[],
        missing=missing[:3],
        details=_details(sources, factual, fuel_mix=fuel_mix),
    )


def _plan_comparison(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    regions = sources.decomp.entities.get("regions") or [sources.current.region]
    headline = f"Comparison requested for {', '.join(regions)}."
    return PlannedAnswer(
        headline=headline,
        direct_answer=[
            f"Comparison requested across {len(regions)} NEM region(s): {', '.join(regions)}.",
            "Use the comparison table for region-by-region price, demand, and headroom.",
        ],
        key_evidence=[
            "Each region is fetched through the same live dispatch evidence path.",
        ],
        drivers=[
            "Regional price separation normally requires constraint and interconnector evidence before assigning cause.",
        ],
        continuation=[],
        missing=_missing_lines(factual),
        details=_details(sources, factual),
    )


def _plan_data_status(
    sources: WhySources,
    factual: FactualVerdict,
    evidence_quality: dict[str, Any] | None,
    provenance: list[dict[str, Any]] | None,
) -> PlannedAnswer:
    eq = evidence_quality or {}
    stale: list[str] = []
    fresh: list[str] = []
    for key, val in eq.items():
        if not isinstance(val, dict):
            continue
        status = str(val.get("status") or "")
        if status in {"fresh", "ok", "found"}:
            fresh.append(key)
        elif status in {"stale", "offline", "none", "thin"}:
            stale.append(f"{key}: {status}")
    if not stale and provenance:
        for src in provenance:
            if not src.get("is_available", True):
                stale.append(f"{src.get('source', 'source')}: unavailable")

    return PlannedAnswer(
        headline="Source freshness summary.",
        direct_answer=stale[:3] or ["No stale source was flagged in the query evidence bundle."],
        key_evidence=fresh[:3] or ["Fresh-source details are available in the Data panel."],
        drivers=[
            "Freshness affects confidence because stale or missing sources block causal claims."
        ],
        continuation=[],
        missing=_missing_lines(factual),
        details=_details(sources, factual, evidence_quality=evidence_quality, provenance=provenance),
    )


def _plan_fuel_source(
    sources: WhySources,
    factual: FactualVerdict,
    fuel_mix: dict[str, Any] | None,
) -> PlannedAnswer:
    c = sources.current
    requested = sources.decomp.entities.get("technologies") or []
    mix = fuel_mix or {}
    rec = mix.get("recommendation") or {}
    preferred = rec.get("preferred_order") or []
    best = rec.get("fuel_type") or "unknown"
    spot = mix.get("spot_price_rrp", c.price_rrp)
    data_tier = mix.get("data_tier") or "unknown"

    direct = []
    if "coal" in requested and best != "coal":
        direct.append(
            f"No: based on the current source model, coal is not the preferred source now; {best} ranks ahead at ${float(spot):.2f}/MWh."
        )
    else:
        direct.append(
            f"The source model currently prefers {best} at ${float(spot):.2f}/MWh."
        )
    if preferred:
        direct.append("Preferred order: " + " > ".join(preferred) + ".")
    if rec.get("reason"):
        direct.append(str(rec["reason"]))

    evidence = [
        f"NEM spot price is ${c.price_rrp:.2f}/MWh; all fuel types clear at the regional spot price.",
        f"Fuel-mix evidence tier is {data_tier}; confidence is {rec.get('confidence', 'low')}.",
    ]
    for note in (rec.get("notes") or [])[:2]:
        evidence.append(str(note))
    for fuel in requested[:3]:
        line = _fuel_compare_line(mix, fuel)
        if line:
            evidence.append(line)

    drivers = [
        "This is a procurement/source suitability answer, not proof that one fuel caused the NSW spot price.",
        "At elevated prices, dispatchable flexibility and opportunity cost matter more than simple marginal-cost ranking.",
    ]
    missing = [
        "unit dispatch by fuel" if data_tier != "dispatch" else "",
        "bid/rebid stack by fuel",
        "retail contract or PPA terms",
    ]
    return PlannedAnswer(
        headline=f"{best.upper()} is the current preferred source, with low-to-medium evidence.",
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=[],
        missing=[m for m in missing if m],
        details=_details(sources, factual, fuel_mix=fuel_mix),
    )


def _plan_portfolio_or_action(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    c = sources.current
    return PlannedAnswer(
        headline="Action answer is advisory only.",
        direct_answer=[
            f"Current market state: ${c.price_rrp:.2f}/MWh, {c.headroom_mw:.0f} MW headroom.",
            f"Suggested action label: {factual.action.value.replace('_', ' ')}.",
        ],
        key_evidence=[f"Confidence is {factual.confidence:.0%}; check missing-before-action items before acting."],
        drivers=_driver_lines(sources),
        continuation=_continuation_lines(sources),
        missing=_missing_lines(factual),
        details=_details(sources, factual),
    )


def _plan_lookup(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    c = sources.current
    return PlannedAnswer(
        headline=f"{c.region} current market state.",
        direct_answer=[
            f"{c.region}: ${c.price_rrp:.2f}/MWh, demand {c.demand_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW.",
            f"Regime: {c.regime}.",
        ],
        key_evidence=[f"Dispatch interval: {_fmt_time(c.valid_time)}."],
        drivers=[],
        continuation=[],
        missing=_missing_lines(factual),
        details=_details(sources, factual),
    )


def _trend_line(sources: WhySources) -> str:
    rows = sorted(
        [r for r in sources.recent_dispatch if r.get("valid_time")],
        key=lambda r: r["valid_time"],
        reverse=True,
    )
    if not rows:
        return ""
    current = sources.current
    bits = [f"now ${current.price_rrp:.0f}"]
    for label, minutes in [("5m", 5), ("10m", 10), ("60m", 60)]:
        row = _nearest_row(rows, current.valid_time, minutes)
        if row and row.get("price_rrp") is not None:
            delta = current.price_rrp - float(row["price_rrp"])
            bits.append(f"{label} ago ${float(row['price_rrp']):.0f} ({delta:+.0f})")
    return "Recent price path: " + "; ".join(bits) + "."


def _price_movement_line(sources: WhySources) -> str:
    rows = sorted(
        [r for r in sources.recent_dispatch if r.get("valid_time") and r.get("price_rrp") is not None],
        key=lambda r: r["valid_time"],
    )
    if not rows:
        return ""
    current = sources.current
    points = []
    for row in rows[-6:]:
        vt = row.get("valid_time")
        label = _short_time(vt)
        points.append(f"{label} ${float(row['price_rrp']):.0f}")
    points.append(f"now ${current.price_rrp:.0f}")
    prices = [float(r["price_rrp"]) for r in rows[-6:]] + [current.price_rrp]
    swing = max(prices) - min(prices) if prices else 0.0
    return f"Recent movement: {' -> '.join(points)}; observed swing about ${swing:.0f}/MWh."


def _nearest_row(rows: list[dict[str, Any]], anchor: datetime, minutes: int) -> dict[str, Any] | None:
    target_seconds = minutes * 60
    best = None
    best_delta = 10**9
    if anchor.tzinfo is not None:
        anchor = anchor.replace(tzinfo=None)
    for row in rows:
        vt = row.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
            except ValueError:
                continue
        if not isinstance(vt, datetime):
            continue
        if vt.tzinfo is not None:
            vt = vt.replace(tzinfo=None)
        age = (anchor - vt).total_seconds()
        if age < 60:
            continue
        delta = abs(age - target_seconds)
        if delta < best_delta:
            best = row
            best_delta = delta
    return best if best_delta <= 15 * 60 else None


def _analog_outcome_line(sources: WhySources) -> str:
    a = sources.analogs
    if a.count <= 0:
        return ""
    outcome = f" {a.outcome_summary}" if a.outcome_summary else ""
    return f"Historical analogs: {a.count} matched.{outcome}"


def _top_analog_lines(items: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in (items or [])[:3]:
        price = item.get("price_rrp")
        when = item.get("valid_time") or "unknown time"
        outcome = item.get("outcome") or "outcome unknown"
        if price is None:
            continue
        lines.append(f"{when}: ${float(price):.0f}/MWh, {outcome}.")
    return lines


def _weather_line(sources: WhySources) -> str:
    w = sources.weather
    c = w.consensus
    parts = []
    if c.get("temperature_c") is not None:
        parts.append(f"{float(c['temperature_c']):.1f}C")
    if c.get("wind_speed_kmh") is not None:
        parts.append(f"wind {float(c['wind_speed_kmh']):.1f} km/h")
    if c.get("cloud_cover_pct") is not None:
        parts.append(f"cloud {float(c['cloud_cover_pct']):.0f}%")
    return f"Weather consensus: {', '.join(parts) or 'no strong signal'} across {w.source_count} source(s)."


def _weather_support_line(sources: WhySources) -> str:
    c = sources.weather.consensus
    temp = c.get("temperature_c")
    wind = c.get("wind_speed_kmh")
    if temp is not None and float(temp) >= 35:
        return f"Hot weather supports demand pressure: {float(temp):.1f}C."
    if wind is not None and float(wind) <= 12:
        return f"Weather context: low wind may be relevant for renewable output at {float(wind):.1f} km/h."
    return "Weather is contextual, but does not by itself confirm the price driver."


def _driver_lines(sources: WhySources) -> list[str]:
    supported = []
    unconfirmed = []
    if sources.current.is_fresh:
        supported.append("live dispatch price/demand/headroom")
    if sources.news.explained:
        supported.append("AEMO notice")
    if sources.drivers.binding_constraints:
        supported.append("binding constraints")
    if sources.drivers.tight_interconnectors:
        supported.append("interconnector congestion")
    if sources.technology.has_unit_evidence:
        supported.append("unit dispatch")
    if sources.analogs.count >= 3:
        supported.append("historical analog pattern")

    for name, present in [
        ("constraints", bool(sources.drivers.binding_constraints)),
        ("interconnectors", bool(sources.drivers.tight_interconnectors)),
        ("unit dispatch", sources.technology.has_unit_evidence),
        ("AEMO notice", sources.news.explained),
    ]:
        if not present:
            unconfirmed.append(name)

    lines = []
    if supported:
        lines.append("Supported/plausible: " + ", ".join(supported[:5]) + ".")
    if unconfirmed:
        lines.append("Unconfirmed blockers: " + ", ".join(unconfirmed[:5]) + ".")
    if not lines:
        lines.append("No causal driver is confirmed from the current evidence.")
    return lines


def _continuation_lines(sources: WhySources) -> list[str]:
    detail = [m for m in sources.forecast.model_detail if m.available and m.p50 is not None]
    if not detail:
        if sources.forecast.available and sources.forecast.direction != "unknown":
            return [f"Only a weak {sources.forecast.direction} fallback signal is available."]
        return ["No calibrated LEAR/QRA/LNN continuation signal is available."]
    lines = []
    falling = rising = flat = 0
    parts = []
    for m in detail:
        if m.direction == "falling":
            falling += 1
        elif m.direction == "rising":
            rising += 1
        else:
            flat += 1
        label = _model_label(m.model)
        if m.p90 is not None:
            parts.append(f"{label} {m.direction}, P50 ${m.p50:.0f}, P90 ${m.p90:.0f}")
        else:
            parts.append(f"{label} {m.direction}, P50 ${m.p50:.0f}")
    if falling > rising and falling >= flat:
        call = "Available models lean lower."
    elif rising > falling and rising >= flat:
        call = "Available models lean higher."
    else:
        call = "Available models are mixed or flat."
    lines.append(call + " " + "; ".join(parts[:3]) + ".")
    disabled = [m for m in sources.forecast.model_detail if not m.available and m.model in {"lnn", "tcn"}]
    if disabled:
        lines.append("; ".join(f"{_model_label(m.model)} unavailable: {m.caveat}" for m in disabled[:2]) + ".")
    return lines


def _missing_lines(factual: FactualVerdict) -> list[str]:
    gaps = list(dict.fromkeys((factual.missing_data or []) + (factual.known_missing_before_action or [])))
    friendly = [_friendly_missing(g) for g in gaps if g]
    return friendly[:3] or ["No major missing-data blocker flagged."]


def _friendly_missing(gap: str) -> str:
    return gap.replace("_", " ")


def _details(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    analogs: list[dict[str, Any]] | None = None,
    evidence_quality: dict[str, Any] | None = None,
    temporal_evidence: list[dict[str, Any]] | None = None,
    provenance: list[dict[str, Any]] | None = None,
    fuel_mix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_refs": [e.model_dump(mode="json") for e in factual.evidence_refs],
        "claim_map": [c.model_dump(mode="json") for c in factual.claim_map],
        "models": [
            {
                "model": m.model,
                "available": m.available,
                "direction": m.direction,
                "p10": m.p10,
                "p50": m.p50,
                "p90": m.p90,
                "caveat": m.caveat,
            }
            for m in sources.forecast.model_detail
        ],
        "analogs": analogs or sources.analogs.top_items,
        "temporal_evidence": temporal_evidence or [],
        "evidence_quality": evidence_quality or {},
        "provenance": provenance or [],
        "fuel_mix": fuel_mix or {},
    }


def _fuel_compare_line(fuel_mix: dict[str, Any], fuel: str) -> str:
    for item in fuel_mix.get("sources") or []:
        if item.get("fuel_type") != fuel:
            continue
        tier = item.get("data_tier") or "unknown"
        typical = item.get("marginal_cost_typical")
        mw = item.get("mw_dispatched")
        cap = item.get("mw_capacity")
        mw_part = (
            f"dispatch {float(mw):.0f} MW"
            if mw is not None else (
                f"capacity {float(cap):.0f} MW" if cap else "no current MW evidence"
            )
        )
        return f"{fuel}: typical marginal-cost prior ${float(typical):.0f}/MWh, {mw_part}, data tier {tier}."
    return ""


def _fuel_driver_context_line(fuel_mix: dict[str, Any] | None) -> str:
    if not fuel_mix:
        return ""
    rec = fuel_mix.get("recommendation") or {}
    preferred = rec.get("preferred_order") or []
    best = rec.get("fuel_type")
    if not best:
        return ""
    order = f" Preferred order: {' > '.join(preferred)}." if preferred else ""
    return f"Fuel-source model currently ranks {best} first.{order}"


def _fuel_reason_line(fuel_mix: dict[str, Any] | None) -> str:
    if not fuel_mix:
        return ""
    rec = fuel_mix.get("recommendation") or {}
    reason = rec.get("reason")
    notes = rec.get("notes") or []
    if reason and notes:
        return f"Fuel/weather context: {reason} {notes[0]}"
    if reason:
        return f"Fuel/weather context: {reason}"
    return ""


def _short_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%H:%M")
        except ValueError:
            return value[11:16] if len(value) >= 16 else value
    return "t"


def _model_label(model: str) -> str:
    return {
        "meta_ensemble": "Meta-ensemble",
        "lear": "LEAR",
        "qra": "QRA",
        "lnn": "LNN",
        "tcn": "TCN",
        "gbm": "GBM",
        "predispatch": "AEMO pre-dispatch",
        "aemo_predispatch": "AEMO model",
    }.get(model, model.upper())


def _fmt_time(dt: datetime) -> str:
    return dt.isoformat()


def _extract_query_price_path(query: str) -> list[float]:
    """Extract sequential price values from a fluctuation query.

    Matches numbers following transition words (from/to/at) or dollar signs,
    e.g. "from 143 to 167 to 165 back down to 140" → [143.0, 167.0, 165.0, 140.0].
    Filters to the plausible NEM spot-price range (0–15500 $/MWh).
    """
    import re
    matches = re.findall(
        r'(?:from|to|at|\$)\s*(\d{1,5}(?:\.\d{1,2})?)',
        query.lower(),
    )
    prices: list[float] = []
    seen: set[float] = set()
    for m in matches:
        try:
            val = float(m)
        except ValueError:
            continue
        if 0.0 <= val <= 15500.0 and val not in seen:
            seen.add(val)
            prices.append(val)
    return prices


def _swing_label(swing: float) -> str:
    if swing >= 500:
        return "significant price excursion"
    if swing >= 100:
        return "moderate price swing"
    return "minor price variation"


def _cap(items: list[str], n: int = 3) -> list[str]:
    clean = [str(i) for i in items if str(i).strip()]
    return clean[:n]
