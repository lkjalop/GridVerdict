"""Formatting and scoring helpers for answer_planner.py.

Extracted from answer_planner.py to keep the planner module under 1500 lines.
All functions here are pure (no DB, no HTTP, no LLM calls).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.agents.why_sources import WhySources
from app.core.schema import FactualVerdict


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
        key=lambda r: _sort_time(r["valid_time"]),
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


def _sort_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            return datetime.min
    return datetime.min


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


def _headroom_str(headroom_mw: float, demand_mw: float | None = None) -> str:
    """Format headroom with plain-English interpretation so non-engineers understand it.

    Headroom = available generation − demand = the grid's spare capacity buffer.
    Low headroom means generators are near their limit and can bid higher.
    """
    hw = headroom_mw
    if hw >= 6000:
        ctx = "ample — well below peak capacity, prices stable"
    elif hw >= 3500:
        ctx = "comfortable — normal operating zone"
    elif hw >= 1500:
        ctx = "moderate — watch for demand surge or generator trips"
    elif hw >= 500:
        ctx = "tight — elevated price risk if another unit trips"
    else:
        ctx = "critical — near grid limit, spike risk"
    return f"{hw:,.0f} MW spare capacity ({ctx})"


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

    # Surface LNN spike-risk head probabilities when available
    _spike_model = next(
        (m for m in detail if m.model in {"lnn", "lnn_cfc"} and m.spike_probs),
        None,
    )
    if _spike_model and _spike_model.spike_probs:
        sp = _spike_model.spike_probs
        _spike_parts = []
        if sp.get("gt_300") is not None and sp["gt_300"] >= 0.05:
            _spike_parts.append(f"P(>$300) {sp['gt_300']:.0%}")
        if sp.get("gt_1000") is not None and sp["gt_1000"] >= 0.02:
            _spike_parts.append(f"P(>$1000) {sp['gt_1000']:.0%}")
        if sp.get("lt_0") is not None and sp["lt_0"] >= 0.05:
            _spike_parts.append(f"P(<$0) {sp['lt_0']:.0%}")
        if _spike_parts:
            lines.append("LNN spike-risk: " + ", ".join(_spike_parts) + ".")

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
    hist_dist: dict[str, Any] | None = None,
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
            for m in (sources.forecast.model_detail or [])
        ],
        "analogs": analogs or sources.analogs.top_items,
        "temporal_evidence": temporal_evidence or [],
        "evidence_quality": evidence_quality or {},
        "provenance": provenance or [],
        "fuel_mix": fuel_mix or {},
        "hist_dist": hist_dist or {},
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


def _normal_source_cost_line(fuel_mix: dict[str, Any]) -> str:
    items = [
        item for item in (fuel_mix.get("sources") or [])
        if item.get("fuel_type") in {"solar", "wind", "coal", "hydro", "gas"}
           and item.get("marginal_cost_typical") is not None
    ]
    if not items:
        return "Normal source-cost ranking is unavailable because fuel priors are missing."
    ranked = sorted(items, key=lambda item: float(item.get("marginal_cost_typical") or 0))
    top = ", ".join(
        f"{item['fuel_type']} ~${float(item['marginal_cost_typical']):.0f}/MWh"
        for item in ranked[:4]
    )
    return f"Normal marginal-cost priors rank cheapest as: {top}."


def _source_price_benchmark_line(fuel_mix: dict[str, Any], best: str, spot: Any) -> str:
    for item in fuel_mix.get("sources") or []:
        if item.get("fuel_type") != best:
            continue
        low = item.get("marginal_cost_low")
        high = item.get("marginal_cost_high")
        if low is None or high is None:
            return ""
        try:
            spot_f = float(spot)
        except (TypeError, ValueError):
            return ""
        if spot_f > float(high):
            relation = "above"
        elif spot_f < float(low):
            relation = "below"
        else:
            relation = "inside"
        return (
            f"Good-price benchmark for {best}: prior band ${float(low):.0f}-${float(high):.0f}/MWh; "
            f"current spot ${spot_f:.0f}/MWh is {relation} that band."
        )
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


# ── Per-sub-question confidence scoring (Sprint U) ───────────────────────────

_SQ_EVIDENCE_REQUIREMENTS: dict[str, Any] = {
    "current_price_reason":          lambda s: 0.90 if s.current.is_fresh else 0.30,
    "fuel_source_comparison":        lambda s: 0.70 if s.technology.has_unit_evidence else 0.35,
    "historical_price_distribution": lambda s: (
        0.80 if getattr(s, "hist_dist_available", False)
        else 0.10
    ),
    "forecast_outlook":              lambda s: 0.75 if s.forecast.available else 0.20,
    "price_forecast":                lambda s: 0.75 if s.forecast.available else 0.20,
    "regime_change":                 lambda s: 0.70 if s.analogs.count >= 5 else 0.15,
    "fcas_opportunity":              lambda s: 0.80 if s.fcas.available else 0.10,
    "interconnector_causality":      lambda s: (
        0.65 if s.drivers.tight_interconnectors else 0.20
    ),
    "price_fluctuation":             lambda s: 0.80 if s.current.is_fresh else 0.25,
    "market_status":                 lambda s: 0.90 if s.current.is_fresh else 0.30,
}


def score_sub_questions(sources: "WhySources") -> dict[str, float]:
    """Return coverage score per sub-question type (0.0–1.0).

    The headline confidence should be min(scores) — the weakest sub-question
    drives the overall answer quality. This makes it visible to the professional
    which part of a multi-part question has thin evidence.
    """
    return {
        sq["type"]: _SQ_EVIDENCE_REQUIREMENTS.get(sq["type"], lambda s: 0.50)(sources)
        for sq in (sources.decomp.sub_questions or [])
        if sq.get("type")
    }


def apply_sub_question_scores(factual: "FactualVerdict", sources: "WhySources") -> "FactualVerdict":
    """Compute per-sub-question confidence and set it on the verdict.

    When sub_questions are present, also adjusts the headline confidence to
    min(sub_question_scores) so the weakest-evidenced part drives the display.
    The prior confidence (from why_builder) remains in why_builder's estimate;
    this function only adjusts when sub_questions push confidence down further.
    """
    scores = score_sub_questions(sources)
    if not scores:
        return factual
    min_score = min(scores.values())
    # Only pull confidence down — never inflate above what why_builder set
    new_conf = round(min(factual.confidence, min_score), 3)
    return factual.model_copy(update={
        "sub_question_scores": scores,
        "confidence": new_conf,
    })
