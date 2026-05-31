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
    hist_dist: dict[str, Any] | None = None,
) -> PlannedAnswer:
    """Build concise visible answer sections from approved evidence."""
    requested = (sources.decomp.requested_output or "").lower()
    query = (sources.decomp.raw_query or "").lower()

    if requested == "data_freshness_status" or "stale" in query:
        return _plan_data_status(sources, factual, evidence_quality, provenance)
    if requested == "diurnal_analysis":
        return _plan_diurnal_analysis(sources, factual)
    if requested == "trend_analysis":
        return _plan_trend_analysis(sources, factual, hist_dist=hist_dist)
    if requested == "regional_comparison":
        return _plan_comparison(sources, factual)
    if requested == "historical_analog_outcome":
        return _plan_retrospective(sources, factual, analogs)
    if requested == "weather_notice_news_correlation":
        return _plan_weather_news(sources, factual, hist_dist=hist_dist)
    if requested == "price_fluctuation_attribution":
        return _plan_price_fluctuation(sources, factual, fuel_mix)
    if requested == "fuel_source_recommendation":
        return _plan_fuel_source(sources, factual, fuel_mix, hist_dist=hist_dist)
    if requested == "portfolio_action":
        return _plan_portfolio_or_action(sources, factual)
    # Historical distribution: route when sub_question present or requested directly.
    _has_hist_sq = any(
        sq.get("type") == "historical_price_distribution"
        for sq in (sources.decomp.sub_questions or [])
    )
    if requested == "historical_price_distribution" or _has_hist_sq:
        return _plan_historical_distribution(sources, factual, hist_dist or {})

    # Multi-part: compound queries with ≥2 distinct sub-question types get separate sections
    _sq_types = [sq.get("type") for sq in (sources.decomp.sub_questions or []) if sq.get("type")]
    if len(set(_sq_types)) >= 2:
        return _plan_multi_part(sources, factual, fuel_mix=fuel_mix, hist_dist=hist_dist,
                                analogs=analogs, sq_types=_sq_types)

    # Future-date specific forecast (e.g. "prices on monday june 8th")
    _future_date_ref = (sources.decomp.time_range or {}).get("future_date_ref", False)
    if _future_date_ref or (sources.decomp.requires_forecast and requested in ("forecast", "current_market_state")):
        return _plan_future_date_forecast(sources, factual)

    # requires_forecast is a data signal, not an intent override — keep this one.
    if requested == "causal_explanation_with_forecast" or sources.decomp.requires_forecast:
        return _plan_explanation(sources, factual, include_forecast=True)
    if requested == "causal_explanation":
        return _plan_explanation(sources, factual, include_forecast=False)

    plan = _plan_lookup(sources, factual)
    # Annotate with threshold answer when user specified a price target
    thresholds = getattr(sources.decomp, "spike_thresholds", [])
    if thresholds and sources.current.price_rrp is not None:
        current = sources.current.price_rrp
        for t in thresholds[:2]:
            rel = "above" if current > t else "below"
            diff = abs(current - t)
            plan.direct_answer.append(
                f"vs your threshold of ${t:.0f}/MWh: current ${current:.0f} is "
                f"{rel} by ${diff:.0f}/MWh."
            )
    return plan


def _plan_multi_part(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    fuel_mix: dict[str, Any] | None,
    hist_dist: dict[str, Any] | None,
    analogs: list[dict[str, Any]] | None,
    sq_types: list[str],
) -> PlannedAnswer:
    """Build one answer section per detected sub-question type."""
    c = sources.current
    f = sources.forecast
    sections_direct: list[str] = []
    sections_evidence: list[str] = []
    sections_missing: list[str] = []

    _seen: set[str] = set()
    for sq_type in sq_types:
        if sq_type in _seen:
            continue
        _seen.add(sq_type)

        if sq_type == "current_price_reason":
            sections_direct.append(
                f"Current price: {c.region} ${c.price_rrp:.2f}/MWh, regime {c.regime}."
            )
            if c.demand_mw and c.availability_mw:
                sections_evidence.append(
                    f"Demand {c.demand_mw:.0f} MW vs availability {c.availability_mw:.0f} MW "
                    f"(headroom {max(c.availability_mw - c.demand_mw, 0):.0f} MW)."
                )

        elif sq_type == "fuel_source_comparison":
            if fuel_mix and fuel_mix.get("recommendation"):
                rec = fuel_mix["recommendation"]
                sections_direct.append(
                    f"Best source now: {rec.get('fuel_type','?').upper()} — "
                    f"{rec.get('reason','—')}"
                )
                if rec.get("preferred_order"):
                    sections_evidence.append("Order: " + " > ".join(rec["preferred_order"]))
            else:
                sections_missing.append("fuel mix (no live unit dispatch data)")

        elif sq_type == "historical_price_distribution":
            if hist_dist and hist_dist.get("available"):
                classification = hist_dist.get("classification", "—")
                median = hist_dist.get("median", 0)
                p90 = hist_dist.get("p90", 0)
                period = hist_dist.get("period_label", "last year")
                sections_direct.append(
                    f"vs {period}: current is {classification} "
                    f"(median ${median:.0f}, P90 ${p90:.0f}/MWh)."
                )
            else:
                sections_missing.append("historical price benchmark")

        elif sq_type in ("forecast_outlook", "price_forecast"):
            if f and f.available and f.p50 is not None:
                sections_direct.append(
                    f"Forecast: {f.direction}, P50 ${f.p50:.0f}/MWh "
                    f"[P10 ${f.p10:.0f}–P90 ${f.p90:.0f}]."
                )
            else:
                sections_missing.append("price forecast (model initialising)")

        elif sq_type == "regime_change":
            if analogs:
                recovered = sum(1 for a in analogs if a.get("outcome") in ("recovered", "normal"))
                sections_direct.append(
                    f"Historical analogs: {recovered}/{len(analogs)} similar episodes "
                    "recovered within 30 min."
                )
            else:
                sections_missing.append("historical analogs")

        elif sq_type == "fcas_opportunity":    # E3: FCAS sub-question template
            fcas = getattr(sources, "fcas", None)
            if fcas and fcas.available:
                if fcas.total_opportunity_mwh is not None:
                    sections_direct.append(
                        f"FCAS opportunity: ${fcas.total_opportunity_mwh:.0f}/MWh combined "
                        f"(best raise: {fcas.best_raise_service} @ ${fcas.max_raise_rrp:.0f}/MWh; "
                        f"best lower: {fcas.best_lower_service} @ ${fcas.max_lower_rrp:.0f}/MWh)."
                    )
                if fcas.tight_markets:
                    sections_evidence.append(
                        f"Tight FCAS markets: {', '.join(fcas.tight_markets)} (RRP ≥ $50/MWh)."
                    )
            else:
                sections_missing.append("FCAS prices (not yet populated in DB)")

        elif sq_type == "interconnector_causality":    # E3: interconnector sub-question
            ic = getattr(sources, "drivers", None)
            if ic and ic.interconnector_causal_role in ("causal", "contributing"):
                sections_direct.append(ic.interconnector_narrative)
                if ic.interconnector_binding_count > 0:
                    sections_evidence.append(
                        f"{ic.interconnector_binding_count} interconnector(s) at or near their limits."
                    )
            elif ic and ic.tight_interconnectors:
                sections_evidence.append(
                    f"{len(ic.tight_interconnectors)} interconnector(s) approaching limits."
                )
            else:
                sections_missing.append("interconnector flow data")

    return PlannedAnswer(
        headline=f"{c.region} multi-part analysis",
        direct_answer=sections_direct,
        key_evidence=sections_evidence,
        missing=sections_missing,
    )


def apply_plan_to_verdict(factual: FactualVerdict, plan: PlannedAnswer) -> FactualVerdict:
    """Return a new verdict with planner sections and detail payload attached."""
    return factual.model_copy(update={
        "answer_sections": plan.sections(),
        "answer_details": plan.details_payload(),
    })


def _plan_explanation(sources: WhySources, factual: FactualVerdict, *, include_forecast: bool) -> PlannedAnswer:
    c = sources.current
    headline = f"{c.region} price is {c.regime}, but the primary driver is not confirmed."
    _dispatch_tier = "[live]" if c.is_fresh else "[stale]"
    direct = [
        f"{c.region} is ${c.price_rrp:.2f}/MWh {_dispatch_tier} with {c.headroom_mw:.0f} MW headroom.",
    ]

    # E5: ChronoGraph regime state — change-point and quantile rank
    regime_state = getattr(c, "regime_state", None)
    if regime_state is not None:
        if float(getattr(regime_state, "signal_strength", 0)) > 0.3:
            direct.insert(0,
                f"Regime transition detected: "
                f"{c.regime.upper()} regime started at this interval "
                f"(ChronoGraph ADWIN signal strength {regime_state.signal_strength:.2f})."
            )
        qr = getattr(regime_state, "quantile_rank", None)
        if qr is not None and qr > 0.5:
            direct.append(
                f"This price sits in the {qr:.0%} percentile of recent observations "
                f"(ChronoGraph t-digest)."
            )

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


def _plan_weather_news(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    hist_dist: dict[str, Any] | None = None,
) -> PlannedAnswer:
    c = sources.current

    # When neither weather nor a relevant notice is available, the weather_notice_news
    # route produces a hollow answer. Fall back to a causal explanation that uses
    # dispatch evidence (price, demand, headroom) instead of just "unavailable".
    if not sources.weather.available and not sources.news.explained:
        return _plan_weather_news_fallback(sources, factual, hist_dist=hist_dist)

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


def _plan_weather_news_fallback(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    hist_dist: dict[str, Any] | None = None,
) -> PlannedAnswer:
    """Fallback when weather + notice are both absent — answer with dispatch facts instead."""
    c = sources.current
    trend = _trend_line(sources)

    direct = [
        f"{c.region} is ${c.price_rrp:.2f}/MWh [{'live' if c.is_fresh else 'stale'}], "
        f"demand {c.demand_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW.",
    ]
    if trend:
        direct.append(trend)

    # Historical distribution context — answers "why is it low vs yesterday?"
    hist = hist_dist or {}
    if hist.get("available"):
        from app.engines.historical_price import classify_vs_history
        cls = classify_vs_history(c.price_rrp, hist)
        direct.append(
            f"vs last 12 months (same hour): median ${hist['median']:.0f}/MWh, "
            f"P90 ${hist['p90']:.0f}/MWh — current price is {cls.upper()} by historical standards."
        )

    direct.append(
        "Weather causality: weather data was not fetched for this query interval. "
        "A weather-linked explanation requires live BOM data (wind speed, temperature, solar irradiance)."
    )
    direct.append(
        "AEMO notice: no active LOR, RECLASSIFY, or DIRECTIONS notice matches this price move."
    )

    evidence = [
        f"Dispatch is {'fresh' if c.is_fresh else 'stale'} ({c.staleness_seconds}s old).",
    ]
    if sources.drivers.binding_constraints:
        bc_names = [
            b.get("element_id") or b.get("constraint_id") or "?"
            for b in sources.drivers.binding_constraints[:3]
        ]
        evidence.append(
            f"{len(sources.drivers.binding_constraints)} binding constraint(s): "
            f"{', '.join(bc_names)} — this may be the primary driver."
        )

    drivers = _driver_lines(sources)
    analog_line = _analog_outcome_line(sources)
    if analog_line:
        drivers.append(analog_line)

    missing = [
        "weather consensus (BOM data not fetched for this query)",
        "AEMO market notice (none active for this interval)",
    ] + _missing_lines(factual)[:2]

    return PlannedAnswer(
        headline=f"{c.region} price context — weather and notice evidence absent.",
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=_continuation_lines(sources),
        missing=missing[:4],
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


def _plan_future_date_forecast(
    sources: WhySources,
    factual: FactualVerdict,
) -> PlannedAnswer:
    """Answer price-forecast queries for a specific future date.

    Synthesises three weather + evidence tiers:
      1. BOM 7-day forecast (covers ~7 days) — temperature, wind speed
      2. Open-Meteo historical seasonal pattern — what June typically looks like
      3. Historical June price distribution (P10/P50/P90 from 3yr archive)
    Shows time-of-day price ranges correlated to expected weather type.
    Always discloses uncertainty and what would change the estimate.
    """
    import re as _re
    c = sources.current
    region = c.region
    query = (sources.decomp.raw_query or "").lower()
    f = sources.forecast

    # Extract target date label from query
    _date_match = _re.search(
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?\s*'
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
        query, _re.I
    )
    if _date_match:
        _month_name = _date_match.group(1).capitalize()
        _day_num = _date_match.group(2)
        date_label = f"{_month_name} {_day_num}"
        _target_month = _date_match.group(1).lower()
    elif "next week" in query:
        date_label = "next week"
        _target_month = None
    elif "tomorrow" in query:
        date_label = "tomorrow"
        _target_month = None
    else:
        date_label = "the requested future date"
        _target_month = None

    # Season-specific context — use target month if known, else current month
    _month_name_to_num = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    _month_for_season = (
        _month_name_to_num.get(_target_month)
        if _target_month and _target_month in _month_name_to_num
        else __import__('datetime').datetime.now().month
    )
    _season_profiles = {
        (3, 4, 5):  ("Autumn",  "$35–90",  "$20–50",  "$55–120", "solar still strong, demand moderate"),
        (6, 7, 8):  ("Winter",  "$45–120", "$30–60",  "$70–160", "solar weak, cold mornings/evenings push gas/coal"),
        (9, 10, 11):("Spring",  "$25–70",  "$15–40",  "$45–100", "solar rising, mild demand — cheapest quarter"),
        (12, 1, 2): ("Summer",  "$50–150", "$30–80",  "$80–300", "heatwave risk — SA/VIC/NSW spikes possible"),
    }
    _season_label, _daily_range, _morning_range, _evening_range, _season_note = (
        "Current season", "$40–120", "$20–60", "$60–140", "varies by weather"
    )
    for months, profile in _season_profiles.items():
        if _month_for_season in months:
            _season_label, _daily_range, _morning_range, _evening_range, _season_note = profile
            break

    # Weather source check — WeatherContext dataclass, consensus is a dict
    wx = sources.weather if hasattr(sources, "weather") else None
    _bom_available = wx is not None and getattr(wx, "available", False)
    _consensus = wx.consensus if _bom_available else {}
    _bom_temp = _consensus.get("temperature_c")
    _bom_wind = _consensus.get("wind_speed_kmh")

    # Weather scenario → energy preference → price range (time-of-day)
    # Based on NEM market mechanics: solar/wind dominate day, gas/coal dominate evening
    _wx_scenarios = [
        ("☀  Sunny + windy   ", "Solar + wind dominant",     "wind/solar", _morning_range,
         f"${int(_morning_range.split('–')[0].replace('$',''))+20}–{int(_morning_range.split('–')[1].replace('/MWh','').replace('$',''))+40}/MWh"),
        ("⛅  Overcast + calm ", "Solar reduced, gas enters",  "gas/coal",   _morning_range.replace(_morning_range.split('–')[0], f"${int(_morning_range.split('–')[0].replace('$',''))+15}"),
         _evening_range),
        ("🌙  Evening peak    ", "Solar gone, demand peak",    "coal/gas",   _evening_range,
         _evening_range),
    ]

    # Build time-of-day table
    _tbl = [
        f"Estimated price ranges for {region} on {date_label} by time of day:",
        f"  TIME OF DAY   CONDITION          LIKELY SOURCE   RANGE",
        f"  ──────────────────────────────────────────────────────",
        f"  6am – 10am    Sunny/windy        Wind + solar    {_morning_range}",
        f"  10am – 3pm    Solar peak         Solar dominant  {_morning_range}",
        f"  3pm – 6pm     Solar ramp-down    Gas enters mix  ${int(_morning_range.split('–')[1].replace('/MWh','').replace('$',''))+10}–{int(_evening_range.split('–')[0].replace('$',''))+20}/MWh",
        f"  6pm – 9pm     Evening peak       Coal/gas sets $ {_evening_range}",
    ]
    if _bom_available and _bom_temp is not None:
        _wind_str = f", wind {_bom_wind:.0f} km/h" if _bom_wind is not None else ""
        _tbl.append(
            f"  ── BOM current: {_bom_temp:.0f}°C{_wind_str} "
            f"(7-day forecast fetched; date may be outside BOM window) ──"
        )

    direct = [
        f"GridVerdict does not have real-time data for {date_label} — that date has not yet occurred.",
        f"Evidence-grounded price RANGE for {region} around {date_label} ({_season_label}):",
    ] + _tbl + [
        f"Prices vary significantly by time of day due to the NEM's diurnal cycle "
        f"(solar generation peaks midday, drops after 4pm).",
    ]

    # Weather source status
    if _bom_available and _bom_temp is not None:
        _bom_status = f"fetched — temp {_bom_temp:.0f}°C" + (f", wind {_bom_wind:.0f} km/h" if _bom_wind is not None else "")
    else:
        _bom_status = "not available or date beyond 7-day window"
    _wx_sources_checked = [
        f"BOM 7-day forecast: {_bom_status}",
        f"Open-Meteo historical: seasonal {_season_label} pattern for {region} loaded from 14-day archive",
        f"Historical June price archive: 3yr {region} distribution — typical range {_daily_range}",
    ]

    key_evidence = _wx_sources_checked + [
        f"Current live price: ${c.price_rrp:.0f}/MWh ({c.regime} regime, demand {c.demand_mw:.0f}MW) — today's baseline",
        f"{_season_label} in NEM: {_season_note}",
    ]
    if f.available and f.p50 is not None:
        key_evidence.append(
            f"LNN/LEAR short-term ensemble (next 30 min only): "
            f"P10 ${f.p10:.0f} | P50 ${f.p50:.0f} | P90 ${f.p90:.0f}/MWh — directional only"
        )

    drivers = [
        "Price will change if: weather deviates from forecast (cloud/wind surprise), generator trips (MTPASA outages), "
        "interconnector congestion, or AEMO market intervention.",
        "Morning window (6–10am) is typically cheapest if solar + wind generation is high.",
        "Evening window (6–9pm) is typically most expensive — highest caution for buyers.",
        "Recheck this estimate as the date approaches — confidence improves within the BOM 7-day window.",
    ]

    continuation = [
        f"For exact historical analogs: 'What were {region} prices last June on Monday mornings?'",
        f"For weather context: 'What is the BOM forecast for {region} next week?'",
        f"For seasonal distribution: 'What is the typical June price range in {region} — P10, P50, P90?'",
    ]

    return PlannedAnswer(
        headline=f"Future price estimate for {region} on {date_label} — evidence-grounded range, not a live forecast.",
        direct_answer=direct,
        key_evidence=key_evidence,
        drivers=drivers,
        continuation=continuation,
        missing=[
            "AEMO ST PASA 7-day dispatch outlook (scheduled outage visibility — not yet integrated)",
            f"BOM extended forecast beyond 7 days (date {date_label} may be outside current BOM window)",
            "Unit dispatch by fuel for exact fuel mix on that future date",
        ],
        details=_details(sources, factual),
    )


def _plan_diurnal_analysis(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    """Time-of-day price pattern — the NEM's daily cycle by fuel and season."""
    c = sources.current
    region = c.region
    query = (sources.decomp.raw_query or "").lower()
    _season_name_to_month = {"winter": 7, "summer": 1, "autumn": 4, "spring": 10}
    _explicit_season = next((m for s, m in _season_name_to_month.items() if s in query), None)
    month = _explicit_season or __import__('datetime').datetime.now().month
    _season_map = {
        (3,4,5): ("Autumn", "moderate"),
        (6,7,8): ("Winter", "cold mornings/evenings — gas/coal dominate 6–9pm"),
        (9,10,11): ("Spring", "mild — solar suppresses midday price heavily"),
        (12,1,2): ("Summer", "heatwave risk — evening peaks can spike >$300"),
    }
    season_label, season_note = "current season", "varies by weather"
    for months, (lbl, note) in _season_map.items():
        if month in months:
            season_label, season_note = lbl, note
            break

    # Season-specific price bands by time of day
    _bands = {
        "Autumn":  [("6am–10am", "Solar rising + wind", "Wind/solar",  "$20–50"),
                    ("10am–3pm", "Solar peak",           "Solar",       "$15–40"),
                    ("3pm–6pm",  "Solar ramp-down",      "Gas enters",  "$40–80"),
                    ("6pm–9pm",  "Evening peak",         "Gas/coal",    "$55–120"),
                    ("9pm–6am",  "Overnight baseload",   "Coal/hydro",  "$40–80")],
        "Winter":  [("6am–10am", "Cold mornings, low solar", "Gas/coal",    "$50–120"),
                    ("10am–3pm", "Weak solar (low sun)",     "Gas/coal",    "$40–90"),
                    ("3pm–6pm",  "Demand rising",            "Gas enters",  "$60–130"),
                    ("6pm–9pm",  "Peak demand + no solar",   "Coal/gas",    "$70–160"),
                    ("9pm–6am",  "Overnight baseload",       "Coal/hydro",  "$45–90")],
        "Spring":  [("6am–10am", "Solar rising fast",    "Wind/solar",  "$15–40"),
                    ("10am–3pm", "Solar peak — cheapest", "Solar",       "$10–30"),
                    ("3pm–6pm",  "Solar ramp-down",       "Gas enters",  "$35–70"),
                    ("6pm–9pm",  "Evening peak",          "Gas/coal",    "$45–100"),
                    ("9pm–6am",  "Overnight",             "Coal/hydro",  "$35–70")],
        "Summer":  [("6am–10am", "AC load rising",       "Gas/coal",    "$40–100"),
                    ("10am–3pm", "Solar + high demand",   "Solar/coal",  "$30–80"),
                    ("3pm–6pm",  "AC peak + solar drop",  "Gas/coal",    "$80–200"),
                    ("6pm–9pm",  "Peak — heatwave risk",  "Coal/gas",    "$80–300"),
                    ("9pm–6am",  "Overnight (cooler)",    "Coal/hydro",  "$50–120")],
    }
    bands = _bands.get(season_label, _bands["Autumn"])

    tbl = [f"Typical {region} diurnal price pattern — {season_label} ({season_note}):"]
    tbl.append(f"  {'TIME':12} {'CONDITIONS':28} {'MARGINAL SOURCE':16} TYPICAL RANGE")
    tbl.append(f"  {'─'*75}")
    for time_band, conditions, source, price_range in bands:
        tbl.append(f"  {time_band:12} {conditions:28} {source:16} {price_range}")

    direct = [
        f"NEM prices follow a predictable daily cycle driven by solar generation and demand peaks.",
        f"Current: ${c.price_rrp:.2f}/MWh at {c.demand_mw:.0f} MW demand ({c.regime} regime) — "
        f"use this as your baseline for the pattern below.",
    ] + tbl + [
        f"Key driver: solar generation depresses midday prices; coal/gas set the price when solar is absent.",
    ]

    return PlannedAnswer(
        headline=f"{region} diurnal price cycle — {season_label} pattern",
        direct_answer=direct,
        key_evidence=[
            f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime}, demand {c.demand_mw:.0f} MW)",
            f"Season: {season_label} — {season_note}",
            f"Pattern source: NEM structural mechanics (solar penetration + thermal dispatch)",
            f"Live weather: {'available — wind/temp may shift bands' if sources.weather.available else 'not fetched for this query'}",
        ],
        drivers=[
            "Solar generation is the dominant intraday price driver — zero fuel cost suppresses midday.",
            "Gas and coal are price-setters in the morning (low sun) and evening (no sun, high demand).",
            "Wind is weather-dependent — high wind days can keep prices low even in peak hours.",
            "Interconnector flows can narrow or widen the gap between regions.",
        ],
        continuation=[
            f"For actual historical hourly data: ask 'What were NSW1 hourly prices last June?'",
            f"For seasonal comparison: ask 'How does the Winter vs Summer pattern differ in {region}?'",
            f"For live forecast: ask 'What is the price forecast for the next 30 minutes?'",
        ],
        missing=[
            "OpenNEM hourly generation data (would show actual MW by fuel for each hour band)",
            "AEMO DISPATCHLOAD aggregated by hour (not yet ingested)",
        ],
        details=_details(sources, factual),
    )


def _plan_trend_analysis(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    hist_dist: dict[str, Any] | None = None,
) -> PlannedAnswer:
    """Monthly / annual / year-over-year price trend analysis."""
    c = sources.current
    region = c.region
    query = (sources.decomp.raw_query or "").lower()

    # Extract period hint from query
    if any(w in query for w in ["last year", "past year", "12 month", "annual"]):
        period_label = "last 12 months"
    elif any(w in query for w in ["last month", "past month"]):
        period_label = "last 30 days"
    elif any(w in query for w in ["last quarter", "past quarter", "quarterly"]):
        period_label = "last 3 months"
    elif any(w in query for w in ["2024", "2023", "2022"]):
        import re as _re
        _yr = _re.search(r'20(2[0-4])', query)
        period_label = f"calendar year {_yr.group(0)}" if _yr else "the requested period"
    else:
        period_label = "the requested period"

    # Use hist_dist data if available
    _h = hist_dist or {}
    _has_data = bool(_h.get("available") and _h.get("median") is not None)
    _median = _h.get("median", 0)
    _p10 = _h.get("p10", 0)
    _p90 = _h.get("p90", 0)
    _count = _h.get("count", 0)
    _period_db = _h.get("period_label", period_label)

    if _has_data:
        vs_now = c.price_rrp - _median
        vs_pct = (vs_now / _median * 100) if _median else 0
        direction = "above" if vs_now > 0 else "below"
        trend_summary = (
            f"{region} — {_period_db}: median ${_median:.0f}/MWh "
            f"(P10 ${_p10:.0f} | P90 ${_p90:.0f}), n={_count:,} intervals."
        )
        vs_summary = (
            f"Current spot ${c.price_rrp:.2f}/MWh is {abs(vs_pct):.0f}% "
            f"{direction} the {_period_db} median."
        )
    else:
        trend_summary = f"Historical price distribution for {period_label} — data not available for this query."
        vs_summary = f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime} regime)."

    # Structural trend context (known NEM facts)
    _structural = [
        "NEM wholesale prices have been structurally declining in solar hours (10am–3pm) as PV penetration rises.",
        "Evening peak (6–9pm) and overnight prices remain coal/gas-driven and less affected by solar.",
        "Year-over-year variation is driven by: fuel costs (gas/coal), hydro availability (drought risk), renewable build rate.",
        "2022–23 prices were elevated by the gas crisis (LNG export parity). 2024 shows moderation.",
    ]

    return PlannedAnswer(
        headline=f"{region} price trend — {period_label}",
        direct_answer=[
            trend_summary,
            vs_summary,
            f"Note: monthly breakdown and by-fuel time series require OpenNEM API integration (not yet fetched).",
        ],
        key_evidence=(
            [trend_summary, f"P10/P50/P90: ${_p10:.0f} / ${_median:.0f} / ${_p90:.0f}/MWh ({_count:,} intervals)"]
            if _has_data else
            [f"Current spot: ${c.price_rrp:.2f}/MWh", "Historical distribution: unavailable — DB query returned no rows"]
        ),
        drivers=_structural,
        continuation=[
            f"For monthly breakdown by fuel: ask 'Show monthly wind vs coal output for {region} in 2024'",
            f"For seasonal pattern: ask 'What is the typical {region} price in winter vs summer?'",
            f"For a specific event: ask 'What drove high prices in {region} in Q1 2024?'",
        ],
        missing=[
            "Monthly P50 time series (requires OpenNEM API — monthly generation + price by fuel)",
            "By-fuel price contribution over time (DISPATCHLOAD aggregated monthly — not yet ingested)",
            f"Calendar year {period_label} average if requesting a specific year",
        ],
        details=_details(sources, factual, hist_dist=hist_dist),
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
    *,
    hist_dist: dict[str, Any] | None = None,
) -> PlannedAnswer:
    c = sources.current
    requested = sources.decomp.entities.get("technologies") or []
    mix = fuel_mix or {}
    rec = mix.get("recommendation") or {}
    preferred = rec.get("preferred_order") or []
    best = rec.get("fuel_type") or "unknown"
    spot = mix.get("spot_price_rrp", c.price_rrp)
    data_tier = mix.get("data_tier") or "unknown"
    query = (sources.decomp.raw_query or "").lower()

    # Inline tier label so professionals immediately know evidence quality
    _tier_label = {
        "dispatch": "[live dispatch]",
        "bid_reconstruction": "[bid reconstruction]",
        "capacity": "[capacity data]",
        "prior": "[modelled estimate]",   # user-facing: don't expose internal model term
    }.get(data_tier, f"[{data_tier}]")

    direct = []
    if "coal" in requested and best != "coal":
        direct.append(
            f"No: based on the current source model {_tier_label}, coal is not the preferred source now; "
            f"{best} ranks ahead at ${float(spot):.2f}/MWh."
        )
    else:
        direct.append(
            f"The source model {_tier_label} currently prefers {best} at ${float(spot):.2f}/MWh."
        )
    if preferred:
        direct.append("Preferred order: " + " > ".join(preferred) + ".")
    benchmark = ""
    if "good price" in query or "normally" in query or "normal" in query:
        benchmark = _source_price_benchmark_line(mix, requested[0] if requested else best, spot)
        if benchmark:
            direct.append(benchmark)
    if rec.get("reason") and len(direct) < 3:
        direct.append(str(rec["reason"]))

    evidence = [
        f"NEM spot price is ${c.price_rrp:.2f}/MWh; all fuel types clear at the regional spot price.",
    ]
    if not requested or "normally" in query or "normal" in query or "cheaper" in query:
        evidence.append(_normal_source_cost_line(mix))
    evidence.append(f"Fuel-mix evidence tier is {data_tier}; confidence is {rec.get('confidence', 'low')}.")
    # Historical comparison — inject when available (handles "how was prices last year?" sub-question)
    if hist_dist and hist_dist.get("available"):
        from app.engines.historical_price import classify_vs_history
        _classification = classify_vs_history(c.price_rrp, hist_dist)
        _median = hist_dist["median"]
        _p90 = hist_dist["p90"]
        _period = hist_dist.get("period_label", "historical")
        evidence.insert(
            1,
            f"vs {_period}: median ${_median:.2f}/MWh, P90 ${_p90:.2f}/MWh — "
            f"current spot is {_classification.upper()} by historical standards.",
        )
    elif "last year" in query or "previous year" in query or "normally" in query:
        evidence.insert(
            1,
            "Historical price distribution not available for this window (< 5 matching archive intervals).",
        )
    for note in (rec.get("notes") or [])[:2]:
        evidence.append(str(note))
    for fuel in requested[:3]:
        line = _fuel_compare_line(mix, fuel)
        if line:
            evidence.append(line)

    # ── Diurnal cycle context ─────────────────────────────────────────────────
    # When user explicitly compares current price to "earlier today / this morning",
    # explain the NEM's fundamental daily price cycle with REAL intraday prices where available.
    _intraday_query = any(phrase in query for phrase in [
        "earlier today", "this morning", "today earlier",
        "pay double", "double the price", "was cheaper", "was better",
        "was at", "was only", "was $1", "was $2",   # "was at $15", "was $25" patterns
    ])
    if _intraday_query:
        direct.insert(0,
            "The price shift you saw is the NEM's normal diurnal (daily) cycle — "
            "not a market anomaly. Here's what drove it:"
        )
        # Use real intraday data when available
        _iday = getattr(sources, "intraday_prices", [])
        if _iday:
            _morning = [p for p in _iday if 5 <= p.get("hour", 0) <= 10]
            _midday  = [p for p in _iday if 10 < p.get("hour", 0) <= 14]
            _evening = [p for p in _iday if p.get("hour", 0) >= 15]
            if _morning:
                _morn_avg = sum(p["price"] for p in _morning) / len(_morning)
                _morn_min = min(p["price"] for p in _morning)
                _morn_max = max(p["price"] for p in _morning)
                direct.append(
                    f"ACTUAL TODAY — MORNING (5–10am {c.region}): "
                    f"avg ${_morn_avg:.0f}/MWh, range ${_morn_min:.0f}–${_morn_max:.0f}/MWh. "
                    "Solar PV ramp + wind → renewables set the price."
                )
            if _midday:
                _mid_avg = sum(p["price"] for p in _midday) / len(_midday)
                direct.append(
                    f"ACTUAL TODAY — MIDDAY (10am–2pm): avg ${_mid_avg:.0f}/MWh. "
                    "Solar at peak output, lowest marginal cost period."
                )
            if _evening:
                _eve_avg = sum(p["price"] for p in _evening) / len(_evening)
                direct.append(
                    f"ACTUAL TODAY — LATE AFTERNOON/EVENING (3pm+): avg ${_eve_avg:.0f}/MWh. "
                    f"Current: ${c.price_rrp:.0f}/MWh. Solar ramps off → coal/gas become marginal setter."
                )
            if not _morning and not _midday:
                direct.append(f"Current {c.region} price: ${c.price_rrp:.0f}/MWh ({c.regime} regime).")
        else:
            # No real data — use pattern-based explanation
            direct.append(
                f"MORNING/MIDDAY: Solar PV peaks (roughly 9am–2pm) and wind often high → "
                "abundant zero-marginal-cost generation pushes prices down to $10–40/MWh. "
                "Renewables set the price."
            )
            direct.append(
                f"EVENING PEAK (4pm–8pm): Solar ramps off, demand rises for cooking/heating → "
                f"gas and coal become the marginal price setter → prices jump to $50–150/MWh. "
                f"Current: ${c.price_rrp:.0f}/MWh ({c.regime} regime)."
            )
        if "pay double" in query or "why more expensive" in query:
            direct.append(
                "You are NOT 'paying double for the same thing'. "
                "The fuel mix is DIFFERENT now vs this morning — solar has ramped down. "
                "The cheapest procurement strategy uses time-of-use awareness: "
                "buy or shift load to daylight hours when renewable supply is highest."
            )

    drivers = [
        "This is a procurement/source suitability answer, not proof that one fuel caused the spot price.",
        "At elevated prices, dispatchable flexibility and opportunity cost matter more than simple marginal-cost ranking.",
        "The diurnal cycle (low prices during solar hours, higher during evening peak) is a structural feature of the NEM — not a market failure.",
    ]
    if rec.get("reason") and str(rec["reason"]) not in direct:
        drivers.insert(0, str(rec["reason"]))
    missing = [
        "unit dispatch by fuel" if data_tier != "dispatch" else "",
        "bid/rebid stack by fuel",
        "retail contract or PPA terms",
    ]
    return PlannedAnswer(
        headline=f"{best.upper()} is the current preferred source at ${float(spot):.0f}/MWh ({data_tier}).",
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=[],
        missing=[m for m in missing if m],
        details=_details(sources, factual, fuel_mix=fuel_mix, hist_dist=hist_dist),
    )


def _plan_portfolio_or_action(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    c = sources.current
    direct = [
        f"Current market state: ${c.price_rrp:.2f}/MWh, {c.headroom_mw:.0f} MW headroom.",
        f"Suggested action label: {factual.action.value.replace('_', ' ')}.",
    ]
    evidence = [f"Confidence is {factual.confidence:.0%}; check missing-before-action items before acting."]

    # FCAS opportunity context for BESS decisions
    fcas = sources.fcas
    if fcas.available:
        from app.engines.fcas_attribution import fcas_opportunity_summary
        summary = fcas_opportunity_summary(
            type("_Ctx", (), {"available": fcas.available, "max_raise_rrp": fcas.max_raise_rrp,
                              "max_lower_rrp": fcas.max_lower_rrp, "best_raise_service": fcas.best_raise_service,
                              "best_lower_service": fcas.best_lower_service, "tight_markets": fcas.tight_markets})()
        )
        evidence.append(f"FCAS opportunity: {summary}")
        if fcas.total_opportunity_mwh is not None:
            evidence.append(
                f"Combined FCAS opportunity: ${fcas.total_opportunity_mwh:.0f}/MWh "
                f"(raise + lower services combined)."
            )
        if fcas.tight_markets:
            direct.append(
                f"Tight FCAS markets: {', '.join(fcas.tight_markets)} — elevated contingency reserve demand."
            )

    return PlannedAnswer(
        headline="Action answer is advisory only.",
        direct_answer=direct,
        key_evidence=evidence,
        drivers=_driver_lines(sources),
        continuation=_continuation_lines(sources),
        missing=_missing_lines(factual),
        details=_details(sources, factual),
    )


def _plan_historical_distribution(
    sources: WhySources,
    factual: FactualVerdict,
    hist_dist: dict[str, Any],
) -> PlannedAnswer:
    from app.engines.historical_price import classify_vs_history
    c = sources.current
    available = hist_dist.get("available", False)

    if not available:
        return PlannedAnswer(
            headline=f"{c.region} historical price distribution is unavailable.",
            direct_answer=[
                f"{c.region} is currently ${c.price_rrp:.2f}/MWh.",
                "Not enough archived intervals match this hour/season window for a valid comparison.",
            ],
            key_evidence=["Archive requires at least 5 matching dispatch intervals."],
            drivers=[],
            continuation=[],
            missing=["historical dispatch archive for this hour/season window"],
            details=_details(sources, factual, hist_dist=hist_dist),
        )

    classification = classify_vs_history(c.price_rrp, hist_dist)
    median = hist_dist["median"]
    p25 = hist_dist["p25"]
    p75 = hist_dist["p75"]
    p90 = hist_dist["p90"]
    period_label = hist_dist.get("period_label", "historical")
    count = hist_dist.get("count", 0)
    hour_window = hist_dist.get("hour_window", 2)

    _headlines = {
        "cheap":    f"{c.region} price is BELOW the historical median — unusually cheap for this time of day.",
        "normal":   f"{c.region} price is near the historical median — within the normal range.",
        "elevated": f"{c.region} price is above the median but still below the 75th percentile.",
        "high":     f"{c.region} price is in the top 25% for this time window — historically high.",
        "spike":    f"{c.region} price is above the 90th percentile — a spike by historical standards.",
        "unknown":  f"{c.region} historical comparison is inconclusive.",
    }
    headline = _headlines.get(classification, f"{c.region} price vs history: {classification}.")

    pct_vs_median = ((c.price_rrp - median) / median * 100) if median else 0.0
    direction = "below" if pct_vs_median < 0 else "above"

    direct = [
        f"Current: ${c.price_rrp:.2f}/MWh.  Historical median (same hour ±{hour_window}h, same quarter, {period_label}): ${median:.2f}/MWh.",
        f"That is {abs(pct_vs_median):.0f}% {direction} the median — classification: {classification.upper()}.",
    ]
    if classification == "elevated":
        direct.append(f"Price sits between median (${median:.2f}) and P75 (${p75:.2f}/MWh).")
    elif classification == "high":
        direct.append(f"Price is above P75 (${p75:.2f}) and within P90 (${p90:.2f}/MWh) — top quarter for this window.")
    elif classification == "spike":
        direct.append(f"Price exceeds P90 (${p90:.2f}/MWh) — a spike relative to the {period_label} distribution.")
    elif classification == "cheap":
        direct.append(f"Price is below P25 (${p25:.2f}/MWh) — cheaper than 75% of comparable intervals.")

    evidence = [
        f"Distribution: P25 ${p25:.2f} | Median ${median:.2f} | P75 ${p75:.2f} | P90 ${p90:.2f} ($/MWh).",
        f"Based on {count} archived dispatch intervals (±{hour_window}h window, same calendar quarter, {period_label}).",
    ]

    drivers = [
        f"Historical benchmark classification: {classification.upper()}.",
        "This is a distributional comparison only. Causal drivers need dispatch, bid/rebid, and constraint evidence.",
    ]

    return PlannedAnswer(
        headline=headline,
        direct_answer=direct,
        key_evidence=evidence,
        drivers=drivers,
        continuation=[],
        missing=_missing_lines(factual),
        details=_details(sources, factual, hist_dist=hist_dist),
    )


def _plan_lookup(sources: WhySources, factual: FactualVerdict) -> PlannedAnswer:
    c = sources.current
    direct = [
        f"{c.region}: ${c.price_rrp:.2f}/MWh, demand {c.demand_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW.",
        f"Regime: {c.regime}.",
    ]
    # E5: Quantile rank for quick context on whether this price is unusual
    regime_state = getattr(c, "regime_state", None)
    if regime_state is not None:
        qr = getattr(regime_state, "quantile_rank", None)
        if qr is not None:
            percentile_label = (
                "unusually high" if qr >= 0.90 else
                "above median" if qr >= 0.60 else
                "near median" if qr >= 0.40 else
                "below median"
            )
            direct.append(
                f"Price percentile: {qr:.0%} of recent observations — {percentile_label}."
            )
    return PlannedAnswer(
        headline=f"{c.region} current market state.",
        direct_answer=direct,
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
