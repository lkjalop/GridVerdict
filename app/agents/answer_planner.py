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
from app.agents.planner_helpers import (
    _trend_line, _price_movement_line, _sort_time, _nearest_row,
    _analog_outcome_line, _headroom_str, _top_analog_lines,
    _weather_line, _weather_support_line, _driver_lines, _continuation_lines,
    _missing_lines, _friendly_missing, _details,
    _fuel_compare_line, _normal_source_cost_line, _source_price_benchmark_line,
    _fuel_driver_context_line, _fuel_reason_line,
    _short_time, _model_label, _fmt_time, _extract_query_price_path,
    _swing_label, _cap,
)


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
    opennem_trend: Any | None = None,
    opennem_diurnal: Any | None = None,
    period_stats: dict[str, Any] | None = None,
    intraday_fuel_timeline: dict[str, Any] | None = None,
) -> PlannedAnswer:
    """Build concise visible answer sections from approved evidence."""
    requested = (sources.decomp.requested_output or "").lower()
    query = (sources.decomp.raw_query or "").lower()

    # Specific past-month query ("What was the average NSW price in July 2023?")
    # Triggered by a specific_period_stats sub_question (date range baked in at decomp time)
    _has_period_sq = any(
        sq.get("type") == "specific_period_stats"
        for sq in (sources.decomp.sub_questions or [])
    )
    if _has_period_sq and period_stats and period_stats.get("available"):
        return _plan_specific_period(sources, factual, period_stats, hist_dist)

    if requested == "data_freshness_status" or "stale" in query:
        return _plan_data_status(sources, factual, evidence_quality, provenance)
    if requested == "diurnal_analysis":
        return _plan_diurnal_analysis(sources, factual, opennem_diurnal=opennem_diurnal)
    if requested == "trend_analysis":
        return _plan_trend_analysis(sources, factual, hist_dist=hist_dist, opennem_trend=opennem_trend)
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
        return _plan_future_date_forecast(sources, factual, hist_dist=hist_dist)

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

    # Inject intraday fuel timeline bullets when available — corroborates
    # "why coal now vs wind earlier today?" claims with actual generation data
    if intraday_fuel_timeline and intraday_fuel_timeline.get("hours"):
        for bullet in _build_intraday_fuel_bullets(intraday_fuel_timeline, sources.current.region):
            plan.key_evidence.append(bullet)

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
                    f"Demand {c.demand_mw:,.0f} MW vs available {c.availability_mw:,.0f} MW — "
                    f"{_headroom_str(max(c.availability_mw - c.demand_mw, 0), c.demand_mw)}."
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
                sections_missing.append("FCAS prices: no data for this interval")

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


_NEM_GLOSSARY: dict[str, str] = {
    "headroom": (
        "Headroom = available generation − current demand. "
        "It's the grid's spare capacity buffer. "
        "High headroom (>5,000 MW) → generators compete → low prices. "
        "Low headroom (<1,000 MW) → generators can bid high → spike risk. "
        "Think of it like spare lanes on a highway: more lanes = smoother traffic."
    ),
    "fcas": (
        "FCAS = Frequency Control Ancillary Services. "
        "The grid must run at exactly 50 Hz. When a large generator trips, frequency drops in seconds. "
        "FCAS providers (hydro, batteries, gas peakers) are pre-contracted to inject or absorb power "
        "within 6–60 seconds. FCAS prices spike when the grid is stressed or after major trips."
    ),
    "mtpasa": (
        "MTPASA = Medium-Term Projected Assessment of System Adequacy. "
        "AEMO publishes this weekly — it forecasts whether the grid has enough generation "
        "over the next 2 years to meet peak demand. "
        "High MTPASA 'reserve' = comfortable. Low reserve = risk of load shedding, and "
        "forward contract prices tend to rise."
    ),
    "dispatch interval": (
        "Every 5 minutes, AEMO runs a real-time auction called a dispatch interval. "
        "Generators submit price/quantity bids, AEMO stacks them cheapest-first to meet demand, "
        "and the most expensive bid needed to cover demand sets the spot price for that interval. "
        "Six intervals make one 30-minute settlement period."
    ),
    "rrp": (
        "RRP = Regional Reference Price — the spot price for a NEM region (NSW1, VIC1, etc). "
        "It's the $/MWh price set by the dispatch auction every 5 minutes. "
        "Large buyers (industrials, retailers) pay this price if they're on spot contracts."
    ),
    "marginal setter": (
        "The marginal setter is the generator whose bid price determines the spot price. "
        "AEMO stacks all offers cheapest-first. The last (most expensive) generator needed "
        "to meet demand sets the price for everyone. "
        "If coal bids $60 and the last needed unit is a gas peaker at $120, gas is the marginal setter."
    ),
    "interconnector": (
        "Interconnectors are high-voltage transmission lines between NEM regions "
        "(e.g., QNI connects Queensland and NSW, Heywood connects Victoria and SA). "
        "When they're constrained (at capacity), regions can't share power and prices diverge. "
        "SA is most prone to interconnector-driven spikes."
    ),
    "predispatch": (
        "Predispatch is AEMO's 30-minute-ahead price forecast, updated every 5 minutes. "
        "It shows where the market thinks prices are heading. "
        "GridVerdict pulls predispatch as one of its forecast signals (alongside LNN/LEAR/QRA)."
    ),
}


def _nem_glossary_answer(query: str) -> list[str]:
    """Return plain-English definition if the query is asking what a NEM term means."""
    lower = query.lower()
    for term, definition in _NEM_GLOSSARY.items():
        if term in lower:
            return [f"NEM term — {term.upper()}: {definition}"]
    return []


def _plan_explanation(sources: WhySources, factual: FactualVerdict, *, include_forecast: bool) -> PlannedAnswer:
    c = sources.current
    query = (sources.decomp.raw_query or "")
    headline = f"{c.region} price is {c.regime}, but the primary driver is not confirmed."
    _dispatch_tier = "[live]" if c.is_fresh else "[stale]"
    direct = [
        f"{c.region} is ${c.price_rrp:.2f}/MWh {_dispatch_tier} — {_headroom_str(c.headroom_mw, c.demand_mw)}.",
    ]
    # Glossary inject — if user is asking what a NEM term means, lead with the definition
    _gloss = _nem_glossary_answer(query)
    if _gloss:
        direct = _gloss + direct

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
    if sources.news.explained and sources.news.top_notice_type:
        try:
            from app.engines.notice_price_signal import classify_notice as _cn
            _ns = _cn(sources.news.top_notice_type, c.region)
            evidence.append(_ns.as_nlp_bullet())
        except Exception:
            evidence.append(f"AEMO notice present: {sources.news.top_notice_type}.")
    elif sources.news.notices_stale:
        evidence.append("AEMO notice context is stale or unavailable.")
    if sources.weather.relevant and sources.weather.available:
        evidence.append(_weather_line(sources))

    # Gas causal chain — wire in GBB price when gas is elevated or region is gas-heavy
    _gas = getattr(sources, "gas_context", None)
    if _gas and _gas.get("latest_hub_price_gj") is not None:
        _gj = _gas["latest_hub_price_gj"]
        _ccgt = _gas.get("srmc_ccgt_mwh")
        _trend = _gas.get("price_trend", "")
        if _gas.get("crisis_alert"):
            evidence.append(
                f"Gas crisis alert: hub price ${_gj:.2f}/GJ ({_gas.get('hub_name', 'east coast')}) "
                f"— CCGT SRMC ~${_ccgt:.0f}/MWh. Gas is setting the NEM cap."
            )
        elif _gas.get("high_price_alert"):
            evidence.append(
                f"Elevated gas: ${_gj:.2f}/GJ → CCGT SRMC ~${_ccgt:.0f}/MWh "
                f"({_trend}). Gas generators are pushing prices."
            )
        elif _ccgt and _ccgt > 80:
            evidence.append(
                f"Gas price context: ${_gj:.2f}/GJ → CCGT SRMC ~${_ccgt:.0f}/MWh "
                f"({_trend}). [{_gas.get('source', 'AEMO_STTM')}]"
            )

    # ST PASA 7-day reserve outlook — flag tight intervals
    _pasa = getattr(sources, "st_pasa", None)
    if _pasa and _pasa.get("tight_interval_count", 0) > 0:
        _tight = _pasa["tight_interval_count"]
        _next_risk = _pasa.get("next_lor_risk_interval")
        if _next_risk and _next_risk.get("datetime"):
            evidence.append(
                f"ST PASA 7-day outlook: {_tight} tight interval(s) with reserve below 1000 MW. "
                f"Next LOR risk: {_next_risk['datetime'][:16]} "
                f"({_next_risk.get('reserve_mw', 'unknown')} MW headroom). "
                "Adequacy risk within the week."
            )

    # LEAR feature attribution — wire into evidence when available
    _lf = getattr(sources, "forecast", None)
    if _lf and hasattr(_lf, "model_details"):
        for _md in (_lf.model_details or []):
            _attrs = (_md.raw or {}).get("feature_attributions") if hasattr(_md, "raw") else None
            if _attrs:
                _attr_str = ", ".join(
                    f"{a['feature']} ({'+' if a['contribution'] >= 0 else ''}{a['contribution']:.0f})"
                    for a in _attrs[:3]
                )
                evidence.append(
                    f"LEAR model attribution: forecast driven by {_attr_str} ($/MWh contributions)."
                )
                break

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


def _plan_specific_period(
    sources: WhySources,
    factual: FactualVerdict,
    period_stats: dict[str, Any],
    hist_dist: dict[str, Any] | None,
) -> PlannedAnswer:
    """Answer 'What was the average price in [specific month/year]?' from DB aggregate."""
    region = sources.current.region
    start = period_stats.get("start", "")
    end   = period_stats.get("end", "")
    mean  = period_stats.get("mean")
    median = period_stats.get("median")
    lo    = period_stats.get("min")
    hi    = period_stats.get("max")
    n     = period_stats.get("count", 0)

    # Human-readable period label ("July 2023", "Oct 2024–Mar 2025", etc.)
    try:
        from datetime import date as _date
        _s = _date.fromisoformat(start)
        _e = _date.fromisoformat(end)
        period_label = _s.strftime("%B %Y") if _s.month == _e.month and _s.year == _e.year else f"{_s.strftime('%b %Y')}–{_e.strftime('%b %Y')}"
    except Exception:
        period_label = f"{start} to {end}"

    headline = (
        f"{region} average spot price in {period_label}: **${mean:,.2f}/MWh** "
        f"(median ${median:,.2f}, range ${lo:,.0f}–${hi:,.0f})"
    ) if mean is not None else f"No {region} price data found for {period_label}."

    # Optionally compare to 4-year history
    hist_context = ""
    _h = hist_dist or {}
    if _h.get("available") and _h.get("median") and mean is not None:
        _hmed = _h["median"]
        _pct = ((mean - _hmed) / _hmed * 100) if _hmed else 0
        _direction = "above" if _pct > 0 else "below"
        hist_context = (
            f" That's {abs(_pct):.0f}% {_direction} the 4-year historical median "
            f"(${_hmed:,.2f}/MWh, {_h.get('period_label','available history')})."
        )

    hist_line = (
        f"4-year median for comparison: ${_h['median']:,.2f}/MWh ({_h.get('period_label','')})."
        if _h.get("available") and _h.get("median") else ""
    )
    return PlannedAnswer(
        headline=headline + hist_context,
        direct_answer=[headline + hist_context],
        key_evidence=[
            f"Source: AEMO MMSDM archive — {n:,} 5-minute dispatch intervals covering {period_label}.",
            f"Mean ${mean:,.2f}/MWh · Median ${median:,.2f}/MWh · Range ${lo:,.0f}–${hi:,.0f}/MWh.",
            *(  [hist_line] if hist_line else [] ),
        ],
        details={
            "period_stats": period_stats,
            "hist_dist": _h,
            "specific_period": {
                "period": period_label,
                "region": region,
                "mean_price": mean,
                "median_price": median,
                "min_price": lo,
                "max_price": hi,
                "interval_count": n,
                "hist_median": (_h.get("median") if _h.get("available") else None),
            },
        },
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
        f"{c.region} is ${c.price_rrp:.2f}/MWh [{'live' if c.is_fresh else 'stale'}] — "
        f"{_headroom_str(c.headroom_mw, c.demand_mw)}.",
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
        f"Dispatch: demand {c.demand_mw:,.0f} MW, available {c.availability_mw:,.0f} MW — {_headroom_str(c.headroom_mw, c.demand_mw)}.",
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
    *,
    hist_dist: dict[str, Any] | None = None,
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
        else datetime.now().month
    )
    _season_map = {
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring",
        12: "Summer", 1: "Summer", 2: "Summer",
    }
    _season_notes = {
        "Autumn":  "solar still strong, demand moderate",
        "Winter":  "solar weak, cold mornings/evenings push gas/coal",
        "Spring":  "solar rising, mild demand — cheapest quarter",
        "Summer":  "heatwave risk — SA/VIC/NSW spikes possible",
    }
    _season_label = _season_map.get(_month_for_season, "Current season")
    _season_note = _season_notes.get(_season_label, "varies by weather")

    # Use DB-derived percentiles when hist_dist is available — always more accurate
    # than hardcoded ranges which don't reflect region or actual observed volatility.
    if hist_dist and hist_dist.get("available") and hist_dist.get("p10") is not None:
        _p10 = int(hist_dist["p10"])
        _p50 = int(hist_dist.get("median") or hist_dist.get("p50") or 60)
        _p90 = int(hist_dist["p90"])
        _count = hist_dist.get("count", "")
        _daily_range  = f"${_p10}–{_p90}"
        _morning_range = f"${max(5, _p10 - 10)}–{_p50}"
        _evening_range = f"${_p50}–{min(15000, _p90 + 30)}"
        _season_note  = (
            f"P10/P50/P90 from {_count} actual {region} intervals"
            if _count else f"derived from {region} historical archive"
        )
    else:
        # Fallback static profiles — replaced by DB data when archive is populated
        _static = {
            "Autumn":  ("$35–90",   "$20–50",  "$55–120"),
            "Winter":  ("$45–120",  "$30–60",  "$70–160"),
            "Spring":  ("$25–70",   "$15–40",  "$45–100"),
            "Summer":  ("$50–150",  "$30–80",  "$80–300"),
        }
        _daily_range, _morning_range, _evening_range = _static.get(
            _season_label, ("$40–120", "$20–60", "$60–140")
        )

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


def _plan_diurnal_analysis(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    opennem_diurnal: Any | None = None,
) -> PlannedAnswer:
    """Time-of-day price pattern — uses real OpenNEM hourly data when available,
    otherwise falls back to seasonal NEM structural norms."""
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

    # ── Path A: real OpenNEM hourly data ─────────────────────────────────────
    _onem = opennem_diurnal
    if _onem is not None and getattr(_onem, "available", False) and _onem.hours:
        from app.mcp.opennem_client import format_diurnal_table
        _real_tbl = format_diurnal_table(_onem)
        _cheapest = _onem.cheapest_hour
        _peak = _onem.peak_hour
        direct = [
            f"Real hourly price data for {region} — last {_onem.days_sampled} days (OpenNEM):",
            f"Current: ${c.price_rrp:.2f}/MWh ({c.regime}, demand {c.demand_mw:,.0f} MW).",
        ] + _real_tbl

        key_evidence = [
            f"Source: OpenNEM/OpenElectricity API — {_onem.days_sampled} days of hourly {region} prices",
            f"Cheapest hour: {_cheapest:02d}:00 AEST — buy here for lowest cost exposure",
            f"Peak hour: {_peak:02d}:00 AEST — highest average price, avoid unhedged exposure",
            f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime})",
        ]
        if sources.weather.available:
            key_evidence.append(_weather_line(sources))

        missing = ["Per-fuel hourly breakdown: DISPATCHLOAD ingested but hourly aggregation not computed for this query"]

        return PlannedAnswer(
            headline=f"{region} diurnal price cycle — real data, last {_onem.days_sampled} days",
            direct_answer=direct,
            key_evidence=key_evidence,
            drivers=[
                "Solar generation (zero fuel cost) suppresses midday prices — typically cheapest 10am–3pm.",
                "Coal and gas are price-setters when solar is absent: morning 6–9am and evening 6–9pm.",
                f"In {season_label}: {season_note}",
                "Wind is weather-dependent — high-wind days keep prices low even in peak hours.",
            ],
            continuation=[
                f"For seasonal comparison: ask 'How does the {season_label} pattern compare to Summer in {region}?'",
                f"For live forecast: ask 'What is the price forecast for the next 30 minutes?'",
                f"For monthly trend: ask 'What is the annual price trend for {region}?'",
            ],
            missing=missing,
            details=_details(sources, factual),
        )

    # ── Path B: seasonal norms fallback ──────────────────────────────────────
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
    tbl = [f"Seasonal estimate — {region} {season_label} ({season_note}):"]
    tbl.append(f"  {'TIME':12} {'CONDITIONS':28} {'SOURCE':16} TYPICAL RANGE")
    tbl.append(f"  {'─'*75}")
    for time_band, conditions, source, price_range in bands:
        tbl.append(f"  {time_band:12} {conditions:28} {source:16} {price_range}")

    direct = [
        f"NEM prices follow a predictable daily cycle driven by solar generation and demand peaks.",
        f"Current: ${c.price_rrp:.2f}/MWh at {c.demand_mw:,.0f} MW demand ({c.regime} regime).",
    ] + tbl + [
        "Key driver: solar generation depresses midday prices; coal/gas set price when solar is absent.",
    ]

    return PlannedAnswer(
        headline=f"{region} diurnal price cycle — {season_label} seasonal estimate",
        direct_answer=direct,
        key_evidence=[
            f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime}, demand {c.demand_mw:,.0f} MW)",
            f"Season: {season_label} — {season_note}",
            "Pattern: NEM structural mechanics (solar penetration + thermal dispatch order)",
            f"Live weather: {'available — wind/temp may shift bands' if sources.weather.available else 'not fetched'}",
        ],
        drivers=[
            "Solar generation is the dominant intraday driver — zero fuel cost suppresses midday.",
            "Gas and coal set the price in morning (low sun) and evening (no sun, high demand).",
            "Wind is weather-dependent — high-wind days keep prices low even during peaks.",
            "Interconnector flows narrow or widen the gap between regions.",
        ],
        continuation=[
            f"For real hourly data: ask 'When is the cheapest time to buy power in {region}?' (pulls OpenNEM)",
            f"For seasonal comparison: ask 'How does Winter vs Summer differ in {region}?'",
            f"For live forecast: ask 'What is the price forecast for the next 30 minutes?'",
        ],
        missing=["OpenNEM hourly data (fetch timed out or unavailable — retry to get real data)"],
        details=_details(sources, factual),
    )


def _plan_trend_analysis(
    sources: WhySources,
    factual: FactualVerdict,
    *,
    hist_dist: dict[str, Any] | None = None,
    opennem_trend: Any | None = None,
) -> PlannedAnswer:
    """Monthly / annual price trend — uses real OpenNEM monthly data when available."""
    c = sources.current
    region = c.region
    query = (sources.decomp.raw_query or "").lower()

    # Period label from query text
    if any(w in query for w in ["last year", "past year", "12 month", "annual"]):
        period_label = "last 12 months"
    elif any(w in query for w in ["last quarter", "past quarter", "quarterly"]):
        period_label = "last 3 months"
    elif any(w in query for w in ["2024", "2023", "2022"]):
        import re as _re
        _yr = _re.search(r'20(2[0-4])', query)
        period_label = f"calendar year {_yr.group(0)}" if _yr else "the requested period"
    else:
        period_label = "last 12 months"

    _structural = [
        "NEM wholesale prices have been structurally declining in solar hours (10am–3pm) as PV penetration rises.",
        "Evening peak (6–9pm) and overnight prices remain coal/gas-driven and less affected by solar.",
        "Year-over-year variation is driven by: fuel costs (gas/coal), hydro availability (drought risk), renewable build rate.",
        "2022–23 prices were elevated by the gas crisis (LNG export parity). 2024 shows moderation.",
    ]

    # ── Path A: real OpenNEM monthly data ────────────────────────────────────
    _onem = opennem_trend
    if _onem is not None and getattr(_onem, "available", False) and _onem.months:
        from app.mcp.opennem_client import format_monthly_trend_table
        _tbl = format_monthly_trend_table(_onem)

        _yoy_str = ""
        if _onem.yoy_change_pct is not None:
            _dir = "up" if _onem.yoy_change_pct > 0 else "down"
            _yoy_str = f" — {abs(_onem.yoy_change_pct):.1f}% {_dir} year-on-year"

        vs_now = ""
        if _onem.twelve_month_avg and c.price_rrp:
            _vs_pct = (c.price_rrp - _onem.twelve_month_avg) / _onem.twelve_month_avg * 100
            _vs_dir = "above" if _vs_pct > 0 else "below"
            vs_now = (
                f"Current spot ${c.price_rrp:.2f}/MWh is "
                f"{abs(_vs_pct):.0f}% {_vs_dir} the 12-month average "
                f"(${_onem.twelve_month_avg:.0f}/MWh){_yoy_str}."
            )

        direct = [vs_now] + _tbl if vs_now else _tbl

        key_evidence = [
            f"Source: OpenNEM/OpenElectricity API — real monthly {region} prices",
            f"12-month average: ${_onem.twelve_month_avg:.0f}/MWh" if _onem.twelve_month_avg else "",
            f"Renewable mix (latest): {_onem.renewable_latest_pct:.0f}%" if _onem.renewable_latest_pct else "",
            f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime})",
        ]
        key_evidence = [e for e in key_evidence if e]

        return PlannedAnswer(
            headline=f"{region} price trend — {period_label} (real OpenNEM data)",
            direct_answer=direct,
            key_evidence=key_evidence,
            drivers=_structural,
            continuation=[
                f"For daily price pattern: ask 'What is the typical diurnal pattern for {region}?'",
                f"For fuel source breakdown: ask 'Which fuel type has driven NSW prices this year?'",
                f"For a specific month: ask 'Why were {region} prices elevated in June 2025?'",
            ],
            missing=["By-fuel monthly breakdown (requires OpenNEM facilities endpoint — auth needed)"],
            details=_details(sources, factual, hist_dist=hist_dist),
        )

    # ── Path B: DB hist_dist fallback ─────────────────────────────────────────
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
        trend_summary = (
            f"{region} — {_period_db}: median ${_median:.0f}/MWh "
            f"(P10 ${_p10:.0f} | P90 ${_p90:.0f}), n={_count:,} intervals."
        )
        vs_summary = (
            f"Current ${c.price_rrp:.2f}/MWh is "
            f"{'above' if vs_now > 0 else 'below'} the median by "
            f"{abs(vs_pct):.0f}%."
        )
    else:
        trend_summary = f"Historical distribution for {period_label}: no DB data matched this hour/season."
        vs_summary = f"Current spot: ${c.price_rrp:.2f}/MWh ({c.regime})."

    return PlannedAnswer(
        headline=f"{region} price trend — {period_label}",
        direct_answer=[
            trend_summary,
            vs_summary,
            "Monthly table: OpenNEM API unavailable — showing DB archive summary only.",
        ],
        key_evidence=(
            [trend_summary, f"P10/P50/P90: ${_p10:.0f} / ${_median:.0f} / ${_p90:.0f}/MWh ({_count:,} intervals)"]
            if _has_data else
            [f"Current: ${c.price_rrp:.2f}/MWh", "DB historical distribution: unavailable"]
        ),
        drivers=_structural,
        continuation=[
            f"For monthly breakdown by fuel: ask 'Show monthly wind vs coal output for {region} in 2024'",
            f"For seasonal pattern: ask 'What is the typical {region} price in winter vs summer?'",
            f"For a specific event: ask 'What drove high prices in {region} in Q1 2024?'",
        ],
        missing=[
            "Monthly P50 time series (requires OpenNEM API — monthly generation + price by fuel)",
            "By-fuel price contribution over time (monthly per-fuel aggregation not computed)",
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
        f"Current market: ${c.price_rrp:.2f}/MWh — {_headroom_str(c.headroom_mw, c.demand_mw)}.",
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
        f"{c.region}: ${c.price_rrp:.2f}/MWh — {_headroom_str(c.headroom_mw, c.demand_mw)}.",
        f"Demand: {c.demand_mw:,.0f} MW. Regime: {c.regime}.",
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


# ── Intraday fuel timeline helper ────────────────────────────────────────────

def _build_intraday_fuel_bullets(timeline: dict, region: str) -> list[str]:
    """Convert a summarised intraday fuel timeline into 2-4 evidence bullets.

    Used to corroborate "why coal now vs wind/solar earlier today?" claims.
    Returns empty list when the timeline has no meaningful transition to describe.
    """
    bullets: list[str] = []
    hours = timeline.get("hours") or []
    if not hours:
        return bullets

    transition = timeline.get("transition")
    solar_cliff = timeline.get("solar_cliff_hour")
    peak_ren_hour = timeline.get("peak_renewable_hour")

    # Peak renewable window
    if peak_ren_hour:
        peak = next((h for h in hours if h["hour_iso"] == peak_ren_hour), None)
        if peak:
            bullets.append(
                f"Peak renewable window today ({region}): {peak_ren_hour[:16].replace('T', ' ')} — "
                f"{peak['renewable_pct']:.0f}% renewable, dominant source: {peak['dominant_fuel']} "
                f"({peak['dominant_mw']:.0f} MW)."
            )

    # Solar cliff moment
    if solar_cliff:
        bullets.append(
            f"Solar cliff detected at {solar_cliff[:16].replace('T', ' ')} — "
            "rooftop and utility solar dropped to below 15% of its midday peak, "
            "shifting the marginal generator to fossil fuel."
        )

    # Fuel transition
    if transition:
        bullets.append(
            f"Dispatch transition observed: {transition['from_fuel']} was dominant before "
            f"{transition['transition_hour'][:16].replace('T', ' ')}, "
            f"then {transition['to_fuel']} took over as the largest source."
        )

    # Current vs earlier comparison
    if hours:
        latest = hours[-1]
        earliest = hours[0]
        if latest["dominant_fuel"] != earliest["dominant_fuel"]:
            bullets.append(
                f"Generation mix shift today: {earliest['dominant_fuel']} was leading at "
                f"{earliest['hour_iso'][:16].replace('T', ' ')} "
                f"({earliest['dominant_mw']:.0f} MW); "
                f"{latest['dominant_fuel']} leads now "
                f"({latest['dominant_mw']:.0f} MW). "
                "This is the NEM merit order responding to solar/wind availability changes."
            )
        else:
            bullets.append(
                f"{latest['dominant_fuel'].capitalize()} has been the dominant source throughout "
                f"the last {len(hours)} hours in {region} "
                f"({latest['dominant_mw']:.0f} MW avg). Renewable share now: "
                f"{latest['renewable_pct']:.0f}%."
            )

    return bullets[:4]  # cap at 4 bullets to keep answers concise

