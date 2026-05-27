"""Why Engine — Part 1/3: Source assembly.

Assembles the 5 evidence inputs into a WhySources bundle:
  1. Current market drivers (from dispatch price + regime)
  2. Forecast drivers (predispatch, QRA/LEAR/LNN where available)
  3. Historical analogs (HippoGraph — Week 2)
  4. Bitemporal trace (from Trace table — available immediately)
  5. News/market notice (from AEMO notices cache)

WhySources is passed to why_builder.py which produces the narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents.scatter_gather import GatherResult
from app.core.interfaces import RegimeState
from app.core.schema import QueryDecomposition


@dataclass
class CurrentDrivers:
    region: str
    price_rrp: float
    demand_mw: float
    availability_mw: float
    headroom_mw: float
    regime: str
    valid_time: datetime
    staleness_seconds: int
    is_fresh: bool
    regime_state: RegimeState | None = None   # None until ChronoGraph warms up


@dataclass
class ModelForecastDetail:
    """Per-model forecast breakdown for the WhyEngine narrative."""
    model: str                        # "lnn", "lear", "qra", "predispatch"
    available: bool = False
    direction: str = "unknown"        # rising | falling | flat | unknown
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    caveat: str | None = None         # e.g. "checkpoint age 4d", "untrained"


@dataclass
class ForecastDrivers:
    available: bool = False
    direction: str = "unknown"      # rising | falling | flat | unknown
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    horizon_intervals: int = 6      # how many 5-min intervals ahead
    model_detail: list[ModelForecastDetail] = field(default_factory=list)  # per-model breakdown


@dataclass
class AnalogSummary:
    count: int = 0
    success_count: int = 0
    window_days: int = 90
    method: str = "state_vector_ppr"
    outcome_summary: str | None = None
    top_items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.count, 1)


@dataclass
class NewsContext:
    explained: bool = False
    notices: list[dict[str, Any]] = field(default_factory=list)
    commentary_items: list[dict[str, Any]] = field(default_factory=list)
    auto_commentary: list[dict[str, Any]] = field(default_factory=list)  # Sprint Q: recent commentary events
    top_notice_type: str | None = None
    top_notice_title: str | None = None
    credibility_tier: int | None = None    # 1=AEMO official, 2=press
    notices_stale: bool = False            # True if cache is older than notices_max_age_s
    news_stale: bool = False               # True if RSS cache is older than nem_news_max_age_s


@dataclass
class WeatherContext:
    relevant: bool = False
    available: bool = False
    consensus: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_count: int = 0
    tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass
class DriverContext:
    events: list[dict[str, Any]] = field(default_factory=list)
    binding_constraints: list[dict[str, Any]] = field(default_factory=list)
    tight_interconnectors: list[dict[str, Any]] = field(default_factory=list)
    has_confirmed_driver: bool = False
    # Sprint C: richer interconnector causality
    interconnector_causal_role: str = "unknown"   # "causal"|"contributing"|"not_relevant"|"unknown"
    interconnector_narrative: str = ""            # one-line causal explanation for why_builder
    interconnector_binding_count: int = 0


@dataclass
class TechnologyContext:
    events: list[dict[str, Any]] = field(default_factory=list)
    by_fuel: dict[str, dict[str, Any]] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    has_unit_evidence: bool = False


@dataclass
class WhySources:
    decomp: QueryDecomposition
    current: CurrentDrivers
    forecast: ForecastDrivers
    analogs: AnalogSummary
    news: NewsContext
    recent_dispatch: list[dict[str, Any]] = field(default_factory=list)
    weather: WeatherContext = field(default_factory=WeatherContext)
    drivers: DriverContext = field(default_factory=DriverContext)
    technology: TechnologyContext = field(default_factory=TechnologyContext)
    source_coverage: float = 0.0   # fraction of scatter_gather tasks that returned data


@dataclass
class SeasonalSources:
    region: str
    season_buckets: list[dict[str, Any]]
    summaries: list[dict[str, Any]]


def _summarise_outcomes(analogs: list[dict]) -> str | None:
    """Produce a one-line outcome summary for the adversarial critic."""
    if not analogs:
        return None
    total = len(analogs)
    recovered = sum(1 for a in analogs if a.get("outcome") == "recovered")
    spiked = sum(1 for a in analogs if a.get("outcome") == "continued_spike")
    unknown = total - recovered - spiked
    parts = []
    if recovered:
        parts.append(f"{recovered}/{total} recovered within 30 min")
    if spiked:
        parts.append(f"{spiked}/{total} continued spike beyond 30 min")
    if unknown:
        parts.append(f"{unknown}/{total} outcome unknown")
    return "; ".join(parts) if parts else None


def _derive_forecast_from_predispatch(
    intervals: list[dict], region: str
) -> ForecastDrivers:
    """Build ForecastDrivers from AEMO pre-dispatch intervals (up to 2 hours ahead).

    Uses the next 6 intervals (30 min) for P10/P50/P90 and direction.
    P10 = min RRP, P50 = median, P90 = max across the window.
    """
    region_up = region.upper()
    rows = sorted(
        [iv for iv in intervals if iv.get("region", "").upper() == region_up],
        key=lambda x: x.get("interval_datetime", ""),
    )
    if not rows:
        return ForecastDrivers(available=False, direction="unknown")

    window = rows[:6]  # up to 30 min ahead (6 × 5-min intervals)
    prices = [r["rrp"] for r in window if "rrp" in r]
    if not prices:
        return ForecastDrivers(available=False, direction="unknown")

    prices_sorted = sorted(prices)
    n = len(prices_sorted)
    p10 = prices_sorted[max(0, int(n * 0.1))]
    p50 = prices_sorted[n // 2]
    p90 = prices_sorted[min(n - 1, int(n * 0.9))]

    if len(prices) >= 2:
        slope = (prices[-1] - prices[0]) / max(len(prices) - 1, 1)
        slope_pct = slope / max(abs(prices[0]), 1.0)
        if slope_pct > 0.05:
            direction = "rising"
        elif slope_pct < -0.05:
            direction = "falling"
        else:
            direction = "flat"
    else:
        direction = "flat"

    return ForecastDrivers(
        available=True,
        direction=direction,
        p10=p10,
        p50=p50,
        p90=p90,
        horizon_intervals=len(window),
    )


def _derive_forecast_direction(classifier) -> ForecastDrivers:
    """Linear trend over ChronoGraph's ADWIN window → rising/falling/flat."""
    if not hasattr(classifier, '_adwin') or classifier._adwin.width < 6:
        return ForecastDrivers(available=False, direction="unknown")

    recent = list(classifier._adwin.values)[-12:]  # last 12 × 5min = 1 hour
    if len(recent) < 4:
        return ForecastDrivers(available=False, direction="unknown")

    n = len(recent)
    mean_x = (n - 1) / 2.0
    mean_y = sum(recent) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(recent))
    den = sum((i - mean_x) ** 2 for i in range(n))
    slope = num / den if den > 1e-10 else 0.0

    slope_pct = slope / max(abs(mean_y), 1.0)
    if slope_pct > 0.05:
        direction = "rising"
    elif slope_pct < -0.05:
        direction = "falling"
    else:
        direction = "flat"

    return ForecastDrivers(available=True, direction=direction, horizon_intervals=12)


def _build_model_detail_from_live_forecast(gather, classifier) -> list[ModelForecastDetail]:
    """Parse GatherResult.live_forecast into per-model ModelForecastDetail entries.

    Returns entries for LNN, LEAR, and QRA. Falls back to placeholder entries
    when live_forecast is unavailable (cold DB, insufficient history, timeout).
    """
    live = gather.live_forecast  # full run_live_forecast() result or None

    if not live or not live.get("available"):
        # Explain why — distinguish "no DB history yet" from timeout
        reason = live.get("reason", "insufficient data") if live else "not available"
        return [
            ModelForecastDetail(model="lnn",  available=False, caveat=_lnn_caveat(gather)),
            ModelForecastDetail(model="lear", available=False, caveat=reason),
            ModelForecastDetail(model="qra",  available=False, caveat=reason),
        ]

    detail: list[ModelForecastDetail] = []
    forecasts_by_model = {fc["model"]: fc for fc in live.get("forecasts", [])}
    errors_by_model = {e["model"]: e["error"] for e in live.get("errors", [])}

    preferred = [
        ("meta_ensemble", "meta_ensemble"),
        ("qra", "qra"),
        ("lear", "lear"),
        ("experimental_lnn", "lnn"),
        ("lnn", "lnn"),
        ("gbm", "gbm"),
        ("tcn", "tcn"),
        ("seasonal_naive", "seasonal_naive"),
        ("persistence", "persistence"),
        ("aemo_predispatch", "aemo_predispatch"),
    ]
    seen_models: set[str] = set()

    for model_key, label in preferred:
        fc = forecasts_by_model.get(model_key)
        if fc and fc.get("p50"):
            seen_models.add(model_key)
            p50_list = fc["p50"]
            p10_list = fc.get("p10", p50_list)
            p90_list = fc.get("p90", p50_list)
            p50 = p50_list[0] if p50_list else None
            p10 = p10_list[0] if p10_list else None
            p90 = p90_list[0] if p90_list else None
            # Derive direction from p50 vs current price
            current = gather.dispatch.price_rrp if gather.dispatch else None
            direction = "unknown"
            if p50 is not None and current is not None:
                pct = (p50 - current) / max(abs(current), 1.0)
                direction = "rising" if pct > 0.03 else "falling" if pct < -0.03 else "flat"
            detail.append(ModelForecastDetail(
                model=label, available=True, direction=direction,
                p10=p10, p50=p50, p90=p90, caveat=fc.get("caveat"),
            ))
        else:
            err = errors_by_model.get(model_key, "not in forecast output")
            if label == "lnn":
                err = _lnn_caveat(gather)
            if model_key in {"experimental_lnn", "lear", "qra"}:
                detail.append(ModelForecastDetail(model=label, available=False, caveat=err))

    for model_key, fc in forecasts_by_model.items():
        if model_key in seen_models or not fc.get("p50"):
            continue
        p50_list = fc["p50"]
        p10_list = fc.get("p10", p50_list)
        p90_list = fc.get("p90", p50_list)
        p50 = p50_list[0] if p50_list else None
        p10 = p10_list[0] if p10_list else None
        p90 = p90_list[0] if p90_list else None
        current = gather.dispatch.price_rrp if gather.dispatch else None
        direction = "unknown"
        if p50 is not None and current is not None:
            pct = (p50 - current) / max(abs(current), 1.0)
            direction = "rising" if pct > 0.03 else "falling" if pct < -0.03 else "flat"
        detail.append(ModelForecastDetail(
            model=model_key, available=True, direction=direction,
            p10=p10, p50=p50, p90=p90, caveat=fc.get("caveat"),
        ))

    return detail


def _lnn_caveat(gather) -> str:
    """Return a useful LNN caveat string based on trainer state."""
    try:
        from app.engines.forecasting.inference import get_trainer
        trainer = get_trainer(gather.dispatch.region if gather.dispatch else "NSW1")
        buf = getattr(trainer, "_buffer_count", None) or 0
        if not trainer.is_trained:
            return f"untrained ({buf}/288 intervals ingested)"
    except Exception:
        pass
    return "unavailable"


def assemble_why_sources(
    decomp: QueryDecomposition,
    gather: GatherResult,
    region: str,
) -> WhySources:
    """Build WhySources from decomposition + gather results."""
    from domain.nem.adapter import NEM_REGIME_THRESHOLDS, classify_regime
    from app.engines.chronograph.regime import get_classifier

    dp = gather.dispatch

    # 1. Current drivers — also wire ChronoGraph classifier (used by forecast below)
    thresholds = NEM_REGIME_THRESHOLDS.get(region, NEM_REGIME_THRESHOLDS["NSW1"])
    classifier = get_classifier(region, thresholds)

    if dp:
        headroom = max(dp.availability_mw - dp.demand_mw, 0.0)
        age = int((datetime.now(timezone.utc) - dp.valid_time).total_seconds())

        # Feed ChronoGraph — enriches the regime with quantile rank + change-point
        regime_state = classifier.observe(dp.price_rrp, dp.valid_time)

        current = CurrentDrivers(
            region=region,
            price_rrp=dp.price_rrp,
            demand_mw=dp.demand_mw,
            availability_mw=dp.availability_mw,
            headroom_mw=headroom,
            regime=regime_state.label,
            valid_time=dp.valid_time,
            staleness_seconds=age,
            is_fresh=gather.dispatch_fresh,
            regime_state=regime_state,
        )
    else:
        current = CurrentDrivers(
            region=region,
            price_rrp=0.0,
            demand_mw=0.0,
            availability_mw=0.0,
            headroom_mw=0.0,
            regime="unknown",
            valid_time=datetime.now(timezone.utc),
            staleness_seconds=9999,
            is_fresh=False,
        )

    # 2. Forecast drivers — built from live_forecast (LEAR/QRA/LNN ensemble)
    #    with AEMO pre-dispatch as fallback, and ChronoGraph trend as last resort
    lnn = gather.forecast  # best-model p10/p50/p90 dict extracted by scatter_gather
    model_detail: list[ModelForecastDetail] = _build_model_detail_from_live_forecast(
        gather, classifier
    )

    # AEMO pre-dispatch detail (always appended as an independent source)
    if gather.predispatch:
        pd_drivers = _derive_forecast_from_predispatch(gather.predispatch, region)
        if pd_drivers.available:
            model_detail.append(ModelForecastDetail(
                model="predispatch",
                available=True,
                direction=pd_drivers.direction,
                p10=pd_drivers.p10,
                p50=pd_drivers.p50,
                p90=pd_drivers.p90,
            ))
        else:
            model_detail.append(ModelForecastDetail(
                model="predispatch", available=False, caveat="no predispatch intervals"
            ))
    else:
        model_detail.append(ModelForecastDetail(
            model="predispatch", available=False, caveat="not fetched"
        ))

    # Build the combined ForecastDrivers from the best available source
    if lnn and lnn.get("p50") is not None:
        direction = _derive_forecast_direction(classifier).direction
        forecast = ForecastDrivers(
            available=True,
            direction=direction,
            p10=lnn["p10"],
            p50=lnn["p50"],
            p90=lnn["p90"],
            horizon_intervals=1,
            model_detail=model_detail,
        )
    elif gather.predispatch:
        pd_f = _derive_forecast_from_predispatch(gather.predispatch, region)
        forecast = ForecastDrivers(
            available=pd_f.available,
            direction=pd_f.direction,
            p10=pd_f.p10,
            p50=pd_f.p50,
            p90=pd_f.p90,
            horizon_intervals=pd_f.horizon_intervals,
            model_detail=model_detail,
        )
    else:
        base = _derive_forecast_direction(classifier)
        forecast = ForecastDrivers(
            available=base.available,
            direction=base.direction,
            p10=base.p10,
            p50=base.p50,
            p90=base.p90,
            horizon_intervals=base.horizon_intervals,
            model_detail=model_detail,
        )

    # 3. Analogs — count outcomes from PPR results
    analog_list = gather.analogs
    success_count = sum(
        1 for a in analog_list if a.get("outcome") == "recovered"
    )
    analogs = AnalogSummary(
        count=len(analog_list),
        success_count=success_count,
        outcome_summary=_summarise_outcomes(analog_list),
        top_items=analog_list[:3],
    )

    # 4. News — credibility_tier comes from the notice dict (set by AEMOMarketNoticesClient)
    notices = gather.notices
    news_explained = bool(notices)
    top_type = notices[0].get("notice_type") if notices else None
    top_title = notices[0].get("reason") if notices else None
    tier = notices[0].get("credibility_tier") if notices else None

    news = NewsContext(
        explained=news_explained,
        notices=notices,
        commentary_items=gather.news_items,
        auto_commentary=gather.commentary_context,
        top_notice_type=top_type,
        top_notice_title=top_title,
        credibility_tier=tier,
        notices_stale=gather.notices_stale,
        news_stale=gather.news_stale,
    )

    from app.mcp.weather_client import weather_query_relevant
    weather_raw = gather.weather or {}
    weather_consensus = weather_raw.get("consensus", {}) if weather_raw else {}
    weather = WeatherContext(
        relevant=weather_query_relevant(decomp.raw_query),
        available=bool(weather_raw and weather_consensus),
        consensus=weather_consensus,
        confidence=float(weather_raw.get("confidence") or 0.0) if weather_raw else 0.0,
        source_count=len(weather_raw.get("readings", [])) if weather_raw else 0,
        tags=list(weather_raw.get("relevance_tags", [])) if weather_raw else [],
        raw=weather_raw or None,
    )

    from app.engines.driver_attribution import summarise_driver_events
    driver_summary = summarise_driver_events(gather.driver_events)

    # Sprint C: interconnector causality
    _interconnector_ctx = None
    try:
        from app.engines.interconnector import (
            classify_interconnector_causality,
            parse_interconnector_events,
        )
        _ic_statuses = parse_interconnector_events(gather.driver_events, region)
        _interconnector_ctx = classify_interconnector_causality(
            _ic_statuses, region, current.price_rrp, current.regime,
        )
    except Exception:
        pass

    drivers = DriverContext(
        events=gather.driver_events,
        binding_constraints=driver_summary["binding_constraints"],
        tight_interconnectors=driver_summary["tight_interconnectors"],
        has_confirmed_driver=driver_summary["has_confirmed_driver"],
        interconnector_causal_role=_interconnector_ctx.causal_role if _interconnector_ctx else "unknown",
        interconnector_narrative=_interconnector_ctx.primary_note if _interconnector_ctx else "",
        interconnector_binding_count=_interconnector_ctx.binding_count if _interconnector_ctx else 0,
    )

    from app.engines.unit_attribution import summarise_unit_dispatch
    technology_summary = summarise_unit_dispatch(gather.unit_events)
    technology = TechnologyContext(
        events=gather.unit_events,
        by_fuel=technology_summary["by_fuel"],
        caveats=technology_summary["caveats"],
        has_unit_evidence=technology_summary["has_unit_evidence"],
    )

    return WhySources(
        decomp=decomp,
        current=current,
        forecast=forecast,
        analogs=analogs,
        recent_dispatch=list(getattr(gather, "recent_dispatch", []) or []),
        news=news,
        weather=weather,
        drivers=drivers,
        technology=technology,
        source_coverage=gather.source_coverage,
    )
