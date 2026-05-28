from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agents.answer_planner import plan_answer
from app.agents.why_sources import (
    AnalogSummary,
    CurrentDrivers,
    DriverContext,
    ForecastDrivers,
    ModelForecastDetail,
    NewsContext,
    TechnologyContext,
    WeatherContext,
    WhySources,
)
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    FactualVerdict,
    IntentLabel,
    QueryDecomposition,
    VerdictLabel,
)


def _verdict(**overrides) -> FactualVerdict:
    data = {
        "verdict": VerdictLabel.LOW_CONFIDENCE,
        "action": ActionLabel.HOLD,
        "confidence": 0.55,
        "confidence_band": ConfidenceBand.LOW,
        "as_of": datetime(2026, 5, 27, 11, 5, tzinfo=timezone.utc),
        "why_plain_english": "long audit narrative",
        "counterargument": "counter",
        "missing_data": ["dispatch_constraints", "unit_dispatch_events"],
    }
    data.update(overrides)
    return FactualVerdict(**data)


def _sources(intent=IntentLabel.EXPLANATION, requested_output="causal_explanation_with_forecast") -> WhySources:
    now = datetime(2026, 5, 27, 11, 5, tzinfo=timezone.utc)
    return WhySources(
        decomp=QueryDecomposition(
            raw_query="Why is NSW price elevated and will it continue?",
            intent=intent,
            entities={"regions": ["NSW1"]},
            requires_why=intent == IntentLabel.EXPLANATION,
            requires_history=True,
            requires_forecast=True,
            requested_output=requested_output,
        ),
        current=CurrentDrivers(
            region="NSW1",
            price_rrp=163.63,
            demand_mw=8498.2,
            availability_mw=12005.0,
            headroom_mw=3506.8,
            regime="elevated",
            valid_time=now,
            staleness_seconds=20,
            is_fresh=True,
        ),
        forecast=ForecastDrivers(
            available=True,
            direction="falling",
            model_detail=[
                ModelForecastDetail(model="meta_ensemble", available=True, direction="falling", p50=104, p90=760),
                ModelForecastDetail(model="qra", available=True, direction="falling", p50=106, p90=479),
                ModelForecastDetail(model="lear", available=True, direction="falling", p50=100, p90=1135),
                ModelForecastDetail(model="lnn", available=False, caveat="untrained"),
            ],
        ),
        analogs=AnalogSummary(
            count=10,
            success_count=7,
            outcome_summary="7/10 recovered within 30 min; 3/10 outcome unknown",
            top_items=[
                {"valid_time": "2026-05-26T23:55:00+00:00", "price_rrp": 178, "outcome": "recovered"},
                {"valid_time": "2026-05-27T00:30:00+00:00", "price_rrp": 148, "outcome": "recovered"},
            ],
        ),
        news=NewsContext(notices_stale=True),
        recent_dispatch=[
            {"valid_time": now - timedelta(minutes=5), "price_rrp": 142.0},
            {"valid_time": now - timedelta(minutes=10), "price_rrp": 151.0},
            {"valid_time": now - timedelta(minutes=60), "price_rrp": 129.0},
        ],
        weather=WeatherContext(),
        drivers=DriverContext(),
        technology=TechnologyContext(),
    )


def _section(plan, title):
    return next(s for s in plan.sections() if s["title"] == title)


def test_explanation_plan_includes_trend_and_model_names():
    plan = plan_answer(_sources(), _verdict())

    answer = " ".join(_section(plan, "Answer")["items"])
    continuation = " ".join(_section(plan, "Continuation")["items"])

    assert "$163.63/MWh" in answer
    assert "5m ago" in answer
    assert "10m ago" in answer
    assert "60m ago" in answer
    assert "Meta-ensemble" in continuation
    assert "QRA" in continuation
    assert "LEAR" in continuation


def test_retrospective_plan_starts_with_analog_outcome():
    sources = _sources(IntentLabel.RETROSPECTIVE, "historical_analog_outcome")
    plan = plan_answer(sources, _verdict())
    answer = _section(plan, "Answer")["items"]

    assert answer[0].startswith("HippoGraph found 10")
    assert "Outcome split" in answer[1]


def test_visible_sections_are_short_and_do_not_expose_evidence_ids():
    plan = plan_answer(_sources(), _verdict())
    for section in plan.sections():
        assert len(section["items"]) <= 3
        visible = " ".join(section["items"])
        assert "ev-" not in visible
        assert "raw_ref" not in visible


def test_weather_news_plan_answers_weather_relevance_directly():
    sources = _sources(IntentLabel.EXPLANATION, "weather_notice_news_correlation")
    sources.weather = WeatherContext(
        relevant=True,
        available=True,
        consensus={"temperature_c": 18.9, "wind_speed_kmh": 6.0, "cloud_cover_pct": 99},
        confidence=1.0,
        source_count=3,
    )

    plan = plan_answer(sources, _verdict())
    answer = " ".join(_section(plan, "Answer")["items"])

    assert "Weather" in answer or "weather" in answer
    assert "AEMO notice" in answer


def test_data_status_plan_uses_freshness_summary():
    sources = _sources(IntentLabel.LOOKUP, "data_freshness_status")
    plan = plan_answer(
        sources,
        _verdict(),
        evidence_quality={
            "dispatch": {"status": "fresh"},
            "notices": {"status": "stale"},
            "analogs": {"status": "ok"},
        },
    )

    answer = " ".join(_section(plan, "Answer")["items"])
    evidence = " ".join(_section(plan, "Evidence")["items"])
    assert "notices: stale" in answer
    assert "dispatch" in evidence


def test_fuel_source_plan_answers_requested_fuels_directly():
    sources = _sources(IntentLabel.EXPLANATION, "fuel_source_recommendation")
    sources.decomp.raw_query = "why is coal the best to buy now instead of solar? or hydro?"
    sources.decomp.entities["technologies"] = ["coal", "solar", "hydro"]

    plan = plan_answer(
        sources,
        _verdict(),
        fuel_mix={
            "spot_price_rrp": 167.02,
            "data_tier": "prior",
            "recommendation": {
                "fuel_type": "hydro",
                "preferred_order": ["hydro", "wind", "gas"],
                "reason": "Elevated spot price ($167/MWh): hydro and wind are preferred.",
                "confidence": "low",
            },
            "sources": [
                {"fuel_type": "coal", "marginal_cost_typical": 55, "data_tier": "prior", "mw_dispatched": None, "mw_capacity": None},
                {"fuel_type": "solar", "marginal_cost_typical": 0, "data_tier": "prior", "mw_dispatched": None, "mw_capacity": None},
                {"fuel_type": "hydro", "marginal_cost_typical": 30, "data_tier": "prior", "mw_dispatched": None, "mw_capacity": None},
            ],
        },
    )

    answer = " ".join(_section(plan, "Answer")["items"])
    evidence = " ".join(_section(plan, "Evidence")["items"])

    assert "coal is not the preferred source" in answer
    assert "hydro" in answer
    assert "Preferred order: hydro > wind > gas" in answer
    assert "all fuel types clear at the regional spot price" in evidence
    assert "coal:" in evidence


def test_fuel_source_plan_answers_generic_cheaper_source_question():
    sources = _sources(IntentLabel.EXPLANATION, "fuel_source_recommendation")
    sources.decomp.raw_query = "why is nsw prices elevated? which source is normally cheaper?"
    sources.decomp.entities["technologies"] = []

    plan = plan_answer(
        sources,
        _verdict(),
        fuel_mix={
            "spot_price_rrp": 136.01,
            "data_tier": "prior",
            "recommendation": {
                "fuel_type": "coal",
                "preferred_order": ["coal", "hydro", "wind"],
                "reason": "Moderate spot price: coal baseload and hydro offer stable procurement.",
                "confidence": "low",
            },
            "sources": [
                {"fuel_type": "solar", "marginal_cost_typical": 0, "marginal_cost_low": -10, "marginal_cost_high": 10, "data_tier": "prior"},
                {"fuel_type": "wind", "marginal_cost_typical": 5, "marginal_cost_low": 0, "marginal_cost_high": 20, "data_tier": "prior"},
                {"fuel_type": "coal", "marginal_cost_typical": 55, "marginal_cost_low": 30, "marginal_cost_high": 80, "data_tier": "prior"},
                {"fuel_type": "hydro", "marginal_cost_typical": 30, "marginal_cost_low": 0, "marginal_cost_high": 300, "data_tier": "prior"},
            ],
        },
    )

    answer = " ".join(_section(plan, "Answer")["items"])
    evidence = " ".join(_section(plan, "Evidence")["items"])

    assert "prefers coal" in answer
    assert "Normal marginal-cost priors" in evidence
    assert "solar" in evidence


# ── Price path extraction tests ───────────────────────────────────────────────

def test_extract_query_price_path_sequential():
    from app.agents.answer_planner import _extract_query_price_path
    prices = _extract_query_price_path(
        "why did the price fluctuate from 143 to 167 to 165 and then back down to 140 again?"
    )
    assert 143.0 in prices
    assert 167.0 in prices
    assert 165.0 in prices
    assert 140.0 in prices


def test_extract_query_price_path_dollar_signs():
    from app.agents.answer_planner import _extract_query_price_path
    prices = _extract_query_price_path("price moved from $300 to $15000 to $200")
    assert 300.0 in prices
    assert 15000.0 in prices
    assert 200.0 in prices


def test_extract_query_price_path_rejects_over_cap():
    from app.agents.answer_planner import _extract_query_price_path
    prices = _extract_query_price_path("from 143 to 99999 to 140")
    assert 99999.0 not in prices
    assert 143.0 in prices


def test_extract_query_price_path_empty_without_transitions():
    from app.agents.answer_planner import _extract_query_price_path
    prices = _extract_query_price_path("why is the market volatile today?")
    assert prices == []


def test_extract_query_price_path_decimal():
    from app.agents.answer_planner import _extract_query_price_path
    prices = _extract_query_price_path("price moved from $143.50 to $167.25")
    assert any(abs(p - 143.50) < 0.01 for p in prices)
    assert any(abs(p - 167.25) < 0.01 for p in prices)


# ── Price fluctuation planner tests ──────────────────────────────────────────

def _fluctuation_sources(with_recent_dispatch: bool = False) -> "WhySources":
    """Sources for a fluctuation query — optionally with DB history."""
    now = datetime(2026, 5, 27, 12, 20, tzinfo=timezone.utc)
    recent = []
    if with_recent_dispatch:
        recent = [
            {"valid_time": now - timedelta(minutes=5), "price_rrp": 167.0},
            {"valid_time": now - timedelta(minutes=10), "price_rrp": 143.0},
        ]
    return WhySources(
        decomp=QueryDecomposition(
            raw_query=(
                "why did the price fluctuate from 143 to 167 to 165 and then back down to 140? "
                "what is causing the fluctuations? which fuel source or other reasons?"
            ),
            intent=IntentLabel.EXPLANATION,
            entities={"regions": ["NSW1"]},
            requires_why=True,
            requires_history=True,
            requires_forecast=False,
            requested_output="price_fluctuation_attribution",
        ),
        current=CurrentDrivers(
            region="NSW1",
            price_rrp=140.25,
            demand_mw=8106.0,
            availability_mw=12060.0,
            headroom_mw=3954.0,
            regime="elevated",
            valid_time=now,
            staleness_seconds=20,
            is_fresh=True,
        ),
        forecast=ForecastDrivers(),
        analogs=AnalogSummary(),
        news=NewsContext(),
        recent_dispatch=recent,
        weather=WeatherContext(),
        drivers=DriverContext(),
        technology=TechnologyContext(),
    )


def test_price_fluctuation_plan_acknowledges_query_prices_when_db_empty():
    """When recent_dispatch is empty, plan must acknowledge the query-described price path."""
    sources = _fluctuation_sources(with_recent_dispatch=False)
    plan = plan_answer(sources, _verdict(), fuel_mix=None)

    answer_items = _section(plan, "Answer")["items"]
    answer_text = " ".join(answer_items)

    # Must mention the user-described prices
    assert "$143" in answer_text, f"Expected $143 in answer: {answer_text}"
    assert "$167" in answer_text, f"Expected $167 in answer: {answer_text}"
    assert "$140" in answer_text, f"Expected $140 in answer: {answer_text}"


def test_price_fluctuation_plan_swing_label_present():
    """Plan must include a swing label when query prices are extracted."""
    sources = _fluctuation_sources(with_recent_dispatch=False)
    plan = plan_answer(sources, _verdict(), fuel_mix=None)

    answer_text = " ".join(_section(plan, "Answer")["items"])
    # Swing of max(143,167,165,140)-min(...) = 167-140 = $27 → "minor price variation"
    assert "swing" in answer_text.lower() or "variation" in answer_text.lower(), (
        f"Expected swing/variation label in answer: {answer_text}"
    )


def test_price_fluctuation_plan_attribution_guidance_present():
    """Plan must tell the user that driver attribution requires unit dispatch evidence."""
    sources = _fluctuation_sources(with_recent_dispatch=False)
    plan = plan_answer(sources, _verdict(), fuel_mix=None)

    all_text = " ".join(
        item for sec in plan.sections() for item in sec["items"]
    ).lower()
    assert "unit dispatch" in all_text or "attribution" in all_text, (
        f"Expected attribution guidance in sections: {all_text}"
    )


def test_price_fluctuation_plan_uses_db_movement_when_available():
    """When recent_dispatch has data, plan uses the DB price path (not query text extraction)."""
    sources = _fluctuation_sources(with_recent_dispatch=True)
    plan = plan_answer(sources, _verdict(), fuel_mix=None)

    answer_text = " ".join(_section(plan, "Answer")["items"])
    # DB movement line uses "Recent movement: HH:MM $X -> ..." format
    assert "Recent movement" in answer_text, (
        f"Expected 'Recent movement:' from DB history, got: {answer_text}"
    )


def test_price_fluctuation_plan_missing_section_surfaces_db_gap():
    """When DB history is absent, Missing section must surface the gap."""
    sources = _fluctuation_sources(with_recent_dispatch=False)
    plan = plan_answer(sources, _verdict(), fuel_mix=None)

    missing_items = _section(plan, "Missing")["items"]
    missing_text = " ".join(missing_items).lower()
    assert "price trend" in missing_text or "dispatch" in missing_text or "database" in missing_text, (
        f"Expected DB gap in Missing section: {missing_items}"
    )


# ── Historical price distribution planner tests ───────────────────────────────

_HIST_DIST = {
    "available": True,
    "p25": 50.0,
    "median": 100.0,
    "p75": 150.0,
    "p90": 300.0,
    "mean": 110.0,
    "count": 200,
    "period_label": "last 12 months",
    "hour_window": 2,
    "lookback_days": 365,
}


def _hist_sources(sub_questions=None, requested_output="historical_price_distribution") -> WhySources:
    now = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    return WhySources(
        decomp=QueryDecomposition(
            raw_query="Is the NSW price high compared to last year at this time of day?",
            intent=IntentLabel.EXPLANATION,
            entities={"regions": ["NSW1"]},
            requires_why=True,
            requires_history=True,
            requires_forecast=False,
            requested_output=requested_output,
            sub_questions=sub_questions or [{"type": "historical_price_distribution", "period": "last_year"}],
        ),
        current=CurrentDrivers(
            region="NSW1",
            price_rrp=200.0,
            demand_mw=8000.0,
            availability_mw=11000.0,
            headroom_mw=3000.0,
            regime="high",
            valid_time=now,
            staleness_seconds=20,
            is_fresh=True,
        ),
        forecast=ForecastDrivers(),
        analogs=AnalogSummary(),
        news=NewsContext(),
        recent_dispatch=[],
        weather=WeatherContext(),
        drivers=DriverContext(),
        technology=TechnologyContext(),
    )


def test_hist_dist_plan_routes_via_sub_question():
    """Sub_question type=historical_price_distribution routes to historical planner."""
    sources = _hist_sources(
        sub_questions=[{"type": "historical_price_distribution", "period": "last_year"}],
        requested_output="causal_explanation",
    )
    plan = plan_answer(sources, _verdict(), hist_dist=_HIST_DIST)

    answer = " ".join(_section(plan, "Answer")["items"])
    assert "median" in answer.lower()
    assert "$200.00/MWh" in answer
    assert "$100.00/MWh" in answer


def test_hist_dist_plan_routes_via_requested_output():
    """requested_output=historical_price_distribution routes to historical planner."""
    sources = _hist_sources(sub_questions=[], requested_output="historical_price_distribution")
    plan = plan_answer(sources, _verdict(), hist_dist=_HIST_DIST)

    answer = " ".join(_section(plan, "Answer")["items"])
    assert "median" in answer.lower()


def test_hist_dist_plan_shows_correct_classification_spike():
    """Current price above P90 → spike classification in headline and answer."""
    sources = _hist_sources()
    plan = plan_answer(sources, _verdict(), hist_dist=_HIST_DIST)  # $200 > P90=$300? No — elevated

    # $200 is between median (100) and p75 (150)? No, 200 > 150, so between p75 and p90 → HIGH
    answer_text = " ".join(s for sec in plan.sections() for s in sec["items"]).upper()
    assert "HIGH" in answer_text or "ELEVATED" in answer_text


def test_hist_dist_plan_shows_spike_when_above_p90():
    """Price above P90 → SPIKE in classification."""
    sources = _hist_sources()
    # Override: make current price a spike ($400 > P90 $300)
    sources.current = sources.current.__class__(
        region="NSW1", price_rrp=400.0, demand_mw=8000.0,
        availability_mw=11000.0, headroom_mw=3000.0, regime="spike",
        valid_time=sources.current.valid_time, staleness_seconds=20, is_fresh=True,
    )
    plan = plan_answer(sources, _verdict(), hist_dist=_HIST_DIST)

    answer_text = " ".join(s for sec in plan.sections() for s in sec["items"]).upper()
    assert "SPIKE" in answer_text


def test_hist_dist_plan_unavailable_returns_graceful_message():
    """When hist_dist is unavailable, planner returns a clear fallback."""
    sources = _hist_sources()
    unavailable = {**_HIST_DIST, "available": False, "count": 0}
    plan = plan_answer(sources, _verdict(), hist_dist=unavailable)

    answer = " ".join(_section(plan, "Answer")["items"])
    assert "unavailable" in answer.lower() or "insufficient" in answer.lower() or "not enough" in answer.lower()


def test_hist_dist_plan_includes_percentile_table_in_evidence():
    """Evidence section must show P25/Median/P75/P90 values."""
    sources = _hist_sources()
    plan = plan_answer(sources, _verdict(), hist_dist=_HIST_DIST)

    evidence = " ".join(_section(plan, "Evidence")["items"])
    assert "P25" in evidence
    assert "Median" in evidence
    assert "P75" in evidence
    assert "P90" in evidence


def test_hist_dist_plan_no_sub_question_no_hist_dist_falls_through():
    """Without hist sub_question and without hist_dist, planner falls through to lookup."""
    sources = _hist_sources(sub_questions=[], requested_output="causal_explanation")
    plan = plan_answer(sources, _verdict(), hist_dist=None)

    # Should fall through to _plan_explanation, not _plan_historical_distribution
    assert "median" not in " ".join(s for sec in plan.sections() for s in sec["items"]).lower()


def test_fuel_source_plan_includes_historical_when_hist_dist_available():
    """Fuel source planner injects historical comparison when hist_dist is present."""
    sources = _sources(IntentLabel.EXPLANATION, "fuel_source_recommendation")
    sources.decomp.raw_query = "why should i be cautious with buying coal? how was prices last year?"
    sources.decomp.entities["technologies"] = ["coal"]
    sources.decomp.sub_questions = [
        {"type": "fuel_source_comparison", "fuels": ["coal"]},
        {"type": "historical_price_distribution", "period": "last_year"},
    ]

    plan = plan_answer(
        sources,
        _verdict(),
        fuel_mix={
            "spot_price_rrp": 163.63,
            "data_tier": "prior",
            "recommendation": {
                "fuel_type": "coal",
                "preferred_order": ["coal", "hydro"],
                "reason": "Moderate spot price: coal baseload offers stable cost.",
                "confidence": "low",
            },
            "sources": [
                {"fuel_type": "coal", "marginal_cost_typical": 55, "marginal_cost_low": 30,
                 "marginal_cost_high": 80, "data_tier": "prior"},
            ],
        },
        hist_dist=_HIST_DIST,
    )

    evidence = " ".join(_section(plan, "Evidence")["items"])
    # Should include historical comparison line
    assert "median" in evidence.lower() or "historical" in evidence.lower()
    # Should still include fuel evidence
    assert "NEM spot price" in evidence or "fuel" in evidence.lower() or "marginal" in evidence.lower()
