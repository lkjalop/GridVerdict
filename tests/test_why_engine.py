"""Tests for why_builder adversarial critic and why_sources forecast assembly.

Proves:
  - Adversarial critic uses evidence (analog outcomes, notice status, LNN forecast,
    headroom) rather than a static regime template.
  - Intent-specific narrative branches produce correct output per IntentLabel.
  - AEMO pre-dispatch intervals are consumed to build ForecastDrivers correctly.
  - Every fresh-dispatch answer carries at least 2 EvidenceRefSchemas.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from app.agents.why_builder import build_why, _build_counterargument, _estimate_confidence
from app.agents.why_sources import (
    CurrentDrivers,
    ForecastDrivers,
    AnalogSummary,
    NewsContext,
    WhySources,
    _derive_forecast_from_predispatch,
    _summarise_outcomes,
)
from app.core.schema import IntentLabel, QueryDecomposition


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc)


def _make_current(
    region="NSW1",
    price=150.0,
    demand=8500.0,
    avail=9200.0,
    regime="elevated",
    fresh=True,
    staleness=30,
):
    return CurrentDrivers(
        region=region,
        price_rrp=price,
        demand_mw=demand,
        availability_mw=avail,
        headroom_mw=max(avail - demand, 0.0),
        regime=regime,
        valid_time=_now(),
        staleness_seconds=staleness,
        is_fresh=fresh,
    )


def _make_analogs(count=5, success_count=4, outcome_summary=None):
    return AnalogSummary(
        count=count,
        success_count=success_count,
        outcome_summary=outcome_summary,
    )


def _make_forecast(
    available=False,
    direction="unknown",
    p10=None,
    p50=None,
    p90=None,
    horizon=6,
):
    return ForecastDrivers(
        available=available,
        direction=direction,
        p10=p10,
        p50=p50,
        p90=p90,
        horizon_intervals=horizon,
    )


def _make_news(explained=False, notice_type=None, title=None, tier=None):
    notices = []
    if explained and notice_type:
        notices = [{"notice_type": notice_type, "reason": title or "", "credibility_tier": tier}]
    return NewsContext(
        explained=explained,
        notices=notices,
        top_notice_type=notice_type,
        top_notice_title=title,
        credibility_tier=tier,
    )


def _make_decomp(intent=IntentLabel.EXPLANATION, region="NSW1"):
    return QueryDecomposition(
        intent=intent,
        regions=[region],
        confidence=0.9,
        requires_history=False,
        requires_forecast=False,
        requires_why=True,
        raw_query=f"test query for {intent.value}",
    )


def _make_sources(
    intent=IntentLabel.EXPLANATION,
    region="NSW1",
    price=150.0,
    demand=8500.0,
    avail=9200.0,
    regime="elevated",
    fresh=True,
    analog_count=5,
    analog_success=4,
    forecast_available=False,
    forecast_direction="unknown",
    p10=None,
    p50=None,
    p90=None,
    news_explained=False,
    notice_type=None,
    tier=None,
    driver_events=None,
    unit_events=None,
):
    from app.engines.driver_attribution import summarise_driver_events
    from app.engines.unit_attribution import summarise_unit_dispatch
    from app.agents.why_sources import DriverContext, TechnologyContext
    driver_events = driver_events or []
    unit_events = unit_events or []
    driver_summary = summarise_driver_events(driver_events)
    technology_summary = summarise_unit_dispatch(unit_events)
    return WhySources(
        decomp=_make_decomp(intent=intent, region=region),
        current=_make_current(
            region=region, price=price, demand=demand, avail=avail,
            regime=regime, fresh=fresh,
        ),
        forecast=_make_forecast(
            available=forecast_available,
            direction=forecast_direction,
            p10=p10, p50=p50, p90=p90,
        ),
        analogs=_make_analogs(count=analog_count, success_count=analog_success),
        news=_make_news(
            explained=news_explained, notice_type=notice_type, tier=tier,
        ),
        drivers=DriverContext(
            events=driver_events,
            binding_constraints=driver_summary["binding_constraints"],
            tight_interconnectors=driver_summary["tight_interconnectors"],
            has_confirmed_driver=driver_summary["has_confirmed_driver"],
        ),
        technology=TechnologyContext(
            events=unit_events,
            by_fuel=technology_summary["by_fuel"],
            caveats=technology_summary["caveats"],
            has_unit_evidence=technology_summary["has_unit_evidence"],
        ),
        source_coverage=0.8,
    )


# ── Adversarial Critic — evidence-driven, not template ───────────────────────

class TestAdversarialCritic:

    def test_high_recovery_rate_signals_transient(self):
        """≥70% analog recovery → critic says spike may be transient."""
        c = _make_current(regime="spike", price=1200.0, demand=9000.0, avail=9400.0)
        analogs = _make_analogs(count=10, success_count=8)   # 80% recovery
        news = _make_news()
        forecast = _make_forecast()

        result = _build_counterargument(c, analogs, news, forecast)

        assert "80%" in result or "transient" in result.lower()
        assert "recovered" in result.lower()

    def test_low_recovery_rate_signals_sticky(self):
        """≤30% analog recovery → critic says regime is sticky."""
        c = _make_current(regime="spike", price=1800.0, demand=9100.0, avail=9300.0)
        analogs = _make_analogs(count=10, success_count=2)   # 20% recovery

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "sticky" in result.lower() or "continued" in result.lower()
        assert "80%" in result or "spiking" in result.lower()

    def test_mixed_recovery_rate_no_strong_signal(self):
        """~50% recovery → critic reports mixed signal, not transient or sticky."""
        c = _make_current(regime="elevated", price=300.0)
        analogs = _make_analogs(count=10, success_count=5)   # 50% recovery

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "mixed" in result.lower()
        assert "50%" in result

    def test_insufficient_analogs_states_no_base_rate(self):
        """Fewer than 3 analogs → critic says no historical base rate."""
        c = _make_current(regime="spike", price=900.0)
        analogs = _make_analogs(count=1, success_count=1)

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "insufficient" in result.lower() or "no historical" in result.lower()

    def test_zero_analogs_states_no_base_rate(self):
        """Zero analogs → critic explicitly says no history."""
        c = _make_current()
        analogs = _make_analogs(count=0, success_count=0)

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "insufficient" in result.lower() or "no historical" in result.lower()

    def test_tier1_notice_challenges_guarantee(self):
        """Tier-1 AEMO notice → critic warns notice can be cancelled rapidly."""
        c = _make_current(regime="spike", price=1500.0)
        analogs = _make_analogs(count=5, success_count=3)
        news = _make_news(explained=True, notice_type="LACK OF RESERVE 1", tier=1)

        result = _build_counterargument(c, analogs, news, _make_forecast())

        assert "cancel" in result.lower() or "downgrade" in result.lower()

    def test_no_notice_suggests_reversal(self):
        """No AEMO notice → critic says no fundamental cause, likely reversal."""
        c = _make_current(regime="elevated", price=250.0)
        analogs = _make_analogs(count=5, success_count=3)
        news = _make_news(explained=False)

        result = _build_counterargument(c, analogs, news, _make_forecast())

        assert "no confirmed" in result.lower() or "reverse" in result.lower()

    def test_lnn_p50_below_current_during_spike_anticipates_easing(self):
        """LNN P50 < current price + spike regime → critic says model anticipates easing."""
        c = _make_current(regime="spike", price=1200.0)
        analogs = _make_analogs(count=5, success_count=3)
        news = _make_news()
        forecast = _make_forecast(
            available=True, direction="falling",
            p10=200.0, p50=400.0, p90=800.0
        )

        result = _build_counterargument(c, analogs, news, forecast)

        assert "400" in result or "easing" in result.lower() or "lnn" in result.lower()

    def test_lnn_p50_above_current_during_normal_signals_pressure(self):
        """LNN P50 > current price + normal regime → critic says pressure building."""
        c = _make_current(regime="normal", price=60.0)
        analogs = _make_analogs(count=5, success_count=3)
        news = _make_news()
        forecast = _make_forecast(
            available=True, direction="rising",
            p10=70.0, p50=120.0, p90=250.0
        )

        result = _build_counterargument(c, analogs, news, forecast)

        assert "120" in result or "pressure" in result.lower() or "building" in result.lower()

    def test_ample_headroom_softens_spike_persistence(self):
        """Headroom > 1500 MW → critic says AEMO can bring capacity online."""
        c = _make_current(regime="spike", price=900.0, demand=6000.0, avail=8000.0)
        # headroom = 2000 MW
        analogs = _make_analogs(count=5, success_count=3)

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "capacity" in result.lower() or "headroom" in result.lower()

    def test_critically_low_headroom_sustains_extreme(self):
        """Headroom < 200 MW → critic says sustained extreme prices are plausible."""
        c = _make_current(regime="extreme", price=14500.0, demand=9950.0, avail=10000.0)
        # headroom = 50 MW
        analogs = _make_analogs(count=5, success_count=2)

        result = _build_counterargument(c, analogs, _make_news(), _make_forecast())

        assert "critically" in result.lower() or "sustained" in result.lower()

    def test_critic_is_not_same_for_spike_vs_normal_same_analogs(self):
        """Same analog evidence but different regimes should produce different critic text."""
        analogs = _make_analogs(count=10, success_count=8)
        news = _make_news()
        forecast = _make_forecast()

        c_spike = _make_current(regime="spike", price=1200.0, demand=9000.0, avail=9200.0)
        c_normal = _make_current(regime="normal", price=55.0, demand=7000.0, avail=9200.0)

        result_spike = _build_counterargument(c_spike, analogs, news, forecast)
        result_normal = _build_counterargument(c_normal, analogs, news, forecast)

        # Both should mention the same analog recovery rate
        assert "80%" in result_spike
        assert "80%" in result_normal
        # But headroom language should differ (spike has 200 MW, normal has 2200 MW)
        assert result_spike != result_normal


# ── Intent-specific narrative paths ──────────────────────────────────────────

class TestWhyBuilderIntentPaths:

    def test_action_recommendation_spike_tight_headroom_dispatch_signal(self):
        """ACTION_RECOMMENDATION + spike + <500 MW headroom → fast-response signal."""
        sources = _make_sources(
            intent=IntentLabel.ACTION_RECOMMENDATION,
            price=1500.0, demand=9400.0, avail=9600.0,   # 200 MW headroom
            regime="spike",
            analog_count=5, analog_success=2,
        )
        out = build_why(sources)

        assert "fast-response" in out.why_plain_english.lower() or \
               "favours" in out.why_plain_english.lower() or \
               "tight headroom" in out.why_plain_english.lower()

    def test_action_recommendation_normal_stable_signal(self):
        """ACTION_RECOMMENDATION + normal regime → no urgent dispatch signal."""
        sources = _make_sources(
            intent=IntentLabel.ACTION_RECOMMENDATION,
            price=65.0, demand=7000.0, avail=9500.0,
            regime="normal",
        )
        out = build_why(sources)

        assert "stable" in out.why_plain_english.lower() or \
               "no urgent" in out.why_plain_english.lower()

    def test_explanation_headroom_and_notice_lists_causes(self):
        """EXPLANATION + low headroom + tier-1 notice → primary drivers listed."""
        sources = _make_sources(
            intent=IntentLabel.EXPLANATION,
            price=900.0, demand=9200.0, avail=9500.0,   # 300 MW headroom
            regime="spike",
            news_explained=True, notice_type="LACK OF RESERVE 2", tier=1,
        )
        out = build_why(sources)

        assert "primary price driver" in out.why_plain_english.lower() or \
               "primary" in out.why_plain_english.lower()

    def test_explanation_no_cause_acknowledges_uncertainty(self):
        """EXPLANATION + no notice + normal headroom → uncertainty acknowledged."""
        sources = _make_sources(
            intent=IntentLabel.EXPLANATION,
            price=400.0, demand=7500.0, avail=9000.0,
            regime="elevated",
            news_explained=False,
        )
        out = build_why(sources)

        assert (
            "no single dominant" in out.why_plain_english.lower()
            or "unannounced" in out.why_plain_english.lower()
        )
        assert "price_cause_unconfirmed" in out.missing_data

    def test_retrospective_with_analogs_mentions_archive(self):
        """RETROSPECTIVE + analogs → archive window and trace replay mentioned."""
        sources = _make_sources(
            intent=IntentLabel.RETROSPECTIVE,
            analog_count=8, analog_success=5,
        )
        out = build_why(sources)

        assert "archive" in out.why_plain_english.lower() or \
               "analog" in out.why_plain_english.lower()

    def test_retrospective_no_analogs_mentions_backfill(self):
        """RETROSPECTIVE + 0 analogs → backfill message."""
        sources = _make_sources(
            intent=IntentLabel.RETROSPECTIVE,
            analog_count=0, analog_success=0,
        )
        out = build_why(sources)

        assert "backfill" in out.why_plain_english.lower() or \
               "no close" in out.why_plain_english.lower()

    def test_counterfactual_with_analogs_uses_outcome_summary(self):
        """COUNTERFACTUAL + analogs with outcome summary → outcome used."""
        analogs = AnalogSummary(
            count=5, success_count=4,
            outcome_summary="4/5 recovered within 30 min",
        )
        sources = WhySources(
            decomp=_make_decomp(intent=IntentLabel.COUNTERFACTUAL),
            current=_make_current(regime="spike", price=800.0),
            forecast=_make_forecast(),
            analogs=analogs,
            news=_make_news(),
        )
        out = build_why(sources)

        assert "4/5" in out.why_plain_english or "analog" in out.why_plain_english.lower()

    def test_counterfactual_no_analogs_defers_to_live_operation(self):
        """COUNTERFACTUAL + 0 analogs → defers to live operation message."""
        sources = _make_sources(
            intent=IntentLabel.COUNTERFACTUAL,
            analog_count=0, analog_success=0,
        )
        out = build_why(sources)

        assert "insufficient analog" in out.why_plain_english.lower() or \
               "live operation" in out.why_plain_english.lower()

    def test_comparison_mentions_transmission_constraints(self):
        """COMPARISON intent → transmission constraints sentence."""
        sources = _make_sources(intent=IntentLabel.COMPARISON)
        out = build_why(sources)

        assert "transmission" in out.why_plain_english.lower()

    def test_constraint_driver_is_cited_when_available(self):
        sources = _make_sources(driver_events=[{
            "source": "AEMO_DISPATCHCONSTRAINT",
            "driver_type": "constraint",
            "element_id": "N^^Q_NIL",
            "region": "NSW1",
            "valid_time": _now(),
            "values": {"marginal_value": 12.5},
            "raw_ref": "constraint-ref",
        }])
        out = build_why(sources)
        assert "N^^Q_NIL" in out.why_plain_english
        assert any(ref.field == "constraint_marginal_value" for ref in out.evidence_refs)

    def test_missing_driver_data_is_first_class_missing_data(self):
        sources = _make_sources()
        out = build_why(sources)
        assert "dispatch_constraints" in out.missing_data
        assert "dispatch_interconnector_flows" in out.missing_data

    def test_trace_replay_mentions_bitemporal(self):
        """TRACE_REPLAY intent → bitemporal audit reference."""
        sources = _make_sources(intent=IntentLabel.TRACE_REPLAY)
        out = build_why(sources)

        assert "bitemporal" in out.why_plain_english.lower() or \
               "audit" in out.why_plain_english.lower()


# ── ForecastDrivers from AEMO pre-dispatch ───────────────────────────────────

class TestForecastFromPredispatch:

    def _make_intervals(self, prices: list[float], region="NSW1") -> list[dict]:
        from datetime import timedelta
        now = _now()
        return [
            {
                "region": region,
                "interval_datetime": (now + timedelta(minutes=5 * i)).isoformat(),
                "rrp": price,
                "demand_mw": 8000.0,
                "raw_ref": "test-ref",
            }
            for i, price in enumerate(prices)
        ]

    def test_rising_prices_direction_rising(self):
        intervals = self._make_intervals([100, 150, 200, 250, 300, 350])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.available is True
        assert result.direction == "rising"

    def test_falling_prices_direction_falling(self):
        intervals = self._make_intervals([400, 350, 300, 250, 200, 150])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.available is True
        assert result.direction == "falling"

    def test_flat_prices_direction_flat(self):
        intervals = self._make_intervals([100, 102, 100, 101, 100, 99])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.available is True
        assert result.direction == "flat"

    def test_p50_is_median(self):
        # sorted: [100, 150, 200, 250, 300] → median at index 2 = 200
        intervals = self._make_intervals([300, 150, 200, 100, 250])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.p50 == 200.0

    def test_p10_is_near_minimum(self):
        # Window caps at 6 intervals: [100, 150, 200, 250, 300, 350]
        # p10 = sorted[max(0, int(6 * 0.1))] = sorted[0] = 100
        intervals = self._make_intervals([100, 150, 200, 250, 300, 350, 400, 450, 500, 600])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.p10 == 100.0

    def test_p90_is_near_maximum(self):
        # Window caps at 6 intervals: [100, 150, 200, 250, 300, 350]
        # p90 = sorted[min(5, int(6 * 0.9))] = sorted[5] = 350
        intervals = self._make_intervals([100, 150, 200, 250, 300, 350, 400, 450, 500, 600])
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.p90 == 350.0

    def test_empty_intervals_not_available(self):
        result = _derive_forecast_from_predispatch([], "NSW1")
        assert result.available is False
        assert result.direction == "unknown"

    def test_wrong_region_filtered_out(self):
        intervals = self._make_intervals([200, 250, 300], region="VIC1")
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.available is False

    def test_mixed_regions_only_uses_target(self):
        vic_intervals = self._make_intervals([1000, 1200, 1400], region="VIC1")
        nsw_intervals = self._make_intervals([80, 85, 90], region="NSW1")
        all_intervals = vic_intervals + nsw_intervals
        result = _derive_forecast_from_predispatch(all_intervals, "NSW1")
        assert result.available is True
        assert result.p50 == 85.0  # median of [80, 85, 90]

    def test_horizon_capped_at_six_intervals(self):
        intervals = self._make_intervals([100] * 20, region="NSW1")
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.horizon_intervals == 6

    def test_single_interval_still_returns_available(self):
        intervals = self._make_intervals([200.0], region="NSW1")
        result = _derive_forecast_from_predispatch(intervals, "NSW1")
        assert result.available is True
        assert result.p50 == 200.0
        assert result.direction == "flat"


# ── Evidence refs on every fresh-dispatch answer ─────────────────────────────

class TestEvidenceRefCoverage:

    def test_fresh_dispatch_produces_price_ref(self):
        sources = _make_sources(fresh=True, price=200.0)
        out = build_why(sources)
        price_refs = [r for r in out.evidence_refs if r.field == "price_rrp"]
        assert len(price_refs) >= 1
        assert abs(price_refs[0].value - 200.0) < 0.01

    def test_fresh_dispatch_produces_demand_ref(self):
        sources = _make_sources(fresh=True, demand=8000.0)
        out = build_why(sources)
        demand_refs = [r for r in out.evidence_refs if r.field == "demand_mw"]
        assert len(demand_refs) >= 1

    def test_stale_dispatch_produces_no_evidence_refs(self):
        sources = _make_sources(fresh=False)
        out = build_why(sources)
        # Missing live data → no dispatch evidence refs, but no crash
        price_refs = [r for r in out.evidence_refs if r.field == "price_rrp"]
        assert len(price_refs) == 0
        assert "live_dispatch_price" in out.missing_data

    def test_evidence_refs_have_required_fields(self):
        sources = _make_sources(fresh=True)
        out = build_why(sources)
        for ref in out.evidence_refs:
            assert ref.source, "EvidenceRefSchema.source must be set"
            assert ref.field, "EvidenceRefSchema.field must be set"
            assert ref.raw_ref, "EvidenceRefSchema.raw_ref must be set"

    def test_evidence_ref_interval_matches_dispatch_time(self):
        sources = _make_sources(fresh=True, price=350.0)
        out = build_why(sources)
        price_refs = [r for r in out.evidence_refs if r.field == "price_rrp"]
        assert len(price_refs) >= 1
        # Interval in ref should be close to the current time (within 5 seconds)
        age = abs((datetime.now(timezone.utc) - price_refs[0].interval).total_seconds())
        assert age < 5.0


# ── Confidence estimation ─────────────────────────────────────────────────────

class TestTechnologyEvidence:

    def test_unit_dispatch_technology_evidence_is_narrated_with_caveats(self):
        sources = _make_sources(unit_events=[
            {
                "source": "AEMO_DISPATCH_UNIT_SOLUTION",
                "duid": "ERARING1",
                "region": "NSW1",
                "fuel_type": "coal",
                "valid_time": datetime.now(timezone.utc),
                "initial_mw": 450.0,
                "total_cleared_mw": 510.0,
                "availability_mw": 650.0,
                "raw_ref": "coal-ref",
            },
            {
                "source": "AEMO_DISPATCH_UNIT_SOLUTION",
                "duid": "POAT220",
                "region": "NSW1",
                "fuel_type": "hydro",
                "valid_time": datetime.now(timezone.utc),
                "initial_mw": 90.0,
                "total_cleared_mw": 140.0,
                "availability_mw": 200.0,
                "raw_ref": "hydro-ref",
            },
        ])
        out = build_why(sources)

        assert "Observed unit dispatch by technology" in out.why_plain_english
        assert "hydro cleared 140 MW" in out.why_plain_english
        assert "water scarcity is not asserted" in out.why_plain_english
        assert "hydro_water_storage" in out.missing_data
        assert any(ref.source == "AEMO_DISPATCH_UNIT_SOLUTION" for ref in out.evidence_refs)


class TestConfidenceEstimation:

    def test_stale_data_low_confidence(self):
        score = _estimate_confidence(live_fresh=False, analog_count=0, news_explained=False)
        assert score == 0.10

    def test_fresh_no_analogs_no_news(self):
        score = _estimate_confidence(live_fresh=True, analog_count=0, news_explained=False)
        assert score == 0.50

    def test_fresh_with_few_analogs(self):
        score = _estimate_confidence(live_fresh=True, analog_count=5, news_explained=False)
        assert score == 0.65

    def test_fresh_with_many_analogs(self):
        score = _estimate_confidence(live_fresh=True, analog_count=10, news_explained=False)
        assert score == 0.75

    def test_news_explained_adds_to_confidence(self):
        score = _estimate_confidence(live_fresh=True, analog_count=5, news_explained=True)
        assert score == 0.80

    def test_all_sources_present_max_confidence(self):
        # 0.50 (live) + 0.25 (≥10 analogs) + 0.15 (news) = 0.90
        score = _estimate_confidence(live_fresh=True, analog_count=10, news_explained=True)
        assert score == 0.90

    def test_confidence_never_exceeds_one(self):
        score = _estimate_confidence(live_fresh=True, analog_count=100, news_explained=True)
        assert score <= 1.0


# ── Outcome summary helper ────────────────────────────────────────────────────

class TestSummariseOutcomes:

    def test_empty_returns_none(self):
        assert _summarise_outcomes([]) is None

    def test_all_recovered(self):
        analogs = [{"outcome": "recovered"}] * 5
        result = _summarise_outcomes(analogs)
        assert result is not None
        assert "5/5" in result
        assert "recovered" in result

    def test_all_continued_spike(self):
        analogs = [{"outcome": "continued_spike"}] * 3
        result = _summarise_outcomes(analogs)
        assert "3/3" in result
        assert "continued spike" in result

    def test_mixed_outcomes(self):
        analogs = [
            {"outcome": "recovered"},
            {"outcome": "recovered"},
            {"outcome": "continued_spike"},
            {"outcome": "unknown_outcome"},
        ]
        result = _summarise_outcomes(analogs)
        assert "2/4" in result
        assert "1/4" in result

    def test_no_outcome_key(self):
        analogs = [{"price": 100.0}, {"price": 200.0}]
        result = _summarise_outcomes(analogs)
        # All 2 are "unknown"
        assert result is not None
        assert "2/2" in result
