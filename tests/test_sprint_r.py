"""Sprint R — Evidence Claim Map + Live Commentary Quality: pytest test suite.

Tests cover:
  1. ClaimType enum — all 13 types present
  2. _build_claim_map — base claims + stale cap + news cap + OOS path
  3. _build_claim_map — Sprint R extended types (CONSTRAINT_BINDING, WEATHER_CORRELATION,
     REBID_EVIDENCE, OUTAGE_EVIDENCE, FCAS_CLAIM)
  4. _build_next_watch — new rules (FCAS, rebid, notice, weather thresholds, watch_closed)
  5. ChangeDetector — 5 new ChangeTypes (CONSTRAINT_ACTIVE, WEATHER_PRESSURE_BUILDING,
     DATA_STALE, DATA_RECOVERED, WATCH_CLOSED)
  6. ChangeDetector — materiality tuning (HEADROOM rate-of-change, FORECAST_RISK P90 guard)
  7. CommentaryEvent — claim_map field present in to_dict()
  8. prose — format_headline for all new ChangeTypes
  9. WhyOutput — claim_map always non-empty (including OOS path)
  10. Snapshot round-trip — new fields preserved
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GRIDVERDICT_DEV_NO_AUTH", "true")
os.environ.setdefault("JWT_SECRET", "sprint-r-test-placeholder-not-used")


# ─── helpers ────────────────────────────────────────────────────────────────

def _snap(
    region="NSW1",
    price=80.0,
    demand=7000.0,
    headroom=800.0,
    regime="normal",
    spike_300=0.0,
    forecast_p90=None,
    notice_ids=None,
    staleness_seconds=0,
    binding_constraint_ids=None,
    weather_pressure=0.0,
):
    from app.engines.commentary.snapshot import RegionSnapshot
    return RegionSnapshot(
        region=region,
        price_rrp=price,
        demand_mw=demand,
        headroom_mw=headroom,
        regime=regime,
        valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
        spike_prob_300=spike_300,
        forecast_p90=forecast_p90,
        notice_ids=notice_ids or [],
        staleness_seconds=staleness_seconds,
        binding_constraint_ids=binding_constraint_ids or [],
        weather_pressure=weather_pressure,
    )


def _fresh_ref(field_name="price_rrp", age_seconds=0):
    """Evidence ref with interval at (now - age_seconds)."""
    from app.core.schema import EvidenceRefSchema
    interval = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return EvidenceRefSchema(
        source="AEMO_DISPATCH_PRICE",
        region="NSW1",
        interval=interval,
        field=field_name,
        value=120.0,
        raw_ref="test_ref",
    )


def _stale_ref():
    return _fresh_ref(age_seconds=1800)  # 30 min old — well past the 15 min threshold


def _claim_tier(label, tier, present=True, ref_ids=None):
    return {
        "label": label,
        "tier": tier,
        "present": present,
        "evidence_ref_ids": ref_ids or [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. ClaimType enum
# ═══════════════════════════════════════════════════════════════════════════

class TestClaimTypeEnum:
    def test_all_sprint_r_types_present(self):
        from app.core.schema import ClaimType
        expected = {
            "fcas_claim", "rebid_evidence", "outage_evidence",
            "weather_correlation", "constraint_binding", "watch_closed", "other",
        }
        values = {ct.value for ct in ClaimType}
        assert expected <= values, f"Missing ClaimType values: {expected - values}"

    def test_original_types_still_present(self):
        from app.core.schema import ClaimType
        originals = {
            "price_assertion", "demand_assertion", "cause_claim",
            "forecast_claim", "action_recommendation", "probability_claim",
            "historical_analog",
        }
        values = {ct.value for ct in ClaimType}
        assert originals <= values


# ═══════════════════════════════════════════════════════════════════════════
# 2. _build_claim_map — base quality invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildClaimMapQuality:
    def test_confirmed_claim_has_evidence_ref_ids(self):
        from app.agents.why_builder import _build_claim_map
        ref = _fresh_ref()
        tiers = [_claim_tier("dispatch_price", "confirmed", ref_ids=[ref.id])]
        result = _build_claim_map(tiers, [ref])
        confirmed = [c for c in result if c.tier.value == "confirmed"]
        assert confirmed, "No confirmed claims produced"
        for c in confirmed:
            assert c.evidence_ref_ids, f"Confirmed claim '{c.label}' has no evidence_ref_ids"

    def test_news_claim_never_exceeds_supported_tier(self):
        from app.agents.why_builder import _build_claim_map
        ref = _fresh_ref()
        tiers = [_claim_tier("aemo_notice", "confirmed", ref_ids=[ref.id])]
        result = _build_claim_map(tiers, [ref])
        notice_claims = [c for c in result if "notice" in c.label.lower() or "aemo" in c.label.lower()]
        for c in notice_claims:
            assert c.tier.value in ("supported", "plausible", "unconfirmed"), (
                f"Notice claim has tier {c.tier.value!r} — must not exceed 'supported'"
            )

    def test_stale_source_downgrades_to_plausible(self):
        from app.agents.why_builder import _build_claim_map
        ref = _stale_ref()
        tiers = [_claim_tier("dispatch_price", "confirmed", ref_ids=[ref.id])]
        # Use a fixed now that is definitely after the stale ref
        now = datetime.now(timezone.utc)
        result = _build_claim_map(tiers, [ref], now=now)
        price_claims = [c for c in result if "dispatch" in c.label.lower() or "price" in c.label.lower()]
        for c in price_claims:
            assert c.tier.value == "plausible", (
                f"Stale evidence should downgrade tier to 'plausible', got '{c.tier.value}'"
            )
            assert c.note and "stale" in c.note.lower(), "Stale claim should have a note"

    def test_fresh_source_keeps_confirmed_tier(self):
        from app.agents.why_builder import _build_claim_map
        ref = _fresh_ref(age_seconds=60)  # 1 min old — fresh
        tiers = [_claim_tier("dispatch_price", "confirmed", ref_ids=[ref.id])]
        now = datetime.now(timezone.utc)
        result = _build_claim_map(tiers, [ref], now=now)
        price_claims = [c for c in result if "price" in c.label.lower() or "dispatch" in c.label.lower()]
        assert price_claims, "No price claims generated"
        assert any(c.tier.value == "confirmed" for c in price_claims), (
            "Fresh evidence should keep 'confirmed' tier"
        )

    def test_oos_path_returns_non_empty_claim_map(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimMapItem, ClaimType, DriverConfidenceTier
        # Simulate OOS path: empty claim_tiers, empty evidence_refs, no sources
        result = _build_claim_map([], [], sources=None)
        # OOS path returns empty list from _build_claim_map;
        # the non-empty check is in build_why's OOS branch
        # Test that build_why OOS path returns a non-empty claim_map
        from app.agents.why_builder import build_why
        from app.core.schema import IntentLabel, QueryDecomposition
        from app.agents.why_sources import (
            WhySources, CurrentDrivers, ForecastDrivers,
            AnalogSummary, NewsContext, WeatherContext, DriverContext, TechnologyContext,
        )
        decomp = QueryDecomposition(
            raw_query="what is the stock price of Tesla?",
            intent=IntentLabel.OUT_OF_SCOPE,
        )
        sources = WhySources(
            decomp=decomp,
            current=CurrentDrivers(
                region="NSW1", price_rrp=80.0, demand_mw=7000.0,
                availability_mw=8000.0, headroom_mw=1000.0,
                regime="normal", valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
                staleness_seconds=0, is_fresh=True,
            ),
            forecast=ForecastDrivers(),
            analogs=AnalogSummary(),
            news=NewsContext(),
            weather=WeatherContext(),
            drivers=DriverContext(),
            technology=TechnologyContext(),
        )
        why = build_why(sources)
        assert why.claim_map, "OOS path must return a non-empty claim_map"
        assert why.claim_map[0].present is False
        assert why.claim_map[0].confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. _build_claim_map — Sprint R extended claim types
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildClaimMapSprintR:
    def _make_sources(self, binding_constraints=None, weather=None, driver_events=None):
        from app.agents.why_sources import (
            WhySources, CurrentDrivers, ForecastDrivers,
            AnalogSummary, NewsContext, WeatherContext, DriverContext, TechnologyContext,
        )
        from app.core.schema import IntentLabel, QueryDecomposition
        dc = DriverContext(
            binding_constraints=binding_constraints or [],
            events=driver_events or [],
        )
        wc = weather or WeatherContext()
        decomp = QueryDecomposition(raw_query="test", intent=IntentLabel.EXPLANATION)
        return WhySources(
            decomp=decomp,
            current=CurrentDrivers(
                region="NSW1", price_rrp=500.0, demand_mw=9000.0,
                availability_mw=9500.0, headroom_mw=500.0,
                regime="spike", valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
                staleness_seconds=0, is_fresh=True,
            ),
            forecast=ForecastDrivers(),
            analogs=AnalogSummary(),
            news=NewsContext(),
            weather=wc,
            drivers=dc,
            technology=TechnologyContext(),
        )

    def test_constraint_binding_present_when_constraints_in_sources(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        sources = self._make_sources(
            binding_constraints=[{"constraint_id": "N^^NSWGN_E", "values": {"marginal_value": 45.0}}]
        )
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.CONSTRAINT_BINDING in types, "CONSTRAINT_BINDING not in claim_map"

    def test_weather_correlation_present_when_weather_relevant(self):
        from app.agents.why_builder import _build_claim_map
        from app.agents.why_sources import WeatherContext
        from app.core.schema import ClaimType
        weather = WeatherContext(
            relevant=True, available=True, confidence=0.8,
            source_count=3, tags=["high_temp"],
            consensus={"temperature_c": 39.5},
        )
        sources = self._make_sources(weather=weather)
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.WEATHER_CORRELATION in types, "WEATHER_CORRELATION not in claim_map"

    def test_weather_correlation_absent_when_weather_not_relevant(self):
        from app.agents.why_builder import _build_claim_map
        from app.agents.why_sources import WeatherContext
        from app.core.schema import ClaimType
        weather = WeatherContext(relevant=False, available=False)
        sources = self._make_sources(weather=weather)
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.WEATHER_CORRELATION not in types

    def test_rebid_evidence_present_when_rebid_events(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        sources = self._make_sources(driver_events=[
            {"type": "REBID", "rebid_mw": 200.0, "duid": "BAYSW"},
            {"type": "REBID", "rebid_mw": 50.0, "duid": "ERGT01"},
        ])
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.REBID_EVIDENCE in types, "REBID_EVIDENCE not in claim_map"

    def test_outage_evidence_present_when_outage_events(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        sources = self._make_sources(driver_events=[
            {"type": "OUTAGE", "duid": "TARONG"},
        ])
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.OUTAGE_EVIDENCE in types, "OUTAGE_EVIDENCE not in claim_map"

    def test_fcas_claim_present_when_fcas_price_above_200(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        sources = self._make_sources(driver_events=[
            {"type": "FCAS", "fcas_price": 350.0},
        ])
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.FCAS_CLAIM in types, "FCAS_CLAIM not in claim_map when FCAS price > $200"

    def test_fcas_claim_absent_when_fcas_price_below_200(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        sources = self._make_sources(driver_events=[
            {"type": "FCAS", "fcas_price": 50.0},
        ])
        result = _build_claim_map([], [], sources=sources)
        types = [c.claim_type for c in result]
        assert ClaimType.FCAS_CLAIM not in types, "FCAS_CLAIM should not fire when FCAS price < $200"


# ═══════════════════════════════════════════════════════════════════════════
# 4. _build_next_watch — new rules
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildNextWatchSprintR:
    def _curr(self, price=80.0, headroom=800.0, regime="normal"):
        from app.agents.why_sources import CurrentDrivers
        return CurrentDrivers(
            region="NSW1", price_rrp=price, demand_mw=7000.0,
            availability_mw=7000.0 + headroom, headroom_mw=headroom,
            regime=regime, valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            staleness_seconds=0, is_fresh=True,
        )

    def _drivers(self, events=None):
        from app.agents.why_sources import DriverContext
        return DriverContext(events=events or [])

    def _forecast(self, available=False):
        from app.agents.why_sources import ForecastDrivers
        return ForecastDrivers(available=available)

    def _analogs(self):
        from app.agents.why_sources import AnalogSummary
        return AnalogSummary()

    def test_fcas_watch_fires_when_fcas_price_above_200(self):
        from app.agents.why_builder import _build_next_watch
        drivers = self._drivers(events=[{"type": "FCAS", "fcas_price": 450.0}])
        items = _build_next_watch(self._curr(), self._forecast(), drivers, self._analogs(), None, None)
        assert any("FCAS" in item for item in items), "FCAS watch should fire when price > $200"
        assert any("450" in item for item in items), "FCAS watch should include the actual price"

    def test_fcas_watch_silent_when_fcas_price_below_200(self):
        from app.agents.why_builder import _build_next_watch
        drivers = self._drivers(events=[{"type": "FCAS", "fcas_price": 100.0}])
        items = _build_next_watch(self._curr(), self._forecast(), drivers, self._analogs(), None, None)
        assert not any("FCAS" in item for item in items)

    def test_rebid_watch_fires_when_rebid_events(self):
        from app.agents.why_builder import _build_next_watch
        drivers = self._drivers(events=[
            {"type": "REBID", "rebid_mw": 120.0, "duid": "BAYSW"},
        ])
        items = _build_next_watch(self._curr(), self._forecast(), drivers, self._analogs(), None, None)
        assert any("rebid" in item.lower() for item in items), "Rebid watch should fire"

    def test_notice_watch_fires_when_recent_notice(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import NewsContext
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(minutes=10)).isoformat()
        news = NewsContext(
            notices=[{"notice_id": "N001", "publish_datetime": recent_ts, "notice_type": "LOR3"}],
            top_notice_type="LOR3",
        )
        items = _build_next_watch(self._curr(), self._forecast(), self._drivers(), self._analogs(), None, news)
        assert any("notice" in item.lower() for item in items), "Notice watch should fire for recent notice"

    def test_notice_watch_silent_when_old_notice(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import NewsContext
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        news = NewsContext(
            notices=[{"notice_id": "N001", "publish_datetime": old_ts}],
        )
        items = _build_next_watch(self._curr(), self._forecast(), self._drivers(), self._analogs(), None, news)
        assert not any("notice follow-up" in item.lower() for item in items), (
            "Notice watch should not fire for notice > 30 min old"
        )

    def test_weather_extreme_heat_watch_fires(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import WeatherContext
        weather = WeatherContext(
            relevant=True, available=True, confidence=0.9,
            consensus={"temperature_c": 41.0},
        )
        items = _build_next_watch(self._curr(), self._forecast(), self._drivers(), self._analogs(), weather, None)
        assert any("heat" in item.lower() or "41" in item for item in items), (
            "Extreme heat watch should fire at 41°C"
        )

    def test_weather_wind_drought_watch_fires(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import WeatherContext
        weather = WeatherContext(
            relevant=True, available=True, confidence=0.8,
            consensus={"wind_speed_kmh": 5.0},  # ≈ 1.4 m/s < 3 m/s threshold
        )
        items = _build_next_watch(self._curr(), self._forecast(), self._drivers(), self._analogs(), weather, None)
        assert any("wind" in item.lower() for item in items), "Wind drought watch should fire"

    def test_watch_closed_fires_when_price_normalised_after_spike(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import NewsContext
        # Current price < 150, and there's a recent PRICE_SPIKE in auto_commentary
        c = self._curr(price=100.0)
        news = NewsContext(
            auto_commentary=[
                {"event_type": "PRICE_SPIKE", "headline": "NSW1 spike", "confidence": 0.9},
            ]
        )
        items = _build_next_watch(c, self._forecast(), self._drivers(), self._analogs(), None, news)
        assert any("watch closed" in item.lower() or "normalised" in item.lower() for item in items), (
            "Watch closed should fire when price < $150 after a recent PRICE_SPIKE event"
        )

    def test_watch_closed_silent_when_no_prior_spike(self):
        from app.agents.why_builder import _build_next_watch
        from app.agents.why_sources import NewsContext
        c = self._curr(price=100.0)
        news = NewsContext(auto_commentary=[])
        items = _build_next_watch(c, self._forecast(), self._drivers(), self._analogs(), None, news)
        assert not any("watch closed" in item.lower() for item in items), (
            "Watch closed should not fire when no prior spike in auto_commentary"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. ChangeDetector — 5 new ChangeTypes
# ═══════════════════════════════════════════════════════════════════════════

class TestNewChangeTypes:
    def test_constraint_active_fires_on_new_constraint(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(binding_constraint_ids=[])
        curr = _snap(binding_constraint_ids=["N^^NSWGN_E"])
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.CONSTRAINT_ACTIVE in types, "CONSTRAINT_ACTIVE should fire on new constraint"

    def test_constraint_active_silent_when_no_new_constraint(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(binding_constraint_ids=["N^^NSWGN_E"])
        curr = _snap(binding_constraint_ids=["N^^NSWGN_E"])
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.CONSTRAINT_ACTIVE not in types

    def test_weather_pressure_building_fires_at_threshold(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(weather_pressure=0.3)
        curr = _snap(weather_pressure=0.8)
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.WEATHER_PRESSURE_BUILDING in types, (
            "WEATHER_PRESSURE_BUILDING should fire when pressure crosses 0.7"
        )

    def test_weather_pressure_building_silent_when_already_elevated(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(weather_pressure=0.75)  # already above threshold
        curr = _snap(weather_pressure=0.85)
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.WEATHER_PRESSURE_BUILDING not in types

    def test_data_stale_fires_after_15min(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(staleness_seconds=300)    # 5 min — fresh
        curr = _snap(staleness_seconds=1200)   # 20 min — stale
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.DATA_STALE in types, "DATA_STALE should fire when staleness crosses 900s"

    def test_data_stale_silent_when_already_stale(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(staleness_seconds=1000)  # already stale
        curr = _snap(staleness_seconds=1500)
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.DATA_STALE not in types

    def test_data_recovered_fires_when_stale_clears(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(staleness_seconds=1200)  # was stale
        curr = _snap(staleness_seconds=60)    # now fresh
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.DATA_RECOVERED in types, "DATA_RECOVERED should fire when staleness drops below 900s"

    def test_watch_closed_fires_when_price_drops_from_spike(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=500.0)
        curr = _snap(price=100.0)
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.WATCH_CLOSED in types, (
            "WATCH_CLOSED should fire when price drops from ≥300 to <150"
        )

    def test_watch_closed_silent_when_price_still_elevated(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=500.0)
        curr = _snap(price=200.0)  # dropped but still above 150
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.WATCH_CLOSED not in types


# ═══════════════════════════════════════════════════════════════════════════
# 6. ChangeDetector — materiality tuning
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialityTuning:
    def test_headroom_tightened_fires_on_rapid_decline_below_50mw_threshold(self):
        """Rate-of-change: >50 MW decline per tick fires HEADROOM_TIGHTENED
        even without crossing the absolute 500/200 MW thresholds."""
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(headroom=700.0)
        curr = _snap(headroom=640.0)  # -60 MW decline, no threshold crossed
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.HEADROOM_TIGHTENED in types, (
            "HEADROOM_TIGHTENED should fire on >50 MW rapid decline"
        )

    def test_headroom_tightened_silent_on_small_decline(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(headroom=700.0)
        curr = _snap(headroom=680.0)  # -20 MW decline — below threshold
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.HEADROOM_TIGHTENED not in types

    def test_forecast_risk_increased_requires_p90_500(self):
        """FORECAST_RISK_INCREASED should not fire unless forecast_p90 >= $500."""
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(spike_300=0.10, forecast_p90=200.0)   # p90 too low
        curr = _snap(spike_300=0.30, forecast_p90=200.0)   # delta > 15pp but p90 < 500
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.FORECAST_RISK_INCREASED not in types, (
            "FORECAST_RISK_INCREASED should not fire when p90 < $500"
        )

    def test_forecast_risk_increased_fires_with_p90_500_and_delta(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(spike_300=0.10, forecast_p90=600.0)
        curr = _snap(spike_300=0.30, forecast_p90=600.0)   # delta=20pp AND p90 >= $500
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.FORECAST_RISK_INCREASED in types, (
            "FORECAST_RISK_INCREASED should fire when delta > 15pp AND p90 >= $500"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. CommentaryEvent — claim_map field in to_dict()
# ═══════════════════════════════════════════════════════════════════════════

class TestCommentaryEventClaimMap:
    def test_claim_map_present_in_to_dict(self):
        from app.engines.commentary.engine import CommentaryEvent
        from datetime import datetime, timezone
        evt = CommentaryEvent(
            id="test-id",
            region="NSW1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            system_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            event_type="price_spike",
            severity="HIGH",
            headline="Test headline",
            contributing_factors=[],
            missing_data=[],
            evidence_refs=[],
            confidence=0.8,
            corroborations={},
            next_watch=["Watch for continuation"],
            counterargument=None,
            snapshot_before=None,
            snapshot_after={"region": "NSW1"},
        )
        d = evt.to_dict()
        assert "claim_map" in d, "claim_map must be present in CommentaryEvent.to_dict()"
        assert isinstance(d["claim_map"], list), "claim_map must be a list"

    def test_claim_map_defaults_to_empty_list(self):
        from app.engines.commentary.engine import CommentaryEvent
        evt = CommentaryEvent(
            id="test-id-2",
            region="NSW1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            system_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            event_type="notice_added",
            severity="MEDIUM",
            headline="Test",
            contributing_factors=[],
            missing_data=[],
            evidence_refs=[],
            confidence=0.5,
            corroborations={},
            next_watch=[],
            counterargument=None,
            snapshot_before=None,
            snapshot_after={},
        )
        assert evt.claim_map == [], "claim_map should default to empty list"


# ═══════════════════════════════════════════════════════════════════════════
# 8. prose — format_headline for new ChangeTypes
# ═══════════════════════════════════════════════════════════════════════════

class TestProseHeadlines:
    def _change(self, ct, prev=None, curr=None, region="NSW1"):
        from app.engines.commentary.detector import MaterialChange
        return MaterialChange(
            change_type=ct,
            region=region,
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            severity="MEDIUM",
            prev_value=prev,
            curr_value=curr,
            threshold_crossed=None,
            description=f"{region}: {ct.value}",
        )

    def _mock_why(self):
        from app.agents.why_builder import WhyOutput
        return WhyOutput(
            why_plain_english="Test narrative.",
            counterargument="Counter.",
            missing_data=[],
            evidence_refs=[],
            confidence=0.7,
            claim_map=[],
        )

    def test_constraint_active_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._change(ChangeType.CONSTRAINT_ACTIVE, prev=0.0, curr=1.0)
        headline = format_headline(change, self._mock_why())
        assert "constraint" in headline.lower(), f"Expected 'constraint' in: {headline}"

    def test_weather_pressure_building_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._change(ChangeType.WEATHER_PRESSURE_BUILDING, prev=0.3, curr=0.9)
        headline = format_headline(change, self._mock_why())
        assert "weather" in headline.lower(), f"Expected 'weather' in: {headline}"

    def test_data_stale_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._change(ChangeType.DATA_STALE, prev=300.0, curr=1500.0)
        headline = format_headline(change, self._mock_why())
        assert "stale" in headline.lower(), f"Expected 'stale' in: {headline}"

    def test_data_recovered_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._change(ChangeType.DATA_RECOVERED, prev=1500.0, curr=60.0)
        headline = format_headline(change, self._mock_why())
        assert "recover" in headline.lower(), f"Expected 'recover' in: {headline}"

    def test_watch_closed_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._change(ChangeType.WATCH_CLOSED, prev=500.0, curr=90.0)
        headline = format_headline(change, self._mock_why())
        assert "normalised" in headline.lower() or "closed" in headline.lower(), (
            f"Expected 'normalised' or 'closed' in: {headline}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. Snapshot round-trip — new Sprint R fields preserved
# ═══════════════════════════════════════════════════════════════════════════

class TestSnapshotRoundTrip:
    def test_new_fields_serialise_and_deserialise(self):
        import json
        from app.engines.commentary.snapshot import RegionSnapshot
        original = RegionSnapshot(
            region="SA1",
            price_rrp=450.0,
            demand_mw=3000.0,
            headroom_mw=200.0,
            regime="spike",
            valid_time=datetime(2026, 5, 26, 15, 0, tzinfo=timezone.utc),
            staleness_seconds=120,
            binding_constraint_ids=["V^^SNW_E", "N^^NSWGN_E"],
            weather_pressure=0.85,
        )
        # Simulate serialisation (what save_snapshot writes)
        payload = json.dumps({
            "region": original.region,
            "price_rrp": original.price_rrp,
            "demand_mw": original.demand_mw,
            "headroom_mw": original.headroom_mw,
            "regime": original.regime,
            "valid_time": original.valid_time.isoformat(),
            "spike_prob_300": original.spike_prob_300,
            "spike_prob_1000": original.spike_prob_1000,
            "notice_ids": original.notice_ids,
            "forecast_p90": original.forecast_p90,
            "staleness_seconds": original.staleness_seconds,
            "binding_constraint_ids": original.binding_constraint_ids,
            "weather_pressure": original.weather_pressure,
        })
        d = json.loads(payload)
        recovered = RegionSnapshot(
            region=d["region"],
            price_rrp=float(d["price_rrp"]),
            demand_mw=float(d["demand_mw"]),
            headroom_mw=float(d["headroom_mw"]),
            regime=str(d["regime"]),
            valid_time=datetime.fromisoformat(d["valid_time"]),
            spike_prob_300=float(d.get("spike_prob_300") or 0.0),
            spike_prob_1000=float(d.get("spike_prob_1000") or 0.0),
            notice_ids=list(d.get("notice_ids") or []),
            forecast_p90=d.get("forecast_p90"),
            staleness_seconds=int(d.get("staleness_seconds") or 0),
            binding_constraint_ids=list(d.get("binding_constraint_ids") or []),
            weather_pressure=float(d.get("weather_pressure") or 0.0),
        )
        assert recovered.staleness_seconds == 120
        assert recovered.binding_constraint_ids == ["V^^SNW_E", "N^^NSWGN_E"]
        assert abs(recovered.weather_pressure - 0.85) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# 10. Cooldown seconds — DATA_STALE has 30 min cooldown
# ═══════════════════════════════════════════════════════════════════════════

class TestCooldownSeconds:
    def test_data_stale_cooldown_is_30_min(self):
        from app.engines.commentary.detector import ChangeType, _COOLDOWN_SECONDS
        assert _COOLDOWN_SECONDS[ChangeType.DATA_STALE] == 1800, (
            "DATA_STALE cooldown should be 1800s (30 min)"
        )

    def test_watch_closed_cooldown_present(self):
        from app.engines.commentary.detector import ChangeType, _COOLDOWN_SECONDS
        assert ChangeType.WATCH_CLOSED in _COOLDOWN_SECONDS

    def test_constraint_active_cooldown_present(self):
        from app.engines.commentary.detector import ChangeType, _COOLDOWN_SECONDS
        assert ChangeType.CONSTRAINT_ACTIVE in _COOLDOWN_SECONDS

    def test_change_decomposition_has_entries_for_all_new_types(self):
        from app.engines.commentary.detector import ChangeType, _CHANGE_DECOMPOSITION
        new_types = [
            ChangeType.CONSTRAINT_ACTIVE,
            ChangeType.WEATHER_PRESSURE_BUILDING,
            ChangeType.DATA_STALE,
            ChangeType.DATA_RECOVERED,
            ChangeType.WATCH_CLOSED,
        ]
        for ct in new_types:
            assert ct in _CHANGE_DECOMPOSITION, (
                f"{ct.value} missing from _CHANGE_DECOMPOSITION"
            )
