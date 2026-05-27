"""Tests for DriverConfidenceTier causality tier system.

Verifies:
  - build_why() always returns exactly 8 driver_tiers
  - correct label set
  - tier values change correctly with different evidence states
  - confirmed/supported tiers only when real evidence is present
  - tier values are valid DriverConfidenceTier members
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.agents.why_builder import build_why, _build_driver_tiers
from app.agents.why_sources import (
    AnalogSummary,
    CurrentDrivers,
    DriverContext,
    ForecastDrivers,
    NewsContext,
    TechnologyContext,
    WeatherContext,
    WhySources,
)
from app.core.schema import DriverConfidenceTier, IntentLabel, QueryDecomposition

_VALID_TIME = datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc)
_TIER_LABELS = {
    "dispatch_price",
    "aemo_notice",
    "weather",
    "dispatch_constraints",
    "interconnector_flows",
    "unit_dispatch",
    "historical_analogs",
    "forecast",
}
_VALID_TIERS = {t.value for t in DriverConfidenceTier}


def _make_decomp(intent: IntentLabel = IntentLabel.EXPLANATION) -> QueryDecomposition:
    return QueryDecomposition(
        raw_query="test query",
        intent=intent,
        region="NSW1",
    )


def _make_current(fresh: bool = True, price: float = 347.0, staleness: int = 0) -> CurrentDrivers:
    return CurrentDrivers(
        region="NSW1",
        price_rrp=price,
        demand_mw=8420.0,
        availability_mw=8850.0,
        headroom_mw=430.0,
        regime="elevated",
        valid_time=_VALID_TIME,
        staleness_seconds=staleness if not fresh else 0,
        is_fresh=fresh,
        regime_state=None,
    )


def _make_sources(
    current: CurrentDrivers | None = None,
    news: NewsContext | None = None,
    weather: WeatherContext | None = None,
    drivers: DriverContext | None = None,
    technology: TechnologyContext | None = None,
    analogs: AnalogSummary | None = None,
    forecast: ForecastDrivers | None = None,
) -> WhySources:
    return WhySources(
        decomp=_make_decomp(),
        current=current or _make_current(),
        forecast=forecast or ForecastDrivers(),
        analogs=analogs or AnalogSummary(),
        news=news or NewsContext(),
        weather=weather or WeatherContext(),
        drivers=drivers or DriverContext(),
        technology=technology or TechnologyContext(),
    )


# ── Structure invariants ──────────────────────────────────────────────

def test_driver_tiers_always_8_entries():
    sources = _make_sources()
    out = build_why(sources)
    assert len(out.driver_tiers) == 8


def test_driver_tiers_correct_labels():
    sources = _make_sources()
    out = build_why(sources)
    labels = {t["label"] for t in out.driver_tiers}
    assert labels == _TIER_LABELS


def test_driver_tiers_all_valid_tier_values():
    sources = _make_sources()
    out = build_why(sources)
    for entry in out.driver_tiers:
        assert entry["tier"] in _VALID_TIERS, f"Invalid tier: {entry['tier']}"


def test_driver_tiers_present_field_is_bool():
    sources = _make_sources()
    out = build_why(sources)
    for entry in out.driver_tiers:
        assert isinstance(entry["present"], bool)


def test_driver_tiers_note_field_is_string():
    sources = _make_sources()
    out = build_why(sources)
    for entry in out.driver_tiers:
        assert isinstance(entry.get("note", ""), str)


# ── dispatch_price tier ───────────────────────────────────────────────

def test_dispatch_price_confirmed_when_fresh():
    sources = _make_sources(current=_make_current(fresh=True))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "dispatch_price")
    assert tier["tier"] == DriverConfidenceTier.CONFIRMED.value
    assert tier["present"] is True


def test_dispatch_price_unconfirmed_when_stale():
    sources = _make_sources(current=_make_current(fresh=False, staleness=900))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "dispatch_price")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value
    assert tier["present"] is False


# ── aemo_notice tier ──────────────────────────────────────────────────

def test_aemo_notice_confirmed_on_tier1():
    news = NewsContext(
        explained=True,
        notices=[{"id": "n1"}],
        credibility_tier=1,
        top_notice_type="LOR2",
    )
    sources = _make_sources(news=news)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "aemo_notice")
    assert tier["tier"] == DriverConfidenceTier.CONFIRMED.value
    assert tier["present"] is True


def test_aemo_notice_plausible_on_tier2():
    news = NewsContext(
        explained=True,
        notices=[{"id": "n2"}],
        credibility_tier=2,
        top_notice_type="MARKET_NOTICE",
    )
    sources = _make_sources(news=news)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "aemo_notice")
    assert tier["tier"] == DriverConfidenceTier.PLAUSIBLE.value


def test_aemo_notice_unconfirmed_when_no_notices():
    sources = _make_sources(news=NewsContext())
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "aemo_notice")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value
    assert tier["present"] is False


# ── weather tier ──────────────────────────────────────────────────────

def test_weather_plausible_when_available_and_relevant():
    weather = WeatherContext(available=True, relevant=True, confidence=0.8, source_count=3)
    sources = _make_sources(weather=weather)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "weather")
    assert tier["tier"] == DriverConfidenceTier.PLAUSIBLE.value
    assert tier["present"] is True


def test_weather_unconfirmed_when_not_available():
    sources = _make_sources(weather=WeatherContext(available=False, relevant=True))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "weather")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


def test_weather_unconfirmed_when_not_relevant():
    sources = _make_sources(weather=WeatherContext(available=True, relevant=False))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "weather")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── dispatch_constraints tier ─────────────────────────────────────────

def test_constraints_supported_when_evidence_present():
    drivers = DriverContext(
        binding_constraints=[{"element_id": "C1", "values": {"marginal_value": 5000}, "source": "AEMO"}]
    )
    sources = _make_sources(drivers=drivers)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "dispatch_constraints")
    assert tier["tier"] == DriverConfidenceTier.SUPPORTED.value
    assert tier["present"] is True


def test_constraints_unconfirmed_when_no_evidence():
    sources = _make_sources(drivers=DriverContext())
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "dispatch_constraints")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── interconnector_flows tier ─────────────────────────────────────────

def test_interconnectors_supported_when_tight():
    drivers = DriverContext(
        tight_interconnectors=[{"element_id": "VIC1-NSW1", "values": {"mw_flow": 950}, "source": "AEMO"}]
    )
    sources = _make_sources(drivers=drivers)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "interconnector_flows")
    assert tier["tier"] == DriverConfidenceTier.SUPPORTED.value


def test_interconnectors_unconfirmed_when_none():
    sources = _make_sources()
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "interconnector_flows")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── unit_dispatch tier ────────────────────────────────────────────────

def test_unit_dispatch_supported_when_evidence_present():
    tech = TechnologyContext(
        has_unit_evidence=True,
        by_fuel={"coal": {"fuel_type": "coal", "total_cleared_mw": 2000, "delta_mw": -100, "raw_refs": ["r1"]}},
    )
    sources = _make_sources(technology=tech)
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "unit_dispatch")
    assert tier["tier"] == DriverConfidenceTier.SUPPORTED.value
    assert tier["present"] is True


def test_unit_dispatch_unconfirmed_without_evidence():
    sources = _make_sources()
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "unit_dispatch")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── historical_analogs tier ───────────────────────────────────────────

def test_analogs_plausible_with_3_or_more():
    sources = _make_sources(analogs=AnalogSummary(count=5, success_count=3))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "historical_analogs")
    assert tier["tier"] == DriverConfidenceTier.PLAUSIBLE.value
    assert tier["present"] is True


def test_analogs_unconfirmed_with_fewer_than_3():
    sources = _make_sources(analogs=AnalogSummary(count=2))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "historical_analogs")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value
    assert tier["present"] is False


def test_analogs_unconfirmed_with_zero():
    sources = _make_sources(analogs=AnalogSummary(count=0))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "historical_analogs")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── forecast tier ─────────────────────────────────────────────────────

def test_forecast_plausible_when_available():
    sources = _make_sources(forecast=ForecastDrivers(available=True, direction="rising", p10=320.0, p50=380.0, p90=450.0))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "forecast")
    assert tier["tier"] == DriverConfidenceTier.PLAUSIBLE.value
    assert tier["present"] is True


def test_forecast_unconfirmed_when_not_available():
    sources = _make_sources(forecast=ForecastDrivers(available=False))
    out = build_why(sources)
    tier = next(t for t in out.driver_tiers if t["label"] == "forecast")
    assert tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value


# ── confirmed/supported never without backing data ────────────────────

def test_no_confirmed_or_supported_when_all_empty():
    """With no evidence at all, only plausible or unconfirmed should appear."""
    sources = _make_sources(current=_make_current(fresh=False, staleness=1200))
    out = build_why(sources)
    high_tier = {t["tier"] for t in out.driver_tiers} & {
        DriverConfidenceTier.CONFIRMED.value,
        DriverConfidenceTier.SUPPORTED.value,
    }
    assert len(high_tier) == 0, f"Expected no confirmed/supported with empty evidence, got {high_tier}"


def test_all_8_confirmed_or_supported_max_evidence():
    """Check that all 8 tiers return non-unconfirmed when maximum evidence is present."""
    sources = _make_sources(
        current=_make_current(fresh=True),
        news=NewsContext(explained=True, notices=[{"id": "n1"}], credibility_tier=1, top_notice_type="LOR2"),
        weather=WeatherContext(available=True, relevant=True, confidence=0.9, source_count=4),
        drivers=DriverContext(
            binding_constraints=[{"element_id": "C1", "values": {"marginal_value": 5000}, "source": "AEMO"}],
            tight_interconnectors=[{"element_id": "IC1", "values": {"mw_flow": 1000}, "source": "AEMO"}],
        ),
        technology=TechnologyContext(
            has_unit_evidence=True,
            by_fuel={"coal": {"fuel_type": "coal", "total_cleared_mw": 2000, "delta_mw": 0, "raw_refs": ["r1"]}},
        ),
        analogs=AnalogSummary(count=5, success_count=4),
        forecast=ForecastDrivers(available=True, direction="rising", p10=340.0, p50=400.0, p90=480.0),
    )
    out = build_why(sources)
    unconfirmed = [t for t in out.driver_tiers if t["tier"] == DriverConfidenceTier.UNCONFIRMED.value]
    assert len(unconfirmed) == 0, f"Expected 0 unconfirmed with full evidence, got {unconfirmed}"


# ── claim_tiers (claim-level causality) ───────────────────────────────

def test_claim_tiers_always_8_entries():
    sources = _make_sources()
    out = build_why(sources)
    assert len(out.claim_tiers) == 8


def test_claim_tiers_correct_labels():
    sources = _make_sources()
    out = build_why(sources)
    labels = {t["label"] for t in out.claim_tiers}
    assert labels == _TIER_LABELS


def test_claim_tiers_has_evidence_ref_ids_field():
    sources = _make_sources()
    out = build_why(sources)
    for entry in out.claim_tiers:
        assert "evidence_ref_ids" in entry
        assert isinstance(entry["evidence_ref_ids"], list)


def test_claim_tiers_confirmed_has_evidence_ref_ids():
    """Core invariant: confirmed tier → evidence_ref_ids non-empty."""
    sources = _make_sources(current=_make_current(fresh=True))
    out = build_why(sources)
    dispatch_tier = next(t for t in out.claim_tiers if t["label"] == "dispatch_price")
    assert dispatch_tier["tier"] == DriverConfidenceTier.CONFIRMED.value
    assert len(dispatch_tier["evidence_ref_ids"]) > 0, (
        "confirmed dispatch_price tier must have at least one evidence_ref_id"
    )


def test_claim_tiers_supported_has_evidence_ref_ids():
    """Core invariant: supported tier → evidence_ref_ids non-empty."""
    drivers = DriverContext(
        binding_constraints=[{
            "element_id": "C1",
            "values": {"marginal_value": 5000},
            "source": "AEMO_DISPATCHCONSTRAINT",
            "region": "NSW1",
            "valid_time": _VALID_TIME,
            "raw_ref": "constraint_archive",
        }]
    )
    sources = _make_sources(drivers=drivers)
    out = build_why(sources)
    constraints_tier = next(t for t in out.claim_tiers if t["label"] == "dispatch_constraints")
    assert constraints_tier["tier"] == DriverConfidenceTier.SUPPORTED.value
    assert len(constraints_tier["evidence_ref_ids"]) > 0, (
        "supported dispatch_constraints tier must have at least one evidence_ref_id"
    )


def test_claim_tiers_plausible_may_have_empty_refs():
    """Plausible tiers (analogs, forecast, weather) don't need hard evidence_ref_ids."""
    sources = _make_sources(analogs=AnalogSummary(count=5, success_count=3))
    out = build_why(sources)
    analog_tier = next(t for t in out.claim_tiers if t["label"] == "historical_analogs")
    assert analog_tier["tier"] == DriverConfidenceTier.PLAUSIBLE.value
    # evidence_ref_ids may be empty for pattern-based evidence — this is by design


def test_claim_tiers_unconfirmed_has_empty_refs():
    """Unconfirmed tiers with no data → no evidence refs."""
    sources = _make_sources(current=_make_current(fresh=False, staleness=900))
    out = build_why(sources)
    dispatch_tier = next(t for t in out.claim_tiers if t["label"] == "dispatch_price")
    assert dispatch_tier["tier"] == DriverConfidenceTier.UNCONFIRMED.value
    assert dispatch_tier["evidence_ref_ids"] == []


def test_claim_tiers_all_high_tiers_have_refs():
    """Global invariant: any confirmed or supported entry → non-empty evidence_ref_ids."""
    sources = _make_sources(
        current=_make_current(fresh=True),
        drivers=DriverContext(
            binding_constraints=[{
                "element_id": "C1",
                "values": {"marginal_value": 5000},
                "source": "AEMO_DISPATCHCONSTRAINT",
                "region": "NSW1",
                "valid_time": _VALID_TIME,
                "raw_ref": "c_archive",
            }],
            tight_interconnectors=[{
                "element_id": "IC1",
                "values": {"mw_flow": 950},
                "source": "AEMO_DISPATCHINTERCONNECTORRES",
                "region": "NSW1",
                "valid_time": _VALID_TIME,
                "raw_ref": "ic_archive",
            }],
        ),
        technology=TechnologyContext(
            has_unit_evidence=True,
            by_fuel={"gas": {"fuel_type": "gas", "total_cleared_mw": 1200, "delta_mw": 50, "raw_refs": ["r1"]}},
        ),
    )
    out = build_why(sources)
    high_tiers = [
        t for t in out.claim_tiers
        if t["tier"] in (DriverConfidenceTier.CONFIRMED.value, DriverConfidenceTier.SUPPORTED.value)
    ]
    for entry in high_tiers:
        assert len(entry["evidence_ref_ids"]) > 0, (
            f"tier={entry['tier']} label={entry['label']} has no evidence_ref_ids — invariant violated"
        )
