from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agents.why_builder import build_seasonal_why
from app.agents.why_sources import SeasonalSources
from app.engines.analog_retriever import summarise_seasonal_events
from app.engines.temporal_utils import extract_season_buckets, resolve_season_to_range


def test_resolve_southern_hemisphere_autumn():
    bucket = resolve_season_to_range("autumn", 2025)
    assert bucket["from_dt"] == "2025-03-01T00:00:00+00:00"
    assert bucket["to_dt"] == "2025-06-01T00:00:00+00:00"


def test_resolve_summer_crosses_calendar_year():
    bucket = resolve_season_to_range("summer", 2025)
    assert bucket["from_dt"] == "2024-12-01T00:00:00+00:00"
    assert bucket["to_dt"] == "2025-03-01T00:00:00+00:00"


def test_extract_last_three_autumns_is_calendar_aware():
    now = datetime(2026, 5, 23, tzinfo=timezone.utc)
    buckets = extract_season_buckets("last three autumns", now=now)
    assert [bucket["label"] for bucket in buckets] == ["autumn 2025", "autumn 2024", "autumn 2023"]


def test_summarise_seasonal_events():
    buckets = [
        resolve_season_to_range("autumn", 2025),
        resolve_season_to_range("autumn", 2024),
    ]
    rows = [
        {"region": "NSW1", "valid_time": "2025-03-01T00:00:00+00:00", "price_rrp": 50.0},
        {"region": "NSW1", "valid_time": "2025-04-01T00:00:00+00:00", "price_rrp": 400.0},
        {"region": "NSW1", "valid_time": "2024-04-01T00:00:00+00:00", "price_rrp": 100.0},
        {"region": "VIC1", "valid_time": "2025-04-01T00:00:00+00:00", "price_rrp": 999.0},
    ]
    summaries = summarise_seasonal_events("NSW1", buckets, rows, spike_threshold=300.0)
    assert summaries[0]["interval_count"] == 2
    assert summaries[0]["mean_price"] == pytest.approx(225.0)
    assert summaries[0]["spike_count"] == 1
    assert summaries[1]["interval_count"] == 1


def test_build_seasonal_why_marks_attribution_limits():
    buckets = [resolve_season_to_range("autumn", 2025)]
    summaries = [{
        "label": "autumn 2025",
        "region": "NSW1",
        "from_dt": buckets[0]["from_dt"],
        "to_dt": buckets[0]["to_dt"],
        "interval_count": 2,
        "mean_price": 225.0,
        "p90_price": 365.0,
        "max_price": 400.0,
        "spike_count": 1,
    }]
    output = build_seasonal_why(SeasonalSources("NSW1", buckets, summaries))
    assert "autumn 2025" in output.why_plain_english
    assert "statistical only" in output.counterargument
    assert "historical_constraints" in output.missing_data
    assert output.evidence_refs
