"""Tests for Market Incident Timeline.

Covers:
  - Engine unit tests (build_incident_timeline with empty DB)
  - HTTP route: GET /api/incidents/timeline
  - Structure invariants (ordering, tiers, evidence_ref_ids)
  - Coverage grade logic
  - Invalid inputs
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.engines.incident_timeline import (
    TimelineEvent,
    IncidentTimeline,
    build_incident_timeline,
    timeline_to_dict,
    _as_utc,
    _classify_regime,
    _ic_is_tight,
    _coverage_grade,
    _build_verdict,
)

_ANCHOR = datetime(2026, 5, 25, 14, 5, 0, tzinfo=timezone.utc)
_WINDOW_START = _ANCHOR - timedelta(minutes=30)

_VALID_TIERS = {"confirmed", "supported", "plausible", "unconfirmed"}
_VALID_CATEGORIES = {
    "price", "headroom", "constraint", "interconnector",
    "unit_dispatch", "rebid", "weather", "missing_data",
}


# ── Helper: minimal async DB session stub ────────────────────────────

class _EmptyDB:
    """Returns empty result sets for all queries (simulates no data ingested)."""

    async def execute(self, *args, **kwargs):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        return result


# ── Engine unit tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_returns_incident_timeline():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR, lookback_minutes=30)
    assert isinstance(tl, IncidentTimeline)


@pytest.mark.asyncio
async def test_build_has_correct_region():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    assert tl.region == "NSW1"


@pytest.mark.asyncio
async def test_build_has_correct_anchor():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    assert tl.anchor_interval == _ANCHOR


@pytest.mark.asyncio
async def test_build_events_is_list():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    assert isinstance(tl.events, list)
    assert len(tl.events) > 0


@pytest.mark.asyncio
async def test_build_events_all_have_required_fields():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        assert hasattr(ev, "event_id")
        assert hasattr(ev, "interval")
        assert hasattr(ev, "category")
        assert hasattr(ev, "description")
        assert hasattr(ev, "tier")
        assert hasattr(ev, "evidence_ref_ids")
        assert hasattr(ev, "missing")


@pytest.mark.asyncio
async def test_build_events_valid_tiers():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        assert ev.tier in _VALID_TIERS, f"Invalid tier: {ev.tier}"


@pytest.mark.asyncio
async def test_build_events_valid_categories():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        assert ev.category in _VALID_CATEGORIES, f"Invalid category: {ev.category}"


@pytest.mark.asyncio
async def test_build_events_ordered_newest_first():
    """Events must be sorted newest → oldest (descending interval)."""
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    times = [e.interval for e in tl.events]
    assert times == sorted(times, reverse=True), "Events are not sorted newest-first"


def test_as_utc_normalizes_naive_datetime():
    naive = datetime(2026, 5, 25, 14, 5, 0)
    normalized = _as_utc(naive)
    assert normalized.tzinfo is timezone.utc
    assert normalized.hour == 14


@pytest.mark.asyncio
async def test_build_with_empty_db_all_events_unconfirmed_or_missing():
    """With no data ingested, every event is unconfirmed or a missing_data entry."""
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        assert ev.tier in ("unconfirmed", "plausible") or ev.missing, (
            f"Expected unconfirmed/missing with no data, got tier={ev.tier} for {ev.category}"
        )


@pytest.mark.asyncio
async def test_build_has_verdict_dict():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    v = tl.verdict
    assert isinstance(v, dict)
    for key in ("confirmed", "supported", "plausible", "unconfirmed", "missing_coverage"):
        assert key in v, f"verdict missing key '{key}'"


@pytest.mark.asyncio
async def test_build_coverage_grade_minimal_with_no_data():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    assert tl.coverage_grade == "minimal"


@pytest.mark.asyncio
async def test_build_has_as_of():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    assert isinstance(tl.as_of, datetime)


# ── Missing_data flag ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_data_events_have_missing_true():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    missing_events = [e for e in tl.events if e.category == "missing_data"]
    for ev in missing_events:
        assert ev.missing is True


@pytest.mark.asyncio
async def test_missing_true_events_always_unconfirmed():
    """Any event with missing=True must have tier 'unconfirmed'."""
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        if ev.missing:
            assert ev.tier == "unconfirmed", (
                f"missing event {ev.category} has tier={ev.tier}, expected unconfirmed"
            )


# ── Confirmed events require evidence_ref_ids ─────────────────────────

@pytest.mark.asyncio
async def test_confirmed_events_have_evidence_refs():
    """Confirmed tier must always have at least one evidence_ref_id."""
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        if ev.tier == "confirmed":
            assert len(ev.evidence_ref_ids) > 0, (
                f"confirmed event {ev.category} has no evidence_ref_ids"
            )


@pytest.mark.asyncio
async def test_supported_events_have_evidence_refs():
    """Supported tier must always have at least one evidence_ref_id."""
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    for ev in tl.events:
        if ev.tier == "supported":
            assert len(ev.evidence_ref_ids) > 0, (
                f"supported event {ev.category} has no evidence_ref_ids"
            )


# ── timeline_to_dict serialisation ───────────────────────────────────

@pytest.mark.asyncio
async def test_timeline_to_dict_has_required_keys():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    d = timeline_to_dict(tl)
    for key in ("region", "anchor_interval", "anchor_price", "anchor_regime",
                "lookback_minutes", "coverage_grade", "verdict", "as_of", "events"):
        assert key in d, f"timeline_to_dict missing key '{key}'"


@pytest.mark.asyncio
async def test_timeline_to_dict_events_serialisable():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    d = timeline_to_dict(tl)
    for ev in d["events"]:
        for key in ("event_id", "interval", "category", "description", "tier",
                    "evidence_summary", "evidence_ref_ids", "delta", "missing"):
            assert key in ev, f"event dict missing key '{key}'"


@pytest.mark.asyncio
async def test_timeline_to_dict_intervals_are_iso_strings():
    tl = await build_incident_timeline(_EmptyDB(), "NSW1", _ANCHOR)
    d = timeline_to_dict(tl)
    for ev in d["events"]:
        try:
            datetime.fromisoformat(ev["interval"])
        except ValueError:
            pytest.fail(f"event interval is not ISO: {ev['interval']}")


# ── Helper function unit tests ─────────────────────────────────────────

def test_classify_regime_extreme():
    assert _classify_regime(15000.0) == "extreme"


def test_classify_regime_spike():
    assert _classify_regime(500.0) == "spike"
    assert _classify_regime(300.0) == "spike"


def test_classify_regime_elevated():
    assert _classify_regime(150.0) == "elevated"


def test_classify_regime_normal():
    assert _classify_regime(80.0) == "normal"


def test_ic_is_tight_near_export_limit():
    vals = {"mw_flow": 950.0, "export_limit": 1000.0, "import_limit": -1000.0}
    assert _ic_is_tight(vals) is True


def test_ic_is_tight_far_from_limit():
    vals = {"mw_flow": 200.0, "export_limit": 1000.0, "import_limit": -1000.0}
    assert _ic_is_tight(vals) is False


def test_ic_is_tight_no_flow():
    assert _ic_is_tight({}) is False


def test_coverage_grade_full():
    verdict = {"confirmed": ["a"], "supported": ["b", "c"], "plausible": [], "unconfirmed": [], "missing_coverage": []}
    assert _coverage_grade(verdict) == "full"


def test_coverage_grade_partial_one_confirmed():
    verdict = {"confirmed": ["a"], "supported": [], "plausible": [], "unconfirmed": [], "missing_coverage": []}
    assert _coverage_grade(verdict) == "partial"


def test_coverage_grade_partial_one_supported():
    verdict = {"confirmed": [], "supported": ["b"], "plausible": [], "unconfirmed": [], "missing_coverage": []}
    assert _coverage_grade(verdict) == "partial"


def test_coverage_grade_minimal():
    verdict = {"confirmed": [], "supported": [], "plausible": ["c"], "unconfirmed": ["d"], "missing_coverage": ["x"]}
    assert _coverage_grade(verdict) == "minimal"


def test_build_verdict_deduplicates():
    events = [
        TimelineEvent("ev1", _ANCHOR, "price", "same desc", "confirmed"),
        TimelineEvent("ev2", _ANCHOR - timedelta(minutes=5), "price", "same desc", "confirmed"),
    ]
    verdict = _build_verdict(events, set())
    assert verdict["confirmed"].count("same desc") == 1


def test_build_verdict_excludes_missing_events():
    events = [
        TimelineEvent("ev1", _ANCHOR, "missing_data", "data absent", "unconfirmed", missing=True),
    ]
    verdict = _build_verdict(events, {"price"})
    assert "data absent" not in verdict["unconfirmed"]
    assert "price" in verdict["missing_coverage"]


# ── HTTP route tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_incident_timeline_route_200(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_incident_timeline_route_has_events(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1")
    data = r.json()
    assert "events" in data
    assert isinstance(data["events"], list)


@pytest.mark.asyncio
async def test_incident_timeline_route_has_verdict(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1")
    data = r.json()
    assert "verdict" in data
    assert "confirmed" in data["verdict"]
    assert "missing_coverage" in data["verdict"]


@pytest.mark.asyncio
async def test_incident_timeline_route_has_coverage_grade(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1")
    data = r.json()
    assert data["coverage_grade"] in ("full", "partial", "minimal")


@pytest.mark.asyncio
async def test_incident_timeline_route_invalid_region(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=INVALID")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_incident_timeline_route_invalid_interval(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1&interval=not-a-date")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_incident_timeline_route_with_valid_interval(client: AsyncClient):
    r = await client.get(
        "/api/incidents/timeline?region=NSW1&interval=2026-05-25T14:05:00Z&lookback_minutes=15"
    )
    assert r.status_code == 200
    data = r.json()
    assert data["lookback_minutes"] == 15


@pytest.mark.asyncio
async def test_incident_timeline_events_ordered_newest_first_via_http(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1")
    data = r.json()
    times = [e["interval"] for e in data["events"]]
    assert times == sorted(times, reverse=True), "HTTP response events not sorted newest-first"


@pytest.mark.asyncio
async def test_incident_timeline_all_regions(client: AsyncClient):
    for region in ("NSW1", "VIC1", "QLD1", "SA1", "TAS1"):
        r = await client.get(f"/api/incidents/timeline?region={region}")
        assert r.status_code == 200, f"Failed for region {region}"


@pytest.mark.asyncio
async def test_incident_timeline_lookback_min_boundary(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1&lookback_minutes=1")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_incident_timeline_lookback_max_boundary(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1&lookback_minutes=120")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_incident_timeline_lookback_over_max(client: AsyncClient):
    r = await client.get("/api/incidents/timeline?region=NSW1&lookback_minutes=121")
    assert r.status_code == 422
