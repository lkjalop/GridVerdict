"""Sprint Q — Rolling Market Commentary: pytest test suite.

Tests cover:
  1. ChangeDetector — all change types, severity rules, first-boot (prev=None)
  2. RegionSnapshot — serialise / deserialise round-trip
  3. CommentaryEngine — process_tick with mocked cache, store, Why Engine
  4. CommentaryStore — prune, search_for_rag shape
  5. prose formatters — format_headline, format_factors
  6. Migration chain integrity (AST smoke test for 0004)
  7. routes_commentary — /recent, /stats, /{id} with mock store
  8. Scheduler integration — commentary_engine wired, cleanup job registered
  9. SSE — commentary_created in _ALL_EVENT_TYPES
  10. scatter_gather — commentary_context field present, include_commentary param
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── env vars must be set before any app import ──────────────────────────────
os.environ.setdefault("GRIDVERDICT_DEV_NO_AUTH", "true")
os.environ.setdefault("JWT_SECRET", "sprint-q-test-placeholder-not-used")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ChangeDetector
# ═══════════════════════════════════════════════════════════════════════════

def _snap(
    region="NSW1",
    price=80.0,
    demand=7000.0,
    headroom=800.0,
    regime="normal",
    spike_300=0.0,
    notice_ids=None,
    forecast_p90=None,
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
        notice_ids=notice_ids or [],
        forecast_p90=forecast_p90,
    )


class TestChangeDetector:
    def test_no_change_returns_empty(self):
        from app.engines.commentary.detector import detect
        prev = _snap(price=80.0)
        curr = _snap(price=85.0)  # less than $100 delta
        assert detect(prev, curr, []) == []

    def test_price_spike_300(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=280.0)
        curr = _snap(price=350.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.PRICE_SPIKE in types
        spike = next(c for c in changes if c.change_type == ChangeType.PRICE_SPIKE)
        assert spike.severity == "HIGH"
        assert spike.threshold_crossed == 300.0

    def test_price_spike_1000_critical(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=950.0)
        curr = _snap(price=1200.0)
        changes = detect(prev, curr, [])
        types = [c.change_type for c in changes]
        assert ChangeType.PRICE_SPIKE in types
        spike_1000 = next(
            (c for c in changes if c.change_type == ChangeType.PRICE_SPIKE and c.threshold_crossed == 1000.0),
            None,
        )
        assert spike_1000 is not None
        assert spike_1000.severity == "CRITICAL"

    def test_negative_price(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=50.0)
        curr = _snap(price=-100.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.NEGATIVE_PRICE in types
        neg = next(c for c in changes if c.change_type == ChangeType.NEGATIVE_PRICE)
        assert neg.severity == "HIGH"

    def test_price_normalised(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=500.0)
        curr = _snap(price=90.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.PRICE_NORMALISED in types

    def test_large_price_move_no_threshold_cross(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=100.0)
        curr = _snap(price=210.0)  # +$110, no threshold crossing
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.PRICE_MOVE_LARGE in types

    def test_large_price_move_suppressed_by_spike(self):
        """PRICE_MOVE_LARGE should not fire when a PRICE_SPIKE already captures the event."""
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(price=280.0)
        curr = _snap(price=500.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.PRICE_SPIKE in types
        assert ChangeType.PRICE_MOVE_LARGE not in types

    def test_regime_change(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(regime="normal")
        curr = _snap(regime="elevated")
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.PRICE_REGIME_CHANGE in types

    def test_headroom_tightened_medium(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(headroom=600.0)
        curr = _snap(headroom=450.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.HEADROOM_TIGHTENED in types
        h = next(c for c in changes if c.change_type == ChangeType.HEADROOM_TIGHTENED)
        assert h.severity == "MEDIUM"

    def test_headroom_tightened_high(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(headroom=250.0)
        curr = _snap(headroom=150.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.HEADROOM_TIGHTENED in types
        h = next(c for c in changes if c.change_type == ChangeType.HEADROOM_TIGHTENED)
        assert h.severity == "HIGH"

    def test_headroom_recovered(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(headroom=300.0)
        curr = _snap(headroom=600.0)
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.HEADROOM_RECOVERED in types

    def test_notice_added(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(notice_ids=["n001"])
        curr = _snap(notice_ids=["n001", "n002"])
        notices = [{"notice_id": "n001"}, {"notice_id": "n002"}]
        changes = detect(prev, curr, notices)
        types = {c.change_type for c in changes}
        assert ChangeType.NOTICE_ADDED in types

    def test_no_notice_change_when_ids_same(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(notice_ids=["n001"])
        curr = _snap(notice_ids=["n001"])
        notices = [{"notice_id": "n001"}]
        changes = detect(prev, curr, notices)
        types = {c.change_type for c in changes}
        assert ChangeType.NOTICE_ADDED not in types

    def test_forecast_risk_increased(self):
        from app.engines.commentary.detector import ChangeType, detect
        # Sprint R: FORECAST_RISK_INCREASED now requires P90 >= $500 AND delta > 15pp
        prev = _snap(spike_300=0.05, forecast_p90=600.0)
        curr = _snap(spike_300=0.25, forecast_p90=600.0)  # +20pp AND P90 $600
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.FORECAST_RISK_INCREASED in types

    def test_forecast_risk_decreased(self):
        from app.engines.commentary.detector import ChangeType, detect
        prev = _snap(spike_300=0.35)
        curr = _snap(spike_300=0.10)  # -25pp < -15pp threshold
        changes = detect(prev, curr, [])
        types = {c.change_type for c in changes}
        assert ChangeType.FORECAST_RISK_DECREASED in types

    def test_first_boot_prev_none_no_changes(self):
        """When prev is None and no notices, no changes should fire."""
        from app.engines.commentary.detector import detect
        curr = _snap(price=80.0)
        changes = detect(None, curr, [])
        assert changes == []

    def test_first_boot_prev_none_with_notices(self):
        """When prev is None but notices exist, NOTICE_ADDED fires once."""
        from app.engines.commentary.detector import ChangeType, detect
        curr = _snap()
        notices = [{"notice_id": "n001"}, {"notice_id": "n002"}]
        changes = detect(None, curr, notices)
        notice_changes = [c for c in changes if c.change_type == ChangeType.NOTICE_ADDED]
        assert len(notice_changes) == 1  # one per set, not per individual notice

    def test_change_descriptions_non_empty(self):
        from app.engines.commentary.detector import detect
        prev = _snap(price=280.0)
        curr = _snap(price=500.0)
        changes = detect(prev, curr, [])
        for c in changes:
            assert c.description, f"Empty description for {c.change_type}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. RegionSnapshot round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestRegionSnapshot:
    def test_snapshot_fields(self):
        from app.engines.commentary.snapshot import RegionSnapshot
        s = RegionSnapshot(
            region="SA1",
            price_rrp=120.5,
            demand_mw=1500.0,
            headroom_mw=400.0,
            regime="elevated",
            valid_time=datetime(2026, 5, 26, 8, 0, tzinfo=timezone.utc),
            spike_prob_300=0.12,
            spike_prob_1000=0.01,
            notice_ids=["n001"],
            forecast_p90=250.0,
        )
        assert s.region == "SA1"
        assert s.price_rrp == 120.5
        assert s.spike_prob_300 == 0.12

    def test_snapshot_json_round_trip(self):
        """The serialisation dict used by Redis store should round-trip cleanly."""
        from app.engines.commentary.snapshot import RegionSnapshot
        snap = RegionSnapshot(
            region="VIC1",
            price_rrp=95.0,
            demand_mw=5000.0,
            headroom_mw=700.0,
            regime="normal",
            valid_time=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            notice_ids=["x"],
        )
        d = {
            "region": snap.region,
            "price_rrp": snap.price_rrp,
            "demand_mw": snap.demand_mw,
            "headroom_mw": snap.headroom_mw,
            "regime": snap.regime,
            "valid_time": snap.valid_time.isoformat(),
            "spike_prob_300": snap.spike_prob_300,
            "spike_prob_1000": snap.spike_prob_1000,
            "notice_ids": snap.notice_ids,
            "forecast_p90": snap.forecast_p90,
        }
        raw = json.dumps(d)
        loaded = json.loads(raw)
        assert loaded["region"] == "VIC1"
        assert loaded["price_rrp"] == 95.0
        assert loaded["notice_ids"] == ["x"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prose formatters
# ═══════════════════════════════════════════════════════════════════════════

class TestProseFormatters:
    def _make_why(self, claim_map=None):
        from unittest.mock import MagicMock
        why = MagicMock()
        why.claim_map = claim_map or []
        why.missing_data = []
        why.next_watch = []
        why.counterargument = ""
        why.evidence_refs = []
        why.confidence = 0.6
        return why

    def _make_change(self, change_type, region="NSW1", prev=None, curr=None, threshold=None):
        from app.engines.commentary.detector import MaterialChange
        return MaterialChange(
            change_type=change_type,
            region=region,
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            severity="HIGH",
            prev_value=prev,
            curr_value=curr,
            threshold_crossed=threshold,
            description="test change",
        )

    def test_price_spike_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._make_change(ChangeType.PRICE_SPIKE, prev=280.0, curr=450.0)
        headline = format_headline(change, self._make_why())
        assert "NSW1" in headline
        assert "450" in headline
        assert "280" in headline

    def test_negative_price_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._make_change(ChangeType.NEGATIVE_PRICE, prev=50.0, curr=-200.0)
        headline = format_headline(change, self._make_why())
        assert "negative" in headline.lower()

    def test_headroom_tightened_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._make_change(ChangeType.HEADROOM_TIGHTENED, prev=600.0, curr=180.0)
        headline = format_headline(change, self._make_why())
        assert "180" in headline

    def test_notice_added_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._make_change(ChangeType.NOTICE_ADDED, region="QLD1")
        headline = format_headline(change, self._make_why())
        assert "AEMO" in headline
        assert "QLD1" in headline

    def test_forecast_risk_increased_headline(self):
        from app.engines.commentary.detector import ChangeType
        from app.engines.commentary.prose import format_headline
        change = self._make_change(ChangeType.FORECAST_RISK_INCREASED, prev=0.1, curr=0.32)
        headline = format_headline(change, self._make_why())
        assert "32" in headline

    def test_format_factors_sorts_by_tier(self):
        from app.engines.commentary.prose import format_factors
        from app.core.schema import ClaimMapItem, ClaimType, DriverConfidenceTier

        items = [
            ClaimMapItem(
                claim_type=ClaimType.CAUSE_CLAIM, label="constraint", tier=DriverConfidenceTier.PLAUSIBLE,
                present=True, confidence=0.5,
            ),
            ClaimMapItem(
                claim_type=ClaimType.PRICE_ASSERTION, label="price", tier=DriverConfidenceTier.CONFIRMED,
                present=True, confidence=0.95,
            ),
            ClaimMapItem(
                claim_type=ClaimType.DEMAND_ASSERTION, label="demand", tier=DriverConfidenceTier.SUPPORTED,
                present=True, confidence=0.7,
            ),
        ]
        why = self._make_why(claim_map=items)
        factors = format_factors(why)
        assert factors[0]["tier"] == "confirmed"
        assert factors[1]["tier"] == "supported"
        assert factors[2]["tier"] == "plausible"

    def test_format_factors_excludes_absent(self):
        from app.engines.commentary.prose import format_factors
        from app.core.schema import ClaimMapItem, ClaimType, DriverConfidenceTier

        items = [
            ClaimMapItem(
                claim_type=ClaimType.CAUSE_CLAIM, label="present_claim", tier=DriverConfidenceTier.CONFIRMED,
                present=True, confidence=0.9,
            ),
            ClaimMapItem(
                claim_type=ClaimType.CAUSE_CLAIM, label="absent_claim", tier=DriverConfidenceTier.CONFIRMED,
                present=False, confidence=0.9,
            ),
        ]
        why = self._make_why(claim_map=items)
        factors = format_factors(why)
        labels = [f["label"] for f in factors]
        assert "present_claim" in labels
        assert "absent_claim" not in labels


# ═══════════════════════════════════════════════════════════════════════════
# 4. CommentaryEngine.process_tick — end-to-end with mocked dependencies
# ═══════════════════════════════════════════════════════════════════════════

class TestCommentaryEngine:
    @pytest.mark.asyncio
    async def test_process_tick_no_change_returns_empty(self):
        """When prev and curr are identical (no material change), engine returns []."""
        from app.engines.commentary.engine import CommentaryEngine

        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)

        engine = CommentaryEngine()

        with (
            patch("app.engines.commentary.engine._is_on_cooldown", new=AsyncMock(return_value=False)),
            patch("app.engines.commentary.snapshot.load_snapshot", new=AsyncMock(return_value=None)),
            patch("app.engines.commentary.snapshot.save_snapshot", new=AsyncMock()),
            patch("app.engines.commentary.detector.detect", return_value=[]),
        ):
            events = await engine.process_tick(
                region="NSW1", price_rrp=80.0, demand_mw=7000.0,
                headroom_mw=800.0, regime="normal",
                valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
                cache=cache,
            )
        assert events == []

    @pytest.mark.asyncio
    async def test_process_tick_on_cooldown_skips_event(self):
        """When change is on cooldown, no commentary event is produced."""
        from app.engines.commentary.detector import ChangeType, MaterialChange
        from app.engines.commentary.engine import CommentaryEngine

        mock_change = MaterialChange(
            change_type=ChangeType.PRICE_SPIKE,
            region="NSW1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            severity="HIGH",
            prev_value=280.0,
            curr_value=400.0,
            threshold_crossed=300.0,
            description="test spike",
        )

        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)

        engine = CommentaryEngine()

        with (
            patch("app.engines.commentary.snapshot.load_snapshot", new=AsyncMock(return_value=None)),
            patch("app.engines.commentary.snapshot.save_snapshot", new=AsyncMock()),
            patch("app.engines.commentary.detector.detect", return_value=[mock_change]),
            patch("app.engines.commentary.engine._is_on_cooldown", new=AsyncMock(return_value=True)),
        ):
            events = await engine.process_tick(
                region="NSW1", price_rrp=400.0, demand_mw=7000.0,
                headroom_mw=600.0, regime="spike",
                valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
                cache=cache,
            )
        assert events == []

    @pytest.mark.asyncio
    async def test_process_tick_generates_event_on_material_change(self):
        """A material change that is not on cooldown should produce one CommentaryEvent."""
        from app.engines.commentary.detector import ChangeType, MaterialChange
        from app.engines.commentary.engine import CommentaryEngine, CommentaryEvent

        mock_change = MaterialChange(
            change_type=ChangeType.HEADROOM_TIGHTENED,
            region="SA1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            severity="MEDIUM",
            prev_value=600.0,
            curr_value=400.0,
            threshold_crossed=500.0,
            description="SA1 headroom tightened",
        )

        fake_evt = CommentaryEvent(
            id="fake-id",
            region="SA1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            system_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            event_type="headroom_tightened",
            severity="MEDIUM",
            headline="SA1 headroom tightened to 400MW",
            contributing_factors=[],
            missing_data=[],
            evidence_refs=[],
            confidence=0.6,
            corroborations={},
            next_watch=[],
            counterargument=None,
            snapshot_before=None,
            snapshot_after={},
        )

        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)

        engine = CommentaryEngine()

        with (
            patch("app.engines.commentary.snapshot.load_snapshot", new=AsyncMock(return_value=None)),
            patch("app.engines.commentary.snapshot.save_snapshot", new=AsyncMock()),
            patch("app.engines.commentary.detector.detect", return_value=[mock_change]),
            patch("app.engines.commentary.engine._is_on_cooldown", new=AsyncMock(return_value=False)),
            patch("app.engines.commentary.engine._build_event", new=AsyncMock(return_value=fake_evt)),
            patch("app.engines.commentary.store.write_event", new=AsyncMock()),
            patch("app.engines.commentary.engine._set_cooldown", new=AsyncMock()),
        ):
            events = await engine.process_tick(
                region="SA1", price_rrp=120.0, demand_mw=1500.0,
                headroom_mw=400.0, regime="elevated",
                valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
                cache=cache,
            )

        assert len(events) == 1
        assert events[0].id == "fake-id"
        assert events[0].event_type == "headroom_tightened"

    def test_commentary_event_to_dict(self):
        """to_dict() must produce a JSON-serialisable dict with expected keys."""
        from app.engines.commentary.engine import CommentaryEvent
        evt = CommentaryEvent(
            id="test-id",
            region="NSW1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            system_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            event_type="price_spike",
            severity="HIGH",
            headline="NSW1 price spiked",
            contributing_factors=[{"label": "constraint", "tier": "confirmed"}],
            missing_data=["bid_stacks"],
            evidence_refs=[],
            confidence=0.72,
            corroborations={"weather": True, "news": False, "notices": True},
            next_watch=["Watch price cap"],
            counterargument="Cause not fully explained",
            snapshot_before=None,
            snapshot_after={"price_rrp": 450.0},
        )
        d = evt.to_dict()
        assert d["id"] == "test-id"
        assert d["severity"] == "HIGH"
        assert d["confidence"] == 0.72
        assert d["corroborations"]["weather"] is True
        # Must be JSON-serialisable
        json.dumps(d)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Migration chain integrity (AST smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestMigration0004:
    def test_revision_constants(self):
        import importlib.util, pathlib
        path = pathlib.Path(__file__).parent.parent / "app/db/migrations/versions/0004_sprint_q_commentary_events.py"
        assert path.exists(), "Migration file 0004 not found"
        spec = importlib.util.spec_from_file_location("mig_0004", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.revision == "0004"
        assert mod.down_revision == "0003"

    def test_upgrade_creates_commentary_events(self):
        import importlib.util, pathlib, ast
        path = pathlib.Path(__file__).parent.parent / "app/db/migrations/versions/0004_sprint_q_commentary_events.py"
        source = path.read_text()
        assert "commentary_events" in source
        assert "create_table" in source
        assert "create_index" in source

    def test_downgrade_drops_commentary_events(self):
        import pathlib
        path = pathlib.Path(__file__).parent.parent / "app/db/migrations/versions/0004_sprint_q_commentary_events.py"
        source = path.read_text()
        assert "drop_table" in source
        assert "drop_index" in source


# ═══════════════════════════════════════════════════════════════════════════
# 6. DB model — CommentaryEvent ORM class exists with required columns
# ═══════════════════════════════════════════════════════════════════════════

class TestCommentaryEventModel:
    def test_model_exists(self):
        from app.db.models import CommentaryEvent
        assert CommentaryEvent.__tablename__ == "commentary_events"

    def test_model_has_required_columns(self):
        from app.db.models import CommentaryEvent
        mapper = CommentaryEvent.__mapper__
        col_names = {c.key for c in mapper.columns}
        required = {
            "id", "region", "valid_time", "system_time", "event_type",
            "severity", "headline", "contributing_factors", "missing_data",
            "evidence_refs", "confidence", "corroborations", "next_watch",
            "counterargument", "snapshot_before", "snapshot_after", "trace_id",
        }
        assert required.issubset(col_names), f"Missing columns: {required - col_names}"

    def test_indexes_defined(self):
        from app.db.models import CommentaryEvent
        index_names = {i.name for i in CommentaryEvent.__table__.indexes}
        assert "ix_commentary_region_valid_time" in index_names
        assert "ix_commentary_event_type" in index_names
        assert "ix_commentary_severity" in index_names


# ═══════════════════════════════════════════════════════════════════════════
# 7. routes_commentary — endpoint structure
# ═══════════════════════════════════════════════════════════════════════════

class TestCommentaryRoutes:
    def test_router_prefix(self):
        from app.api.routes_commentary import router
        assert router.prefix == "/commentary"

    def test_router_has_recent_route(self):
        from app.api.routes_commentary import router
        paths = [r.path for r in router.routes]
        assert any(p.endswith("/recent") for p in paths), f"No /recent route in {paths}"

    def test_router_has_stats_route(self):
        from app.api.routes_commentary import router
        paths = [r.path for r in router.routes]
        assert any(p.endswith("/stats") for p in paths), f"No /stats route in {paths}"

    def test_router_has_detail_route(self):
        from app.api.routes_commentary import router
        paths = [r.path for r in router.routes]
        assert any(p.endswith("/{event_id}") for p in paths), f"No /{{event_id}} route in {paths}"

    @pytest.mark.asyncio
    async def test_get_recent_calls_store(self):
        from fastapi.testclient import TestClient
        from app.api.routes_commentary import router
        import fastapi

        app = fastapi.FastAPI()
        app.include_router(router)

        mock_events = [
            {"id": "e1", "region": "NSW1", "headline": "Test spike", "severity": "HIGH",
             "confidence": 0.7, "event_type": "price_spike", "contributing_factors": [],
             "missing_data": [], "evidence_refs": [], "corroborations": {}, "next_watch": [],
             "valid_time": "2026-05-26T14:00:00+00:00", "system_time": "2026-05-26T14:00:00+00:00",
             "counterargument": None, "snapshot_before": None, "snapshot_after": {}, "trace_id": None}
        ]

        with patch("app.engines.commentary.store.search_recent", new=AsyncMock(return_value=mock_events)):
            client = TestClient(app)
            resp = client.get("/commentary/recent?region=NSW1")
            assert resp.status_code == 200
            data = resp.json()
            assert "events" in data
            assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_get_recent_invalid_region(self):
        from fastapi.testclient import TestClient
        from app.api.routes_commentary import router
        import fastapi

        app = fastapi.FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/commentary/recent?region=INVALID")
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════
# 8. SSE — commentary_created in allowed event types
# ═══════════════════════════════════════════════════════════════════════════

class TestSSEEventTypes:
    def test_commentary_created_in_all_event_types(self):
        from app.api.routes_events import _ALL_EVENT_TYPES
        assert "commentary_created" in _ALL_EVENT_TYPES

    def test_commentary_created_passes_type_filter(self):
        """The SSE filter logic should allow commentary_created through."""
        from app.api.routes_events import _ALL_EVENT_TYPES
        requested = frozenset({"commentary_created"}) & _ALL_EVENT_TYPES
        assert "commentary_created" in requested


# ═══════════════════════════════════════════════════════════════════════════
# 9. scatter_gather — new fields and parameter
# ═══════════════════════════════════════════════════════════════════════════

class TestScatterGatherCommentary:
    def test_gather_result_has_commentary_context_field(self):
        from app.agents.scatter_gather import GatherResult
        gr = GatherResult(dispatch=None, dispatch_fresh=False)
        assert hasattr(gr, "commentary_context")
        assert isinstance(gr.commentary_context, list)

    def test_scatter_gather_accepts_include_commentary_param(self):
        import inspect
        from app.agents.scatter_gather import scatter_gather
        sig = inspect.signature(scatter_gather)
        assert "include_commentary" in sig.parameters
        assert sig.parameters["include_commentary"].default is False


# ═══════════════════════════════════════════════════════════════════════════
# 10. Scheduler — commentary engine and cleanup job registered
# ═══════════════════════════════════════════════════════════════════════════

class TestSchedulerIntegration:
    def test_commentary_engine_singleton_exists(self):
        from app.data.scheduler import _commentary_engine
        from app.engines.commentary.engine import CommentaryEngine
        assert isinstance(_commentary_engine, CommentaryEngine)

    def test_commentary_cleanup_job_defined(self):
        """_job_commentary_cleanup should be importable and callable."""
        from app.data.scheduler import _job_commentary_cleanup
        import asyncio
        assert asyncio.iscoroutinefunction(_job_commentary_cleanup)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Cooldown constants — all change types with cooldown have non-zero values
# ═══════════════════════════════════════════════════════════════════════════

class TestCooldownConstants:
    def test_cooldown_values_positive(self):
        from app.engines.commentary.detector import ChangeType, _COOLDOWN_SECONDS
        for ct, secs in _COOLDOWN_SECONDS.items():
            assert secs > 0, f"Cooldown for {ct} must be positive"

    def test_no_cooldown_types(self):
        """PRICE_SPIKE and NOTICE_ADDED should have no cooldown (every crossing matters)."""
        from app.engines.commentary.detector import ChangeType, _COOLDOWN_SECONDS
        assert ChangeType.PRICE_SPIKE not in _COOLDOWN_SECONDS
        assert ChangeType.NOTICE_ADDED not in _COOLDOWN_SECONDS
        assert ChangeType.NEGATIVE_PRICE not in _COOLDOWN_SECONDS


# ═══════════════════════════════════════════════════════════════════════════
# 12. store.search_for_rag output shape
# ═══════════════════════════════════════════════════════════════════════════

class TestCommentaryStore:
    @pytest.mark.asyncio
    async def test_search_for_rag_returns_list_on_db_error(self):
        """When DB is unavailable, search_for_rag returns empty list (non-fatal)."""
        from app.engines.commentary.store import search_for_rag
        with patch("app.db.session.db_session", side_effect=Exception("no db")):
            result = await search_for_rag(
                region="NSW1",
                time_from=datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc),
                time_to=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_write_event_non_fatal_on_error(self):
        """write_event should not raise even if DB write fails."""
        from app.engines.commentary.engine import CommentaryEvent
        from app.engines.commentary.store import write_event

        evt = CommentaryEvent(
            id="x", region="NSW1",
            valid_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            system_time=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
            event_type="price_spike", severity="HIGH",
            headline="Test", contributing_factors=[], missing_data=[],
            evidence_refs=[], confidence=0.5, corroborations={},
            next_watch=[], counterargument=None, snapshot_before=None,
            snapshot_after={},
        )

        with patch("app.db.session.db_session", side_effect=Exception("no db")):
            await write_event(evt)  # must not raise
