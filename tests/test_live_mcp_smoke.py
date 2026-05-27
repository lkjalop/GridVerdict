"""Live MCP smoke tests — hit real NEMWeb endpoints.

Skipped by default. Run with:
    GRIDVERDICT_LIVE_TESTS=1 pytest tests/test_live_mcp_smoke.py -v -s

These tests make real HTTP requests to nemweb.com.au. They:
  - Verify we can fetch a live DispatchIS snapshot (price, demand, valid_time)
  - Verify we can fetch active market notices (may be empty list, that's OK)
  - Verify the data shapes match what the pipeline expects
  - Assert no hallucinated or placeholder values are returned

Do NOT run in CI without rate-limit guards and VPN/proxy configuration.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

_LIVE = os.getenv("GRIDVERDICT_LIVE_TESTS", "").lower() in ("1", "true")
pytestmark = pytest.mark.skipif(not _LIVE, reason="Set GRIDVERDICT_LIVE_TESTS=1 to run live smoke tests")


# ── fetch_latest_snapshot ────────────────────────────────────────────────────

class TestLiveDispatchSnapshot:
    @pytest.fixture
    def live_client(self):
        from app.data.aemo_live_client import AEMOLiveClient
        return AEMOLiveClient()

    async def test_snapshot_returns_all_five_regions(self, live_client):
        """All 5 NEM regions must be present in the snapshot."""
        snapshot = await live_client.fetch_latest_snapshot()
        regions = set(snapshot.regions.keys())
        expected = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
        assert expected <= regions, f"Missing regions: {expected - regions}"

    async def test_prices_are_in_valid_range(self, live_client):
        """Dispatch price must be within NEM market floor/cap."""
        snapshot = await live_client.fetch_latest_snapshot()
        for region, dp in snapshot.regions.items():
            assert -1000.0 <= dp.price_rrp <= 16600.0, (
                f"{region} price {dp.price_rrp} outside [-1000, 16600] market limits"
            )

    async def test_valid_time_is_recent(self, live_client):
        """Dispatch valid_time must be within the last 15 minutes."""
        snapshot = await live_client.fetch_latest_snapshot()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=15)
        for region, dp in snapshot.regions.items():
            assert dp.valid_time >= cutoff, (
                f"{region} valid_time {dp.valid_time} is stale (>15 min old)"
            )

    async def test_demand_is_positive(self, live_client):
        snapshot = await live_client.fetch_latest_snapshot()
        for region, dp in snapshot.regions.items():
            assert dp.demand_mw > 0, f"{region} demand_mw is non-positive: {dp.demand_mw}"

    async def test_raw_ref_is_populated(self, live_client):
        """raw_ref must be a non-empty string (SHA-256 or URL fragment)."""
        snapshot = await live_client.fetch_latest_snapshot()
        for region, dp in snapshot.regions.items():
            assert dp.raw_ref, f"{region} raw_ref is empty — evidence chain broken"

    async def test_snapshot_dispatch_price_shape(self, live_client):
        """DispatchPrice object must have all required fields."""
        snapshot = await live_client.fetch_latest_snapshot()
        dp = next(iter(snapshot.regions.values()))
        assert hasattr(dp, "region")
        assert hasattr(dp, "price_rrp")
        assert hasattr(dp, "demand_mw")
        assert hasattr(dp, "availability_mw")
        assert hasattr(dp, "valid_time")
        assert hasattr(dp, "raw_ref")


# ── fetch_active_notices ─────────────────────────────────────────────────────

class TestLiveMarketNotices:
    @pytest.fixture(scope="class")
    def notices_client(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        return AEMOMarketNoticesClient()

    def test_fetch_returns_list(self, notices_client):
        """fetch_active_notices must return a list (may be empty — that's OK)."""
        result = notices_client.fetch_active_notices()
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_notice_shape_if_present(self, notices_client):
        """Any returned notice must have id, title, and creation_time fields."""
        result = notices_client.fetch_active_notices()
        for notice in result:
            assert "id" in notice or "notice_id" in notice, f"Notice missing id: {notice}"
            assert "title" in notice or "subject" in notice, f"Notice missing title: {notice}"

    def test_regional_filter_works(self, notices_client):
        """Filtering by region must return subset of or equal to unfiltered results."""
        all_notices = notices_client.fetch_active_notices()
        nsw_notices = notices_client.fetch_active_notices(region="NSW1")
        assert len(nsw_notices) <= len(all_notices), (
            "Regional filter returned more notices than unfiltered — logic error"
        )

    def test_no_injection_patterns_in_notices(self, notices_client):
        """Notices must not contain prompt-injection patterns."""
        from app.security.observer import get_observer
        obs = get_observer()
        result = notices_client.fetch_active_notices()
        if result:
            check = obs.pass_tool_output(result)
            assert not check.should_halt(), (
                f"Live notices triggered security halt: {[s.name for s in check.signals]}"
            )


# ── scatter_gather live end-to-end ────────────────────────────────────────────

class TestLiveScatterGather:
    """Run the full scatter_gather pipeline against live NEMWeb.

    Does NOT hit the DB or HippoGraph — analogs and forecast will be absent
    (graph cold, LNN not trained). That's fine — this validates live HTTP paths.
    """

    async def test_scatter_gather_live_returns_dispatch(self):
        from app.data.aemo_live_client import AEMOLiveClient
        from app.data.cache import MarketCache
        from app.agents.scatter_gather import scatter_gather

        client = AEMOLiveClient()
        cache = MarketCache()

        result = await scatter_gather("NSW1", client, cache)

        assert result.dispatch is not None, (
            "Live scatter_gather must return dispatch — check NEMWeb connectivity"
        )
        assert result.dispatch.price_rrp is not None
        assert -1000.0 <= result.dispatch.price_rrp <= 16600.0

    async def test_scatter_gather_live_elapsed_under_10s(self):
        """Full live scatter_gather must complete in < 10 seconds."""
        from app.data.aemo_live_client import AEMOLiveClient
        from app.data.cache import MarketCache
        from app.agents.scatter_gather import scatter_gather

        client = AEMOLiveClient()
        cache = MarketCache()

        result = await scatter_gather("NSW1", client, cache)

        assert result.elapsed_ms < 10_000, (
            f"scatter_gather took {result.elapsed_ms:.0f}ms — exceeds 10s limit"
        )
