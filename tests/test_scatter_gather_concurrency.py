"""scatter_gather concurrency and resilience tests.

Tests that the pipeline:
  - Fires T2-T6 truly in parallel (elapsed ≈ max(task_time), not sum)
  - Degrades gracefully when one task times out (TimeoutError)
  - Degrades gracefully when one task returns 429 (rate limit)
  - Degrades gracefully when multiple tasks fail
  - Returns tasks_ok count that accurately reflects failures
  - Never propagates exceptions to the caller
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.scatter_gather import GatherResult, scatter_gather
from app.data.aemo_live_client import AEMOLiveClient, DispatchPrice, LiveMarketSnapshot
from app.data.cache import MarketCache


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fresh_dispatch(region: str = "NSW1", price: float = 180.0) -> DispatchPrice:
    return DispatchPrice(
        region=region,
        valid_time=_now() - timedelta(seconds=60),
        system_time=_now(),
        price_rrp=price,
        demand_mw=8400.0,
        availability_mw=9200.0,
        raw_ref="concurrency-test-ref",
    )


def _mock_client(region: str = "NSW1", price: float = 180.0) -> AEMOLiveClient:
    dp = _fresh_dispatch(region, price)
    snapshot = LiveMarketSnapshot(interval=dp.valid_time, fetched_at=_now(), regions={region: dp})
    client = AsyncMock(spec=AEMOLiveClient)
    client.fetch_latest_snapshot = AsyncMock(return_value=snapshot)
    client.fetch_predispatch = AsyncMock(return_value={})
    return client


def _mock_cache() -> MarketCache:
    cache = AsyncMock(spec=MarketCache)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)
    cache.age_seconds = AsyncMock(return_value=None)  # treat as stale — safe default
    return cache


# ── Baseline: all tasks succeed ───────────────────────────────────────────────

class TestScatterGatherHappyPath:
    async def test_returns_gather_result(self):
        client = _mock_client("NSW1")
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[{"ppr_score": 0.8}])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value={"p10": 120.0, "p50": 180.0, "p90": 250.0})),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert isinstance(result, GatherResult)
        assert result.dispatch is not None
        assert result.dispatch.price_rrp == 180.0
        assert result.tasks_ok >= 1

    async def test_source_coverage_full_when_all_succeed(self):
        client = _mock_client("NSW1")
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value={"p10": 100.0, "p50": 150.0, "p90": 200.0})),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert result.source_coverage > 0.5


# ── Timeout resilience ────────────────────────────────────────────────────────

class TestScatterGatherTimeout:
    async def test_single_task_timeout_does_not_crash(self):
        """T3 (analogs) timing out must not prevent the pipeline from returning."""
        client = _mock_client("NSW1")
        cache = _mock_cache()

        async def _slow_analogs(*args, **kwargs):
            await asyncio.sleep(30)   # will exceed _TASK_TIMEOUT
            return []

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(side_effect=asyncio.TimeoutError())),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert isinstance(result, GatherResult), "Must return GatherResult even when T3 times out"
        assert result.analogs == [], "Timed-out analogs should default to empty list"

    async def test_dispatch_timeout_returns_none_dispatch(self):
        """T1 (dispatch) timing out → dispatch=None but result still returned."""
        client = AsyncMock(spec=AEMOLiveClient)
        client.fetch_latest_snapshot = AsyncMock(side_effect=asyncio.TimeoutError())
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert isinstance(result, GatherResult)
        assert result.dispatch is None
        assert result.dispatch_fresh is False

    async def test_multiple_timeouts_tasks_ok_reflects_failures(self):
        """tasks_ok count must decrease for each failed task."""
        client = AsyncMock(spec=AEMOLiveClient)
        client.fetch_latest_snapshot = AsyncMock(side_effect=asyncio.TimeoutError())
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices",
                  new=AsyncMock(side_effect=asyncio.TimeoutError())),
            patch("app.agents.scatter_gather._task_analogs",
                  new=AsyncMock(side_effect=asyncio.TimeoutError())),
            patch("app.agents.scatter_gather._task_forecast",
                  new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch",
                  new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment",
                  new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        # T1, T2, T3 failed → tasks_ok should be ≤ 3 (T4 None also counts as failure)
        assert result.tasks_ok <= 3, (
            f"Expected ≤3 tasks_ok with 3 timeouts, got {result.tasks_ok}"
        )
        assert result.source_coverage < 0.6


# ── Rate-limit (429) resilience ───────────────────────────────────────────────

class TestScatterGatherRateLimit:
    async def test_429_on_notices_does_not_crash(self):
        """T2 returning 429-simulated exception must be swallowed."""
        import httpx
        client = _mock_client("NSW1")
        cache = _mock_cache()

        rate_limit_exc = Exception("429 Too Many Requests")

        with (
            patch("app.agents.scatter_gather._task_notices",
                  new=AsyncMock(side_effect=rate_limit_exc)),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert isinstance(result, GatherResult)
        assert result.notices == [], "Rate-limited notices should default to []"

    async def test_429_on_predispatch_does_not_reduce_dispatch_quality(self):
        """T5 (predispatch) 429 must not affect T1 (dispatch) result."""
        client = _mock_client("NSW1", price=350.0)
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch",
                  new=AsyncMock(side_effect=Exception("429 rate limited"))),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert result.dispatch is not None, "dispatch must still be populated when predispatch 429s"
        assert result.dispatch.price_rrp == 350.0
        assert result.predispatch == []

    async def test_all_external_tasks_fail_dispatch_still_usable(self):
        """Full external failure: dispatch is the fallback of last resort."""
        client = _mock_client("NSW1", price=125.0)
        cache = _mock_cache()

        exc = Exception("Service unavailable")

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(side_effect=exc)),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(side_effect=exc)),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(side_effect=exc)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(side_effect=exc)),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(side_effect=exc)),
        ):
            result = await scatter_gather("NSW1", client, cache)

        assert result.dispatch is not None, "dispatch must survive total external failure"
        assert result.dispatch.price_rrp == 125.0
        assert result.analogs == []
        assert result.notices == []
        assert result.forecast is None


# ── Parallelism sanity check ──────────────────────────────────────────────────

class TestScatterGatherParallelism:
    async def test_t2_t6_run_in_parallel_not_serial(self):
        """T2-T6 must complete in ≈max(delays), not sum(delays).

        Each mock sleeps 0.05s. If serial: ~0.25s. If parallel: ~0.05s.
        """
        DELAY = 0.05

        async def _slow(*args, **kwargs):
            await asyncio.sleep(DELAY)
            return []

        async def _slow_none(*args, **kwargs):
            await asyncio.sleep(DELAY)
            return None

        client = _mock_client("NSW1")
        cache = _mock_cache()

        with (
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(side_effect=_slow)),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(side_effect=_slow)),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(side_effect=_slow_none)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(side_effect=_slow)),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(side_effect=_slow)),
        ):
            import time
            t0 = time.monotonic()
            result = await scatter_gather("NSW1", client, cache)
            elapsed = time.monotonic() - t0

        # Serial would be ~0.25s; parallel should be well under 0.20s
        assert elapsed < DELAY * 4, (
            f"T2-T6 appear to run serially ({elapsed:.3f}s ≥ {DELAY * 4:.3f}s)"
        )
        assert result.elapsed_ms > 0
