"""Sprint A: MCP source contract + provenance tests.

Tests cover:
  - SourceStatus construction (from_data, unavailable, to_dict)
  - SourceStatus.freshness_label property
  - SourceTimer context manager
  - call_tool_with_retry success + retry + exhaust
  - _wrap_task success, None result, exception, empty list, dict unavailable
  - GatherResult.source_statuses field presence
  - scatter_gather populates source_statuses (via mocked tasks)
  - QueryResponse provenance field is serialised
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.source_status import (
    FRESHNESS_THRESHOLDS,
    SourceStatus,
    SourceTimer,
)


# ── SourceStatus.from_data ─────────────────────────────────────────────────────

class TestSourceStatusFromData:
    def test_fresh_data_full_confidence(self):
        vt = datetime.now(timezone.utc) - timedelta(seconds=30)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        assert s.coverage_status == "full"
        assert s.confidence >= 0.9
        assert s.freshness_s < 60
        assert s.error is None

    def test_stale_data_degrades_confidence(self):
        threshold = FRESHNESS_THRESHOLDS["AEMO_DISPATCH_PRICE"]   # 300 s
        vt = datetime.now(timezone.utc) - timedelta(seconds=threshold + 10)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        # confidence = 0 when freshness_ratio >= 1
        assert s.confidence == 0.0
        # coverage degrades to partial when confidence <= 0.2
        assert s.coverage_status == "partial"

    def test_exactly_half_threshold_partial_confidence(self):
        threshold = FRESHNESS_THRESHOLDS["AEMO_DISPATCH_PRICE"]   # 300
        vt = datetime.now(timezone.utc) - timedelta(seconds=threshold / 2)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        assert 0.4 < s.confidence < 0.6   # ~50% through threshold → ~50% confidence

    def test_partial_flag_forces_partial_status(self):
        vt = datetime.now(timezone.utc)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt, partial=True)
        assert s.coverage_status == "partial"

    def test_extra_confidence_multiplied(self):
        vt = datetime.now(timezone.utc)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt, extra_confidence=0.5)
        assert s.confidence == pytest.approx(0.5, abs=0.01)

    def test_none_valid_time_uses_now(self):
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", None)
        # freshness should be near 0 when valid_time = now
        assert s.freshness_s < 2.0
        assert s.confidence > 0.9

    def test_naive_datetime_gets_utc_attached(self):
        vt = datetime.utcnow()  # no tzinfo
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        assert s.freshness_s < 2.0

    def test_unknown_source_uses_default_threshold(self):
        vt = datetime.now(timezone.utc) - timedelta(seconds=300)
        s = SourceStatus.from_data("UNKNOWN_SOURCE_XYZ", vt)
        # default threshold is 600 s, so 300s = 50% → confidence ~0.5
        assert 0.4 < s.confidence < 0.6

    def test_raw_ref_truncated_at_300(self):
        long_ref = "x" * 500
        vt = datetime.now(timezone.utc)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt, raw_ref=long_ref)
        assert len(s.raw_ref) == 300

    def test_latency_ms_stored(self):
        vt = datetime.now(timezone.utc)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt, latency_ms=123.456)
        assert s.latency_ms == pytest.approx(123.5, abs=0.1)


# ── SourceStatus.unavailable ───────────────────────────────────────────────────

class TestSourceStatusUnavailable:
    def test_confidence_zero(self):
        s = SourceStatus.unavailable("AEMO_DISPATCH_PRICE", "timeout")
        assert s.confidence == 0.0
        assert s.coverage_status == "unavailable"

    def test_error_truncated_at_200(self):
        s = SourceStatus.unavailable("X", "e" * 300)
        assert len(s.error) == 200

    def test_latency_stored(self):
        s = SourceStatus.unavailable("X", "err", latency_ms=55.0)
        assert s.latency_ms == 55.0

    def test_source_stored(self):
        s = SourceStatus.unavailable("AEMO_MARKET_NOTICES", "fail")
        assert s.source == "AEMO_MARKET_NOTICES"


# ── SourceStatus.to_dict ───────────────────────────────────────────────────────

class TestSourceStatusToDict:
    def test_to_dict_has_all_keys(self):
        s = SourceStatus.unavailable("X", "err")
        d = s.to_dict()
        expected = {"source", "valid_time", "system_time", "raw_ref",
                    "freshness_s", "confidence", "coverage_status", "error", "latency_ms"}
        assert set(d.keys()) == expected

    def test_to_dict_values_round_trip(self):
        vt = datetime.now(timezone.utc)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt, latency_ms=10.0)
        d = s.to_dict()
        assert d["source"] == "AEMO_DISPATCH_PRICE"
        assert d["confidence"] == s.confidence
        assert d["error"] is None


# ── SourceStatus.freshness_label ───────────────────────────────────────────────

class TestFreshnessLabel:
    def test_fresh_label(self):
        vt = datetime.now(timezone.utc) - timedelta(seconds=10)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        assert s.freshness_label == "fresh"

    def test_stale_label_just_over_half(self):
        threshold = FRESHNESS_THRESHOLDS["AEMO_DISPATCH_PRICE"]
        vt = datetime.now(timezone.utc) - timedelta(seconds=threshold * 0.6)
        s = SourceStatus.from_data("AEMO_DISPATCH_PRICE", vt)
        assert s.freshness_label == "stale"

    def test_unavailable_label(self):
        s = SourceStatus.unavailable("AEMO_DISPATCH_PRICE", "err")
        assert s.freshness_label == "unavailable"


# ── SourceTimer ────────────────────────────────────────────────────────────────

class TestSourceTimer:
    def test_measures_elapsed(self):
        import time
        with SourceTimer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10.0
        assert t.elapsed_ms < 500.0

    def test_zero_before_exit(self):
        timer = SourceTimer()
        timer.__enter__()
        assert timer.elapsed_ms == 0.0
        timer.__exit__(None, None, None)
        assert timer.elapsed_ms > 0.0


# ── call_tool_with_retry ───────────────────────────────────────────────────────

class TestCallToolWithRetry:
    @pytest.fixture(autouse=True)
    def _patch_call_tool(self):
        self._mock_ct = AsyncMock()
        with patch("app.mcp.router.call_tool", self._mock_ct):
            yield

    @pytest.mark.asyncio
    async def test_success_returns_result_and_status(self):
        from app.mcp.router import call_tool_with_retry
        self._mock_ct.return_value = {"foo": "bar"}
        result, status = await call_tool_with_retry("weather_consensus", region="NSW1")
        assert result == {"foo": "bar"}
        assert status.source == "WEATHER_CONSENSUS"
        assert status.coverage_status in ("full", "partial")
        assert self._mock_ct.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        from app.mcp.router import call_tool_with_retry, MCPCallError
        self._mock_ct.side_effect = [MCPCallError("fail"), {"data": 1}]
        result, status = await call_tool_with_retry("weather_consensus", max_retries=1, backoff_s=0.0)
        assert result == {"data": 1}
        assert self._mock_ct.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_unavailable(self):
        from app.mcp.router import call_tool_with_retry, MCPCallError
        self._mock_ct.side_effect = MCPCallError("always fails")
        result, status = await call_tool_with_retry("weather_consensus", max_retries=1, backoff_s=0.0)
        assert result is None
        assert status.coverage_status == "unavailable"
        assert status.confidence == 0.0
        assert self._mock_ct.call_count == 2


# ── _wrap_task ─────────────────────────────────────────────────────────────────

class TestWrapTask:
    @pytest.mark.asyncio
    async def test_success_returns_result_and_full_status(self):
        from app.agents.scatter_gather import _wrap_task

        async def good_coro():
            return [{"notice": "test"}]

        data, status = await _wrap_task("AEMO_MARKET_NOTICES", good_coro())
        assert data == [{"notice": "test"}]
        assert status.source == "AEMO_MARKET_NOTICES"
        assert status.coverage_status == "full"

    @pytest.mark.asyncio
    async def test_empty_list_returns_partial(self):
        from app.agents.scatter_gather import _wrap_task

        async def empty_coro():
            return []

        data, status = await _wrap_task("HIPPOGRAPH_ANALOGS", empty_coro())
        assert data == []
        assert status.coverage_status == "partial"

    @pytest.mark.asyncio
    async def test_none_result_returns_unavailable(self):
        from app.agents.scatter_gather import _wrap_task

        async def none_coro():
            return None

        data, status = await _wrap_task("LNN_FORECAST", none_coro())
        assert data is None
        assert status.coverage_status == "unavailable"

    @pytest.mark.asyncio
    async def test_dict_with_available_false_returns_partial(self):
        from app.agents.scatter_gather import _wrap_task

        async def not_available():
            return {"available": False, "reason": "no data"}

        data, status = await _wrap_task("LIVE_QUANTILE_FORECAST", not_available())
        assert data is not None
        assert status.coverage_status == "partial"

    @pytest.mark.asyncio
    async def test_exception_returns_unavailable(self):
        from app.agents.scatter_gather import _wrap_task

        async def failing_coro():
            raise RuntimeError("network error")

        data, status = await _wrap_task("AEMO_PREDISPATCH", failing_coro())
        assert data is None
        assert status.coverage_status == "unavailable"
        assert "network error" in status.error

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        from app.agents.scatter_gather import _wrap_task
        import time

        async def slow_coro():
            await asyncio.sleep(0.01)
            return {"ok": True}

        _, status = await _wrap_task("WEATHER_CONSENSUS", slow_coro())
        assert status.latency_ms >= 10.0


# ── GatherResult.source_statuses ──────────────────────────────────────────────

class TestGatherResultSourceStatuses:
    def test_default_source_statuses_empty(self):
        from app.agents.scatter_gather import GatherResult
        gr = GatherResult(dispatch=None, dispatch_fresh=False)
        assert gr.source_statuses == {}

    def test_source_statuses_accepts_dict(self):
        from app.agents.scatter_gather import GatherResult
        s = SourceStatus.unavailable("AEMO_DISPATCH_PRICE", "err")
        gr = GatherResult(dispatch=None, dispatch_fresh=False, source_statuses={"AEMO_DISPATCH_PRICE": s})
        assert gr.source_statuses["AEMO_DISPATCH_PRICE"].confidence == 0.0


# ── scatter_gather populates source_statuses ───────────────────────────────────

class TestScatterGatherSourceStatuses:
    @pytest.mark.asyncio
    async def test_source_statuses_populated_on_success(self):
        from app.agents.scatter_gather import scatter_gather, GatherResult
        from app.data.aemo_live_client import DispatchPrice
        from unittest.mock import AsyncMock, MagicMock, patch

        now = datetime.now(timezone.utc)
        dp = DispatchPrice(
            region="NSW1",
            price_rrp=100.0,
            demand_mw=7000.0,
            availability_mw=9000.0,
            valid_time=now,
            system_time=now,
            raw_ref="test-ref",
        )

        mock_client = AsyncMock()
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.age_seconds.return_value = None

        with (
            patch("app.agents.scatter_gather._task_dispatch", AsyncMock(return_value=dp)),
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", mock_client, mock_cache)

        assert "AEMO_DISPATCH_PRICE" in result.source_statuses
        assert isinstance(result.source_statuses["AEMO_DISPATCH_PRICE"], SourceStatus)

    @pytest.mark.asyncio
    async def test_dispatch_failure_gives_unavailable_status(self):
        from app.agents.scatter_gather import scatter_gather
        from unittest.mock import AsyncMock, patch

        mock_client = AsyncMock()
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.age_seconds.return_value = None

        with (
            patch("app.agents.scatter_gather._task_dispatch",
                  AsyncMock(side_effect=RuntimeError("AEMO down"))),
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_analogs", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_forecast", new=AsyncMock(return_value=None)),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            result = await scatter_gather("NSW1", mock_client, mock_cache)

        s = result.source_statuses.get("AEMO_DISPATCH_PRICE")
        assert s is not None
        assert s.coverage_status == "unavailable"
        assert s.confidence == 0.0


# ── QueryResponse provenance field ────────────────────────────────────────────

class TestQueryResponseProvenance:
    def test_provenance_field_optional(self):
        from app.api.routes_query import QueryResponse
        from app.core.schema import FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand

        verdict = FactualVerdict(
            verdict=VerdictLabel.INSUFFICIENT_DATA,
            action=ActionLabel.MONITOR,
            confidence=0.0,
            confidence_band=ConfidenceBand.VERY_LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english="test",
            counterargument="No contrary evidence available.",
        )
        qr = QueryResponse(
            query_id="q1",
            session_id="s1",
            intent="lookup",
            verdict=verdict,
            decomposition={},
            viewport_type="answer",
        )
        assert qr.provenance is None

    def test_provenance_with_source_statuses(self):
        from app.api.routes_query import QueryResponse
        from app.core.schema import FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand

        verdict = FactualVerdict(
            verdict=VerdictLabel.INSUFFICIENT_DATA,
            action=ActionLabel.MONITOR,
            confidence=0.0,
            confidence_band=ConfidenceBand.VERY_LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english="test",
            counterargument="No contrary evidence available.",
        )
        s = SourceStatus.unavailable("AEMO_DISPATCH_PRICE", "err")
        qr = QueryResponse(
            query_id="q1",
            session_id="s1",
            intent="lookup",
            verdict=verdict,
            decomposition={},
            viewport_type="answer",
            provenance=[s.to_dict()],
        )
        assert qr.provenance is not None
        assert len(qr.provenance) == 1
        assert qr.provenance[0]["source"] == "AEMO_DISPATCH_PRICE"


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _async_return(value):
    """Return a coroutine that yields value (for patching task functions)."""
    return value
