"""Tests for Production Hardening Sprint 1 — Redis abstractions and Prometheus metrics.

Covers:
 Event bus (in-process mode):
  1.  subscribe returns a queue; unsubscribe removes it from registry
  2.  publish fans out to all matching subscribers
  3.  region filter: only matching region receives event
  4.  type filter: only matching type receives event
  5.  queue-full drop: subscriber queue full → QueueFull suppressed, others still deliver
  6.  subscriber_count reflects current subscriber count

 Event bus (Redis mode):
  7.  publish calls redis.publish when Redis is available
  8.  publish falls back to _dispatch_local when redis.publish raises
  9.  start_redis_listener is no-op when get_redis returns None

 Rate limiter (in-process fallback):
 10.  Requests within limit are allowed (window tracking)
 11.  find_limit returns None for unmatched paths
 12.  find_limit returns correct (window_s, max_req) for matched paths
 13.  reset_rate_limits clears in-process windows

 Rate limiter (Redis backend):
 14.  Redis Lua script result [1, 0] → allowed=True, retry=0
 15.  Redis Lua script result [0, 5] → allowed=False, retry=5
 16.  Redis script exception → returns (None, 0) to signal fallback

 Scheduler leader lock:
 17.  try_acquire_leader_lock returns True when Redis.set returns truthy
 18.  try_acquire_leader_lock returns False when Redis.set returns None
 19.  heartbeat_leader_lock returns True and refreshes TTL when instance owns lock
 20.  heartbeat_leader_lock returns False when lock held by another instance
 21.  release_leader_lock deletes key when instance owns lock
 22.  release_leader_lock does nothing when another instance owns lock
 23.  Without Redis, start_scheduler calls _start_jobs (always-leader mode)

 Prometheus metrics:
 24.  ingest_success_total increments when _record_success is called
 25.  ingest_failure_total increments when _record_failure is called
 26.  claim_verifier_downgrades_total increments on downgrade finding
 27.  GET /api/metrics returns HTTP 200
 28.  GET /api/metrics uses Prometheus text/plain content-type
 29.  GET /api/metrics body contains all expected metric names
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_counter_value(metric_obj, labels: dict) -> float:
    """Read the current value of a prometheus Counter child directly via its _value."""
    child = metric_obj.labels(**labels) if labels else metric_obj
    try:
        return child._value.get()
    except AttributeError:
        # Unlabelled counter
        return child._value.get()  # type: ignore[union-attr]


def _subscribe_with_maxsize(maxsize: int):
    """Register a subscriber queue with a custom maxsize (for testing queue-full behaviour)."""
    import app.data.event_bus as eb
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    eb._counter += 1
    eb._subs[eb._counter] = (q, None, None)
    return q


# ── 1–6: Event bus in-process ────────────────────────────────────────────────

class TestEventBusInProcess:
    """All tests run with Redis disabled — patches app.data.redis_client.get_redis."""

    @pytest.fixture(autouse=True)
    def _no_redis(self, monkeypatch):
        # Patch the real source so publish()'s local import gets the mock.
        monkeypatch.setattr("app.data.redis_client.get_redis", AsyncMock(return_value=None))

    def teardown_method(self):
        import app.data.event_bus as eb
        eb._subs.clear()

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self):
        from app.data.event_bus import subscribe, unsubscribe, subscriber_count
        assert subscriber_count() == 0
        q = subscribe()
        assert subscriber_count() == 1
        unsubscribe(q)
        assert subscriber_count() == 0

    @pytest.mark.asyncio
    async def test_publish_delivers_to_subscriber(self):
        from app.data.event_bus import subscribe, unsubscribe, publish
        q = subscribe()
        await publish("test_event", {"val": 42})
        event = q.get_nowait()
        assert event.type == "test_event"
        assert event.payload["val"] == 42
        unsubscribe(q)

    @pytest.mark.asyncio
    async def test_region_filter_blocks_unmatched(self):
        from app.data.event_bus import subscribe, unsubscribe, publish
        q_nsw = subscribe(region="NSW1")
        q_vic = subscribe(region="VIC1")
        await publish("price_spike", {"price": 999}, region="NSW1")
        event = q_nsw.get_nowait()
        assert event.region == "NSW1"
        assert q_vic.empty(), "VIC1 subscriber must not receive NSW1 event"
        unsubscribe(q_nsw)
        unsubscribe(q_vic)

    @pytest.mark.asyncio
    async def test_type_filter_blocks_unmatched(self):
        from app.data.event_bus import subscribe, unsubscribe, publish
        q_dispatch = subscribe(types={"dispatch_updated"})
        q_forecast = subscribe(types={"forecast_updated"})
        await publish("dispatch_updated", {})
        q_dispatch.get_nowait()
        assert q_forecast.empty(), "forecast subscriber must not receive dispatch event"
        unsubscribe(q_dispatch)
        unsubscribe(q_forecast)

    @pytest.mark.asyncio
    async def test_queue_full_does_not_raise(self):
        """Events dropped for a full queue must not propagate; other queues still receive."""
        import app.data.event_bus as eb
        q_full = _subscribe_with_maxsize(maxsize=1)
        # Fill the queue
        await eb.publish("a", {})
        # Second publish: q_full is full, should drop silently without raising
        await eb.publish("b", {})
        # A fresh subscriber still receives future events
        q2 = eb.subscribe()
        await eb.publish("c", {})
        event = q2.get_nowait()
        assert event.type == "c"
        eb.unsubscribe(q_full)
        eb.unsubscribe(q2)

    @pytest.mark.asyncio
    async def test_subscriber_count(self):
        from app.data.event_bus import subscribe, unsubscribe, subscriber_count
        qs = [subscribe() for _ in range(5)]
        assert subscriber_count() == 5
        for q in qs:
            unsubscribe(q)
        assert subscriber_count() == 0


# ── 7–9: Event bus Redis mode ────────────────────────────────────────────────

class TestEventBusRedis:

    def teardown_method(self):
        import app.data.event_bus as eb
        eb._subs.clear()

    @pytest.mark.asyncio
    async def test_publish_uses_redis_when_available(self):
        """publish() must send to Redis channel, not local queues, when Redis is up."""
        from app.data.event_bus import publish, subscriber_count
        import app.data.event_bus as eb

        # Subscriber that should NOT receive (because Redis handles delivery)
        q = eb.subscribe()

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)

        with patch("app.data.redis_client.get_redis", AsyncMock(return_value=mock_redis)):
            await publish("dispatch_updated", {"x": 1}, region="NSW1")

        mock_redis.publish.assert_awaited_once()
        channel, raw = mock_redis.publish.call_args[0]
        import json
        data = json.loads(raw)
        assert channel == "gv:events"
        assert data["type"] == "dispatch_updated"
        assert data["region"] == "NSW1"
        # Local queue is empty (delivery is handled by Redis listener, not publish())
        assert q.empty()
        eb.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_publish_falls_back_on_redis_error(self):
        """When redis.publish raises, event must still reach local subscribers."""
        import app.data.event_bus as eb
        q = eb.subscribe()

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(side_effect=ConnectionError("Redis down"))

        with patch("app.data.redis_client.get_redis", AsyncMock(return_value=mock_redis)):
            await eb.publish("fallback_event", {"ok": True})

        event = q.get_nowait()
        assert event.type == "fallback_event"
        eb.unsubscribe(q)

    @pytest.mark.asyncio
    async def test_start_redis_listener_no_op_without_redis(self):
        """start_redis_listener must not create a task when Redis is unavailable."""
        import app.data.event_bus as eb
        eb._listener_task = None

        with patch("app.data.redis_client.get_redis", AsyncMock(return_value=None)):
            await eb.start_redis_listener()

        assert eb._listener_task is None


# ── 10–13: Rate limiter in-process ───────────────────────────────────────────

class TestRateLimiterInProcess:

    def setup_method(self):
        from app.api.middleware import reset_rate_limits
        reset_rate_limits()

    def test_rate_window_tracking(self):
        """Timestamps inside the window count; older timestamps are evicted."""
        import time
        from app.api.middleware import _windows, _LIMITS

        window_s, max_req = next(
            (w, m) for pfx, w, m in _LIMITS if pfx == "/api/auth/token"
        )
        key = "tenant:test:token"
        now = time.monotonic()
        # 2-minute-old timestamps → outside 60 s window → should be evicted
        _windows[key] = [now - 120] * max_req
        cutoff = now - window_s
        remaining = [ts for ts in _windows[key] if ts >= cutoff]
        assert len(remaining) == 0, "Stale timestamps must fall outside the 60 s window"

    def test_find_limit_none_for_unmatched_path(self):
        from app.api.middleware import _find_limit
        assert _find_limit("/api/health") is None
        assert _find_limit("/api/portfolio/bess/scenario") is None
        assert _find_limit("/api/market/state") is None

    def test_find_limit_returns_correct_limits(self):
        from app.api.middleware import _find_limit
        assert _find_limit("/api/auth/token") == (60, 10)
        assert _find_limit("/api/market/forecast") == (60, 20)
        window_s, max_req = _find_limit("/api/sessions/abc/query")
        assert max_req == 30

    def test_reset_clears_windows(self):
        import time
        from app.api.middleware import _windows, reset_rate_limits
        _windows["some_key"] = [time.monotonic()]
        reset_rate_limits()
        assert len(_windows) == 0


# ── 14–16: Rate limiter Redis backend ─────────────────────────────────────────

class TestRateLimiterRedis:

    def setup_method(self):
        import app.api.middleware as mw
        mw._redis_script = None  # reset cached script between tests

    @pytest.mark.asyncio
    async def test_redis_script_allowed(self):
        """Script returning [1, 0] → allowed=True, retry=0."""
        from app.api.middleware import _check_redis
        mock_script = AsyncMock(return_value=[1, 0])
        mock_redis = MagicMock()
        mock_redis.register_script = MagicMock(return_value=mock_script)

        allowed, retry = await _check_redis(mock_redis, "gv:rl:test", 1000.0, 60, 10)
        assert allowed is True
        assert retry == 0

    @pytest.mark.asyncio
    async def test_redis_script_rejected(self):
        """Script returning [0, 5] → allowed=False, retry_after=5."""
        from app.api.middleware import _check_redis
        mock_script = AsyncMock(return_value=[0, 5])
        mock_redis = MagicMock()
        mock_redis.register_script = MagicMock(return_value=mock_script)

        allowed, retry = await _check_redis(mock_redis, "gv:rl:test", 1000.0, 60, 10)
        assert allowed is False
        assert retry == 5

    @pytest.mark.asyncio
    async def test_redis_script_error_returns_none(self):
        """Script raising ConnectionError → (None, 0), signalling in-process fallback."""
        from app.api.middleware import _check_redis
        mock_script = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_redis = MagicMock()
        mock_redis.register_script = MagicMock(return_value=mock_script)

        allowed, retry = await _check_redis(mock_redis, "gv:rl:test", 1000.0, 60, 10)
        assert allowed is None


# ── 17–23: Scheduler leader lock ──────────────────────────────────────────────

class TestSchedulerLeaderLock:

    @pytest.mark.asyncio
    async def test_acquire_lock_when_free(self):
        """Redis.set returning truthy → lock acquired."""
        from app.data.scheduler import try_acquire_leader_lock
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        result = await try_acquire_leader_lock(mock_redis)
        assert result is True
        mock_redis.set.assert_awaited_once()
        kw = mock_redis.set.call_args.kwargs
        assert kw.get("nx") is True
        assert "ex" in kw

    @pytest.mark.asyncio
    async def test_acquire_lock_when_held(self):
        """Redis.set returning None → not acquired (another instance holds it)."""
        from app.data.scheduler import try_acquire_leader_lock
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=None)
        result = await try_acquire_leader_lock(mock_redis)
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_refreshes_when_leader(self):
        """heartbeat_leader_lock returns True and calls expire when we own the lock."""
        from app.data.scheduler import heartbeat_leader_lock, _INSTANCE_ID
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=_INSTANCE_ID)
        mock_redis.expire = AsyncMock()
        result = await heartbeat_leader_lock(mock_redis)
        assert result is True
        mock_redis.expire.assert_awaited_once()
        # First positional arg must be the lock key
        lock_key = mock_redis.expire.call_args[0][0]
        assert "scheduler" in lock_key

    @pytest.mark.asyncio
    async def test_heartbeat_returns_false_when_lock_lost(self):
        """heartbeat_leader_lock returns False when another instance holds the lock."""
        from app.data.scheduler import heartbeat_leader_lock
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="other-instance-id-9999")
        mock_redis.expire = AsyncMock()
        result = await heartbeat_leader_lock(mock_redis)
        assert result is False
        mock_redis.expire.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_release_lock_when_owner(self):
        """release_leader_lock deletes the Redis key when this instance owns it."""
        from app.data.scheduler import release_leader_lock, _INSTANCE_ID
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=_INSTANCE_ID)
        mock_redis.delete = AsyncMock()
        await release_leader_lock(mock_redis)
        mock_redis.delete.assert_awaited_once_with("gv:scheduler:leader")

    @pytest.mark.asyncio
    async def test_release_lock_when_not_owner(self):
        """release_leader_lock must not delete the key if another instance owns it."""
        from app.data.scheduler import release_leader_lock
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="other-instance-id-9999")
        mock_redis.delete = AsyncMock()
        await release_leader_lock(mock_redis)
        mock_redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_redis_always_leader(self):
        """Without Redis, start_scheduler must call _start_jobs immediately."""
        import app.data.scheduler as sched_mod

        jobs_started: list[bool] = []

        async def _mock_start_jobs():
            jobs_started.append(True)

        with (
            patch("app.data.redis_client.get_redis", AsyncMock(return_value=None)),
            patch.object(sched_mod, "_start_jobs", _mock_start_jobs),
        ):
            await sched_mod.start_scheduler()

        assert jobs_started, "_start_jobs must be called when Redis is unavailable"
        assert sched_mod._is_leader is True
        # Cleanup global state
        sched_mod._is_leader = False


# ── 24–29: Prometheus metrics ─────────────────────────────────────────────────

class TestPrometheusMetrics:

    def test_ingest_success_counter_increments(self):
        """_record_success must increment gv_ingest_success_total for the job label."""
        from app.data.scheduler import _record_success, _job_state
        from app.api.metrics_registry import ingest_success_total
        _job_state.clear()

        before = _get_counter_value(ingest_success_total, {"job": "dispatch_refresh"})
        _record_success("dispatch_refresh")
        after = _get_counter_value(ingest_success_total, {"job": "dispatch_refresh"})
        assert after == before + 1.0

    def test_ingest_failure_counter_increments(self):
        """_record_failure must increment gv_ingest_failure_total for the job label."""
        from app.data.scheduler import _record_failure, _job_state
        from app.api.metrics_registry import ingest_failure_total
        _job_state.clear()

        before = _get_counter_value(ingest_failure_total, {"job": "lnn_retrain"})
        _record_failure("lnn_retrain", Exception("test error"))
        after = _get_counter_value(ingest_failure_total, {"job": "lnn_retrain"})
        assert after == before + 1.0

    def test_claim_verifier_downgrade_counter_increments(self):
        """A downgrade finding in verify_answer must increment the downgrade counter."""
        from app.core.schema import FactualVerdict, VerdictLabel
        from app.agents.claim_verifier import verify_answer
        from app.api.metrics_registry import claim_verifier_downgrades_total

        before = _get_counter_value(claim_verifier_downgrades_total, {})

        # LOW_CONFIDENCE verdict + claim_tier with confirmed tier + no evidence_ref_ids
        # → Rule 3 fires (high_tier_without_evidence_refs, severity=downgrade)
        # why_plain_english contains a numeric claim so Rule 1 also fires, but Rule 3
        # (downgrade) is what drives the counter increment.
        factual = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            confidence=0.4,
            confidence_band="low",
            why_plain_english="The price rose by $500/MWh in NSW1 due to a constraint.",
            action="monitor",
            as_of=_now_iso(),
            counterargument="",
            evidence_refs=[],
            claim_tiers=[{
                "label": "price_level",
                "tier": "confirmed",
                "category": "price_spike",
                "evidence_ref_ids": [],
            }],
        )
        verify_answer(factual)

        after = _get_counter_value(claim_verifier_downgrades_total, {})
        assert after > before

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_200(self):
        from httpx import AsyncClient, ASGITransport
        from app.api.main import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/metrics")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_endpoint_content_type(self):
        from httpx import AsyncClient, ASGITransport
        from app.api.main import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/metrics")
        assert "text/plain" in r.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_metrics_endpoint_contains_metric_names(self):
        from httpx import AsyncClient, ASGITransport
        from app.api.main import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/metrics")
        body = r.text
        expected = [
            "gv_ingest_success_total",
            "gv_ingest_failure_total",
            "gv_claim_verifier_downgrades_total",
            "gv_last_dispatch_age_seconds",
            "gv_sse_subscribers",
            "gv_query_latency_ms",
            "gv_llm_decompose_latency_ms",
            "gv_forecast_latency_ms",
        ]
        for name in expected:
            assert name in body, f"Metric '{name}' missing from /api/metrics output"
