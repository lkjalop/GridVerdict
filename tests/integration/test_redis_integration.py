"""Redis integration tests — require a live Redis instance.

Run with:
    docker compose -f docker-compose.test.yml up -d
    GRIDVERDICT_REDIS_TEST=1 pytest tests/integration/test_redis_integration.py -v

Tests prove:
  1. Two app workers receive the same Redis bus event (pub/sub fanout)
  2. Rate limiting is enforced across two workers (Redis SET NX + EXPIRE pattern)
  3. Scheduler leader lock prevents duplicate job acquisition (only one worker wins)
  4. Leader failover works after TTL expiry (second instance acquires when TTL expires)

Skip guard: all tests are skipped unless GRIDVERDICT_REDIS_TEST=1 is set.
This ensures the suite stays green in CI/CD environments without Docker.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import timedelta

import pytest
import pytest_asyncio

# ── Skip guard ────────────────────────────────────────────────────────────────

_REDIS_TEST = os.getenv("GRIDVERDICT_REDIS_TEST", "").lower() in ("1", "true")
pytestmark = pytest.mark.skipif(
    not _REDIS_TEST,
    reason="Set GRIDVERDICT_REDIS_TEST=1 (and docker compose -f docker-compose.test.yml up -d) to run",
)

# Test Redis URL: uses port 6380 from docker-compose.test.yml
_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6380/0")

# Settings override for the duration of these tests
_LOCK_TTL = 3   # short TTL (seconds) so failover tests are fast
_HEARTBEAT = 1


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def redis_a():
    """Simulated Worker A Redis connection."""
    import redis.asyncio as aioredis
    client = aioredis.from_url(
        _REDIS_URL, encoding="utf-8", decode_responses=True,
        socket_connect_timeout=3, socket_timeout=3,
    )
    await client.ping()
    yield client
    await client.aclose()


@pytest_asyncio.fixture(scope="module")
async def redis_b():
    """Simulated Worker B Redis connection — separate client, same Redis instance."""
    import redis.asyncio as aioredis
    client = aioredis.from_url(
        _REDIS_URL, encoding="utf-8", decode_responses=True,
        socket_connect_timeout=3, socket_timeout=3,
    )
    await client.ping()
    yield client
    await client.aclose()


@pytest_asyncio.fixture(autouse=True)
async def flush_test_keys(redis_a):
    """Flush only the keys written by these tests to keep isolation."""
    test_prefix = "gv:test:"
    keys = await redis_a.keys(f"{test_prefix}*")
    if keys:
        await redis_a.delete(*keys)
    yield
    keys = await redis_a.keys(f"{test_prefix}*")
    if keys:
        await redis_a.delete(*keys)


# ── 1. Two-worker pub/sub fanout ──────────────────────────────────────────────

class TestTwoWorkerBusFanout:
    """Proves that an event published on Worker A arrives at Worker B via Redis pub/sub."""

    @pytest.mark.asyncio
    async def test_worker_b_receives_event_published_by_worker_a(self, redis_a, redis_b):
        channel = f"gv:test:events:{uuid.uuid4().hex[:8]}"

        received: list[dict] = []

        async def _listener():
            pubsub = redis_b.pubsub()
            await pubsub.subscribe(channel)
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    import json
                    received.append(json.loads(msg["data"]))
                    await pubsub.unsubscribe(channel)
                    break

        listener_task = asyncio.create_task(_listener())
        # Give subscription time to register before publishing
        await asyncio.sleep(0.1)

        import json
        payload = {"type": "dispatch_update", "region": "NSW1", "price_rrp": 185.0}
        await redis_a.publish(channel, json.dumps(payload))

        # Wait up to 2 seconds for delivery
        await asyncio.wait_for(listener_task, timeout=2.0)

        assert len(received) == 1, "Worker B must receive exactly 1 event"
        assert received[0]["type"] == "dispatch_update"
        assert received[0]["region"] == "NSW1"
        assert received[0]["price_rrp"] == 185.0

    @pytest.mark.asyncio
    async def test_multiple_events_all_delivered(self, redis_a, redis_b):
        """All published events arrive at the subscriber in order."""
        channel = f"gv:test:events:{uuid.uuid4().hex[:8]}"
        import json

        received: list[dict] = []

        async def _listener():
            pubsub = redis_b.pubsub()
            await pubsub.subscribe(channel)
            count = 0
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    received.append(json.loads(msg["data"]))
                    count += 1
                    if count >= 3:
                        await pubsub.unsubscribe(channel)
                        break

        listener_task = asyncio.create_task(_listener())
        await asyncio.sleep(0.1)

        for i in range(3):
            await redis_a.publish(channel, json.dumps({"seq": i, "region": "VIC1"}))
            await asyncio.sleep(0.01)

        await asyncio.wait_for(listener_task, timeout=3.0)

        assert len(received) == 3, f"Expected 3 events, got {len(received)}"
        assert [m["seq"] for m in received] == [0, 1, 2], "Events must arrive in order"


# ── 2. Rate limiting across workers ───────────────────────────────────────────

class TestCrossWorkerRateLimit:
    """Proves rate limiting is enforced across workers via Redis SET NX + EXPIRE."""

    @pytest.mark.asyncio
    async def test_worker_a_sets_rate_limit_worker_b_sees_it(self, redis_a, redis_b):
        """Worker A sets a rate-limit key; Worker B correctly reads it as rate-limited."""
        rate_key = f"gv:test:ratelimit:{uuid.uuid4().hex[:8]}"
        window_s = 10

        # Worker A: consume the first request (set key NX with expiry)
        acquired = await redis_a.set(rate_key, "1", nx=True, ex=window_s)
        assert acquired, "Worker A should acquire the rate-limit key"

        # Worker A: try a second time — must fail (already set)
        second = await redis_a.set(rate_key, "1", nx=True, ex=window_s)
        assert second is None, "Second attempt on Worker A must be blocked"

        # Worker B: check the key — must also find it rate-limited
        third = await redis_b.set(rate_key, "1", nx=True, ex=window_s)
        assert third is None, "Worker B must see the same rate-limit key"

    @pytest.mark.asyncio
    async def test_rate_limit_expires_and_resets(self, redis_a, redis_b):
        """After TTL expiry, the rate-limit window resets and a new request is allowed."""
        rate_key = f"gv:test:ratelimit:{uuid.uuid4().hex[:8]}"
        window_s = 1   # very short TTL for test speed

        # Consume the rate limit
        await redis_a.set(rate_key, "1", nx=True, ex=window_s)
        blocked = await redis_b.set(rate_key, "1", nx=True, ex=window_s)
        assert blocked is None, "Must be blocked before expiry"

        # Wait for TTL to expire
        await asyncio.sleep(1.2)

        # Now Worker B can acquire again
        after_expiry = await redis_b.set(rate_key, "1", nx=True, ex=window_s)
        assert after_expiry is not None, "Worker B must be allowed after rate-limit TTL expires"

    @pytest.mark.asyncio
    async def test_counter_increment_across_workers(self, redis_a, redis_b):
        """INCR from different workers converges to the correct total."""
        counter_key = f"gv:test:counter:{uuid.uuid4().hex[:8]}"
        await redis_a.expire(counter_key, 30)  # set expiry for cleanup

        # Both workers increment 5 times each
        tasks_a = [asyncio.create_task(redis_a.incr(counter_key)) for _ in range(5)]
        tasks_b = [asyncio.create_task(redis_b.incr(counter_key)) for _ in range(5)]
        await asyncio.gather(*tasks_a, *tasks_b)

        final = int(await redis_a.get(counter_key))
        assert final == 10, f"Expected counter=10, got {final}"


# ── 3. Scheduler leader lock prevents duplicate jobs ─────────────────────────

class TestSchedulerLeaderLock:
    """Proves the leader lock allows only one worker to be leader at a time."""

    @pytest.mark.asyncio
    async def test_only_one_worker_acquires_leader_lock(self, redis_a, redis_b):
        """When two workers race for the lock, exactly one wins."""
        from app.data.scheduler import try_acquire_leader_lock, release_leader_lock

        lock_key = f"gv:test:leader:{uuid.uuid4().hex[:8]}"

        # Patch the lock key for test isolation
        import app.data.scheduler as sched_module
        original_key = sched_module._LEADER_LOCK_KEY
        sched_module._LEADER_LOCK_KEY = lock_key

        try:
            # Reset instance IDs to distinct values
            id_a = str(uuid.uuid4())
            id_b = str(uuid.uuid4())
            sched_module._INSTANCE_ID = id_a

            # Both workers attempt to acquire simultaneously
            result_a = await try_acquire_leader_lock(redis_a)

            # Switch to worker B's instance ID
            sched_module._INSTANCE_ID = id_b
            result_b = await try_acquire_leader_lock(redis_b)

            # Exactly one must have won
            assert result_a or result_b, "At least one worker must acquire the lock"
            assert not (result_a and result_b), "Both workers must not hold the lock simultaneously"

            winner = redis_a if result_a else redis_b
            winner_id = id_a if result_a else id_b
            sched_module._INSTANCE_ID = winner_id
            await release_leader_lock(winner)

        finally:
            sched_module._LEADER_LOCK_KEY = original_key

    @pytest.mark.asyncio
    async def test_lock_holder_can_refresh(self, redis_a):
        """The leader can extend its lock via heartbeat."""
        from app.data.scheduler import try_acquire_leader_lock, heartbeat_leader_lock, release_leader_lock

        lock_key = f"gv:test:leader:{uuid.uuid4().hex[:8]}"

        import app.data.scheduler as sched_module
        original_key = sched_module._LEADER_LOCK_KEY
        sched_module._LEADER_LOCK_KEY = lock_key
        original_id = sched_module._INSTANCE_ID
        sched_module._INSTANCE_ID = str(uuid.uuid4())

        try:
            acquired = await try_acquire_leader_lock(redis_a)
            assert acquired, "Must acquire lock"

            still_leader = await heartbeat_leader_lock(redis_a)
            assert still_leader, "Heartbeat must succeed when we hold the lock"

            await release_leader_lock(redis_a)

        finally:
            sched_module._LEADER_LOCK_KEY = original_key
            sched_module._INSTANCE_ID = original_id

    @pytest.mark.asyncio
    async def test_non_holder_heartbeat_returns_false(self, redis_a, redis_b):
        """Heartbeat from a non-holder (wrong instance ID) returns False."""
        from app.data.scheduler import try_acquire_leader_lock, heartbeat_leader_lock, release_leader_lock

        lock_key = f"gv:test:leader:{uuid.uuid4().hex[:8]}"

        import app.data.scheduler as sched_module
        original_key = sched_module._LEADER_LOCK_KEY
        sched_module._LEADER_LOCK_KEY = lock_key
        original_id = sched_module._INSTANCE_ID

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        try:
            # Worker A acquires
            sched_module._INSTANCE_ID = id_a
            acquired = await try_acquire_leader_lock(redis_a)
            assert acquired

            # Worker B tries to heartbeat — should fail (not the holder)
            sched_module._INSTANCE_ID = id_b
            still_leader = await heartbeat_leader_lock(redis_b)
            assert not still_leader, "Non-holder heartbeat must return False"

            # Cleanup
            sched_module._INSTANCE_ID = id_a
            await release_leader_lock(redis_a)

        finally:
            sched_module._LEADER_LOCK_KEY = original_key
            sched_module._INSTANCE_ID = original_id


# ── 4. Leader failover after TTL expiry ───────────────────────────────────────

class TestLeaderFailover:
    """Proves that the leader lock is acquirable by another instance after TTL expiry."""

    @pytest.mark.asyncio
    async def test_failover_after_ttl_expiry(self, redis_a, redis_b):
        """When the leader disappears without releasing, another instance takes over after TTL."""
        import app.data.scheduler as sched_module
        import redis.asyncio as aioredis

        lock_key = f"gv:test:leader:{uuid.uuid4().hex[:8]}"
        short_ttl = 2   # 2-second TTL for test speed

        original_key = sched_module._LEADER_LOCK_KEY
        original_ttl = sched_module._settings.redis_leader_lock_ttl_s
        original_id = sched_module._INSTANCE_ID

        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        try:
            sched_module._LEADER_LOCK_KEY = lock_key
            # Temporarily override TTL to speed up the test
            sched_module._settings = type(sched_module._settings)(
                **{**sched_module._settings.model_dump(), "redis_leader_lock_ttl_s": short_ttl}
            )

            # Worker A acquires lock
            sched_module._INSTANCE_ID = id_a
            acquired_a = await sched_module.try_acquire_leader_lock(redis_a)
            assert acquired_a, "Worker A must acquire lock"

            # Worker B cannot acquire while A holds it
            sched_module._INSTANCE_ID = id_b
            blocked = await sched_module.try_acquire_leader_lock(redis_b)
            assert blocked is False or blocked is None, "Worker B must be blocked"

            # Worker A disappears (no heartbeat, no explicit release)
            # Wait for TTL to expire
            await asyncio.sleep(short_ttl + 0.5)

            # Worker B can now acquire (failover)
            sched_module._INSTANCE_ID = id_b
            acquired_b = await sched_module.try_acquire_leader_lock(redis_b)
            assert acquired_b, "Worker B must acquire lock after Worker A's TTL expires"

            # Cleanup
            await sched_module.release_leader_lock(redis_b)

        finally:
            sched_module._LEADER_LOCK_KEY = original_key
            sched_module._INSTANCE_ID = original_id
            # Restore original settings — reconstruct with original TTL
            try:
                sched_module._settings = type(sched_module._settings)(
                    **{**sched_module._settings.model_dump(), "redis_leader_lock_ttl_s": original_ttl}
                )
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_leader_released_allows_immediate_failover(self, redis_a, redis_b):
        """When the leader explicitly releases, the next worker can immediately take over."""
        from app.data.scheduler import try_acquire_leader_lock, release_leader_lock

        lock_key = f"gv:test:leader:{uuid.uuid4().hex[:8]}"

        import app.data.scheduler as sched_module
        original_key = sched_module._LEADER_LOCK_KEY
        original_id = sched_module._INSTANCE_ID
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        try:
            sched_module._LEADER_LOCK_KEY = lock_key

            sched_module._INSTANCE_ID = id_a
            acquired_a = await try_acquire_leader_lock(redis_a)
            assert acquired_a

            # A releases explicitly
            await release_leader_lock(redis_a)

            # B acquires immediately
            sched_module._INSTANCE_ID = id_b
            acquired_b = await try_acquire_leader_lock(redis_b)
            assert acquired_b, "Worker B must acquire lock immediately after explicit release"

            await release_leader_lock(redis_b)

        finally:
            sched_module._LEADER_LOCK_KEY = original_key
            sched_module._INSTANCE_ID = original_id
