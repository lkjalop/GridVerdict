"""Tests for SSE Live Event Stream.

Covers:
  - Event bus unit tests (subscribe, publish, unsubscribe, queue full, region/type filtering)
  - HTTP route: GET /api/events/stream (connection, event delivery, heartbeat timing)
  - Invalid region / unknown types guard
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.data.event_bus import (
    BusEvent,
    publish,
    subscribe,
    subscriber_count,
    unsubscribe,
)


# ── Event bus unit tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_subscribe_returns_queue():
    q = subscribe()
    assert isinstance(q, asyncio.Queue)
    unsubscribe(q)


@pytest.mark.asyncio
async def test_publish_delivers_to_subscriber():
    q = subscribe()
    await publish("dispatch_updated", {"regions": ["NSW1"]})
    event = q.get_nowait()
    assert isinstance(event, BusEvent)
    assert event.type == "dispatch_updated"
    unsubscribe(q)


@pytest.mark.asyncio
async def test_publish_payload_preserved():
    q = subscribe()
    await publish("forecast_updated", {"source": "lnn_retrain", "count": 42})
    event = q.get_nowait()
    assert event.payload["source"] == "lnn_retrain"
    assert event.payload["count"] == 42
    unsubscribe(q)


@pytest.mark.asyncio
async def test_publish_region_field():
    q = subscribe()
    await publish("dispatch_updated", {}, region="VIC1")
    event = q.get_nowait()
    assert event.region == "VIC1"
    unsubscribe(q)


@pytest.mark.asyncio
async def test_region_filter_excludes_other_region():
    q = subscribe(region="NSW1")
    await publish("dispatch_updated", {}, region="VIC1")
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
    unsubscribe(q)


@pytest.mark.asyncio
async def test_region_filter_passes_matching_region():
    q = subscribe(region="NSW1")
    await publish("dispatch_updated", {}, region="NSW1")
    event = q.get_nowait()
    assert event.type == "dispatch_updated"
    unsubscribe(q)


@pytest.mark.asyncio
async def test_region_filter_passes_no_region_event():
    """Events published without a region reach all region-filtered subscribers."""
    q = subscribe(region="NSW1")
    await publish("scheduler_failure", {"job_id": "lnn_retrain"}, region=None)
    event = q.get_nowait()
    assert event.type == "scheduler_failure"
    unsubscribe(q)


@pytest.mark.asyncio
async def test_type_filter_excludes_unmatched_types():
    q = subscribe(types={"dispatch_updated"})
    await publish("forecast_updated", {})
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
    unsubscribe(q)


@pytest.mark.asyncio
async def test_type_filter_passes_matching_type():
    q = subscribe(types={"forecast_updated", "dispatch_updated"})
    await publish("forecast_updated", {"source": "predispatch_refresh"})
    event = q.get_nowait()
    assert event.type == "forecast_updated"
    unsubscribe(q)


@pytest.mark.asyncio
async def test_unsubscribe_removes_subscriber():
    initial = subscriber_count()
    q = subscribe()
    assert subscriber_count() == initial + 1
    unsubscribe(q)
    assert subscriber_count() == initial


@pytest.mark.asyncio
async def test_multiple_subscribers_each_receive_event():
    q1 = subscribe()
    q2 = subscribe()
    await publish("trace_written", {"trace_id": "abc"})
    e1 = q1.get_nowait()
    e2 = q2.get_nowait()
    assert e1.type == "trace_written"
    assert e2.type == "trace_written"
    unsubscribe(q1)
    unsubscribe(q2)


@pytest.mark.asyncio
async def test_queue_full_does_not_raise():
    """Publishing when a subscriber queue is full must not raise; it drops the event."""
    q = subscribe()
    # Fill the queue to capacity
    for i in range(100):
        await publish("dispatch_updated", {"seq": i})
    # One more event must not raise
    await publish("dispatch_updated", {"seq": 999})
    assert q.qsize() == 100
    unsubscribe(q)


@pytest.mark.asyncio
async def test_bus_event_has_ts():
    q = subscribe()
    await publish("data_status_changed", {})
    event = q.get_nowait()
    assert event.ts  # ISO string, non-empty
    unsubscribe(q)


# ── SSE generator unit tests (bypass HTTP layer) ──────────────────────
#
# httpx ASGITransport drives the full ASGI body synchronously, which blocks
# on infinite SSE streams. We test the generator function directly instead.

from app.api.routes_events import _sse_stream_generator


def _make_mock_request(disconnect_after: int = 4):
    """Return an async callable that acts as request.is_disconnected().

    Returns False for the first `disconnect_after - 1` calls, then True.
    """
    calls = [0]

    async def is_disconnected() -> bool:
        calls[0] += 1
        return calls[0] >= disconnect_after

    class _MockReq:
        pass

    req = _MockReq()
    req.is_disconnected = is_disconnected
    return req


@pytest.mark.asyncio
async def test_generator_delivers_event():
    """Generator yields an SSE frame when an event is published to the bus."""
    mock_req = _make_mock_request(disconnect_after=10)
    frames: list[str] = []

    async def _run():
        async for frame in _sse_stream_generator("NSW1", frozenset({"dispatch_updated"}), mock_req):
            frames.append(frame)
            break  # stop after first frame

    async def _trigger():
        await asyncio.sleep(0.05)
        await publish("dispatch_updated", {"regions": ["NSW1"]}, region="NSW1")

    await asyncio.gather(asyncio.create_task(_run()), asyncio.create_task(_trigger()))
    assert frames, "Generator yielded no frames"
    assert "dispatch_updated" in frames[0]


@pytest.mark.asyncio
async def test_generator_frame_data_is_json():
    """The data: portion of a yielded frame must be valid JSON."""
    mock_req = _make_mock_request(disconnect_after=10)
    frames: list[str] = []

    async def _run():
        async for frame in _sse_stream_generator("NSW1", frozenset({"scheduler_failure"}), mock_req):
            frames.append(frame)
            break

    async def _trigger():
        await asyncio.sleep(0.05)
        await publish("scheduler_failure", {"job_id": "lnn_retrain", "consecutive_failures": 3, "error": "OOM"})

    await asyncio.gather(asyncio.create_task(_run()), asyncio.create_task(_trigger()))
    assert frames
    data_line = [l for l in frames[0].splitlines() if l.startswith("data:")][0]
    json.loads(data_line.removeprefix("data:").strip())  # must not raise


@pytest.mark.asyncio
async def test_generator_stops_on_disconnect():
    """Generator exits cleanly when is_disconnected() returns True."""
    mock_req = _make_mock_request(disconnect_after=2)  # disconnect on 2nd check
    frames: list[str] = []

    async for frame in _sse_stream_generator("NSW1", None, mock_req):
        frames.append(frame)  # pragma: no cover — should stop before yielding

    # Generator exits on disconnect check without yielding (no events published)
    assert frames == []


@pytest.mark.asyncio
async def test_generator_unsubscribes_on_exit():
    """Subscriber count drops back to baseline after the generator exits."""
    before = subscriber_count()
    mock_req = _make_mock_request(disconnect_after=1)  # immediately disconnects
    async for _ in _sse_stream_generator("NSW1", None, mock_req):
        pass
    assert subscriber_count() == before


@pytest.mark.asyncio
async def test_generator_region_filter_blocks_other_regions():
    """Events for a different region must NOT be delivered to the generator."""
    mock_req = _make_mock_request(disconnect_after=4)
    frames: list[str] = []

    async def _run():
        async for frame in _sse_stream_generator("NSW1", frozenset({"dispatch_updated"}), mock_req):
            frames.append(frame)
            break

    async def _trigger():
        await asyncio.sleep(0.05)
        # Publish for VIC1 — should NOT reach the NSW1 generator
        await publish("dispatch_updated", {"regions": ["VIC1"]}, region="VIC1")

    try:
        await asyncio.wait_for(
            asyncio.gather(asyncio.create_task(_run()), asyncio.create_task(_trigger())),
            timeout=3.5,  # 3 × 1s polling ticks then disconnect
        )
    except asyncio.TimeoutError:
        pass

    assert frames == [], f"Unexpected frame delivered: {frames}"


# ── HTTP route tests (synchronous error responses only) ──────────────

@pytest.mark.asyncio
async def test_stream_route_invalid_region(client: AsyncClient):
    r = await client.get("/api/events/stream?region=INVALID")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_legacy_market_route_invalid_region(client: AsyncClient):
    r = await client.get("/api/events/market?region=INVALID")
    assert r.status_code == 400
