"""SSE endpoints — live market and system event streams.

GET /events/market?region=NSW1  (legacy)
  Polling-based market_update every 30s. Kept for compatibility.

GET /events/stream?region=NSW1&types=dispatch_updated,...
  Event-driven SSE — pushes events from the in-process event bus as they occur.
  Supported event types:
    dispatch_updated      — new dispatch interval ingested
    forecast_updated      — predispatch or LNN forecast refreshed
    data_status_changed   — calibration or backfill job completed
    scheduler_failure     — a scheduler job has failed 3+ consecutive times
    source_stale          — a market data source has gone stale
    trace_written         — a new audit trace was persisted
    incident_timeline_updated — incident timeline data has changed
  types defaults to all of the above when omitted.
  Heartbeat comment (': heartbeat') every 30s when no events arrive.

Auth: same bearer token as every other route, accepted via Authorization header
or `token` query param (EventSource cannot set custom headers).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import timezone, datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import AEMOClient, Cache
from app.api.auth import decode_token
from app.data.aemo_live_client import AEMOLiveClient
from app.data.cache import MarketCache
from domain.nem.adapter import classify_regime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["events"])

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_POLL_INTERVAL_S = 30
_CACHE_KEY = "dispatch_snapshot"
_HEARTBEAT_INTERVAL_S = 30       # total seconds between heartbeat SSE comments
_DISCONNECT_POLL_S = 1.0          # how often to check for client disconnect

_ALL_EVENT_TYPES = frozenset({
    "dispatch_updated",
    "forecast_updated",
    "data_status_changed",
    "scheduler_failure",
    "source_stale",
    "trace_written",
    "incident_timeline_updated",
    "commentary_created",   # Sprint Q: rolling market commentary
    "spike_alert",          # Proactive spike detection — frontend injects into chat
    "spike_resolved",       # Price normalised after spike
})


def _check_auth(request: Request, token: str | None) -> None:
    """Validate bearer token from header or query param. Raises HTTPException on failure."""
    from config.settings import get_settings
    settings = get_settings()
    if settings.gridverdict_dev_no_auth:
        return
    auth_header = request.headers.get("Authorization", "")
    raw_token = token or (
        auth_header.removeprefix("Bearer ").strip()
        if auth_header.startswith("Bearer ") else None
    )
    if not raw_token:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    decode_token(raw_token)


async def _sse_stream_generator(
    region: str,
    types: frozenset[str],
    request: Request,
):
    """Yield SSE frames from the event bus until the client disconnects.

    Uses a 1-second internal polling interval so disconnect is detected quickly
    (within ~1s) instead of waiting for the full heartbeat interval.
    A heartbeat comment is sent every _HEARTBEAT_INTERVAL_S seconds when idle.
    """
    from app.data.event_bus import subscribe, unsubscribe

    queue = subscribe(region=region, types=types)
    ticks_until_heartbeat = _HEARTBEAT_INTERVAL_S  # count down disconnect-poll ticks
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_DISCONNECT_POLL_S)
                data = {**event.payload, "ts": event.ts}
                if event.region:
                    data["region"] = event.region
                yield f"event: {event.type}\ndata: {json.dumps(data)}\n\n"
                ticks_until_heartbeat = _HEARTBEAT_INTERVAL_S
            except asyncio.TimeoutError:
                ticks_until_heartbeat -= 1
                if ticks_until_heartbeat <= 0:
                    yield ": heartbeat\n\n"
                    ticks_until_heartbeat = _HEARTBEAT_INTERVAL_S
    finally:
        unsubscribe(queue)


async def _legacy_market_generator(
    region: str,
    client: AEMOLiveClient,
    cache: MarketCache,
    request: Request,
):
    """Legacy polling generator — emits market_update every 30s."""
    while True:
        if await request.is_disconnected():
            break
        try:
            raw = await cache.get(_CACHE_KEY)
            if raw is None:
                snapshot = await client.fetch_latest_snapshot()
                await cache.set(_CACHE_KEY, snapshot.to_dict())
            else:
                from app.data.aemo_live_client import LiveMarketSnapshot
                snapshot = LiveMarketSnapshot.from_dict(raw)

            dp = snapshot.get(region)
            if dp is not None:
                headroom = max(dp.availability_mw - dp.demand_mw, 0.0)
                regime = classify_regime(dp.price_rrp, region)
                staleness = snapshot.staleness_seconds(region)
                payload = {
                    "region": region,
                    "price_rrp": round(dp.price_rrp, 2),
                    "demand_mw": round(dp.demand_mw, 1),
                    "availability_mw": round(dp.availability_mw, 1),
                    "headroom_mw": round(headroom, 1),
                    "regime": regime,
                    "staleness_seconds": staleness,
                    "valid_time": dp.valid_time.isoformat(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                yield f"event: market_update\ndata: {json.dumps(payload)}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'detail': f'No data for {region}'})}\n\n"
        except Exception as exc:
            logger.warning("SSE market_update error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        await asyncio.sleep(_POLL_INTERVAL_S)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@router.get("/stream")
async def stream_events(
    region: str = Query(default="NSW1", description="NEM region code"),
    types: str | None = Query(default=None, description="Comma-separated event types (omit for all)"),
    token: str | None = Query(default=None, description="Bearer token for EventSource clients"),
    request: Request = None,
):
    """Stream live system events as Server-Sent Events (event-driven, multi-type)."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )

    _check_auth(request, token)

    requested = _ALL_EVENT_TYPES
    if types:
        requested = frozenset(t.strip() for t in types.split(",") if t.strip()) & _ALL_EVENT_TYPES

    return StreamingResponse(
        _sse_stream_generator(region, requested, request),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/market")
async def stream_market(
    region: str = Query(default="NSW1", description="NEM region code"),
    token: str | None = Query(default=None, description="Bearer token (for EventSource clients)"),
    request: Request = None,
    client: AEMOLiveClient = AEMOClient,
    cache: MarketCache = Cache,
):
    """Legacy polling SSE — streams market_update every 30s. Kept for compatibility."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )

    _check_auth(request, token)

    return StreamingResponse(
        _legacy_market_generator(region, client, cache, request),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
