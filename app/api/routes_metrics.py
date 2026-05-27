"""GET /api/metrics — Prometheus scrape endpoint.

Updates live gauges (dispatch age, SSE subscribers) immediately before
generating the response so scrapers see current values without a push
mechanism.

No authentication required — metrics contain no PII and are typically
scraped from within a private network.
"""
from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus-format metrics scrape endpoint (text/plain; version=0.0.4)."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from app.api.metrics_registry import last_dispatch_age_seconds, sse_subscribers

    # Refresh gauges that are cheaper to pull than push
    try:
        from app.data.cache import get_cache
        age = await get_cache().age_seconds("dispatch_snapshot")
        if age is not None:
            last_dispatch_age_seconds.set(age)
    except Exception:
        pass

    try:
        from app.data.event_bus import subscriber_count
        sse_subscribers.set(subscriber_count())
    except Exception:
        pass

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
