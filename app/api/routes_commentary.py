"""Commentary routes — auto-generated market event feed.

GET /commentary/recent?region=NSW1&limit=20&min_severity=MEDIUM
  Returns recent commentary events for the Live Feed panel.
  No auth required (same policy as /market/state).

GET /commentary/{id}
  Full commentary event with all evidence refs. Used by 'Full analysis' click.

GET /commentary/stats?region=NSW1&hours=24
  Event count by type and severity — drives the unread badge counter.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/commentary", tags=["commentary"])

_SUPPORTED_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


@router.get("/recent")
async def get_recent_commentary(
    region: str = Query(default="NSW1", description="NEM region code"),
    limit: int = Query(default=20, ge=1, le=100),
    min_severity: str | None = Query(default=None, description="Minimum severity filter (LOW/MEDIUM/HIGH/CRITICAL)"),
):
    """Return recent commentary events for the Live Feed panel."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {sorted(_SUPPORTED_REGIONS)}",
        )
    if min_severity and min_severity.upper() not in _VALID_SEVERITIES:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity '{min_severity}'. Valid: {sorted(_VALID_SEVERITIES)}",
        )

    from app.engines.commentary.store import search_recent
    events = await search_recent(
        region=region,
        limit=limit,
        min_severity=min_severity.upper() if min_severity else None,
    )
    return {"region": region, "events": events, "count": len(events)}


@router.get("/stats")
async def get_commentary_stats(
    region: str = Query(default="NSW1", description="NEM region code"),
    hours: int = Query(default=24, ge=1, le=168),
):
    """Event count by type and severity over the last N hours."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'",
        )

    from app.engines.commentary.store import get_stats
    return await get_stats(region=region, hours=hours)


@router.get("/{event_id}")
async def get_commentary_event(event_id: str):
    """Full commentary event detail — used by 'Full analysis' button."""
    from app.engines.commentary.store import get_event
    evt = await get_event(event_id)
    if evt is None:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Commentary event not found")
    return evt
