"""Market Incident Timeline route.

GET /api/incidents/timeline
  ?region=NSW1
  ?interval=2026-05-25T14:05:00Z   (ISO 8601; defaults to latest dispatch interval)
  ?lookback_minutes=30              (1–120; default 30)

Returns a chronological incident report showing what changed in the 30 minutes
before a price event, with evidence tier and evidence_ref for each driver.

The timeline answers:
  "What changed around this price move, in what order,
   and what evidence supports each driver?"
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/incidents", tags=["incidents"])

_SUPPORTED_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}


@router.get("/timeline")
async def get_incident_timeline(
    region: str = Query(default="NSW1"),
    interval: str = Query(
        default="",
        description="ISO 8601 anchor interval (e.g. 2026-05-25T14:05:00Z). "
                    "Omit to use latest available dispatch interval.",
    ),
    lookback_minutes: int = Query(default=30, ge=1, le=120),
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a Market Incident Timeline for a region + anchor interval.

    The timeline reconstructs the causal sequence of events:
      price spike → headroom → constraints → interconnectors →
      unit dispatch → rebids → weather demand pressure

    Each event carries a causality tier (confirmed / supported / plausible /
    unconfirmed) and evidence_ref_ids linking back to the source documents.

    A coverage_grade (full / partial / minimal) summarises how much of the
    causal chain is backed by ingested data.
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {sorted(_SUPPORTED_REGIONS)}",
        )

    # Resolve anchor interval
    anchor_time = _resolve_anchor(interval, region, db)

    # Build timeline
    from app.engines.incident_timeline import build_incident_timeline, timeline_to_dict
    tl = await build_incident_timeline(
        session=db,
        region=region,
        anchor_time=anchor_time,
        lookback_minutes=lookback_minutes,
    )

    return timeline_to_dict(tl)


@router.get("/brief/{region}")
async def get_incident_brief(
    region: str,
    anchor: str = Query(
        default="",
        description="ISO 8601 anchor time (e.g. 2026-05-25T14:05:00Z). "
                    "Omit to use current time.",
    ),
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a structured Market Incident Brief for a NEM region.

    Assembles all available evidence into a single situational report:
      - Current market state (price, demand, headroom, regime)
      - Driver evidence (constraints, interconnectors, unit events)
      - Active AEMO notices
      - Historical analog events
      - Forecast model state (LEAR / QRA / LNN bands)
      - BESS portfolio implication (charge / dispatch / standby)
      - Source freshness across all layers
      - Claim verification (internal consistency checks)
      - Model provenance (version + training_data_ref)
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {sorted(_SUPPORTED_REGIONS)}",
        )

    anchor_time = _resolve_anchor(anchor, region, db)

    from app.engines.incident_brief import build_incident_brief
    return await build_incident_brief(session=db, region=region, anchor_time=anchor_time)


def _resolve_anchor(interval_str: str, region: str, db) -> datetime:
    """Parse an ISO interval string, or fall back to now UTC."""
    if interval_str:
        try:
            dt = datetime.fromisoformat(interval_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interval format '{interval_str}'. Use ISO 8601, e.g. 2026-05-25T14:05:00Z",
            )
    return datetime.now(timezone.utc)
