"""Rebid detection route — compare BIDDAYOFFER vs BIDPEROFFER to detect
strategic availability withdrawal ahead of price spikes.

A rebid is identified when:
  - A BIDPEROFFER row has max_avail_mw significantly lower than the BIDDAYOFFER
    for the same DUID, bid_type, and settlement_date
  - The period_id corresponds to an interval where spot price was elevated
  - The change was submitted after the day-ahead offer (offer_date delta > 0)

GET /api/market/rebids
  ?region=NSW1
  &date=2024-01-15        (ISO date, defaults to latest settlement date with data)
  &threshold_mw=50        (minimum MW reduction to flag as rebid, default 50)
  &min_price=100          (only flag if corresponding spot price ≥ this, default 100)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db
from app.engines.rebid_engine import RebidEvent, detect_rebids

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"])

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_PERIOD_MINUTES = 30   # BIDPEROFFER is 30-min trading periods


@router.get("/rebids")
async def get_rebids(
    region: str = Query(default="NSW1"),
    date: str = Query(default="", description="ISO date YYYY-MM-DD; defaults to latest data"),
    threshold_mw: float = Query(default=50.0, ge=0),
    min_price: float = Query(default=100.0, ge=0),
    user: TokenPayload = Depends(get_current_user),
):
    """Detect strategic rebids for a settlement date.

    Returns a list of rebid events where generators withdrew availability
    intraday, with the corresponding spot price context and fuel type.
    When no bid data has been ingested, returns an empty list with a
    data-availability note (does not 500).
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'",
        )

    settlement_dt = _parse_date(date)

    try:
        from app.db.session import db_session
        async with db_session() as session:
            events = await detect_rebids(session, region, settlement_dt, threshold_mw, min_price)
            if settlement_dt is None and not events:
                return _empty_response(region, None, "no_bid_data")
            effective_dt = settlement_dt or (
                events[0].period_start_utc if events else None
            )
            return _format_response(region, effective_dt, threshold_mw, min_price, events)
    except Exception as exc:
        logger.warning("Rebid detection failed: %s", exc)
        return _empty_response(region, settlement_dt, str(exc))


def _format_response(
    region: str,
    settlement_dt: datetime | None,
    threshold_mw: float,
    min_price: float,
    events: list[RebidEvent],
) -> dict[str, Any]:
    rebids = [
        {
            "duid": e.duid,
            "station_name": e.station_name,
            "fuel_type": e.fuel_type,
            "period_id": e.period_id,
            "period_start_utc": e.period_start_utc.isoformat() if e.period_start_utc else None,
            "day_avail_mw": e.day_avail_mw,
            "intra_avail_mw": e.intra_avail_mw,
            "withdrawal_mw": e.withdrawal_mw,
            "cheap_band_withdrawn_mw": e.cheap_band_withdrawn_mw,
            "spot_price_context": e.spot_price_context,
            "offer_delta_minutes": e.offer_delta_minutes,
            "rebid_flag": e.rebid_flag,
            "severity": e.severity,
        }
        for e in events[:100]
    ]
    return {
        "region": region,
        "settlement_date": settlement_dt.date().isoformat() if settlement_dt else None,
        "threshold_mw": threshold_mw,
        "min_price_filter": min_price,
        "rebid_count": len(events),
        "rebids": rebids,
        "data_available": True,
        "summary": _summarise(events),
        "caveat": (
            "Rebid detection compares BIDDAYOFFER vs BIDPEROFFER availability. "
            "A withdrawal is flagged when intraday max_avail_mw is reduced by ≥ threshold_mw. "
            "This is observed behaviour — motivation (forced outage vs strategic) requires "
            "separate constraint and SCADA evidence."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _summarise(events: list[RebidEvent]) -> dict[str, Any]:
    if not events:
        return {"total_withdrawn_mw": 0.0, "strategic_count": 0, "by_fuel": {}}
    by_fuel: dict[str, float] = {}
    strategic = 0
    for e in events:
        fuel = e.fuel_type or "other"
        by_fuel[fuel] = by_fuel.get(fuel, 0.0) + e.withdrawal_mw
        if e.rebid_flag == "strategic":
            strategic += 1
    return {
        "total_withdrawn_mw": round(sum(e.withdrawal_mw for e in events), 1),
        "strategic_count": strategic,
        "by_fuel": {k: round(v, 1) for k, v in sorted(by_fuel.items(), key=lambda x: -x[1])},
    }


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        from datetime import date
        d = date.fromisoformat(date_str)
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    except ValueError:
        return None


@router.get("/participants/{duid}/profile")
async def get_participant_profile(
    duid: str,
    region: str | None = Query(default=None, description="NEM region filter (e.g. NSW1)"),
    window_days: int = Query(default=30, ge=7, le=365, description="Rolling lookback window (days)"),
    threshold_mw: float = Query(default=50.0, ge=0, description="Minimum MW withdrawal to count as a rebid"),
    min_price: float = Query(default=100.0, ge=0, description="Only count rebids when spot price ≥ this ($/MWh)"),
    user: TokenPayload = Depends(get_current_user),
    session=Depends(get_db),
) -> dict[str, Any]:
    """Return a behavioral rebid profile for a market participant (DUID).

    Aggregates BIDDAYOFFER vs BIDPEROFFER availability divergence over a rolling
    window to classify the participant as habitual, occasional, or low_activity.

    A "habitual" participant has ≥5 qualifying rebid events with ≥40 % of those
    withdrawing cheap-band capacity. "occasional" requires ≥2 events. Returns
    data_available=False and behavioral_tier="low_activity" when no bid data
    has been ingested for this DUID.
    """
    from app.engines.participant_profiler import profile_participant

    if region:
        region = region.upper()
        if region not in _SUPPORTED_REGIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
            )

    try:
        profile = await profile_participant(
            session, duid.upper(), region=region,
            window_days=window_days,
            threshold_mw=threshold_mw,
            min_price=min_price,
        )
        return {
            "duid": profile.duid,
            "station_name": profile.station_name,
            "participant": profile.participant,
            "fuel_type": profile.fuel_type,
            "region": profile.region,
            "window_days": profile.window_days,
            "behavioral_tier": profile.behavioral_tier,
            "rebid_count": profile.rebid_count,
            "days_with_rebids": profile.days_with_rebids,
            "avg_mw_withdrawn": profile.avg_mw_withdrawn,
            "max_mw_withdrawn": profile.max_mw_withdrawn,
            "strategic_fraction": profile.strategic_fraction,
            "spike_correlation": profile.spike_correlation,
            "data_available": profile.data_available,
            "caveat": profile.caveat,
        }
    except Exception as exc:
        logger.warning("Participant profile failed for %s: %s", duid, exc)
        return {
            "duid": duid.upper(),
            "behavioral_tier": "low_activity",
            "data_available": False,
            "note": "Profile unavailable — bid data may not be ingested yet.",
        }


def _empty_response(region: str, settlement_dt: datetime | None, reason: str) -> dict[str, Any]:
    return {
        "region": region,
        "settlement_date": settlement_dt.date().isoformat() if settlement_dt else None,
        "rebid_count": 0,
        "rebids": [],
        "data_available": False,
        "note": (
            "No bid offer data ingested yet. "
            "Trigger BIDDAYOFFER/BIDPEROFFER backfill to populate rebid detection. "
            f"(reason: {reason})"
        ),
        "caveat": "Rebid detection requires BIDDAYOFFER and BIDPEROFFER tables to be populated.",
    }
