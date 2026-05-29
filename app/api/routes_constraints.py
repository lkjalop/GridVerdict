"""Constraint and interconnector timeline routes.

GET /api/market/constraints
  ?region=NSW1&hours=4&driver_type=constraint
  Returns binding constraints (DISPATCHCONSTRAINT) and interconnector flows
  (DISPATCHINTERCONNECTORRES) for a time window, formatted for a swimlane
  timeline chart.

  driver_type filter: "constraint" | "interconnector" | "all" (default: "all")

GET /api/market/constraints/elements
  Returns the set of constraint IDs and interconnector IDs that have been
  binding in the last N days — for the UI filter dropdown.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.auth import TokenPayload
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"])

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_INTERCONNECTORS = {
    "V-SA", "V-S-MNSP1", "NSW1-QLD1", "VIC1-NSW1",
    "T-V-MNSP1", "BASSLINK", "MURRAYLINK",
}


@router.get("/constraints")
async def get_constraints(
    region: str = Query(default="NSW1"),
    hours: int = Query(default=4, ge=1, le=48),
    driver_type: str = Query(default="all", description="constraint | interconnector | all"),
    element_id: str = Query(default="", description="Filter to specific constraint/interconnector ID"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return binding constraint and interconnector events for a time window.

    Each event in the timeline has:
      - element_id: constraint set ID or interconnector ID
      - driver_type: 'constraint' or 'interconnector'
      - valid_time: interval timestamp
      - marginal_value: $/MWh shadow price (constraint) or 0 (interconnector)
      - mw_flow / export_limit / import_limit: interconnector fields
      - binding: True if marginal_value > 0 (constraint) or near-limit (interconnector)
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'",
        )

    driver_type = driver_type.lower()
    if driver_type not in ("constraint", "interconnector", "all"):
        raise HTTPException(status_code=400, detail="driver_type must be 'constraint', 'interconnector', or 'all'")

    try:
        from sqlalchemy import text
        from app.db.session import db_session

        async with db_session() as session:
            return await _query_timeline(session, region, hours, driver_type, element_id.strip())
    except Exception as exc:
        logger.warning("Constraint timeline query failed: %s", exc)
        return _empty_response(region, hours, "query_failed")


@router.get("/constraints/elements")
async def get_constraint_elements(
    region: str = Query(default="NSW1"),
    days: int = Query(default=7, ge=1, le=90),
    user: TokenPayload = Depends(get_current_user),
):
    """Return active constraint IDs and interconnector IDs for the filter dropdown."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported region '{region}'")

    try:
        from sqlalchemy import text
        from app.db.session import db_session

        async with db_session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            res = await session.execute(text("""
                SELECT DISTINCT element_id, driver_type,
                       COUNT(*) AS event_count
                FROM market_driver_events
                WHERE region = :region
                  AND valid_time >= :cutoff
                GROUP BY element_id, driver_type
                ORDER BY event_count DESC
                LIMIT 200
            """), {"region": region, "cutoff": cutoff})
            rows = res.fetchall()

        return {
            "region": region,
            "days": days,
            "elements": [
                {"element_id": r[0], "driver_type": r[1], "event_count": r[2]}
                for r in rows
            ],
        }
    except Exception as exc:
        logger.debug("Constraint elements query failed: %s", exc)
        return {"region": region, "days": days, "elements": []}


# ── Internal ──────────────────────────────────────────────────────────────────

async def _query_timeline(
    session: Any,
    region: str,
    hours: int,
    driver_type: str,
    element_id: str,
) -> dict[str, Any]:
    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    type_filter = ""
    if driver_type == "constraint":
        type_filter = "AND driver_type = 'constraint'"
    elif driver_type == "interconnector":
        type_filter = "AND driver_type = 'interconnector'"

    elem_filter = ""
    params: dict = {"region": region, "cutoff": cutoff}
    if element_id:
        elem_filter = "AND element_id = :element_id"
        params["element_id"] = element_id

    res = await session.execute(text(f"""
        SELECT element_id, driver_type, valid_time, values, source
        FROM market_driver_events
        WHERE region = :region
          AND valid_time >= :cutoff
          {type_filter}
          {elem_filter}
        ORDER BY valid_time DESC
        LIMIT 2000
    """), params)
    rows = res.fetchall()

    if not rows:
        return _empty_response(region, hours, "no_data")

    events = []
    for element_id_r, dtype, vt, vals, source in rows:
        if not isinstance(vt, datetime):
            vt = datetime.fromisoformat(str(vt))
        vals = vals or {}

        if dtype == "constraint":
            mv = float(vals.get("marginal_value", 0) or 0)
            binding = mv > 0
            event = {
                "element_id": element_id_r,
                "driver_type": "constraint",
                "valid_time": vt.isoformat(),
                "marginal_value": round(mv, 2),
                "violation_degree": float(vals.get("violation_degree", 0) or 0),
                "binding": binding,
                "source": source,
            }
        else:  # interconnector
            flow = float(vals.get("mw_flow", 0) or 0)
            export_lim = float(vals.get("export_limit", 0) or 0)
            import_lim = float(vals.get("import_limit", 0) or 0)
            congested = _is_congested(flow, export_lim, import_lim)
            event = {
                "element_id": element_id_r,
                "driver_type": "interconnector",
                "valid_time": vt.isoformat(),
                "mw_flow": round(flow, 1),
                "export_limit": round(export_lim, 1),
                "import_limit": round(import_lim, 1),
                "export_headroom_mw": round(export_lim - flow, 1) if export_lim > 0 else None,
                "congested": congested,
                "binding": congested,
                "source": source,
            }
        events.append(event)

    # Build timeline summary: per element, count binding intervals
    summary = _summarise_timeline(events)

    return {
        "region": region,
        "hours": hours,
        "driver_type_filter": driver_type,
        "element_filter": element_id or None,
        "event_count": len(events),
        "events": events,
        "summary": summary,
        "data_available": True,
        "caveat": (
            "Binding constraint marginal values indicate dispatch binding pressure. "
            "Interconnector congestion is inferred from flow vs limit proximity. "
            "These are observed signals — causal attribution requires market notice cross-reference."
        ),
    }


def _is_congested(flow: float, export_lim: float, import_lim: float, threshold: float = 0.90) -> bool:
    if export_lim > 0 and flow / export_lim >= threshold:
        return True
    if import_lim < 0 and flow != 0:
        return abs(flow / import_lim) >= threshold
    return False


def _summarise_timeline(events: list[dict]) -> list[dict]:
    by_element: dict[str, dict] = {}
    for e in events:
        eid = e["element_id"]
        if eid not in by_element:
            by_element[eid] = {
                "element_id": eid,
                "driver_type": e["driver_type"],
                "intervals": 0,
                "binding_intervals": 0,
                "max_marginal_value": 0.0,
                "max_flow_mw": 0.0,
            }
        s = by_element[eid]
        s["intervals"] += 1
        if e.get("binding"):
            s["binding_intervals"] += 1
        if e["driver_type"] == "constraint":
            s["max_marginal_value"] = max(s["max_marginal_value"], e.get("marginal_value", 0))
        else:
            s["max_flow_mw"] = max(s["max_flow_mw"], abs(e.get("mw_flow", 0)))

    return sorted(
        by_element.values(),
        key=lambda x: (x["binding_intervals"], x["max_marginal_value"]),
        reverse=True,
    )


def _empty_response(region: str, hours: int, reason: str) -> dict[str, Any]:
    reason_text = {
        "query_failed": "constraint query failed",
        "no_data": "no data in this window",
    }.get(reason, reason)
    return {
        "region": region,
        "hours": hours,
        "event_count": 0,
        "events": [],
        "summary": [],
        "data_available": False,
        "note": (
            "No constraint or interconnector data found for this window. "
            "Archive backfill populates DISPATCHCONSTRAINT and DISPATCHINTERCONNECTORRES. "
            f"(reason: {reason_text})"
        ),
    }
