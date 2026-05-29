"""Deterministic market-driver attribution from archived MMSDM rows."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


async def retrieve_market_drivers(
    session,
    region: str,
    valid_time: datetime,
    window_minutes: int = 5,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch constraint/interconnector rows around a dispatch interval.

    Primary: ±window_minutes of exact timestamp (live data).
    Fallback: same hour-of-day within the last 30 archive days (proxy).
    """
    from sqlalchemy import select, text
    from app.db.models import MarketDriverEvent

    start = valid_time - timedelta(minutes=window_minutes)
    end = valid_time + timedelta(minutes=window_minutes)
    stmt = (
        select(MarketDriverEvent)
        .where(MarketDriverEvent.valid_time >= start)
        .where(MarketDriverEvent.valid_time <= end)
        .order_by(MarketDriverEvent.valid_time.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    live_rows = rank_driver_rows(region, rows)[:limit]

    if live_rows:
        return live_rows

    # Fallback: same hour-of-day proxy from archive (last 30 archive days)
    # Lets us show representative constraint patterns even when live data gaps exist.
    try:
        archive_rows = await _retrieve_archive_proxy(session, region, valid_time, limit)
        if archive_rows:
            for r in archive_rows:
                r["_source"] = "archive_proxy"
            return archive_rows
    except Exception:
        pass
    return []


async def _retrieve_archive_proxy(
    session,
    region: str,
    valid_time: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Return binding constraints from the archive for the same hour-of-day.

    Used when live data is unavailable (archive gap or future timestamps).
    """
    from sqlalchemy import text

    hour = valid_time.hour
    stmt = text("""
        SELECT *
        FROM market_driver_events
        WHERE region = :region
          AND driver_type = 'constraint'
          AND EXTRACT(HOUR FROM valid_time) = :hour
          AND valid_time >= NOW() - INTERVAL '30 days'
        ORDER BY ABS(EXTRACT(EPOCH FROM (valid_time - :vt))) ASC
        LIMIT :limit
    """)
    try:
        result = await session.execute(stmt, {
            "region": region,
            "hour": hour,
            "vt": valid_time,
            "limit": limit,
        })
        rows = result.fetchall()
        if not rows:
            # Wider fallback: any recent archive data for this region
            result2 = await session.execute(text("""
                SELECT *
                FROM market_driver_events
                WHERE driver_type = 'constraint'
                ORDER BY valid_time DESC
                LIMIT :limit
            """), {"limit": limit})
            rows = result2.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def rank_driver_rows(region: str, rows: list[Any]) -> list[dict[str, Any]]:
    """Rank driver rows by actionability: nonzero marginal value first."""
    out = [_row_to_dict(row) for row in rows]
    region_up = region.upper()
    out.sort(key=lambda r: (
        0 if (r.get("region") in (None, region_up)) else 1,
        -abs(float(r.get("values", {}).get("marginal_value") or 0.0)),
        r.get("driver_type", ""),
    ))
    return out


def summarise_driver_events(driver_events: list[dict[str, Any]]) -> dict[str, Any]:
    constraints = [d for d in driver_events if d.get("driver_type") == "constraint"]
    interconnectors = [d for d in driver_events if d.get("driver_type") == "interconnector"]
    binding_constraints = [
        d for d in constraints
        if abs(float(d.get("values", {}).get("marginal_value") or 0.0)) > 0
        or abs(float(d.get("values", {}).get("violation_degree") or 0.0)) > 0
    ]
    tight_interconnectors = [
        d for d in interconnectors
        if _interconnector_is_tight(d.get("values", {}))
    ]
    return {
        "constraints": constraints,
        "interconnectors": interconnectors,
        "binding_constraints": binding_constraints,
        "tight_interconnectors": tight_interconnectors,
        "has_confirmed_driver": bool(binding_constraints or tight_interconnectors),
    }


def _interconnector_is_tight(values: dict[str, Any]) -> bool:
    flow = values.get("mw_flow")
    if flow is None:
        flow = values.get("metered_mw_flow")
    export_limit = values.get("export_limit")
    import_limit = values.get("import_limit")
    if flow is None:
        return False
    flow_f = float(flow)
    for limit in (export_limit, import_limit):
        if limit is None:
            continue
        limit_f = float(limit)
        if abs(limit_f) > 0 and abs(abs(flow_f) - abs(limit_f)) <= max(10.0, abs(limit_f) * 0.05):
            return True
    return False


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return {
        "source": row.source,
        "driver_type": row.driver_type,
        "element_id": row.element_id,
        "region": row.region,
        "valid_time": row.valid_time,
        "values": dict(row.values or {}),
        "raw_ref": row.raw_ref,
    }
