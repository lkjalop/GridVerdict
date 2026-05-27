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
    """Fetch constraint/interconnector rows around a dispatch interval."""
    from sqlalchemy import select
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
    return rank_driver_rows(region, rows)[:limit]


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
