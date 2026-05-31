"""Historical price distribution retrieval for benchmark-style queries.

Handles questions like:
  "is this price high compared to last year?"
  "what's the normal NSW price at this time?"
  "compare current to historical distribution"

Returns percentiles (median, P75, P90) for the matching hour/season window
from the dispatch archive (market_events table, source=AEMO_DISPATCH_PRICE).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Period → lookback window in days
_PERIOD_DAYS: dict[str, int] = {
    "last_year": 365,
    "last_quarter": 90,
    "last_month": 30,
    "last_week": 7,
    "historical": 1460,   # 4 years — covers full MMSDM archive
    "all_time": 1460,     # explicit alias for full-archive queries
    "multi_year": 730,    # 2 years for "how have prices changed recently"
}

_DEFAULT_DAYS = 365


async def get_historical_price_distribution(
    db: AsyncSession,
    region: str,
    anchor_time: datetime,
    period: str = "last_year",
    hour_window: int = 2,       # ±hours around anchor hour for same-time-of-day comparison
    include_season: bool = True,  # restrict to same calendar quarter
) -> dict[str, Any]:
    """Compute price distribution for the same time-of-day window over the lookback period.

    Returns a dict with:
      - median, p25, p75, p90: price percentiles ($/MWh)
      - count: number of intervals
      - anchor_price: current price for comparison
      - classification: "cheap" / "normal" / "elevated" / "high" / "spike"
      - period_label: human-readable period description
      - hour_window: hours used for same-time-of-day match
    """
    if anchor_time.tzinfo is None:
        anchor_time = anchor_time.replace(tzinfo=timezone.utc)

    lookback_days = _PERIOD_DAYS.get(period, _DEFAULT_DAYS)
    cutoff = anchor_time - timedelta(days=lookback_days)
    anchor_hour = anchor_time.hour
    hour_low = (anchor_hour - hour_window) % 24
    hour_high = (anchor_hour + hour_window) % 24

    period_label = {
        "last_year": "last 12 months",
        "last_quarter": "last 90 days",
        "last_month": "last 30 days",
        "last_week": "last 7 days",
        "historical": "available history (4 years)",
        "all_time": "available history (4 years)",
        "multi_year": "last 2 years",
    }.get(period, "historical")

    try:
        # Build hour-of-day filter (handles midnight wrap-around)
        if hour_low <= hour_high:
            hour_filter = "AND EXTRACT(HOUR FROM valid_time AT TIME ZONE 'UTC') BETWEEN :h_low AND :h_high"
        else:
            hour_filter = "AND (EXTRACT(HOUR FROM valid_time AT TIME ZONE 'UTC') >= :h_low OR EXTRACT(HOUR FROM valid_time AT TIME ZONE 'UTC') <= :h_high)"

        # Season filter: same calendar quarter (optional)
        season_filter = ""
        if include_season:
            anchor_quarter = (anchor_time.month - 1) // 3 + 1
            season_filter = "AND EXTRACT(QUARTER FROM valid_time AT TIME ZONE 'UTC') = :quarter"

        params: dict[str, Any] = {
            "region": region.upper(),
            "cutoff": cutoff,
            "h_low": hour_low,
            "h_high": hour_high,
        }
        if include_season:
            params["quarter"] = anchor_quarter

        sql = text(f"""
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price_rrp) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price_rrp) AS median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_rrp) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY price_rrp) AS p90,
                AVG(price_rrp) AS mean,
                COUNT(*) AS cnt
            FROM market_events
            WHERE source = 'AEMO_DISPATCH_PRICE'
              AND region = :region
              AND valid_time >= :cutoff
              AND valid_time < NOW()
              {hour_filter}
              {season_filter}
        """)

        row = (await db.execute(sql, params)).fetchone()
        if not row or not row.cnt or row.cnt < 5:
            return _empty_result(period_label, hour_window)

        return {
            "median": round(float(row.median or 0), 2),
            "p25": round(float(row.p25 or 0), 2),
            "p75": round(float(row.p75 or 0), 2),
            "p90": round(float(row.p90 or 0), 2),
            "mean": round(float(row.mean or 0), 2),
            "count": int(row.cnt),
            "period_label": period_label,
            "hour_window": hour_window,
            "lookback_days": lookback_days,
            "available": True,
        }

    except Exception as exc:
        logger.debug("Historical price distribution unavailable: %s", exc)
        return _empty_result(period_label, hour_window)


def _empty_result(period_label: str, hour_window: int) -> dict[str, Any]:
    return {
        "median": None, "p25": None, "p75": None, "p90": None, "mean": None,
        "count": 0, "period_label": period_label, "hour_window": hour_window,
        "lookback_days": 0, "available": False,
    }


def classify_vs_history(current_price: float, dist: dict[str, Any]) -> str:
    """Classify current price relative to historical distribution.

    Returns one of: cheap / normal / elevated / high / spike
    """
    if not dist.get("available") or dist.get("median") is None:
        return "unknown"
    p25 = dist["p25"]
    median = dist["median"]
    p75 = dist["p75"]
    p90 = dist["p90"]
    if current_price <= p25:
        return "cheap"
    if current_price <= median:
        return "normal"
    if current_price <= p75:
        return "elevated"
    if current_price <= p90:
        return "high"
    return "spike"
