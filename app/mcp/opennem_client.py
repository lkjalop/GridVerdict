"""OpenNEM/OpenElectricity API client — public endpoints, no auth required.

Provides real monthly price trends, renewable proportions, and diurnal
price patterns for NEM regions. Used by _plan_trend_analysis and
_plan_diurnal_analysis to replace seasonal norms with actual market data.

Public API base: https://api.opennem.org.au/v4/market/network/NEM
Docs: https://api.opennem.org.au/openapi.json (OpenElectricity v4.5)

Available public metrics: price, energy, demand, renewable_proportion,
generation_renewable_energy, renewable_with_storage_proportion
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.openelectricity.org.au/v4/market/network/NEM"
_TIMEOUT = httpx.Timeout(8.0)
_HEADERS = {"User-Agent": "GridVerdict/1.0", "Accept": "application/json"}

# NEM region codes accepted by the API
_VALID_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class MonthlyPricePoint:
    month: str          # "2025-06"
    price_mwh: float    # $/MWh average for the month
    renewable_pct: float | None = None  # % renewable generation


@dataclass
class DiurnalHour:
    hour: int           # 0–23 local time
    avg_price: float    # $/MWh
    min_price: float
    max_price: float
    n_obs: int          # number of observations averaged


@dataclass
class TrendContext:
    region: str
    months: list[MonthlyPricePoint] = field(default_factory=list)
    current_month_price: float | None = None
    yoy_change_pct: float | None = None      # vs same month last year
    twelve_month_avg: float | None = None
    renewable_latest_pct: float | None = None
    data_source: str = "OpenNEM/OpenElectricity API"
    available: bool = False
    error: str | None = None


@dataclass
class DiurnalContext:
    region: str
    hours: list[DiurnalHour] = field(default_factory=list)
    cheapest_hour: int | None = None
    peak_hour: int | None = None
    solar_trough_hour: int | None = None  # hour with lowest price (solar peak)
    days_sampled: int = 0
    data_source: str = "OpenNEM/OpenElectricity API (last 14 days)"
    available: bool = False
    error: str | None = None


# ── Internal helpers ────────────────────────────────────────────────────────

def _parse_result_series(data: dict) -> list[tuple[str, float]]:
    """Parse [[timestamp, value], ...] series into (month_str, value) pairs."""
    rows = []
    for result in data.get("results", []):
        for point in result.get("data", []):
            if len(point) == 2 and point[1] is not None:
                ts_str = point[0][:7]  # "2025-06"
                try:
                    rows.append((ts_str, float(point[1])))
                except (TypeError, ValueError):
                    pass
    return rows


def _month_to_dt(month_str: str) -> datetime:
    return datetime.strptime(month_str, "%Y-%m").replace(tzinfo=timezone.utc)


# ── Public fetch functions ────────────────────────────────────────────────

async def fetch_monthly_price(region: str, period: str = "1Y") -> list[tuple[str, float]]:
    """Return [(month_str, price_mwh), ...] sorted ascending."""
    if region not in _VALID_REGIONS:
        raise ValueError(f"Invalid region: {region}")
    params = {
        "metrics": "price",
        "interval": "1M",
        "period": period,
        "network_region": region,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_BASE, params=params, headers=_HEADERS)
        resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"OpenNEM API error: {payload.get('error')}")
    rows = []
    for dataset in payload.get("data", []):
        rows.extend(_parse_result_series(dataset))
    return sorted(rows, key=lambda r: r[0])


async def fetch_monthly_renewable_pct(region: str, period: str = "1Y") -> list[tuple[str, float]]:
    """Return [(month_str, renewable_pct), ...] sorted ascending."""
    params = {
        "metrics": "renewable_proportion",
        "interval": "1M",
        "period": period,
        "network_region": region,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_BASE, params=params, headers=_HEADERS)
        resp.raise_for_status()
    payload = resp.json()
    rows = []
    for dataset in payload.get("data", []):
        rows.extend(_parse_result_series(dataset))
    return sorted(rows, key=lambda r: r[0])


async def fetch_hourly_price(region: str, period: str = "7D") -> list[tuple[datetime, float]]:
    """Return [(utc_datetime, price_mwh), ...] for diurnal analysis."""
    params = {
        "metrics": "price",
        "interval": "1h",
        "period": period,
        "network_region": region,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(_BASE, params=params, headers=_HEADERS)
        resp.raise_for_status()
    payload = resp.json()
    rows = []
    for dataset in payload.get("data", []):
        for result in dataset.get("results", []):
            for point in result.get("data", []):
                if len(point) == 2 and point[1] is not None:
                    try:
                        dt = datetime.fromisoformat(point[0])
                        rows.append((dt, float(point[1])))
                    except (ValueError, TypeError):
                        pass
    return sorted(rows, key=lambda r: r[0])


# ── High-level aggregators ────────────────────────────────────────────────

async def get_trend_context(region: str) -> TrendContext:
    """Fetch and aggregate 12-month price trend + renewable proportion.

    Returns TrendContext with monthly series, YoY change, and 12M average.
    Falls back gracefully — always returns a TrendContext even on error.
    """
    ctx = TrendContext(region=region)
    try:
        price_rows, ren_rows = await _fetch_trend_parallel(region)

        if not price_rows:
            ctx.error = "OpenNEM returned empty price series"
            return ctx

        # Build renewable pct lookup
        ren_map = dict(ren_rows)

        ctx.months = [
            MonthlyPricePoint(
                month=m,
                price_mwh=round(p, 2),
                renewable_pct=round(ren_map[m], 1) if m in ren_map else None,
            )
            for m, p in price_rows
        ]

        prices = [m.price_mwh for m in ctx.months]
        ctx.twelve_month_avg = round(sum(prices) / len(prices), 2)
        ctx.current_month_price = ctx.months[-1].price_mwh if ctx.months else None

        # YoY: compare last month vs 12 months prior
        if len(ctx.months) >= 12:
            last = ctx.months[-1].price_mwh
            year_ago = ctx.months[-12].price_mwh
            ctx.yoy_change_pct = round((last - year_ago) / year_ago * 100, 1) if year_ago else None

        ctx.renewable_latest_pct = (
            ctx.months[-1].renewable_pct if ctx.months and ctx.months[-1].renewable_pct else None
        )
        ctx.available = True

    except Exception as exc:
        logger.debug("OpenNEM trend fetch failed (non-fatal): %s", exc)
        ctx.error = str(exc)

    return ctx


async def _fetch_trend_parallel(region: str):
    import asyncio
    return await asyncio.gather(
        fetch_monthly_price(region, period="1Y"),
        fetch_monthly_renewable_pct(region, period="1Y"),
    )


async def get_diurnal_context(region: str) -> DiurnalContext:
    """Fetch 14 days of hourly data and compute per-hour price statistics.

    Returns DiurnalContext with avg/min/max per hour and key markers
    (cheapest hour, peak hour, solar trough). Falls back gracefully.
    """
    ctx = DiurnalContext(region=region)
    try:
        rows = await fetch_hourly_price(region, period="7D")
        if not rows:
            ctx.error = "OpenNEM returned empty hourly series"
            return ctx

        # Group by local hour — OpenNEM timestamps are already AEST (+10:00), .hour is local
        from collections import defaultdict
        hour_prices: dict[int, list[float]] = defaultdict(list)
        for dt, price in rows:
            hour_prices[dt.hour].append(price)

        ctx.hours = []
        for h in range(24):
            pts = hour_prices.get(h, [])
            if not pts:
                continue
            ctx.hours.append(DiurnalHour(
                hour=h,
                avg_price=round(sum(pts) / len(pts), 2),
                min_price=round(min(pts), 2),
                max_price=round(max(pts), 2),
                n_obs=len(pts),
            ))

        if ctx.hours:
            ctx.cheapest_hour = min(ctx.hours, key=lambda h: h.avg_price).hour
            ctx.peak_hour = max(ctx.hours, key=lambda h: h.avg_price).hour
            # Solar trough: cheapest between 9am and 3pm
            solar_window = [h for h in ctx.hours if 9 <= h.hour <= 15]
            if solar_window:
                ctx.solar_trough_hour = min(solar_window, key=lambda h: h.avg_price).hour

        ctx.days_sampled = max(1, len(rows) // 24)
        ctx.available = True

    except Exception as exc:
        logger.debug("OpenNEM diurnal fetch failed (non-fatal): %s", exc)
        ctx.error = str(exc)

    return ctx


# ── Text formatters for planners ──────────────────────────────────────────

def format_monthly_trend_table(ctx: TrendContext) -> list[str]:
    """Format the monthly price series as a compact text table for answers."""
    if not ctx.available or not ctx.months:
        return [f"Monthly price data: unavailable ({ctx.error or 'fetch failed'})"]

    lines = [f"Monthly average spot price — {ctx.region} (last 12 months, OpenNEM):"]
    lines.append(f"  {'MONTH':8} {'$/MWh':8} {'RENEW%':8}")
    lines.append(f"  {'─'*26}")
    for m in ctx.months[-12:]:
        ren = f"{m.renewable_pct:.0f}%" if m.renewable_pct is not None else "n/a"
        lines.append(f"  {m.month:8} {m.price_mwh:7.1f}  {ren}")

    if ctx.twelve_month_avg:
        lines.append(f"  {'─'*26}")
        lines.append(f"  {'12M avg':8} {ctx.twelve_month_avg:7.1f}")

    if ctx.yoy_change_pct is not None:
        direction = "up" if ctx.yoy_change_pct > 0 else "down"
        lines.append(
            f"Year-over-year: {abs(ctx.yoy_change_pct):.1f}% {direction} "
            f"vs same month last year."
        )

    if ctx.renewable_latest_pct is not None:
        lines.append(
            f"Current renewable mix: {ctx.renewable_latest_pct:.0f}% of generation."
        )

    return lines


def format_diurnal_table(ctx: DiurnalContext) -> list[str]:
    """Format hourly price pattern as a compact text table."""
    if not ctx.available or not ctx.hours:
        return [f"Hourly price pattern: unavailable ({ctx.error or 'fetch failed'})"]

    lines = [
        f"Real hourly price pattern — {ctx.region} (last {ctx.days_sampled} days, OpenNEM AEST):",
        f"  {'HOUR':6} {'AVG $/MWh':10} {'RANGE':16} NOTE",
        f"  {'─'*50}",
    ]
    for h in ctx.hours:
        note = ""
        if h.hour == ctx.cheapest_hour:
            note = "← cheapest"
        elif h.hour == ctx.peak_hour:
            note = "← peak"
        elif h.hour == ctx.solar_trough_hour:
            note = "← solar peak"
        label = f"{h.hour:02d}:00"
        lines.append(
            f"  {label:6} {h.avg_price:9.1f}  "
            f"${h.min_price:.0f}–${h.max_price:.0f}{'':6} {note}"
        )

    if ctx.cheapest_hour is not None:
        lines.append(
            f"Cheapest hour (last {ctx.days_sampled}d): "
            f"{ctx.cheapest_hour:02d}:00 AEST — "
            f"${next(h.avg_price for h in ctx.hours if h.hour == ctx.cheapest_hour):.1f}/MWh avg."
        )

    return lines
