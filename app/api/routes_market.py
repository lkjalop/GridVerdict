"""Market state routes — live AEMO dispatch data.

GET /market/state?region=NSW1   — current price, demand, regime, staleness
GET /market/regions             — list of supported NEM regions
POST /market/refresh            — force cache refresh from NEMWeb (admin only)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from statistics import mean

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import AEMOClient, Cache, CurrentUser, get_current_user
from app.api.auth import TokenPayload
from app.core.schema import MarketStateResponse
from app.data.aemo_live_client import AEMOLiveClient, LiveMarketSnapshot
from app.data.cache import MarketCache
from config.settings import get_settings
from domain.nem.adapter import classify_regime
from app.api.metrics_registry import forecast_latency_ms

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["market"])
_settings = get_settings()

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_CACHE_KEY = "dispatch_snapshot"

# Rolling 30-day price percentile estimation — seeded from typical NEM ranges.
# ChronoGraph replaces this with a real t-digest once it has data.
_TYPICAL_PRICE = {"NSW1": 80.0, "VIC1": 82.0, "QLD1": 78.0, "SA1": 95.0, "TAS1": 70.0}


@router.get("/regions")
async def list_regions():
    return {"regions": _SUPPORTED_REGIONS}


@router.get("/state", response_model=MarketStateResponse)
async def get_market_state(
    region: str = Query(default="NSW1", description="NEM region code"),
    user: TokenPayload = Depends(get_current_user),
    client: AEMOLiveClient = AEMOClient,
    cache: MarketCache = Cache,
):
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )

    snapshot = await _get_or_fetch_snapshot(client, cache)
    dp = snapshot.get(region)

    if dp is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No dispatch price data available for {region}",
        )

    staleness = snapshot.staleness_seconds(region)
    is_stale = staleness > _settings.live_dispatch_max_age_s
    headroom = max(dp.availability_mw - dp.demand_mw, 0.0)
    regime = classify_regime(dp.price_rrp, region)
    percentile = _estimate_percentile(dp.price_rrp, region)

    # Active notices — read from scheduler cache (populated every ~60s by the background job)
    active_notices: list[dict] = []
    try:
        cached_notices = await cache.get(f"notices_{region}") or await cache.get("notices")
        if isinstance(cached_notices, list):
            active_notices = cached_notices[:10]  # cap to 10 for the API response
    except Exception:
        pass

    return MarketStateResponse(
        region=region,
        price_rrp=round(dp.price_rrp, 2),
        demand_mw=round(dp.demand_mw, 1),
        availability_mw=round(dp.availability_mw, 1),
        headroom_mw=round(headroom, 1),
        regime=regime,
        price_percentile=round(percentile, 3),
        valid_time=dp.valid_time,
        system_time=dp.system_time,
        source=dp.raw_ref[:16] + "...",
        is_stale=is_stale,
        staleness_seconds=staleness,
        active_notices=active_notices,
    )


@router.get("/history")
async def get_market_history(
    region: str = Query(default="NSW1", description="NEM region code"),
    range: str = Query(default="24h", description="Lookback window: 1h, 3h, 6h, 12h, 24h"),
    bucket: str = Query(default="5m", description="Average bucket: 5m, 1h, 3h, 6h, 12h, 24h"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return dispatch price history for the right-panel swimlane chart.

    Uses the persisted market_events table. The endpoint anchors the window to
    the latest stored interval rather than wall-clock now so archive views remain
    usable when live ingestion has temporarily paused.
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )

    range_minutes = _parse_window_minutes(range, default=24 * 60)
    bucket_minutes = _parse_window_minutes(bucket, default=5)
    if bucket_minutes < 5:
        bucket_minutes = 5

    try:
        from sqlalchemy import text
        from app.db.session import db_session

        async with db_session() as session:
            latest_res = await session.execute(
                text("""
                    SELECT MAX(valid_time)
                    FROM market_events
                    WHERE region = :region
                      AND source = 'AEMO_DISPATCH_PRICE'
                """),
                {"region": region},
            )
            latest = latest_res.scalar()
            if latest is None:
                return {"region": region, "range": range, "bucket": bucket, "points": []}
            if not isinstance(latest, datetime):
                latest = datetime.fromisoformat(str(latest))
            start = latest - timedelta(minutes=range_minutes)
            start_param = _db_datetime_param(start)
            latest_param = _db_datetime_param(latest)
            rows_res = await session.execute(
                text("""
                    SELECT valid_time, price_rrp, demand_mw, availability_mw
                    FROM market_events
                    WHERE region = :region
                      AND source = 'AEMO_DISPATCH_PRICE'
                      AND valid_time >= :start
                      AND valid_time <= :latest
                    ORDER BY valid_time
                """),
                {"region": region, "start": start_param, "latest": latest_param},
            )
            rows = rows_res.fetchall()
    except Exception as exc:
        logger.warning("Market history query failed: %s", exc)
        rows = []

    points = _bucket_history_rows(rows, bucket_minutes, region)
    return {
        "region": region,
        "range": range,
        "bucket": bucket,
        "bucket_minutes": bucket_minutes,
        "points": points,
    }


@router.get("/forecast")
async def get_market_forecast(
    region: str = Query(default="NSW1", description="NEM region code"),
    lookback_days: int = Query(default=14, ge=2, le=90),
    horizon_intervals: int = Query(default=6, ge=1, le=24),
    user: TokenPayload = Depends(get_current_user),
    cache: MarketCache = Cache,
):
    """Return real LEAR/QRA/LNN forecast bands when enough live history exists."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )
    from app.mcp.router import call_tool

    _ft0 = time.perf_counter()
    try:
        result = await call_tool(
            "live_quantile_forecast",
            region=region,
            lookback_days=lookback_days,
            horizon_intervals=horizon_intervals,
        )
    except Exception as exc:
        logger.warning("Live forecast unavailable for %s: %s", region, exc)
        result = {
            "region": region,
            "available": False,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "reason": str(exc),
            "forecasts": [],
            "errors": [{"model": "live_quantile_forecast", "error": str(exc)}],
        }
    finally:
        forecast_latency_ms.observe((time.perf_counter() - _ft0) * 1000)
    if isinstance(result, dict) and result.get("available"):
        await cache.set(f"live_forecast_{region}", result)
    return result


@router.get("/trust")
async def get_forecast_trust(
    region: str = Query(default="NSW1", description="NEM region code"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return the forecast trust panel: per-model accuracy, provenance, and availability.

    Metrics come from the most recent in-process backtest evaluation.
    If no evaluation has been run yet, metric fields are null.
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )
    from app.engines.forecast_trust import build_forecast_trust
    return build_forecast_trust(region)


_SOURCE_STATUS: dict[str, dict] = {
    "AEMO_DISPATCH_PRICE": {
        "status": "operational",
        "caveat": (
            "MMSDM DISPATCHPRICE archive. Complete 5-min settlement prices for all NEM regions. "
            "Sourced from public NEMWeb archive; hashes verifiable against raw ZIP files."
        ),
    },
    "AEMO_PREDISPATCH_30MIN": {
        "status": "scaffolded",
        "caveat": (
            "Live 30-min ahead predispatch ingested by scheduler since deployment. "
            "Historical window uses price[t-6] (30-min lagged price) as a non-leaking proxy "
            "-- NOT a real forward-looking forecast. Proxy intervals are marked in feature builder."
        ),
    },
    "driver/constraint": {
        "status": "operational",
        "caveat": (
            "MMSDM DISPATCHCONSTRAINT archive. Binding constraints only (marginal_value != 0 "
            "or violation_degree != 0). ~90% storage reduction vs full table -- non-binding rows excluded."
        ),
    },
    "driver/interconnector": {
        "status": "partial",
        "caveat": (
            "MMSDM DISPATCHINTERCONNECTORRES archive. Flow data present; MARGINALVALUE is 0 "
            "for all rows in this table. Use mw_flow vs export_limit proximity for congestion detection."
        ),
    },
    "unit_dispatch_events": {
        "status": "unavailable",
        "caveat": (
            "DISPATCH_UNIT_SOLUTION not included in backfill_tables "
            "(DISPATCHPRICE,DISPATCHINTERCONNECTORRES,DISPATCHCONSTRAINT,DUDETAILSUMMARY). "
            "Platform cannot currently attribute price-setting to a specific fuel type or bidding behaviour."
        ),
    },
    "generator_units": {
        "status": "operational",
        "caveat": (
            "DUDETAILSUMMARY metadata. 872 DUIDs registered with fuel type, region, max capacity. "
            "Used for fuel-type attribution in why-engine narrative."
        ),
    },
    "bid_offers": {
        "status": "scaffolded",
        "caveat": (
            "BIDDAYOFFER (day-ahead bids) and BIDPEROFFER (intraday rebids) ingestion implemented. "
            "Historical backfill for Nov 2023–Feb 2024 window not yet triggered. "
            "Once populated, enables rebid detection: identify when generators strategically "
            "reduced low-price availability ahead of price spikes."
        ),
    },
}


@router.get("/coverage")
async def get_coverage(
    user: TokenPayload = Depends(get_current_user),
):
    """Return data-coverage status per NEM region and per data source.

    Per-region status values (dispatch price):
      operational  -- >=30 days of dispatch rows in DB
      partial      -- 1-29 days of dispatch rows
      not_ingested -- 0 rows

    Per-source status values:
      operational  -- complete data from authoritative AEMO source
      partial      -- data present but known gaps or quality limitations
      scaffolded   -- feature exists but backed by proxy/estimate, not real data
      unavailable  -- source not ingested in current configuration
    """
    try:
        from sqlalchemy import text
        from app.db.session import db_session

        result: dict[str, dict] = {}
        async with db_session() as session:
            rows = await session.execute(
                text("""
                    SELECT region,
                           COUNT(*) AS cnt,
                           MIN(valid_time) AS lo,
                           MAX(valid_time) AS hi
                    FROM market_events
                    WHERE source = 'AEMO_DISPATCH_PRICE'
                    GROUP BY region
                """)
            )
            by_region = {r[0]: {"count": r[1], "lo": r[2], "hi": r[3]} for r in rows.fetchall()}

        for region in _SUPPORTED_REGIONS:
            info = by_region.get(region)
            if info is None or info["count"] == 0:
                status_val = "not_ingested"
                days = 0
            else:
                # Each region-interval is one 5-min row; 288 per day
                days = info["count"] // 288
                status_val = "operational" if days >= 30 else "partial"

            result[region] = {
                "status": status_val,
                "days": days,
                "intervals": info["count"] if info else 0,
                "earliest": info["lo"].isoformat() if info and info["lo"] else None,
                "latest": info["hi"].isoformat() if info and info["hi"] else None,
            }

        return {"coverage": result, "sources": _SOURCE_STATUS}
    except Exception as exc:
        logger.warning("Coverage query failed: %s", exc)
        # Return unknown status rather than 500 — UI degrades gracefully
        return {
            "coverage": {
                r: {"status": "unknown", "days": 0, "intervals": 0,
                    "earliest": None, "latest": None}
                for r in _SUPPORTED_REGIONS
            },
            "sources": _SOURCE_STATUS,
        }


@router.get("/fuel-mix")
async def get_fuel_mix(
    region: str = Query(default="NSW1", description="NEM region code"),
    user: TokenPayload = Depends(get_current_user),
    cache: MarketCache = Cache,
):
    """Per-fuel-type capacity/dispatch breakdown and source recommendation.

    Returns battery, wind, hydro, coal, gas, solar capacity mix with a
    deterministic "best buy source" recommendation based on current spot price.
    Data tier: dispatch (DISPATCH_UNIT_SOLUTION) > capacity (DUDETAILSUMMARY) > priors.
    """
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )
    try:
        from app.db.session import db_session
        from app.engines.fuel_mix import get_fuel_mix as _get_fuel_mix

        weather = await cache.get(f"weather_{region}")
        async with db_session() as session:
            return await _get_fuel_mix(region, session, weather=weather if isinstance(weather, dict) else None)
    except Exception as exc:
        logger.warning("Fuel mix query failed: %s", exc)
        from app.engines.fuel_mix import get_fuel_mix as _get_fuel_mix
        return await _get_fuel_mix(region, None)


@router.post("/refresh", status_code=status.HTTP_204_NO_CONTENT)
async def force_refresh(
    user: TokenPayload = Depends(get_current_user),
    client: AEMOLiveClient = AEMOClient,
    cache: MarketCache = Cache,
):
    await cache.invalidate(_CACHE_KEY)
    try:
        snapshot = await client.fetch_latest_snapshot()
        await cache.set(_CACHE_KEY, snapshot.to_dict())
    except Exception as exc:
        logger.error("Force refresh failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


async def _get_or_fetch_snapshot(
    client: AEMOLiveClient, cache: MarketCache
) -> LiveMarketSnapshot:
    """Return cached snapshot or fetch fresh from NEMWeb."""
    from app.data.aemo_live_client import LiveMarketSnapshot, DispatchPrice, _parse_aemo_dt

    cached = await cache.get(_CACHE_KEY)
    if cached is not None:
        return _dict_to_snapshot(cached)

    try:
        snapshot = await client.fetch_latest_snapshot()
        await cache.set(_CACHE_KEY, snapshot.to_dict())
        return snapshot
    except Exception as exc:
        logger.error("NEMWeb fetch failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch live data from NEMWeb: {exc}",
        ) from exc


def _dict_to_snapshot(data: dict) -> "LiveMarketSnapshot":
    """Reconstruct a LiveMarketSnapshot from the cached dict representation."""
    from app.data.aemo_live_client import LiveMarketSnapshot, DispatchPrice

    regions = {}
    for region_code, rdata in data.get("regions", {}).items():
        regions[region_code] = DispatchPrice(
            region=rdata["region"],
            valid_time=datetime.fromisoformat(rdata["valid_time"]),
            system_time=datetime.now(timezone.utc),
            price_rrp=rdata["price_rrp"],
            demand_mw=rdata["demand_mw"],
            availability_mw=rdata["availability_mw"],
            raw_ref=data.get("raw_ref", "cache"),
        )

    interval_str = data.get("interval", datetime.now(timezone.utc).isoformat())
    fetched_str = data.get("fetched_at", datetime.now(timezone.utc).isoformat())

    return LiveMarketSnapshot(
        interval=datetime.fromisoformat(interval_str),
        fetched_at=datetime.fromisoformat(fetched_str),
        regions=regions,
        raw_ref=data.get("raw_ref", "cache"),
    )


def _estimate_percentile(price: float, region: str) -> float:
    """Return where the current price sits in the rolling 30-day distribution.

    Uses the ChronoGraph t-digest (RegimeClassifier._digest) once it has
    accumulated ≥24 observations (~2 h of dispatch data). Falls back to the
    linear heuristic while the digest is still cold.
    """
    try:
        from app.engines.chronograph.regime import get_classifier
        from domain.nem.adapter import _REGIME_THRESHOLDS
        thresholds = _REGIME_THRESHOLDS.get(region, _REGIME_THRESHOLDS["NSW1"])
        classifier = get_classifier(region, thresholds)
        if classifier.observation_count >= 24:
            return round(classifier._digest.cdf(price), 4)
    except Exception:
        pass

    # Heuristic fallback (no history yet)
    typical = _TYPICAL_PRICE.get(region, 80.0)
    if price <= typical:
        return min(0.50, price / (typical * 2))
    from domain.nem.adapter import _REGIME_THRESHOLDS
    thresholds = _REGIME_THRESHOLDS.get(region, _REGIME_THRESHOLDS["NSW1"])
    if price < thresholds["elevated"]:
        return 0.50 + 0.20 * ((price - typical) / (thresholds["elevated"] - typical))
    if price < thresholds["spike"]:
        return 0.70 + 0.15 * ((price - thresholds["elevated"]) / (thresholds["spike"] - thresholds["elevated"]))
    if price < thresholds["extreme"]:
        return 0.85 + 0.10 * ((price - thresholds["spike"]) / (thresholds["extreme"] - thresholds["spike"]))
    return 0.99


def _parse_window_minutes(value: str, default: int) -> int:
    value = (value or "").strip().lower()
    try:
        if value.endswith("m"):
            return int(value[:-1])
        if value.endswith("h"):
            return int(value[:-1]) * 60
        return int(value)
    except (TypeError, ValueError):
        return default


def _bucket_history_rows(rows, bucket_minutes: int, region: str) -> list[dict]:
    if not rows:
        return []
    buckets: dict[datetime, list] = {}
    for row in rows:
        vt = row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(str(row[0]))
        minute_of_day = vt.hour * 60 + vt.minute
        bucket_start_min = (minute_of_day // bucket_minutes) * bucket_minutes
        bucket_start = vt.replace(
            hour=bucket_start_min // 60,
            minute=bucket_start_min % 60,
            second=0,
            microsecond=0,
        )
        buckets.setdefault(bucket_start, []).append(row)

    points = []
    for ts in sorted(buckets):
        group = buckets[ts]
        prices = [float(r[1]) for r in group if r[1] is not None]
        demands = [float(r[2]) for r in group if r[2] is not None]
        avails = [float(r[3]) for r in group if r[3] is not None]
        if not prices:
            continue
        points.append({
            "time": ts.isoformat(),
            "price_rrp": round(mean(prices), 2),
            "demand_mw": round(mean(demands), 1) if demands else None,
            "availability_mw": round(mean(avails), 1) if avails else None,
            "regime": classify_regime(mean(prices), region),
            "samples": len(group),
        })
    return points


def _db_datetime_param(dt: datetime) -> datetime | str:
    """Return a raw-SQL datetime bind value compatible with the active DB.

    SQLite stores SQLAlchemy DateTime values as naive strings, so text queries
    need a UTC-naive string. PostgreSQL/asyncpg expects a real datetime object
    and rejects strings for timestamp binds.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    from config.settings import get_settings

    if get_settings().database_url.startswith("sqlite"):
        return dt.isoformat(sep=" ")
    return dt


# ── Procurement windows endpoint ─────────────────────────────────────

@router.get("/procurement-windows")
async def get_procurement_windows(
    region: str = Query(default="NSW1"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return cheapest buying windows based on 30-day historical hourly averages.

    Used by the Portfolio tab Procurement Optimizer widget.  Returns:
    - current_price + percentile rank vs 30-day distribution
    - hourly_avg: 24 buckets with avg price, flagged cheapest/expensive
    - cheapest_windows: top-3 recommended buying time windows
    """
    from sqlalchemy import select, func, extract
    from app.db.session import db_session
    from app.db.models import MarketEvent
    from app.data.cache import get_cache

    region = region.upper()
    now = datetime.now(timezone.utc)
    lookback = now - timedelta(days=30)

    try:
        # Get current price from cache
        cache = get_cache()
        snap = await cache.get("dispatch_snapshot") or {}
        region_snap = (snap.get("regions") or {}).get(region, {})
        current_price = float(region_snap.get("price_rrp") or 0.0)
        regime = region_snap.get("regime", "")

        async with db_session() as session:
            # 30-day hourly averages
            result = await session.execute(
                select(
                    extract("hour", MarketEvent.valid_time).label("hour"),
                    func.avg(MarketEvent.price_rrp).label("avg"),
                    func.min(MarketEvent.price_rrp).label("min"),
                    func.max(MarketEvent.price_rrp).label("max"),
                    func.count().label("n"),
                )
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                    MarketEvent.valid_time >= lookback,
                )
                .group_by(extract("hour", MarketEvent.valid_time))
                .order_by(extract("hour", MarketEvent.valid_time))
            )
            hourly = {int(r.hour): {"avg": float(r.avg), "min": float(r.min), "max": float(r.max), "n": r.n}
                      for r in result.fetchall()}

            # Percentile rank of current price
            pct_result = await session.execute(
                select(func.count())
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                    MarketEvent.valid_time >= lookback,
                    MarketEvent.price_rrp <= current_price,
                )
            )
            below = pct_result.scalar() or 0
            total_result = await session.execute(
                select(func.count())
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                    MarketEvent.valid_time >= lookback,
                )
            )
            total = max(total_result.scalar() or 1, 1)
            percentile_rank = round(below / total * 100)

        if not hourly:
            return {"region": region, "current_price": current_price, "regime": regime,
                    "percentile_rank": percentile_rank, "hourly_avg": [], "cheapest_windows": []}

        # Normalise for bar chart (pct = fraction of max avg price)
        max_avg = max(h["avg"] for h in hourly.values()) or 1.0
        sorted_avgs = sorted((h["avg"], hr) for hr, h in hourly.items())
        cheap_hours = {hr for _, hr in sorted_avgs[:8]}   # cheapest 8 hours
        expensive_hours = {hr for _, hr in sorted_avgs[-4:]}  # most expensive 4 hours

        hourly_avg = [
            {
                "hour": hr,
                "avg": round(hourly.get(hr, {}).get("avg", 0), 2),
                "pct": round(hourly.get(hr, {}).get("avg", 0) / max_avg, 3),
                "cheapest": hr in cheap_hours,
                "expensive": hr in expensive_hours,
            }
            for hr in range(24)
        ]

        # Build recommended windows (group consecutive cheap hours)
        windows: list[dict] = []
        for start in [0, 2, 23, 1, 3, 22, 10, 11]:   # rough overnight/midday priorities
            if start in cheap_hours:
                avg_p = hourly.get(start, {}).get("avg", 0)
                next_h = (start + 1) % 24
                avg_p2 = hourly.get(next_h, {}).get("avg", 0)
                window_avg = round((avg_p + avg_p2) / 2, 1)
                label = f"{start:02d}:00–{(start+2)%24:02d}:00"
                note = (
                    "overnight baseload" if 0 <= start <= 5
                    else "solar ramp" if 9 <= start <= 13
                    else "early morning"
                )
                windows.append({"label": label, "avg_price": window_avg, "note": note})
                if len(windows) >= 3:
                    break

        windows.sort(key=lambda w: w["avg_price"])
        return {
            "region": region,
            "current_price": round(current_price, 2),
            "regime": regime,
            "percentile_rank": percentile_rank,
            "hourly_avg": hourly_avg,
            "cheapest_windows": windows[:3],
        }
    except Exception as exc:
        logger.warning("Procurement windows failed for %s: %s", region, exc)
        return {"region": region, "current_price": None, "percentile_rank": None,
                "hourly_avg": [], "cheapest_windows": [], "error": str(exc)}


# ── Causal chain endpoint ─────────────────────────────────────────────

@router.get("/causal-chain")
async def get_causal_chain(
    region: str = Query(default="NSW1", description="NEM region"),
    valid_time: str = Query(..., description="ISO-format dispatch interval e.g. 2026-06-01T10:35:00+00:00"),
    price: float = Query(..., description="Spot price in $/MWh at that interval"),
    user: TokenPayload = Depends(get_current_user),
):
    """Build a causal evidence chain: price → marginal fuel → demand driver → weather.

    Called asynchronously by the frontend after the main query returns so it
    does not add latency to the primary answer path.  Returns partial chains
    gracefully when data is unavailable for the given interval.
    """
    from datetime import datetime as _dt
    from app.db.session import db_session
    from app.engines.why_evidence_chains import build_causal_chain

    region = region.upper()

    try:
        vt = _dt.fromisoformat(valid_time.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"Invalid valid_time: {valid_time!r}")

    try:
        async with db_session() as session:
            chain = await build_causal_chain(
                region=region,
                valid_time=vt,
                spot_price=float(price),
                session=session,
            )
        return chain.to_dict()
    except Exception as exc:
        logger.warning("Causal chain build failed for %s@%s: %s", region, valid_time, exc)
        return {
            "region": region,
            "valid_time": valid_time,
            "spot_price": price,
            "nodes": [],
            "root_cause": "",
            "confidence": "low",
            "data_gaps": [f"Causal chain unavailable: {exc}"],
        }
