"""ScatterGather — parallel evidence collection, ~330ms standard path.

Fires all data-fetch tasks concurrently using asyncio.gather.
Returns a GatherResult consumed by the query route for verdict derivation.

Tasks:
  T1 — Live dispatch price (AEMO NEMWeb)
  T2 — Active market notices (AEMO notices cache)
  T3 — Historical analogs (HippoGraph PPR)
  T4 — LNN quantile forecast (in-process)
  T5 — AEMO pre-dispatch 30-min ahead intervals
  T6 — NEM news RSS sentiment
  T7 — Weather consensus (optional, when query is weather-relevant)

All tasks are timeout-gated. A failed task contributes is_available=False
to coverage scoring — it never raises and never blocks the pipeline.
Each task result carries a SourceStatus provenance record (freshness,
confidence, coverage_status, error, latency_ms).
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.data.aemo_live_client import AEMOLiveClient, DispatchPrice, LiveMarketSnapshot
from app.data.cache import MarketCache
from app.mcp.source_status import SourceStatus

logger = logging.getLogger(__name__)

_TASK_TIMEOUT = 8.0   # seconds per parallel task


@dataclass
class GatherResult:
    dispatch: DispatchPrice | None
    dispatch_fresh: bool
    notices: list[dict[str, Any]] = field(default_factory=list)
    analogs: list[dict[str, Any]] = field(default_factory=list)
    forecast: dict[str, float] | None = None       # {"p10","p50","p90"} from best model (LNN or LEAR)
    live_forecast: dict[str, Any] | None = None    # full run_live_forecast() result (LEAR+QRA+LNN)
    predispatch: list[dict[str, Any]] = field(default_factory=list)
    news_items: list[dict[str, Any]] = field(default_factory=list)
    weather: dict[str, Any] | None = None
    driver_events: list[dict[str, Any]] = field(default_factory=list)
    unit_events: list[dict[str, Any]] = field(default_factory=list)
    recent_dispatch: list[dict[str, Any]] = field(default_factory=list)
    tasks_ok: int = 0
    tasks_total: int = 6
    elapsed_ms: float = 0.0
    # Staleness flags set when cache data is older than the configured threshold
    notices_stale: bool = False
    news_stale: bool = False
    # Sprint Q: pre-computed commentary events used as RAG context
    commentary_context: list[dict[str, Any]] = field(default_factory=list)
    # Per-source provenance: keyed by source constant (e.g. "AEMO_DISPATCH_PRICE")
    source_statuses: dict[str, SourceStatus] = field(default_factory=dict)

    @property
    def source_coverage(self) -> float:
        """Fraction of data sources that returned successfully."""
        return self.tasks_ok / max(self.tasks_total, 1)

    @property
    def freshness_score(self) -> float:
        if self.dispatch is None:
            return 0.0
        age = (datetime.now(timezone.utc) - self.dispatch.valid_time).total_seconds()
        if age <= 60:
            return 1.0
        if age <= 300:
            return 1.0 - (age - 60) / 240 * 0.5   # 1.0 → 0.5 over 5min
        return 0.2


async def _wrap_task(source_key: str, coro) -> tuple[Any, SourceStatus]:
    """Wrap a task coroutine, catching all exceptions and returning a SourceStatus."""
    start = _time.monotonic()
    try:
        result = await coro
        elapsed = (_time.monotonic() - start) * 1000
        if result is None:
            return None, SourceStatus.unavailable(source_key, "no data available", elapsed)
        # Extract valid_time from result if present
        vt = getattr(result, "valid_time", None)
        if vt is None and isinstance(result, dict):
            for key in ("valid_time", "as_of"):
                raw = result.get(key)
                if raw:
                    try:
                        vt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    except ValueError:
                        pass
                    break
        partial = (isinstance(result, list) and len(result) == 0) or (
            isinstance(result, dict) and not result.get("available", True)
        )
        return result, SourceStatus.from_data(
            source=source_key,
            data_valid_time=vt,
            latency_ms=elapsed,
            partial=partial,
        )
    except Exception as exc:
        elapsed = (_time.monotonic() - start) * 1000
        logger.debug("ScatterGather %s task failed: %s", source_key, exc)
        return None, SourceStatus.unavailable(source_key, str(exc)[:200], elapsed)


async def scatter_gather(
    region: str,
    client: AEMOLiveClient,
    cache: MarketCache,
    include_weather: bool = False,
    include_commentary: bool = False,
) -> GatherResult:
    """Run all evidence tasks in parallel and merge results."""
    start = asyncio.get_event_loop().time()

    # T1 — dispatch first so its result is available for T3 (analogs)
    _t1_start = _time.monotonic()
    t1_list = await asyncio.gather(
        _task_dispatch(region, client, cache),
        return_exceptions=True,
    )
    t1_raw = t1_list[0]
    _t1_elapsed = (_time.monotonic() - _t1_start) * 1000

    if isinstance(t1_raw, DispatchPrice):
        dispatch_for_analogs = t1_raw
        dispatch_status = SourceStatus.from_data(
            "AEMO_DISPATCH_PRICE",
            t1_raw.valid_time,
            latency_ms=_t1_elapsed,
        )
    else:
        dispatch_for_analogs = None
        _err = str(t1_raw) if isinstance(t1_raw, Exception) else "no dispatch data"
        dispatch_status = SourceStatus.unavailable("AEMO_DISPATCH_PRICE", _err, _t1_elapsed)
        if isinstance(t1_raw, Exception):
            logger.warning("ScatterGather T1 (dispatch) failed: %s", t1_raw)

    # T2–T7 — parallel tasks wrapped so they never raise
    _ptasks = [
        ("AEMO_MARKET_NOTICES",    _task_notices(region, cache)),
        ("HIPPOGRAPH_ANALOGS",     _task_analogs(region, dispatch_for_analogs)),
        ("LIVE_QUANTILE_FORECAST", _task_forecast(region, cache)),
        ("AEMO_PREDISPATCH",       _task_predispatch(region, client, cache)),
        ("NEM_NEWS_RSS",           _task_news_sentiment(region, cache)),
    ]
    if include_weather:
        _ptasks.append(("WEATHER_CONSENSUS", _task_weather(region, cache)))
    if include_commentary:
        _ptasks.append(("COMMENTARY_CONTEXT", _task_commentary_context(region)))

    _praw = await asyncio.gather(
        *[_wrap_task(src, coro) for src, coro in _ptasks],
        return_exceptions=True,
    )

    source_statuses: dict[str, SourceStatus] = {"AEMO_DISPATCH_PRICE": dispatch_status}
    _pdata: list[Any] = []
    for (src, _), raw in zip(_ptasks, _praw):
        if isinstance(raw, tuple):
            data, status = raw
        else:
            data = None
            status = SourceStatus.unavailable(src, str(raw))
        _pdata.append(data)
        source_statuses[src] = status

    notices_result    = _pdata[0] if isinstance(_pdata[0], list) else []
    analogs_result    = _pdata[1] if isinstance(_pdata[1], list) else []
    live_forecast_result = _pdata[2] if isinstance(_pdata[2], dict) else None
    predispatch_result = _pdata[3] if isinstance(_pdata[3], list) else []
    news_items_result = _pdata[4] if isinstance(_pdata[4], list) else []
    _weather_idx = 5
    weather_result = (
        _pdata[_weather_idx]
        if include_weather and len(_pdata) > _weather_idx and isinstance(_pdata[_weather_idx], dict)
        else None
    )
    _commentary_idx = 5 + (1 if include_weather else 0)
    commentary_result: list[dict[str, Any]] = (
        _pdata[_commentary_idx]
        if include_commentary and len(_pdata) > _commentary_idx and isinstance(_pdata[_commentary_idx], list)
        else []
    )

    elapsed = (asyncio.get_event_loop().time() - start) * 1000

    dispatch_result = t1_raw if isinstance(t1_raw, DispatchPrice) else None

    # Extract LNN-compatible p10/p50/p90 dict from the live forecast result for
    # backward-compat consumers (why_sources.py LNN branch still reads gather.forecast)
    forecast_result: dict[str, float] | None = None
    if live_forecast_result and live_forecast_result.get("available"):
        _primary = live_forecast_result.get("primary_model", "")
        for _fc in live_forecast_result.get("forecasts", []):
            if _fc.get("model") == _primary and _fc.get("p50"):
                _p50_list = _fc["p50"]
                _p10_list = _fc.get("p10", _p50_list)
                _p90_list = _fc.get("p90", _p50_list)
                forecast_result = {
                    "p10": _p10_list[0] if _p10_list else None,
                    "p50": _p50_list[0] if _p50_list else None,
                    "p90": _p90_list[0] if _p90_list else None,
                }
                break

    ok_flags = [
        dispatch_result is not None,
        _pdata[0] is not None,
        _pdata[1] is not None,
        _pdata[2] is not None,
        _pdata[3] is not None,
        _pdata[4] is not None,
    ]
    if include_weather:
        ok_flags.append(_pdata[_weather_idx] is not None if len(_pdata) > _weather_idx else False)
    if include_commentary:
        ok_flags.append(len(commentary_result) > 0)
    tasks_ok = sum(ok_flags)

    dispatch_fresh = False
    if dispatch_result:
        age = (datetime.now(timezone.utc) - dispatch_result.valid_time).total_seconds()
        dispatch_fresh = age <= 300

    # Detect stale notices/news — pull age from cache metadata keys
    from config.settings import get_settings as _get_settings
    _cfg = _get_settings()
    notices_stale = await _is_cache_stale(cache, f"notices_{region}_fetched_at", _cfg.notices_max_age_s)
    news_stale = await _is_cache_stale(cache, "nem_news_fetched_at", _cfg.nem_news_max_age_s)

    return GatherResult(
        dispatch=dispatch_result,
        dispatch_fresh=dispatch_fresh,
        notices=notices_result,
        analogs=analogs_result,
        forecast=forecast_result,
        live_forecast=live_forecast_result,
        predispatch=predispatch_result,
        news_items=news_items_result,
        weather=weather_result,
        commentary_context=commentary_result,
        tasks_ok=tasks_ok,
        tasks_total=6 + (1 if include_weather else 0) + (1 if include_commentary else 0),
        elapsed_ms=elapsed,
        notices_stale=notices_stale,
        news_stale=news_stale,
        source_statuses=source_statuses,
    )


async def _task_dispatch(
    region: str, client: AEMOLiveClient, cache: MarketCache
) -> DispatchPrice:
    """T1 — fetch or use cached dispatch price."""
    cached_raw = await cache.get("dispatch_snapshot")
    if cached_raw:
        from app.api.routes_market import _dict_to_snapshot
        snapshot = _dict_to_snapshot(cached_raw)
        dp = snapshot.get(region)
        if dp:
            return dp

    async with asyncio.timeout(_TASK_TIMEOUT):
        snapshot = await client.fetch_latest_snapshot()
        await cache.set("dispatch_snapshot", snapshot.to_dict())
        dp = snapshot.get(region)
        if dp is None:
            raise RuntimeError(f"No data for {region} in snapshot")
        return dp


async def _task_notices(region: str, cache: MarketCache) -> list[dict[str, Any]]:
    """T2 — fetch active AEMO market notices (from cache only for now)."""
    cached = await cache.get(f"notices_{region}")
    if cached and isinstance(cached, list):
        return cached
    # AEMO notices client will be wired via app/mcp/aemo_notices_client.py
    # Returning empty list is valid — contributes 0 to news_tier
    return []


_FORECAST_CACHE_TTL = 300   # 5 min — one dispatch cycle


async def _task_forecast(region: str, cache: MarketCache | None = None) -> dict[str, Any] | None:
    """T4 — LEAR/QRA/LNN ensemble forecast.

    Runs run_live_forecast() (trains on persisted DB history) with a 5-minute
    in-process cache so model training only happens once per dispatch cycle.
    Falls back to LNN-only when DB has insufficient history.
    Weather context from the cache is forwarded so future feature rows include
    current temperature and wind speed (cols 15-16).
    """
    _cache_key = f"live_forecast_{region}"

    # Fast path: return cached result if still fresh
    if cache is not None:
        cached = await cache.get(_cache_key)
        if isinstance(cached, dict) and cached.get("available"):
            return cached

    # Pull cached weather for this region (best-effort; None is safe)
    weather_context: dict | None = None
    if cache is not None:
        cached_weather = await cache.get(f"weather_{region}")
        if isinstance(cached_weather, dict):
            weather_context = cached_weather

    try:
        from app.engines.forecasting.live_forecast import run_live_forecast
        async with asyncio.timeout(30.0):
            result = await run_live_forecast(
                region, lookback_days=14, horizon_intervals=1,
                weather_context=weather_context,
            )

        if cache is not None and result.get("available"):
            await cache.set(_cache_key, result)
        return result
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("ScatterGather T4 (LEAR/QRA forecast) timed out for %s — falling back to LNN", region)
    except Exception as exc:
        logger.debug("ScatterGather T4 (LEAR/QRA forecast) failed for %s: %s", region, exc)

    # Fallback: LNN only (no DB needed)
    try:
        from app.engines.forecasting.inference import get_forecast
        lnn_raw = await asyncio.get_event_loop().run_in_executor(None, get_forecast, region)
        if lnn_raw:
            return {"available": True, "primary_model": "experimental_lnn",
                    "forecasts": [{"model": "experimental_lnn",
                                   "p10": [lnn_raw["p10"]], "p50": [lnn_raw["p50"]],
                                   "p90": [lnn_raw["p90"]], "caveat": "LNN only"}],
                    "errors": []}
    except Exception:
        pass
    return None


async def _task_analogs(
    region: str,
    dispatch: DispatchPrice | None = None,
) -> list[dict[str, Any]]:
    """T3 — retrieve historical analogs from HippoGraph via PPR."""
    from app.engines.analog_retriever import get_analogs
    from domain.nem.adapter import classify_regime

    if dispatch is None:
        return []

    regime = classify_regime(dispatch.price_rrp, region)
    return await asyncio.get_event_loop().run_in_executor(
        None,
        get_analogs,
        region,
        dispatch.price_rrp,
        dispatch.demand_mw,
        dispatch.availability_mw,
        regime,
        dispatch.valid_time,
    )


async def _task_predispatch(
    region: str, client: AEMOLiveClient, cache: MarketCache
) -> list[dict[str, Any]]:
    """T5 — fetch AEMO pre-dispatch 30-min ahead intervals for the region."""
    cache_key = f"predispatch_{region}"
    cached = await cache.get(cache_key)
    if cached and isinstance(cached, list):
        return cached

    async with asyncio.timeout(_TASK_TIMEOUT):
        pd_by_region = await client.fetch_predispatch()

    intervals = pd_by_region.get(region.upper(), [])
    result = [
        {
            "region": iv.region,
            "interval_datetime": iv.interval_datetime.isoformat(),
            "rrp": iv.rrp,
            "demand_mw": iv.demand_mw,
            "raw_ref": iv.raw_ref,
        }
        for iv in intervals
    ]
    if result:
        await cache.set(cache_key, result)
    return result


async def _task_news_sentiment(region: str, cache: MarketCache) -> list[dict[str, Any]]:
    """T6 — read public RSS market commentary from cache only."""
    cached = await cache.get("nem_news")
    if not isinstance(cached, list):
        return []
    region_l = region.lower()
    region_prefix = region[:3].lower()
    result = []
    for item in cached:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        if region_l in text or region_prefix in text or "nem" in text or "aemo" in text:
            result.append(item)
    return result[:10]


async def _task_weather(region: str, cache: MarketCache) -> dict[str, Any] | None:
    """T7 — read cached weather consensus or fetch on demand when weather is relevant."""
    cache_key = f"weather_{region}"
    cached = await cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    async with asyncio.timeout(_TASK_TIMEOUT):
        from app.mcp.weather_client import WeatherConsensusClient
        result = await WeatherConsensusClient().fetch_region_consensus(region)
        await cache.set(cache_key, result)
        return result


async def _task_commentary_context(region: str) -> list[dict[str, Any]]:
    """T8 — Sprint Q: retrieve recent commentary events for RAG context.

    Returns up to 5 pre-computed Why Engine analyses from the last 4 hours.
    Only included when include_commentary=True (user queries needing history).
    The CommentaryEngine itself always passes include_commentary=False to avoid
    recursive DB load during auto-commentary generation.
    """
    try:
        from datetime import timedelta
        from app.engines.commentary.store import search_for_rag
        now = datetime.now(timezone.utc)
        return await search_for_rag(
            region=region,
            time_from=now - timedelta(hours=4),
            time_to=now,
            min_confidence=0.5,
        )
    except Exception as exc:
        logger.debug("ScatterGather T8 (commentary context) failed for %s: %s", region, exc)
        return []


async def _is_cache_stale(cache: MarketCache, timestamp_key: str, max_age_s: int) -> bool:
    """True if the named timestamp key is absent or older than max_age_s."""
    age = await cache.age_seconds(timestamp_key)
    if age is None:
        return True   # never been written — treat as stale
    return age > max_age_s
