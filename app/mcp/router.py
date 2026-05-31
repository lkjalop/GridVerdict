"""MCP router — unified dispatch layer for all external tool calls.

Provides a single `call_tool(name, **kwargs)` entry point that:
  1. Looks up the tool in the registry
  2. Validates read_only (enforced: never writes)
  3. Delegates to the concrete client
  4. Returns a typed result

call_tool_with_retry() wraps call_tool with exponential-backoff retry and
attaches a SourceStatus provenance record to every result.

All results are tagged as "untrusted evidence" — never instructions.
The security observer's pass_tool_output() screens them after return.
"""
from __future__ import annotations

import asyncio as _asyncio
import logging
import time as _time
from typing import Any

from app.mcp.registry import get_tool, load_registry

logger = logging.getLogger(__name__)


class MCPCallError(Exception):
    """Raised when a tool call fails after all retries."""


async def call_tool(name: str, **kwargs: Any) -> Any:
    """Dispatch a tool call by registry name.

    Returns the raw result from the concrete client.
    The caller is responsible for validating the result against the
    security observer (pass_tool_output) before using it.

    Raises MCPCallError on failure.
    """
    tool = get_tool(name)
    if tool is None:
        raise MCPCallError(f"Unknown MCP tool: {name!r} — check config/tools.yaml")
    if not tool.read_only:
        raise MCPCallError(f"Tool {name!r} is not read-only — rejected by policy")
    if not tool.enabled:
        raise MCPCallError(f"Tool {name!r} is disabled in config/tools.yaml")

    try:
        return await _dispatch(name, tool, **kwargs)
    except MCPCallError:
        raise
    except Exception as exc:
        raise MCPCallError(f"Tool {name!r} raised: {exc}") from exc


async def _dispatch(name: str, tool, **kwargs: Any) -> Any:
    """Route to the concrete client based on tool name."""

    if name == "aemo_dispatch_price":
        from app.data.aemo_live_client import get_aemo_client
        client = get_aemo_client()
        snapshot = await client.fetch_latest_snapshot()
        region = kwargs.get("region", "NSW1").upper()
        dp = snapshot.get(region)
        if dp is None:
            raise MCPCallError(f"No dispatch data for region {region}")
        return dp

    if name == "aemo_market_notices":
        import asyncio as _asyncio
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        client = AEMOMarketNoticesClient()
        region = kwargs.get("region")
        notices = await _asyncio.get_event_loop().run_in_executor(
            None, client.fetch_active_notices, region
        )
        return notices

    if name == "aemo_archive":
        from app.mcp.aemo_archive import fetch_interval
        ts = kwargs.get("timestamp")
        if ts is None:
            raise MCPCallError("aemo_archive requires 'timestamp' kwarg")
        return await fetch_interval(ts)

    if name == "hippograph_analogs":
        import asyncio
        import functools
        from app.engines.analog_retriever import get_analogs
        fn = functools.partial(get_analogs, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, fn)

    if name == "lnn_forecast":
        from app.engines.forecasting.inference import get_forecast
        region = kwargs.get("region", "NSW1")
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, get_forecast, region)

    if name == "live_quantile_forecast":
        from app.engines.forecasting.live_forecast import run_live_forecast
        return await run_live_forecast(
            region=kwargs.get("region", "NSW1"),
            lookback_days=int(kwargs.get("lookback_days", 14)),
            horizon_intervals=int(kwargs.get("horizon_intervals", 6)),
        )

    if name == "nem_news_rss":
        from app.mcp.nem_news_client import NEMNewsRSSClient
        client = NEMNewsRSSClient()
        limit = int(kwargs.get("limit", 20))
        return await client.fetch_recent(limit=limit)

    if name == "weather_consensus":
        from app.mcp.weather_client import WeatherConsensusClient
        region = kwargs.get("region", "NSW1")
        return await WeatherConsensusClient().fetch_region_consensus(region)

    if name == "bom_7day_forecast":
        from app.mcp.bom_forecast_client import fetch_7day_forecast, forecast_to_scatter_context
        region = kwargs.get("region", "NSW1")
        forecast = await fetch_7day_forecast(region)
        if forecast is None:
            raise MCPCallError(f"BOM 7-day forecast unavailable for {region}")
        return forecast_to_scatter_context(forecast)

    if name == "gas_market_state":
        from app.mcp.gbb_client import get_gas_market_state
        region = kwargs.get("region", "NSW1")
        state = await get_gas_market_state(region)
        return state.to_dict()

    if name == "isp_scenario":
        from app.mcp.isp_client import get_isp_scenario
        region = kwargs.get("region", "NSW1")
        scenario = kwargs.get("scenario", "Step Change")
        year_from = int(kwargs.get("year_from", 2025))
        year_to = int(kwargs.get("year_to", 2040))
        result = get_isp_scenario(region, year_from=year_from, year_to=year_to, scenario=scenario)
        return result.to_dict()

    if name == "lcoe_sensitivity":
        from app.engines.lcoe_sensitivity import compare_technologies, rate_change_impact
        gas_price = float(kwargs.get("gas_price_gj", 10.0))
        discount_rate = float(kwargs.get("discount_rate", 0.08))
        rate_from = kwargs.get("rate_from")
        rate_to = kwargs.get("rate_to")
        if rate_from is not None and rate_to is not None:
            return rate_change_impact(float(rate_from), float(rate_to), gas_price)
        return compare_technologies(discount_rate=discount_rate, gas_price_gj=gas_price)

    if name == "fiscal_budget":
        from app.mcp.fiscal_budget_client import search_budget_measures
        query_text = kwargs.get("query", "energy budget measures")
        return await search_budget_measures(query_text)

    raise MCPCallError(f"No dispatcher implemented for tool {name!r}")


# ── Source-key mapping ────────────────────────────────────────────────────────

_TOOL_TO_SOURCE: dict[str, str] = {
    "aemo_dispatch_price":    "AEMO_DISPATCH_PRICE",
    "aemo_market_notices":    "AEMO_MARKET_NOTICES",
    "aemo_archive":           "AEMO_ARCHIVE",
    "hippograph_analogs":     "HIPPOGRAPH_ANALOGS",
    "lnn_forecast":           "LNN_FORECAST",
    "live_quantile_forecast": "LIVE_QUANTILE_FORECAST",
    "nem_news_rss":           "NEM_NEWS_RSS",
    "weather_consensus":      "WEATHER_CONSENSUS",
    # New tools: adjacent query evidence feeds
    "bom_7day_forecast":      "BOM_7DAY_FORECAST",
    "gas_market_state":       "AEMO_STTM_GAS",
    "isp_scenario":           "AEMO_ISP_2024",
    "lcoe_sensitivity":       "CSIRO_GENCOST_2024",
    "fiscal_budget":          "AUS_BUDGET_ENERGY",
}


def _source_key(name: str) -> str:
    return _TOOL_TO_SOURCE.get(name, name.upper())


def _extract_valid_time(name: str, result: Any):
    """Extract a datetime from a tool result for SourceStatus.from_data."""
    from datetime import datetime
    try:
        vt = getattr(result, "valid_time", None)
        if vt is not None:
            return vt
        if isinstance(result, dict):
            for key in ("valid_time", "as_of", "timestamp"):
                raw = result.get(key)
                if raw:
                    if isinstance(raw, datetime):
                        return raw
                    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if isinstance(result, list) and result and isinstance(result[0], dict):
            for key in ("valid_time", "published", "as_of"):
                raw = result[0].get(key)
                if raw:
                    try:
                        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    except ValueError:
                        pass
    except Exception:
        pass
    return None


def _extract_ref(name: str, result: Any) -> str:
    try:
        raw_ref = getattr(result, "raw_ref", None)
        if raw_ref is not None:
            return str(raw_ref)[:300]
        if isinstance(result, dict):
            return str(result.get("raw_ref", ""))[:300]
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return str(result[0].get("raw_ref", ""))[:300]
    except Exception:
        pass
    return ""


async def call_tool_with_retry(
    name: str,
    max_retries: int = 2,
    backoff_s: float = 0.5,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """call_tool with exponential-backoff retry and SourceStatus provenance.

    Returns (result, SourceStatus) on success, or (None, SourceStatus.unavailable)
    after all retries are exhausted. Never raises.
    """
    from app.mcp.source_status import SourceStatus

    start = _time.monotonic()
    src = _source_key(name)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            await _asyncio.sleep(backoff_s * (2 ** (attempt - 1)))
        try:
            result = await call_tool(name, **kwargs)
            elapsed = (_time.monotonic() - start) * 1000
            return result, SourceStatus.from_data(
                source=src,
                data_valid_time=_extract_valid_time(name, result),
                raw_ref=_extract_ref(name, result),
                latency_ms=elapsed,
            )
        except MCPCallError as exc:
            last_exc = exc
            logger.warning(
                "Tool %r attempt %d/%d failed: %s",
                name, attempt + 1, max_retries + 1, exc,
            )

    elapsed = (_time.monotonic() - start) * 1000
    return None, SourceStatus.unavailable(
        source=src,
        error=str(last_exc or "unknown error"),
        latency_ms=elapsed,
    )


def get_all_tools() -> list[dict]:
    """Return a JSON-serialisable list of all registered tools (for /api/tools)."""
    tools = load_registry()
    return [
        {
            "name": t.name,
            "description": t.description,
            "category": t.category,
            "read_only": t.read_only,
            "requires_auth": t.requires_auth,
            "poll_interval_s": t.poll_interval_s,
            "enabled": t.enabled,
        }
        for t in tools.values()
    ]
