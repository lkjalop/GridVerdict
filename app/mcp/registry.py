"""MCP tool registry — central catalogue of all data-fetch capabilities.

Each entry describes one external data source with:
  - name / description / category
  - rate limits and staleness thresholds
  - whether it requires credentials
  - read-only flag (enforced: GridVerdict never writes to external systems)

The registry is loaded from config/tools.yaml at startup.  Routes can
query it to build the "sources" section of the API docs and to validate
that scatter_gather task names match registered tools.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_TOOLS_CONFIG = Path(__file__).parent.parent.parent / "config" / "tools.yaml"


@dataclass
class MCPTool:
    name: str
    description: str
    category: str           # dispatch | notices | archive | forecast | analogs
    endpoint: str           # URL or internal identifier
    read_only: bool = True  # always True — GridVerdict never writes externally
    requires_auth: bool = False
    poll_interval_s: int = 300   # how often to refresh
    staleness_threshold_s: int = 600
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


_registry: dict[str, MCPTool] = {}


def load_registry(config_path: str | Path | None = None) -> dict[str, MCPTool]:
    """Load tool registry from tools.yaml. Returns the registry dict."""
    global _registry
    path = Path(config_path) if config_path else _TOOLS_CONFIG
    if not path.exists():
        logger.warning("tools.yaml not found at %s — using built-in defaults", path)
        _registry = _default_registry()
        return _registry

    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        tools_list = raw.get("tools", [])
        _registry = {t["name"]: MCPTool(**t) for t in tools_list if "name" in t}
        logger.info("MCP registry loaded: %d tools from %s", len(_registry), path)
    except Exception as exc:
        logger.warning("tools.yaml load failed: %s — using defaults", exc)
        _registry = _default_registry()

    return _registry


def get_tool(name: str) -> MCPTool | None:
    if not _registry:
        load_registry()
    return _registry.get(name)


def list_tools(category: str | None = None, enabled_only: bool = True) -> list[MCPTool]:
    if not _registry:
        load_registry()
    tools = list(_registry.values())
    if enabled_only:
        tools = [t for t in tools if t.enabled]
    if category:
        tools = [t for t in tools if t.category == category]
    return tools


def _default_registry() -> dict[str, MCPTool]:
    return {
        t.name: t for t in [
            MCPTool(
                name="aemo_dispatch_price",
                description="Live 5-minute dispatch price from NEMWeb (public, no auth)",
                category="dispatch",
                endpoint="https://nemweb.com.au/Reports/Current/DispatchIS_Reports/",
                poll_interval_s=300,
                staleness_threshold_s=600,
            ),
            MCPTool(
                name="aemo_market_notices",
                description="Active AEMO market notices (LOR, outages, directions)",
                category="notices",
                endpoint="https://nemweb.com.au/Reports/Current/Market_Notice/",
                poll_interval_s=60,
                staleness_threshold_s=120,
            ),
            MCPTool(
                name="aemo_archive",
                description="NEMWeb archive for historical dispatch price backfill",
                category="archive",
                endpoint="https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/",
                poll_interval_s=3600,
                staleness_threshold_s=7200,
            ),
            MCPTool(
                name="hippograph_analogs",
                description="In-process HippoGraph PPR analog retrieval",
                category="analogs",
                endpoint="internal:hippograph",
                poll_interval_s=0,    # on-demand, not polled
                staleness_threshold_s=300,
            ),
            MCPTool(
                name="lnn_forecast",
                description="LTC neural network quantile forecast (in-process)",
                category="forecast",
                endpoint="internal:lnn_trainer",
                poll_interval_s=0,
                staleness_threshold_s=300,
            ),
            MCPTool(
                name="live_quantile_forecast",
                description="Live LEAR/QRA/LNN quantile forecast bands",
                category="forecast",
                endpoint="internal:live_forecast",
                poll_interval_s=300,
                staleness_threshold_s=600,
            ),
            MCPTool(
                name="aemo_predispatch",
                description="AEMO pre-dispatch RRP intervals",
                category="forecast",
                endpoint="https://nemweb.com.au/Reports/Current/PredispatchIS_Reports/",
                poll_interval_s=300,
                staleness_threshold_s=600,
            ),
            MCPTool(
                name="nem_news_rss",
                description="Public RSS/Atom energy market commentary filtered for NEM relevance",
                category="sentiment",
                endpoint="rss:configured",
                poll_interval_s=300,
                staleness_threshold_s=600,
            ),
            MCPTool(
                name="weather_consensus",
                description=(
                    "Read-only weather consensus from BOM observations, Open-Meteo, "
                    "and MET Norway for NEM-relevant demand/renewables context"
                ),
                category="weather",
                endpoint="multi:open_meteo,met_no,bom_observations",
                poll_interval_s=300,
                staleness_threshold_s=900,
            ),
        ]
    }
