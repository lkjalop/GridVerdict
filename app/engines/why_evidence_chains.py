"""Causal evidence chain builder — infers WHY a price occurred.

Distinct from why_builder._build_causal_chain_steps (which audits data availability).
This module infers the actual causal path: price → marginal generator → fuel context
→ demand/weather driver → external factor.

Called via GET /api/v1/causal-chain?region=NSW1&valid_time=...&price=56.0
as a secondary enrichment after the main query returns.  Non-fatal throughout.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Utilisation threshold for "near full output" → likely marginal or infra-marginal
_UTIL_THRESHOLD = 0.88

# Human-readable interconnector labels (matches store.py _IC_LABELS)
_IC_LABELS: dict[str, str] = {
    "V-S-MNSP1": "Heywood (VIC↔SA)", "HYMAROO": "Heywood (VIC↔SA)",
    "N-Q-MNSP1": "QNI (NSW↔QLD)",    "TERRANORA": "Terranora (NSW↔QLD)",
    "V-N-MNSP1": "VIC–NSW",
    "T-V-MNSP1": "Basslink (TAS↔VIC)",
}


@dataclass
class CausalNode:
    step: str           # "price" | "marginal_fuel" | "demand_driver" | "external_factor"
    label: str          # display label
    value: str          # quantified value ("$56/MWh", "Gas CT", "+340 MW")
    tier: str           # "confirmed" | "inferred" | "probable" | "missing"
    detail: str         # one-sentence explanation
    source: str         # data source used
    arrow: str = "because"   # connector to next node


@dataclass
class CausalChain:
    region: str
    valid_time: str
    spot_price: float
    nodes: list[CausalNode] = field(default_factory=list)
    root_cause: str = ""          # single-sentence conclusion
    confidence: str = "low"       # "high" | "medium" | "low"
    data_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "valid_time": self.valid_time,
            "spot_price": self.spot_price,
            "nodes": [
                {
                    "step": n.step, "label": n.label, "value": n.value,
                    "tier": n.tier, "detail": n.detail, "source": n.source,
                    "arrow": n.arrow,
                }
                for n in self.nodes
            ],
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "data_gaps": self.data_gaps,
        }


async def build_causal_chain(
    region: str,
    valid_time: datetime,
    spot_price: float,
    session,
) -> CausalChain:
    """Build a causal chain from spot price back to root cause.

    Uses a 2-table join heuristic for marginal generator identification.
    Non-fatal: returns a partial chain when data is unavailable.
    """
    chain = CausalChain(
        region=region,
        valid_time=valid_time.isoformat(),
        spot_price=spot_price,
    )

    # Node 1 — always available
    chain.nodes.append(CausalNode(
        step="price",
        label="Dispatch price",
        value=f"${spot_price:.0f}/MWh",
        tier="confirmed",
        detail=f"{region} spot price at {valid_time.strftime('%H:%M UTC')}",
        source="AEMO_DISPATCH_PRICE",
        arrow="set by",
    ))

    # Node 2 — marginal generator (heuristic)
    marginal = await _find_marginal_generator(region, valid_time, spot_price, session)
    if marginal:
        chain.nodes.append(CausalNode(
            step="marginal_fuel",
            label=f"Marginal unit: {marginal['fuel_type'] or 'unknown'} generator",
            value=f"{marginal['duid']} — {marginal['station_name'] or marginal['duid']}",
            tier="inferred",
            detail=(
                f"{marginal['duid']} was at {marginal['utilisation_pct']:.0f}% output "
                f"({marginal['total_cleared_mw']:.0f}/{marginal['availability_mw']:.0f} MW); "
                f"highest offer band ≤${spot_price:.0f}/MWh → likely price-setter"
            ),
            source="DISPATCH_UNIT_SOLUTION × BIDDAYOFFER (heuristic)",
            arrow="because",
        ))
    else:
        chain.data_gaps.append("Marginal generator — unit dispatch data not yet available for this interval")

    # Node 3 — demand / headroom context
    demand_node = await _build_demand_node(region, valid_time, session)
    if demand_node:
        chain.nodes.append(demand_node)
    else:
        chain.data_gaps.append("Demand deviation — no historical comparison available")

    # Node 4 — interconnector constraint if relevant
    ic_node = await _build_interconnector_node(region, valid_time, session)
    if ic_node:
        chain.nodes.append(ic_node)

    # Node 5 — weather / external factor
    wx_node = await _build_weather_node(region, valid_time, session)
    if wx_node:
        chain.nodes.append(wx_node)
    else:
        chain.data_gaps.append("Weather observation — not yet persisted for this interval")

    # Synthesise root cause + confidence
    chain.root_cause, chain.confidence = _synthesise(chain.nodes, spot_price, region)
    return chain


# ── Private helpers ───────────────────────────────────────────────────

async def _find_marginal_generator(
    region: str,
    valid_time: datetime,
    spot_price: float,
    session,
) -> dict[str, Any] | None:
    """Heuristic: find the likely price-setting unit at this interval.

    Heuristic: unit with utilisation > 88% AND highest BIDDAYOFFER price band ≤ spot.
    Falls back to highest-utilisation unit with any matching bid if BIDDAYOFFER
    data isn't available for this interval (common for live events).
    """
    try:
        from sqlalchemy import select, text, func
        from app.db.models import UnitDispatchEvent, GeneratorUnit, BidOffer

        # Find near-fully-dispatched units in this region at this time
        vt_start = valid_time - timedelta(minutes=3)
        vt_end   = valid_time + timedelta(minutes=3)

        result = await session.execute(
            select(
                UnitDispatchEvent.duid,
                UnitDispatchEvent.total_cleared_mw,
                UnitDispatchEvent.availability_mw,
                GeneratorUnit.station_name,
                GeneratorUnit.fuel_type,
                GeneratorUnit.max_capacity_mw,
            )
            .join(GeneratorUnit, UnitDispatchEvent.duid == GeneratorUnit.duid)
            .where(
                UnitDispatchEvent.region == region,
                UnitDispatchEvent.valid_time >= vt_start,
                UnitDispatchEvent.valid_time <= vt_end,
                UnitDispatchEvent.total_cleared_mw.isnot(None),
                UnitDispatchEvent.availability_mw > 10,
                GeneratorUnit.fuel_type.isnot(None),
                GeneratorUnit.fuel_type.notin_(["battery", "demand_response"]),
            )
            .order_by(
                (UnitDispatchEvent.total_cleared_mw /
                 UnitDispatchEvent.availability_mw).desc()
            )
            .limit(30)
        )
        units = result.fetchall()
        if not units:
            return None

        # Filter to high-utilisation units
        high_util = [
            u for u in units
            if u.availability_mw > 0
            and (u.total_cleared_mw / u.availability_mw) >= _UTIL_THRESHOLD
        ]
        if not high_util:
            high_util = units[:5]   # fallback: top 5 by utilisation

        # Try to cross-reference with BIDDAYOFFER price bands
        day_start = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        best_unit = None
        best_band_price = -1.0

        duid_list = [u.duid for u in high_util[:10]]
        if duid_list:
            bid_result = await session.execute(
                select(BidOffer.duid, BidOffer.price_bands)
                .where(
                    BidOffer.duid.in_(duid_list),
                    BidOffer.settlement_date >= day_start,
                    BidOffer.settlement_date < day_end,
                    BidOffer.source == "BIDDAYOFFER",
                )
            )
            bid_map: dict[str, dict] = {}
            for row in bid_result.fetchall():
                if row.duid not in bid_map:
                    bid_map[row.duid] = row.price_bands or {}

            for u in high_util:
                bands = bid_map.get(u.duid, {})
                if bands:
                    # Find highest band price that is ≤ spot_price
                    qualifying = [
                        float(p) for p in bands.values()
                        if p is not None and float(p) <= spot_price * 1.05  # 5% tolerance
                    ]
                    if qualifying:
                        max_qual = max(qualifying)
                        if max_qual > best_band_price:
                            best_band_price = max_qual
                            best_unit = u

        if best_unit is None and high_util:
            best_unit = high_util[0]

        if best_unit is None:
            return None

        util_pct = (
            best_unit.total_cleared_mw / best_unit.availability_mw * 100
            if best_unit.availability_mw > 0 else 0.0
        )
        return {
            "duid": best_unit.duid,
            "station_name": best_unit.station_name,
            "fuel_type": best_unit.fuel_type,
            "total_cleared_mw": best_unit.total_cleared_mw,
            "availability_mw": best_unit.availability_mw,
            "utilisation_pct": util_pct,
            "bid_price": best_band_price if best_band_price > 0 else None,
        }
    except Exception as exc:
        logger.debug("Marginal generator heuristic failed: %s", exc)
        return None


async def _build_demand_node(
    region: str,
    valid_time: datetime,
    session,
) -> CausalNode | None:
    """Compare demand at valid_time to the 30-day average for that hour."""
    try:
        from sqlalchemy import select, func, extract
        from app.db.models import MarketEvent

        # Get current demand
        vt_start = valid_time - timedelta(minutes=3)
        vt_end   = valid_time + timedelta(minutes=3)
        curr_row = await session.execute(
            select(MarketEvent.demand_mw, MarketEvent.availability_mw)
            .where(
                MarketEvent.region == region,
                MarketEvent.source == "AEMO_DISPATCH_PRICE",
                MarketEvent.valid_time >= vt_start,
                MarketEvent.valid_time <= vt_end,
            )
            .limit(1)
        )
        curr = curr_row.fetchone()
        if curr is None or curr.demand_mw is None:
            return None

        # 30-day average for this hour
        hour = valid_time.hour
        lookback = valid_time - timedelta(days=30)
        avg_row = await session.execute(
            select(func.avg(MarketEvent.demand_mw).label("avg_demand"))
            .where(
                MarketEvent.region == region,
                MarketEvent.source == "AEMO_DISPATCH_PRICE",
                MarketEvent.valid_time >= lookback,
                MarketEvent.valid_time < valid_time,
                extract("hour", MarketEvent.valid_time) == hour,
            )
        )
        avg_result = avg_row.fetchone()
        avg_demand = float(avg_result.avg_demand) if avg_result and avg_result.avg_demand else None

        demand_mw = float(curr.demand_mw)
        avail_mw = float(curr.availability_mw or demand_mw)
        headroom = max(avail_mw - demand_mw, 0.0)

        if avg_demand:
            deviation = demand_mw - avg_demand
            dev_str = f"{'+' if deviation >= 0 else ''}{deviation:.0f} MW vs 30-day {hour:02d}:xx avg"
            tier = "confirmed" if abs(deviation) > 200 else "inferred"
        else:
            dev_str = f"no historical average available"
            tier = "probable"

        return CausalNode(
            step="demand_driver",
            label="Grid demand",
            value=f"{demand_mw:,.0f} MW demand ({headroom:.0f} MW headroom)",
            tier=tier,
            detail=(
                f"{region} demand {demand_mw:,.0f} MW with {headroom:.0f} MW headroom remaining. "
                f"{dev_str}."
            ),
            source="AEMO_DISPATCH_PRICE (30-day hour average)",
            arrow="driven by",
        )
    except Exception as exc:
        logger.debug("Demand node build failed: %s", exc)
        return None


async def _build_interconnector_node(
    region: str,
    valid_time: datetime,
    session,
) -> CausalNode | None:
    """Add interconnector node when a link is near its capacity limit."""
    try:
        from sqlalchemy import select
        from app.db.models import MarketDriverEvent

        vt_start = valid_time - timedelta(minutes=8)
        vt_end   = valid_time + timedelta(minutes=8)

        ic_result = await session.execute(
            select(
                MarketDriverEvent.element_id,
                MarketDriverEvent.values,
            )
            .where(
                MarketDriverEvent.driver_type == "interconnector",
                MarketDriverEvent.valid_time >= vt_start,
                MarketDriverEvent.valid_time <= vt_end,
            )
        )
        rows = ic_result.fetchall()
        if not rows:
            return None

        # Find the most congested interconnector relevant to this region
        region_ics = {
            "NSW1": ["V-N-MNSP1", "N-Q-MNSP1", "N-Q-MNSP2", "TERRANORA"],
            "VIC1": ["V-N-MNSP1", "V-S-MNSP1", "T-V-MNSP1"],
            "QLD1": ["N-Q-MNSP1", "N-Q-MNSP2", "TERRANORA"],
            "SA1":  ["V-S-MNSP1", "HEYWOOD"],
            "TAS1": ["T-V-MNSP1", "BASSLINK"],
        }
        relevant = {ic.upper() for ic in region_ics.get(region, [])}

        best: dict | None = None
        best_ratio = 0.0
        for row in rows:
            ic_id = row.element_id.upper()
            if ic_id not in relevant:
                continue
            vals = row.values or {}
            flow = vals.get("metered_mw_flow") or vals.get("mw_flow") or 0.0
            limit = max(
                abs(vals.get("export_limit") or 0),
                abs(vals.get("import_limit") or 0),
            )
            if limit > 0:
                ratio = abs(flow) / limit
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = {"ic_id": ic_id, "flow": flow, "limit": limit, "ratio": ratio}

        if best is None or best_ratio < 0.75:
            return None

        label = _IC_LABELS.get(best["ic_id"], best["ic_id"])
        direction = "exporting" if best["flow"] > 0 else "importing"
        severity = "at capacity limit" if best_ratio > 0.95 else f"{best_ratio*100:.0f}% of limit"

        return CausalNode(
            step="interconnector",
            label=f"Interconnector constraint: {label}",
            value=f"{abs(best['flow']):.0f} MW flow ({severity})",
            tier="confirmed" if best_ratio > 0.92 else "inferred",
            detail=(
                f"{label} is {direction} {abs(best['flow']):.0f} MW "
                f"({severity}). When links are full, {region} cannot import cheaper power."
            ),
            source="DISPATCHINTERCONNECTORRES",
            arrow="constrained by",
        )
    except Exception as exc:
        logger.debug("Interconnector node failed: %s", exc)
        return None


async def _build_weather_node(
    region: str,
    valid_time: datetime,
    session,
) -> CausalNode | None:
    """Add weather observation node when temperature deviation is material."""
    try:
        from sqlalchemy import select
        from app.db.models import WeatherObservation

        window_start = valid_time - timedelta(minutes=40)
        window_end   = valid_time + timedelta(minutes=40)

        wx_result = await session.execute(
            select(WeatherObservation)
            .where(
                WeatherObservation.region == region,
                WeatherObservation.observed_at >= window_start,
                WeatherObservation.observed_at <= window_end,
            )
            .order_by(WeatherObservation.observed_at)
            .limit(1)
        )
        wx = wx_result.scalar_one_or_none()
        if wx is None or wx.temperature_c is None:
            return None

        temp = wx.temperature_c
        dev = wx.temp_deviation_c
        cloud = wx.cloud_cover_pct
        wind = wx.wind_speed_kmh

        parts: list[str] = [f"{temp:.1f}°C"]
        if dev is not None and abs(dev) >= 2.0:
            parts.append(f"{'+' if dev > 0 else ''}{dev:.1f}°C vs seasonal norm")
        if cloud is not None and cloud > 50:
            parts.append(f"cloud {cloud:.0f}%")
        if wind is not None:
            parts.append(f"wind {wind:.0f} km/h")

        weather_str = " · ".join(parts)
        tier = "confirmed" if (dev is not None and abs(dev) >= 3.0) else "inferred"

        # Build the narrative
        if dev is not None and dev > 3.0:
            detail = f"Temperature {dev:.1f}°C above norm drives additional cooling/heating demand — increases residual load on thermal generators."
        elif dev is not None and dev < -3.0:
            detail = f"Temperature {abs(dev):.1f}°C below norm reduces demand — increases renewable share and lowers pressure on gas peakers."
        elif cloud is not None and cloud > 65:
            detail = f"High cloud cover ({cloud:.0f}%) reduces solar output — increases residual demand on dispatchable sources."
        else:
            detail = f"Weather at dispatch time: {weather_str}. No extreme deviation detected."

        return CausalNode(
            step="external_factor",
            label="Weather at dispatch",
            value=weather_str,
            tier=tier,
            detail=detail,
            source="BOM/Open-Meteo weather consensus",
            arrow="influenced by",
        )
    except Exception as exc:
        logger.debug("Weather node failed: %s", exc)
        return None


def _synthesise(nodes: list[CausalNode], spot_price: float, region: str) -> tuple[str, str]:
    """Build a one-sentence root cause and confidence from the chain nodes."""
    confirmed = sum(1 for n in nodes if n.tier == "confirmed")
    inferred  = sum(1 for n in nodes if n.tier == "inferred")
    total = len(nodes)

    marginal_node = next((n for n in nodes if n.step == "marginal_fuel"), None)
    weather_node  = next((n for n in nodes if n.step == "external_factor"), None)
    ic_node       = next((n for n in nodes if n.step == "interconnector"), None)
    demand_node   = next((n for n in nodes if n.step == "demand_driver"), None)

    # Build the conclusion sentence
    parts: list[str] = []
    if spot_price > 300:
        parts.append(f"${spot_price:.0f}/MWh price spike in {region}")
    else:
        parts.append(f"${spot_price:.0f}/MWh in {region}")

    if marginal_node:
        fuel = marginal_node.value.split("—")[0].strip() if "—" in marginal_node.value else marginal_node.value
        parts.append(f"set by {marginal_node.label.split(':')[-1].strip()}")
    if ic_node and ic_node.tier in ("confirmed", "inferred"):
        parts.append(f"with {ic_node.label.split(':')[-1].strip()} constraining imports")
    if weather_node and weather_node.tier == "confirmed":
        parts.append(f"driven by {weather_node.detail.split('—')[0].strip().lower()}")
    elif demand_node and demand_node.tier == "confirmed":
        parts.append("above-average demand")

    root = ". ".join(parts[:3]) + "." if parts else f"${spot_price:.0f}/MWh in {region}."

    # Confidence
    if confirmed >= 3:
        conf = "high"
    elif confirmed + inferred >= 3:
        conf = "medium"
    else:
        conf = "low"

    return root, conf
