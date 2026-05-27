"""Market Incident Brief — structured situational report for a NEM region.

Assembles a point-in-time market brief from all available evidence layers:
  - Current market state (price, demand, headroom, regime)
  - Driver evidence (constraints, interconnectors, unit outages)
  - AEMO notices (market, system, infrastructure)
  - Historical analogs (similar price events)
  - Forecast model state (LEAR / QRA / LNN)
  - BESS portfolio implication (charge/dispatch direction)
  - Source freshness summary
  - Security + claim verification
  - Trace provenance (trace_id, model versions)
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

async def build_incident_brief(
    session: AsyncSession,
    region: str,
    anchor_time: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a full market incident brief for `region` at `anchor_time`."""
    t0 = time.perf_counter()
    now = datetime.now(timezone.utc)
    anchor = anchor_time or now
    trace_id = f"brief-{uuid.uuid4().hex[:12]}"

    brief: dict[str, Any] = {
        "trace_id": trace_id,
        "region": region,
        "generated_at": now.isoformat(),
        "anchor_time": anchor.isoformat(),
    }

    # ── 1. Market state ───────────────────────────────────────────────────────
    market = await _market_state(session, region, anchor)
    brief["market"] = market

    # ── 2. Driver evidence ────────────────────────────────────────────────────
    drivers = await _driver_evidence(session, region, anchor)
    brief["drivers"] = drivers

    # ── 3. AEMO notices ───────────────────────────────────────────────────────
    notices = await _aemo_notices(region)
    brief["notices"] = notices

    # ── 4. Historical analogs ─────────────────────────────────────────────────
    analogs = await _historical_analogs(session, region, market)
    brief["analogs"] = analogs

    # ── 5. Forecast model state ───────────────────────────────────────────────
    forecast = await _forecast_state(region)
    brief["forecast"] = forecast

    # ── 6. BESS portfolio implication ─────────────────────────────────────────
    bess = _bess_implication(market)
    brief["bess"] = bess

    # ── 7. Source freshness ───────────────────────────────────────────────────
    freshness = _source_freshness(market, notices, forecast, now)
    brief["source_freshness"] = freshness

    # ── 8. Claim verification ─────────────────────────────────────────────────
    verification = _claim_verification(market, drivers)
    brief["verification"] = verification

    # ── 9. Model provenance ───────────────────────────────────────────────────
    provenance = _model_provenance()
    brief["provenance"] = provenance

    brief["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return brief


# ── Section builders ──────────────────────────────────────────────────────────

async def _market_state(session: AsyncSession, region: str, anchor: datetime) -> dict:
    """Latest dispatch snapshot for this region near anchor_time."""
    from sqlalchemy import select, text as sql_text
    from app.db.models import MarketEvent

    try:
        cutoff = anchor - timedelta(hours=1)
        result = await session.execute(
            select(MarketEvent)
            .where(
                MarketEvent.region == region,
                MarketEvent.source == "AEMO_DISPATCH_PRICE",
                MarketEvent.valid_time >= cutoff,
                MarketEvent.valid_time <= anchor,
            )
            .order_by(MarketEvent.valid_time.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return {
                "status": "unavailable",
                "price_rrp": None,
                "demand_mw": None,
                "availability_mw": None,
                "headroom_mw": None,
                "regime": "unknown",
                "valid_time": None,
                "age_seconds": None,
            }

        now = datetime.now(timezone.utc)
        vt = row.valid_time
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=timezone.utc)
        headroom = max((row.availability_mw or 0) - (row.demand_mw or 0), 0.0)

        from domain.nem.adapter import classify_regime
        regime = classify_regime(row.price_rrp, region)

        return {
            "status": "ok",
            "price_rrp": round(row.price_rrp, 2),
            "demand_mw": round(row.demand_mw, 1),
            "availability_mw": round(row.availability_mw, 1),
            "headroom_mw": round(headroom, 1),
            "regime": regime,
            "valid_time": vt.isoformat(),
            "age_seconds": int((now - vt).total_seconds()),
        }
    except Exception as exc:
        logger.debug("Market state unavailable: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _driver_evidence(session: AsyncSession, region: str, anchor: datetime) -> dict:
    """Constraints, interconnectors, and unit dispatch near the anchor interval."""
    window_start = anchor - timedelta(minutes=30)
    window_end = anchor + timedelta(minutes=5)

    try:
        from app.engines.driver_attribution import retrieve_market_drivers
        events = await retrieve_market_drivers(session, region, anchor)
    except Exception as exc:
        logger.debug("Driver attribution unavailable: %s", exc)
        events = []

    confirmed = [e for e in events if e.get("tier") == "confirmed"]
    supported = [e for e in events if e.get("tier") == "supported"]
    plausible  = [e for e in events if e.get("tier") not in ("confirmed", "supported")]

    return {
        "window": {"from": window_start.isoformat(), "to": window_end.isoformat()},
        "confirmed_count": len(confirmed),
        "supported_count": len(supported),
        "plausible_count": len(plausible),
        "events": events[:20],
    }


async def _aemo_notices(region: str) -> dict:
    """Active AEMO notices for this region from cache."""
    try:
        from app.data.cache import get_cache
        cache = get_cache()
        notices = await cache.get(f"notices_{region}") or []
        fetched_at = await cache.get(f"notices_{region}_fetched_at")
        return {
            "count": len(notices),
            "fetched_at": fetched_at,
            "items": notices[:10],
        }
    except Exception as exc:
        logger.debug("Notices unavailable: %s", exc)
        return {"count": 0, "fetched_at": None, "items": [], "error": str(exc)}


async def _historical_analogs(session: AsyncSession, region: str, market: dict) -> dict:
    """Similar historical price events retrieved from HippoGraph."""
    price = market.get("price_rrp")
    demand = market.get("demand_mw")
    if price is None:
        return {"count": 0, "items": []}

    try:
        from app.engines.analog_retriever import retrieve_analogs
        analogs = await retrieve_analogs(
            session, region, price_rrp=price, demand_mw=demand, top_k=5
        )
        return {
            "count": len(analogs),
            "items": analogs[:5],
        }
    except Exception as exc:
        logger.debug("Analogs unavailable: %s", exc)
        return {"count": 0, "items": [], "note": "HippoGraph index not populated"}


async def _forecast_state(region: str) -> dict:
    """Current LEAR/QRA/LNN forecast bands for the next 6 intervals."""
    try:
        from app.mcp.router import call_tool
        result = await call_tool(
            "live_quantile_forecast",
            region=region,
            lookback_days=14,
            horizon_intervals=6,
        )
        if isinstance(result, dict):
            models = result.get("models", []) or []
            intervals = result.get("intervals", []) or []
            available = bool(models and intervals)
            return {
                "available": available,
                "models": models,
                "intervals": intervals[:6],
                "generated_at": result.get("generated_at"),
                "note": None if available else "Forecast tool returned no model intervals",
            }
        return {"available": False, "note": "Forecast tool returned non-dict"}
    except Exception as exc:
        logger.debug("Forecast unavailable: %s", exc)
        return {"available": False, "note": str(exc)}


def _bess_implication(market: dict) -> dict:
    """Simple rule-based BESS dispatch direction given current market state."""
    price = market.get("price_rrp")
    regime = market.get("regime", "unknown")
    headroom = market.get("headroom_mw")

    if price is None:
        return {"direction": "unknown", "reason": "no market data", "confidence": "low", "simulation_only": True}

    if price >= 300:
        direction = "dispatch"
        reason = f"Price spike (${price:.0f}/MWh) — discharge to capture high revenue"
        confidence = "high" if price >= 1000 else "medium"
    elif price < 0:
        direction = "charge"
        reason = f"Negative price (${price:.0f}/MWh) — absorb excess to capture rebate"
        confidence = "high"
    elif headroom is not None and headroom < 500:
        direction = "standby"
        reason = f"Low headroom ({headroom:.0f} MW) — reserve capacity for imminent spike"
        confidence = "medium"
    else:
        direction = "charge"
        reason = f"Normal conditions (${price:.0f}/MWh, {regime}) — opportunistic charging"
        confidence = "low"

    return {
        "direction": direction,
        "reason": reason,
        "confidence": confidence,
        "price_rrp": price,
        "regime": regime,
        "simulation_only": True,
    }


def _source_freshness(market: dict, notices: dict, forecast: dict, now: datetime) -> dict:
    """Summarise data freshness across all evidence layers."""
    dispatch_age = market.get("age_seconds")
    notices_fetched = notices.get("fetched_at")

    notices_age = None
    if notices_fetched:
        try:
            ft = datetime.fromisoformat(notices_fetched)
            if ft.tzinfo is None:
                ft = ft.replace(tzinfo=timezone.utc)
            notices_age = int((now - ft).total_seconds())
        except Exception:
            pass

    return {
        "dispatch": {
            "age_seconds": dispatch_age,
            "status": (
                "fresh" if dispatch_age is not None and dispatch_age < 300
                else "stale" if dispatch_age is not None and dispatch_age < 900
                else "unavailable"
            ),
        },
        "notices": {
            "age_seconds": notices_age,
            "status": (
                "fresh" if notices_age is not None and notices_age < 120
                else "stale" if notices_age is not None and notices_age < 600
                else "unavailable"
            ),
        },
        "forecast": {
            "available": forecast.get("available", False),
        },
    }


def _claim_verification(market: dict, drivers: dict) -> dict:
    """Basic claim verification: are the key evidence claims internally consistent?"""
    checks = []
    price = market.get("price_rrp")
    headroom = market.get("headroom_mw")
    confirmed = drivers.get("confirmed_count", 0)
    total_drivers = drivers.get("confirmed_count", 0) + drivers.get("supported_count", 0)

    if price is not None and price >= 300 and headroom is not None and headroom > 2000:
        checks.append({
            "claim": "high price with large headroom",
            "verdict": "anomaly",
            "note": "Price spike ($%.0f) but headroom is %.0f MW — check constraints" % (price, headroom),
        })

    if price is not None and price >= 300 and total_drivers == 0:
        checks.append({
            "claim": "high price but no driver evidence",
            "verdict": "insufficient_data",
            "note": "No constraint/interconnector events found — driver attribution incomplete",
        })

    if not checks:
        checks.append({
            "claim": "market state consistency",
            "verdict": "ok",
            "note": "No internal contradictions detected",
        })

    return {
        "checks": checks,
        "overall": "ok" if all(c["verdict"] == "ok" for c in checks) else "warnings",
    }


def _model_provenance() -> dict:
    """Return current model registry entries for provenance record."""
    try:
        from app.engines.forecasting.model_registry import get_all_models
        models = get_all_models()
        return {
            "models": [
                {
                    "name": m.get("name") or m.get("model_name"),
                    "version": m.get("version"),
                    "training_data_ref": m.get("training_data_ref"),
                    "registered_at": m.get("registered_at"),
                }
                for m in models
            ]
        }
    except Exception as exc:
        logger.debug("Model registry unavailable: %s", exc)
        return {"models": [], "note": str(exc)}
