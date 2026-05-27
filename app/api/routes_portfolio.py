"""Portfolio scenario API — BESS dispatch recommendation.

POST /api/portfolio/bess/scenario
  Body: BessScenarioRequest (position + market snapshot)
  Returns: ScenarioResult

All responses are simulation-only. No real market actions are executed.
The market snapshot can be populated manually by the operator or via
the /api/market/live endpoint values.

GET /api/portfolio/bess/market-prefill?region=NSW1
  Returns a MarketSnapshot pre-populated from live market data so the
  frontend can auto-fill the market context fields without the operator
  having to type current values manually.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Response

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db
from app.portfolio.schema import (
    BessScenarioRequest,
    FleetDispatchPlan,
    FleetScenarioRequest,
    ScenarioResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]


@router.post("/bess/scenario")
async def bess_scenario(
    body: BessScenarioRequest,
    user: TokenPayload = Depends(get_current_user),
    session=Depends(get_db),
) -> ScenarioResult:
    """Run a BESS dispatch scenario and return a recommendation.

    Accepts the operator's battery position and the current market context.
    Returns a deterministic recommendation with full economic breakdown,
    confidence level, rationale, and missing-data checklist.

    Every call is logged to the decision audit log (simulation_only=True).
    """
    from app.portfolio.bess_engine import compute_economics
    from app.portfolio.dispatch_policy import evaluate
    from app.audit.audit_logger import log_bess_decision
    from app.engines.forecasting.model_registry import get_active_version, get_training_data_ref

    economics = compute_economics(body.position, body.market)
    result = evaluate(body.position, body.market, economics)
    logger.info(
        "BESS scenario %s → %s (conf:%s, net:$%.2f)",
        body.market.region,
        result.action.value,
        result.confidence,
        economics.net_expected_value,
    )
    try:
        await log_bess_decision(
            session,
            tenant_id=getattr(user, "tenant_id", "system"),
            result=result,
            market=body.market,
            user_id=getattr(user, "sub", None),
            model_version=get_active_version("bess-policy"),
            training_data_ref=get_training_data_ref("bess-policy"),
        )
        await session.commit()
    except Exception as exc:
        logger.warning("Audit log failed for bess_scenario: %s", exc)
    return result


@router.post("/bess/fleet/scenario")
async def bess_fleet_scenario(
    body: FleetScenarioRequest,
    user: TokenPayload = Depends(get_current_user),
    session=Depends(get_db),
) -> FleetDispatchPlan:
    """Run a coordinated multi-asset BESS fleet dispatch scenario.

    Evaluates each asset independently against the shared market snapshot,
    applies fleet-level export cap allocation (ranked by net value), and
    returns an aggregated dispatch plan with per-asset recommendations.

    Fleet export cap (fleet_export_limit_mw): when set, assets are ranked by
    net expected value and allocated in order until the limit is reached.
    Lower-ranked assets are held to respect the grid connection cap.

    Every call is logged to the decision audit log (simulation_only=True).
    """
    from app.portfolio.fleet_coordinator import evaluate_fleet
    from app.audit.audit_logger import log_fleet_decision
    from app.engines.forecasting.model_registry import get_active_version, get_training_data_ref

    plan = evaluate_fleet(body)
    logger.info(
        "BESS fleet scenario %s → %d assets, total %.1f MW, net $%.2f (conf:%s)",
        body.market.region,
        len(plan.assets),
        plan.total_dispatch_mw,
        plan.total_net_value,
        plan.fleet_confidence,
    )
    try:
        await log_fleet_decision(
            session,
            tenant_id=getattr(user, "tenant_id", "system"),
            plan=plan,
            user_id=getattr(user, "sub", None),
            model_version=get_active_version("fleet-policy"),
            training_data_ref=get_training_data_ref("fleet-policy"),
        )
        await session.commit()
    except Exception as exc:
        logger.warning("Audit log failed for bess_fleet_scenario: %s", exc)
    return plan


@router.get("/audit/export")
async def audit_export(
    start: str | None = Query(default=None, description="ISO8601 start datetime (inclusive)"),
    end: str | None = Query(default=None, description="ISO8601 end datetime (inclusive)"),
    region: str | None = Query(default=None, description="NEM region filter (e.g. NSW1)"),
    decision_type: str | None = Query(default=None, description="bess_dispatch | fleet_dispatch"),
    format: str = Query(default="json", description="json | csv"),
    limit: int = Query(default=500, ge=1, le=5000),
    user: TokenPayload = Depends(get_current_user),
    session=Depends(get_db),
) -> Response:
    """Export decision audit log for regulatory review.

    All records are simulation_only=True. Returns JSON (default) or CSV.
    CSV is suitable for direct import into regulatory reporting tools.

    Date filtering uses ISO8601 strings (e.g. "2024-01-01T00:00:00Z").
    """
    from datetime import datetime, timezone
    from app.audit.export import query_audit_log, rows_to_json, rows_to_csv

    def _parse_dt(s: str | None) -> datetime | None:
        if s is None:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    tenant_id = getattr(user, "tenant_id", "system")

    rows = await query_audit_log(
        session, tenant_id=tenant_id,
        start_dt=start_dt, end_dt=end_dt,
        region=region, decision_type=decision_type,
        limit=limit,
    )

    if format.lower() == "csv":
        content = rows_to_csv(rows)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=decision_audit_log.csv"},
        )

    data = rows_to_json(rows)
    import json as _json
    return Response(
        content=_json.dumps({"count": len(data), "records": data}, default=str),
        media_type="application/json",
    )


@router.get("/bess/market-prefill")
async def bess_market_prefill(
    region: str = Query(default="NSW1", description="NEM region code"),
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """Return live market data pre-formatted as a MarketSnapshot for the BESS panel.

    The frontend uses this to auto-fill the market context fields so the
    operator only needs to enter their asset position (SOC, capacity, etc).
    """
    from fastapi import HTTPException, status
    from app.data.aemo_live_client import get_aemo_client
    from app.data.cache import get_cache
    from app.data.event_bus import publish

    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {_SUPPORTED_REGIONS}",
        )

    try:
        cache = get_cache()
        raw = await cache.get("dispatch_snapshot")
        if raw is None:
            client = get_aemo_client()
            snapshot = await client.fetch_latest_snapshot()
            raw = snapshot.to_dict()
        else:
            from app.data.aemo_live_client import LiveMarketSnapshot
            snapshot = LiveMarketSnapshot.from_dict(raw)

        dp = snapshot.get(region)
        if dp is None:
            return _empty_prefill(region)

        from domain.nem.adapter import classify_regime
        regime = classify_regime(dp.price_rrp, region)
        headroom = max(dp.availability_mw - dp.demand_mw, 0.0)

        return {
            "region": region,
            "price_rrp": round(dp.price_rrp, 2),
            "price_regime": regime,
            "headroom_mw": round(headroom, 1),
            "forecast_direction": None,     # requires LNN — not auto-populated
            "fcas_raise_6sec_rrp": None,    # requires FCAS DB — not auto-populated
            "fcas_raise_reg_rrp": None,
            "rebid_evidence_tier": None,
            "outage_evidence_tier": None,
            "evidence_quality": "plausible",
            "data_source": "live_dispatch",
            "note": (
                "Market context auto-filled from live dispatch data. "
                "FCAS prices and forecast direction require additional queries."
            ),
        }

    except Exception as exc:
        logger.warning("BESS market prefill failed for %s: %s", region, exc)
        return _empty_prefill(region)


def _empty_prefill(region: str) -> dict:
    return {
        "region": region,
        "price_rrp": None,
        "price_regime": "unknown",
        "headroom_mw": None,
        "forecast_direction": None,
        "fcas_raise_6sec_rrp": None,
        "fcas_raise_reg_rrp": None,
        "rebid_evidence_tier": None,
        "outage_evidence_tier": None,
        "evidence_quality": "insufficient",
        "data_source": "unavailable",
        "note": "Live market data unavailable — enter values manually.",
    }
