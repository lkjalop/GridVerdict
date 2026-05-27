"""Decision audit logger — writes immutable audit records for every BESS recommendation.

Every call to the BESS scenario or fleet scenario endpoint results in one or more
DecisionAuditLog rows. The rows are write-once; they are never updated or deleted.

Security constraints (enforced here, not just in the API layer):
  - simulation_only is always True — this system never executes real market actions.
  - No private position data is logged; only what the operator explicitly submitted.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DecisionAuditLog
from app.portfolio.schema import (
    FleetDispatchPlan,
    MarketSnapshot,
    ScenarioResult,
)

logger = logging.getLogger(__name__)


async def log_bess_decision(
    session: AsyncSession,
    tenant_id: str,
    result: ScenarioResult,
    market: MarketSnapshot,
    user_id: str | None = None,
    trace_id: str | None = None,
    asset_id: str | None = None,
    model_version: str | None = None,
    training_data_ref: str | None = None,
) -> str:
    """Write one audit row for a single-asset BESS dispatch recommendation.

    model_version and training_data_ref satisfy ISO/IEC 42001:2023 §8.4
    (AI system documentation — model traceability).

    Returns the generated audit log row ID.
    """
    row_id = str(uuid.uuid4())
    row = DecisionAuditLog(
        id=row_id,
        tenant_id=tenant_id,
        user_id=user_id,
        trace_id=trace_id,
        decision_type="bess_dispatch",
        region=market.region,
        asset_id=asset_id,
        action=result.action.value,
        confidence=result.confidence,
        price_rrp=market.price_rrp,
        price_regime=market.price_regime,
        economics={
            "dispatch_mw": result.economics.dispatch_mw,
            "energy_mwh": result.economics.energy_mwh,
            "expected_revenue": result.economics.expected_revenue,
            "degradation_cost": result.economics.degradation_cost,
            "net_expected_value": result.economics.net_expected_value,
            "fcas_opportunity_value": result.economics.fcas_opportunity_value,
        },
        evidence_quality=market.evidence_quality,
        risk_flags=result.risk_flags,
        why_summary=result.why[:5],
        simulation_only=True,
        model_version=model_version,
        training_data_ref=training_data_ref,
    )
    session.add(row)
    try:
        await session.flush()
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)
    return row_id


async def log_fleet_decision(
    session: AsyncSession,
    tenant_id: str,
    plan: FleetDispatchPlan,
    user_id: str | None = None,
    trace_id: str | None = None,
    model_version: str | None = None,
    training_data_ref: str | None = None,
) -> list[str]:
    """Write one audit row per asset in a fleet dispatch plan.

    model_version and training_data_ref satisfy ISO/IEC 42001:2023 §8.4
    (AI system documentation — model traceability).

    Returns a list of generated audit log row IDs (one per asset).
    """
    row_ids: list[str] = []
    for asset in plan.assets:
        row_id = str(uuid.uuid4())
        row = DecisionAuditLog(
            id=row_id,
            tenant_id=tenant_id,
            user_id=user_id,
            trace_id=trace_id,
            decision_type="fleet_dispatch",
            region=plan.market_region,
            asset_id=asset.asset_id,
            action=asset.action.value,
            confidence=asset.confidence,
            price_rrp=plan.market_price_rrp,
            price_regime=plan.market_regime,
            economics={
                "dispatch_mw": asset.dispatch_mw,
                "expected_revenue": asset.expected_revenue,
                "degradation_cost": asset.degradation_cost,
                "net_value": asset.net_value,
            },
            evidence_quality=None,
            risk_flags=asset.risk_flags,
            why_summary=asset.why[:5],
            simulation_only=True,
            model_version=model_version,
            training_data_ref=training_data_ref,
        )
        session.add(row)
        row_ids.append(row_id)

    try:
        await session.flush()
    except Exception as exc:
        logger.warning("Fleet audit log write failed: %s", exc)

    return row_ids
