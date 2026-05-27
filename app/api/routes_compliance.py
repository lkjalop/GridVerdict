"""Compliance API routes — ISO/IEC 42001 and ISO/IEC 27001 evidence artefacts.

GET /api/compliance/42001/risk-register     — AI risk register (ISO 42001 §6.1 / §8.4)
GET /api/compliance/42001/model-registry    — Registered AI models with versions + provenance
GET /api/compliance/27001/controls          — Annex A control mapping from SecurityObserver signals
GET /api/compliance/27001/observer-events   — Persisted security observer events (DB-backed)
GET /api/compliance/aescsf/self-assessment  — AESCSF v2.0 self-assessment (maturity levels)

All endpoints require authentication. Responses are plain JSON — suitable for
import into GRC tools (Vanta, Drata, Tugboat Logic) or regulatory submissions.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compliance", tags=["compliance"])


# ── ISO/IEC 42001 — AI Management ─────────────────────────────────────────────

@router.get("/42001/risk-register")
async def get_ai_risk_register(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the ISO/IEC 42001:2023 AI risk register.

    Documents AI-specific risks, inherent ratings, implemented controls,
    and residual risk classifications. Suitable for regulatory submissions
    and enterprise security questionnaires.
    """
    from app.compliance.ai_risk_register import get_risk_register, get_risk_summary
    return {
        "summary": get_risk_summary(),
        "risks": get_risk_register(),
    }


@router.get("/42001/model-registry")
async def get_model_registry(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Return all registered AI models with version and training data provenance.

    Satisfies ISO/IEC 42001:2023 §8.4 (AI system documentation).
    Dynamic entries reflect the most recent model fit; static entries cover
    rule-based components that do not train.
    """
    from app.engines.forecasting.model_registry import get_all_models
    models = get_all_models()
    return {
        "standard": "ISO/IEC 42001:2023 §8.4",
        "model_count": len(models),
        "models": models,
        "note": (
            "Dynamic model entries (LEAR, QRA, LNN, MetaEnsemble) are registered "
            "at fit time and reset on server restart. Static entries (bess-policy, "
            "fleet-policy) are always present — they are rule-based and do not train."
        ),
    }


# ── ISO/IEC 27001 — Information Security ──────────────────────────────────────

@router.get("/27001/controls")
async def get_27001_controls(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the ISO/IEC 27001:2022 Annex A control coverage map.

    Shows which Annex A controls are evidenced by the SecurityObserver signal
    mapping and which signals cover each control.
    """
    from app.compliance.iso27001_controls import get_all_controls, get_coverage_summary
    return {
        "coverage_summary": get_coverage_summary(),
        "controls": get_all_controls(),
    }


@router.get("/27001/observer-events")
async def get_observer_events(
    start: str | None = Query(default=None, description="ISO8601 start datetime"),
    end: str | None = Query(default=None, description="ISO8601 end datetime"),
    verdict: str | None = Query(default=None, description="pass | warn | halt"),
    control_ref: str | None = Query(default=None, description="Annex A control ref filter (e.g. A.8.28)"),
    limit: int = Query(default=200, ge=1, le=2000),
    user: TokenPayload = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return persisted SecurityObserver events with ISO 27001 control refs.

    Suitable for 27001 audit evidence: provides a tamper-evident log of every
    security signal with the Annex A control it evidences.
    """
    from sqlalchemy import select
    from app.db.models import ObserverEvent

    tenant_id = getattr(user, "tenant_id", "system")
    stmt = (
        select(ObserverEvent)
        .where(ObserverEvent.tenant_id == tenant_id)
        .order_by(ObserverEvent.created_at.desc())
        .limit(limit)
    )

    def _parse_dt(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    if start_dt:
        stmt = stmt.where(ObserverEvent.created_at >= start_dt)
    if end_dt:
        stmt = stmt.where(ObserverEvent.created_at <= end_dt)
    if verdict:
        stmt = stmt.where(ObserverEvent.verdict == verdict)
    if control_ref:
        stmt = stmt.where(ObserverEvent.control_ref == control_ref)

    rows = (await session.execute(stmt)).scalars().all()
    events = [
        {
            "id": r.id,
            "phase": r.phase,
            "risk_score": r.risk_score,
            "risk_band": r.risk_band,
            "verdict": r.verdict,
            "relevance_class": r.relevance_class,
            "control_ref": r.control_ref,
            "signals": r.signals,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    halts = sum(1 for e in events if e["verdict"] == "halt")
    warns = sum(1 for e in events if e["verdict"] == "warn")
    return {
        "standard": "ISO/IEC 27001:2022",
        "count": len(events),
        "halt_count": halts,
        "warn_count": warns,
        "events": events,
    }


# ── AESCSF — Australian Energy Sector Cybersecurity Framework ─────────────────

@router.get("/aescsf/self-assessment")
async def get_aescsf_self_assessment(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the AESCSF v2.0 self-assessment with maturity levels and gaps.

    The Australian Energy Sector Cybersecurity Framework is the primary
    cybersecurity framework for NEM participants. This self-assessment
    documents GridVerdict's current maturity per AESCSF domain.
    """
    from app.compliance.aescsf import get_self_assessment, get_maturity_summary
    return {
        "maturity_summary": get_maturity_summary(),
        "assessment": get_self_assessment(),
    }
