"""Security observer API routes.

GET  /api/security/status      — current observer state and recent signals
POST /api/security/check/input — ad-hoc input text inspection (dev/admin use)
GET  /api/security/signals     — paginated log of all observer signals for this session
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.security.observer import get_observer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["security"])

# Rolling in-process signal log (bounded to last 500 entries)
_signal_log: list[dict[str, Any]] = []
_MAX_LOG = 500


# ── Schemas ───────────────────────────────────────────────────────────────────

class InputCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field("", description="Optional tenant scoping")


class SecurityStatusResponse(BaseModel):
    status: str           # healthy | elevated | alert
    total_signals: int
    recent_halts: int
    recent_warns: int
    last_signal_at: str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/security/status", response_model=SecurityStatusResponse)
async def security_status() -> SecurityStatusResponse:
    """Return aggregate security observer status for the dashboard."""
    recent = _signal_log[-50:] if _signal_log else []
    halts = sum(1 for e in recent if e["verdict"] == "halt")
    warns = sum(1 for e in recent if e["verdict"] == "warn")
    last_at = _signal_log[-1]["recorded_at"] if _signal_log else None

    if halts > 0:
        status = "alert"
    elif warns > 5:
        status = "elevated"
    else:
        status = "healthy"

    return SecurityStatusResponse(
        status=status,
        total_signals=len(_signal_log),
        recent_halts=halts,
        recent_warns=warns,
        last_signal_at=last_at,
    )


@router.post("/security/check/input")
async def check_input(req: InputCheckRequest) -> dict:
    """Run pass_1 on arbitrary text and return the ObserverResult.

    Intended for developer tooling and admin dashboards — not for end-users.
    """
    obs = get_observer()
    result = obs.pass_input(req.text, req.tenant_id)
    entry = {
        "phase": result.phase,
        "risk_score": result.risk_score,
        "risk_band": result.risk_band,
        "verdict": result.verdict,
        "signals": [s.description for s in result.signals],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_signal(entry)
    return result.to_dict()


@router.get("/security/signals")
async def list_signals(limit: int = 50, offset: int = 0) -> dict:
    """Return paginated observer signal log (newest first)."""
    items = list(reversed(_signal_log))
    page = items[offset: offset + limit]
    return {
        "total": len(_signal_log),
        "offset": offset,
        "limit": limit,
        "items": page,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def append_observer_result(result, phase_context: str = "") -> None:
    """Called from routes_query to log every observer pass result."""
    entry = {
        "phase": result.phase or phase_context,
        "risk_score": result.risk_score,
        "risk_band": result.risk_band,
        "verdict": result.verdict,
        "signals": [s.description for s in result.signals],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_signal(entry)


def _append_signal(entry: dict) -> None:
    _signal_log.append(entry)
    if len(_signal_log) > _MAX_LOG:
        del _signal_log[0]
