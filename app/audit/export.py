"""Regulatory export for decision audit log entries.

Exports DecisionAuditLog rows as JSON (list of dicts) or CSV.
Supports filtering by tenant, date range, region, and decision type.

CSV format is designed for direct import into regulatory reporting tools.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DecisionAuditLog

_CSV_FIELDS = [
    "id", "tenant_id", "user_id", "trace_id",
    "decision_type", "region", "asset_id",
    "action", "confidence", "price_rrp", "price_regime",
    "evidence_quality", "simulation_only",
    # ISO/IEC 42001:2023 §8.4 — model traceability fields
    "model_version", "training_data_ref",
    "created_at",
]


async def query_audit_log(
    session: AsyncSession,
    tenant_id: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    region: str | None = None,
    decision_type: str | None = None,
    limit: int = 1000,
) -> list[DecisionAuditLog]:
    """Query audit log rows with optional filters."""
    stmt = (
        select(DecisionAuditLog)
        .where(DecisionAuditLog.tenant_id == tenant_id)
        .order_by(DecisionAuditLog.created_at.desc())
        .limit(limit)
    )
    if start_dt is not None:
        stmt = stmt.where(DecisionAuditLog.created_at >= start_dt)
    if end_dt is not None:
        stmt = stmt.where(DecisionAuditLog.created_at <= end_dt)
    if region is not None:
        stmt = stmt.where(DecisionAuditLog.region == region.upper())
    if decision_type is not None:
        stmt = stmt.where(DecisionAuditLog.decision_type == decision_type)

    result = await session.execute(stmt)
    return list(result.scalars().all())


def rows_to_json(rows: list[DecisionAuditLog]) -> list[dict]:
    """Serialise rows to a list of dicts suitable for JSON export."""
    out = []
    for row in rows:
        out.append({
            "id": row.id,
            "tenant_id": row.tenant_id,
            "user_id": row.user_id,
            "trace_id": row.trace_id,
            "decision_type": row.decision_type,
            "region": row.region,
            "asset_id": row.asset_id,
            "action": row.action,
            "confidence": row.confidence,
            "price_rrp": row.price_rrp,
            "price_regime": row.price_regime,
            "economics": row.economics,
            "evidence_quality": row.evidence_quality,
            "risk_flags": row.risk_flags,
            "why_summary": row.why_summary,
            "simulation_only": row.simulation_only,
            "model_version": row.model_version,
            "training_data_ref": row.training_data_ref,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })
    return out


def rows_to_csv(rows: list[DecisionAuditLog]) -> str:
    """Serialise rows to CSV string with a fixed header."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=_CSV_FIELDS + ["economics_json", "risk_flags_json", "why_summary_json"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "id": row.id,
            "tenant_id": row.tenant_id,
            "user_id": row.user_id or "",
            "trace_id": row.trace_id or "",
            "decision_type": row.decision_type,
            "region": row.region,
            "asset_id": row.asset_id or "",
            "action": row.action,
            "confidence": row.confidence,
            "price_rrp": row.price_rrp if row.price_rrp is not None else "",
            "price_regime": row.price_regime or "",
            "evidence_quality": row.evidence_quality or "",
            "simulation_only": str(row.simulation_only),
            "model_version": row.model_version or "",
            "training_data_ref": row.training_data_ref or "",
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "economics_json": json.dumps(row.economics) if row.economics else "",
            "risk_flags_json": json.dumps(row.risk_flags) if row.risk_flags else "",
            "why_summary_json": json.dumps(row.why_summary) if row.why_summary else "",
        })
    return buf.getvalue()
