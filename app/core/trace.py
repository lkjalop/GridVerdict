"""Bitemporal trace — write and read every query's full decision path.

Every query that completes a verdict cycle writes a Trace row containing:
  - valid_time: when the market data was true (from dispatch interval timestamp)
  - system_time: when GridVerdict learned about it (fetch time)
  - The decomposition, tool calls, prefill data, answer, and observer result

This supports:
  - Trace replay: re-run the pipeline on a past valid_time
  - Counterfactual: "what would you have said at T with different data?"
  - Bitemporal queries: "show me all traces where valid_time was X but
    GridVerdict didn't know until system_time Y"
  - Audit log: full reproducibility of every verdict

Framework boundary: imports only from app.core.schema and SQLAlchemy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Trace

logger = logging.getLogger(__name__)


async def write_trace(
    session: AsyncSession,
    trace_id: str,
    tenant_id: str,
    query_id: str | None,
    valid_time: datetime,
    decomposition: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    answer: dict[str, Any],
    observer_result: dict[str, Any] | None = None,
    model_profile: str = "cost_optimized",
    prefill: dict[str, Any] | None = None,
) -> Trace:
    """Persist a complete query trace to the traces table.

    safe to call inside an existing transaction — does not commit.
    """
    system_time = datetime.now(timezone.utc)

    source_manifest = _build_source_manifest(tool_calls, answer)

    trace = Trace(
        id=trace_id,
        tenant_id=tenant_id,
        query_id=query_id,
        valid_time=valid_time,
        system_time=system_time,
        model_profile=model_profile,
        source_manifest=source_manifest,
        tool_calls=tool_calls,
        decomposition=decomposition,
        prefill=prefill or {},
        answer=answer,
        validator_result=observer_result or {},
    )
    session.add(trace)
    logger.debug("Trace %s written (valid=%s system=%s)", trace_id, valid_time.isoformat(), system_time.isoformat())

    try:
        from app.data.event_bus import publish as _pub
        await _pub("trace_written", {
            "trace_id": trace_id,
            "query_id": query_id,
            "tenant_id": tenant_id,
            "valid_time": valid_time.isoformat(),
        })
    except Exception:
        pass

    return trace


async def read_trace(session: AsyncSession, trace_id: str, tenant_id: str) -> Trace | None:
    """Fetch a single trace by ID, scoped to tenant."""
    result = await session.execute(
        select(Trace).where(
            Trace.id == trace_id,
            Trace.tenant_id == tenant_id,
        )
    )
    return result.scalar_one_or_none()


async def list_traces(
    session: AsyncSession,
    tenant_id: str,
    query_id: str | None = None,
    limit: int = 50,
) -> list[Trace]:
    """List traces for a tenant, optionally filtered by query."""
    q = select(Trace).where(Trace.tenant_id == tenant_id)
    if query_id:
        q = q.where(Trace.query_id == query_id)
    q = q.order_by(Trace.system_time.desc()).limit(limit)
    result = await session.execute(q)
    return list(result.scalars().all())


async def list_traces_by_valid_time(
    session: AsyncSession,
    tenant_id: str,
    valid_from: datetime,
    valid_to: datetime,
    limit: int = 100,
) -> list[Trace]:
    """Bitemporal query: traces whose valid_time falls in [valid_from, valid_to]."""
    result = await session.execute(
        select(Trace)
        .where(
            Trace.tenant_id == tenant_id,
            Trace.valid_time >= valid_from,
            Trace.valid_time <= valid_to,
        )
        .order_by(Trace.valid_time.asc())
        .limit(limit)
    )
    return list(result.scalars().all())


def to_trace_dict(trace: Trace) -> dict[str, Any]:
    """Serialise a Trace model to a plain dict for API responses."""
    return {
        "id": trace.id,
        "query_id": trace.query_id,
        "valid_time": trace.valid_time.isoformat(),
        "system_time": trace.system_time.isoformat(),
        "model_profile": trace.model_profile,
        "source_manifest": trace.source_manifest,
        "decomposition": trace.decomposition,
        "answer": trace.answer,
        "validator_result": trace.validator_result,
        "tool_calls": trace.tool_calls,
    }


def _build_source_manifest(
    tool_calls: list[dict[str, Any]],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Summarise what data sources contributed to this trace."""
    sources = list({tc.get("source", "unknown") for tc in tool_calls if tc.get("source")})
    evidence_count = len(answer.get("evidence_refs", []))
    verdict = answer.get("verdict", "UNKNOWN")
    confidence = answer.get("confidence", 0.0)
    return {
        "sources": sources,
        "evidence_count": evidence_count,
        "verdict": verdict,
        "confidence": confidence,
    }
