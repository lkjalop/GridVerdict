"""CommentaryStore — DB persistence and retrieval for commentary events.

All methods are non-fatal: a DB failure never blocks the scheduler.
The store is used in three contexts:
  1. write_event()       — called by CommentaryEngine after each event
  2. search_recent()     — called by the REST API for frontend display
  3. search_for_rag()    — called by scatter_gather for RAG context
  4. prune_old_events()  — called by the daily cleanup scheduler job
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.engines.commentary.engine import CommentaryEvent

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


async def write_event(evt: "CommentaryEvent") -> None:
    """Persist a CommentaryEvent to commentary_events. Non-fatal on failure."""
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as CommentaryEventModel
        async with db_session() as session:
            row = CommentaryEventModel(
                id=evt.id,
                region=evt.region,
                valid_time=evt.valid_time,
                system_time=evt.system_time,
                event_type=evt.event_type,
                severity=evt.severity,
                headline=evt.headline,
                contributing_factors=evt.contributing_factors,
                missing_data=evt.missing_data,
                evidence_refs=evt.evidence_refs,
                claim_map=evt.claim_map,
                confidence=evt.confidence,
                corroborations=evt.corroborations,
                next_watch=evt.next_watch,
                counterargument=evt.counterargument,
                snapshot_before=evt.snapshot_before,
                snapshot_after=evt.snapshot_after,
                trace_id=evt.trace_id,
            )
            session.add(row)
            await session.commit()
    except Exception as exc:
        logger.debug("Failed to write commentary event %s: %s", getattr(evt, "id", "?"), exc)


async def search_recent(
    region: str,
    limit: int = 20,
    min_severity: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent commentary events for the Live Feed panel."""
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import select, desc

        min_val = _SEVERITY_ORDER.get(min_severity or "LOW", 0)
        allowed_severities = [s for s, v in _SEVERITY_ORDER.items() if v >= min_val]

        async with db_session() as session:
            q = (
                select(Model)
                .where(
                    Model.region == region,
                    Model.severity.in_(allowed_severities),
                )
                .order_by(desc(Model.valid_time))
                .limit(limit)
            )
            result = await session.execute(q)
            return [_to_dict(r) for r in result.scalars().all()]
    except Exception as exc:
        logger.debug("search_recent failed for %s: %s", region, exc)
        return []


async def has_recent_baseline(region: str, valid_time: datetime) -> bool:
    """Return True if a startup baseline already exists near this interval.

    This prevents duplicate "baseline captured" cards when the app restarts
    without Redis. It is deliberately non-fatal: if the DB check fails, the
    caller may still emit a baseline so the Live Feed does not look dead.
    """
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import select

        if valid_time.tzinfo is not None:
            valid_time = valid_time.astimezone(timezone.utc).replace(tzinfo=None)
        lo = valid_time - timedelta(minutes=10)
        hi = valid_time + timedelta(minutes=10)

        async with db_session() as session:
            result = await session.execute(
                select(Model.id)
                .where(
                    Model.region == region,
                    Model.event_type == "market_baseline",
                    Model.valid_time >= lo,
                    Model.valid_time <= hi,
                )
                .limit(1)
            )
            return result.scalar_one_or_none() is not None
    except Exception as exc:
        logger.debug("has_recent_baseline failed for %s: %s", region, exc)
        return False


async def get_event(event_id: str) -> dict[str, Any] | None:
    """Fetch a single commentary event by ID."""
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import select

        async with db_session() as session:
            result = await session.execute(
                select(Model).where(Model.id == event_id)
            )
            row = result.scalar_one_or_none()
            return _to_dict(row) if row else None
    except Exception as exc:
        logger.debug("get_event failed for %s: %s", event_id, exc)
        return None


async def get_stats(region: str, hours: int = 24) -> dict[str, Any]:
    """Summary of recent events by type and severity for the badge counter."""
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import select, func

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with db_session() as session:
            q = (
                select(Model.event_type, Model.severity, func.count().label("n"))
                .where(Model.region == region, Model.valid_time >= cutoff)
                .group_by(Model.event_type, Model.severity)
            )
            rows = (await session.execute(q)).all()

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        total = 0
        for event_type, severity, count in rows:
            by_type[event_type] = by_type.get(event_type, 0) + count
            by_severity[severity] = by_severity.get(severity, 0) + count
            total += count

        return {
            "region": region,
            "hours": hours,
            "total": total,
            "by_type": by_type,
            "by_severity": by_severity,
        }
    except Exception as exc:
        logger.debug("get_stats failed for %s: %s", region, exc)
        return {"region": region, "hours": hours, "total": 0, "by_type": {}, "by_severity": {}}


async def search_for_rag(
    region: str,
    time_from: datetime,
    time_to: datetime,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Return recent commentary events as structured RAG context for WhySources.

    Each returned item is formatted for direct inclusion in the Why Engine
    evidence bundle. Used by scatter_gather's T8 commentary task.
    """
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import select, desc

        async with db_session() as session:
            q = (
                select(Model)
                .where(
                    Model.region == region,
                    Model.valid_time >= time_from,
                    Model.valid_time <= time_to,
                    Model.confidence >= min_confidence,
                )
                .order_by(desc(Model.valid_time))
                .limit(5)
            )
            result = await session.execute(q)
            return [_to_rag_context(r) for r in result.scalars().all()]
    except Exception as exc:
        logger.debug("search_for_rag failed for %s: %s", region, exc)
        return []


async def prune_old_events(days: int = 7) -> int:
    """Delete commentary events older than `days`. Returns count deleted."""
    try:
        from app.db.session import db_session
        from app.db.models import CommentaryEvent as Model
        from sqlalchemy import delete

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with db_session() as session:
            result = await session.execute(
                delete(Model).where(Model.valid_time < cutoff)
            )
            await session.commit()
            return result.rowcount or 0
    except Exception as exc:
        logger.debug("prune_old_events failed: %s", exc)
        return 0


def _to_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "region": row.region,
        "valid_time": row.valid_time.isoformat(),
        "system_time": row.system_time.isoformat(),
        "event_type": row.event_type,
        "severity": row.severity,
        "headline": row.headline,
        "contributing_factors": row.contributing_factors or [],
        "missing_data": row.missing_data or [],
        "evidence_refs": row.evidence_refs or [],
        "claim_map": getattr(row, "claim_map", None) or [],
        "confidence": row.confidence,
        "corroborations": row.corroborations or {},
        "next_watch": row.next_watch or [],
        "counterargument": row.counterargument,
        "snapshot_before": row.snapshot_before,
        "snapshot_after": row.snapshot_after,
        "trace_id": row.trace_id,
    }


def _to_rag_context(row: Any) -> dict[str, Any]:
    return {
        "source": "commentary_events",
        "event_type": row.event_type,
        "valid_time": row.valid_time.isoformat(),
        "headline": row.headline,
        "contributing_factors": row.contributing_factors or [],
        "confidence": row.confidence,
        "evidence_refs": row.evidence_refs or [],
        "claim_map": getattr(row, "claim_map", None) or [],
        "missing_data": row.missing_data or [],
    }
