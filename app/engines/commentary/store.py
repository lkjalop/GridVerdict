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

# Canonical interconnector labels for NEM display
_IC_LABELS: dict[str, str] = {
    "V-S-MNSP1":  "VIC↔SA",
    "HYMAROO":    "VIC↔SA",
    "V-SA":       "VIC↔SA",
    "N-Q-MNSP1":  "NSW↔QLD",
    "TERRANORA":  "NSW↔QLD",
    "N-Q-MNSP2":  "NSW↔QLD",
    "V-N-MNSP1":  "VIC↔NSW",
    "QNI":        "NSW↔QLD",
    "T-V-MNSP1":  "TAS↔VIC",
    "BASSLINK":   "TAS↔VIC",
    "T-VMNSP1":   "TAS↔VIC",
}


async def _fetch_enrichment(session, events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Batch-query interconnector, FCAS, and fuel mix for a list of events.

    Returns a dict keyed by ISO valid_time string → enrichment payload.
    One extra DB call per search_recent/get_event invocation (not per event).
    """
    from sqlalchemy import select, text
    from app.db.models import MarketDriverEvent, FcasPriceEvent, UnitDispatchEvent

    if not events:
        return {}

    # Collect unique (region, valid_time) pairs
    valid_times = list({e["valid_time"] for e in events})
    regions = list({e["region"] for e in events})

    enrichment: dict[str, dict[str, Any]] = {vt: {} for vt in valid_times}

    try:
        # ── Interconnector flows ──────────────────────────────────────
        ic_result = await session.execute(
            select(
                MarketDriverEvent.valid_time,
                MarketDriverEvent.element_id,
                MarketDriverEvent.values,
            ).where(
                MarketDriverEvent.driver_type == "interconnector",
                MarketDriverEvent.valid_time.in_(valid_times),
            )
        )
        for row in ic_result.fetchall():
            vt_key = row.valid_time.isoformat() if hasattr(row.valid_time, "isoformat") else str(row.valid_time)
            # normalise to the ISO string used as the key
            vt_key = _normalise_vt_key(row.valid_time, valid_times)
            if vt_key is None:
                continue
            ic_label = _IC_LABELS.get(row.element_id.upper(), row.element_id)
            flow = row.values.get("metered_mw_flow") or row.values.get("mw_flow")
            export_limit = row.values.get("export_limit")
            import_limit = row.values.get("import_limit")
            capacity = max(abs(export_limit or 0), abs(import_limit or 0)) or None
            cap_pct = round(abs(flow) / capacity * 100) if flow is not None and capacity else None
            direction = "export" if (flow or 0) > 0 else "import"
            enrichment[vt_key].setdefault("interconnectors", []).append({
                "id": ic_label,
                "flow_mw": round(flow) if flow is not None else None,
                "direction": direction,
                "cap_pct": cap_pct,
            })
    except Exception as exc:
        logger.debug("Enrichment interconnector query failed: %s", exc)

    try:
        # ── FCAS prices ───────────────────────────────────────────────
        fcas_result = await session.execute(
            select(
                FcasPriceEvent.valid_time,
                FcasPriceEvent.region,
                FcasPriceEvent.raise_reg_rrp,
                FcasPriceEvent.raise_6sec_rrp,
                FcasPriceEvent.lower_reg_rrp,
                FcasPriceEvent.lower_6sec_rrp,
            ).where(
                FcasPriceEvent.region.in_(regions),
                FcasPriceEvent.valid_time.in_(valid_times),
            )
        )
        for row in fcas_result.fetchall():
            vt_key = _normalise_vt_key(row.valid_time, valid_times)
            if vt_key is None:
                continue
            enrichment[vt_key]["fcas"] = {
                "raise_reg": round(row.raise_reg_rrp, 2) if row.raise_reg_rrp is not None else None,
                "raise_fast": round(row.raise_6sec_rrp, 2) if row.raise_6sec_rrp is not None else None,
                "lower_reg": round(row.lower_reg_rrp, 2) if row.lower_reg_rrp is not None else None,
            }
    except Exception as exc:
        logger.debug("Enrichment FCAS query failed: %s", exc)

    try:
        # ── Fuel mix (% of total dispatch by fuel type) ───────────────
        fuel_result = await session.execute(
            select(
                UnitDispatchEvent.valid_time,
                UnitDispatchEvent.fuel_type,
                text("SUM(total_cleared_mw) as total_mw"),
            ).where(
                UnitDispatchEvent.valid_time.in_(valid_times),
                UnitDispatchEvent.total_cleared_mw.isnot(None),
                UnitDispatchEvent.fuel_type.isnot(None),
            ).group_by(
                UnitDispatchEvent.valid_time,
                UnitDispatchEvent.fuel_type,
            )
        )
        # Aggregate per valid_time
        fuel_by_vt: dict[str, dict[str, float]] = {}
        for row in fuel_result.fetchall():
            vt_key = _normalise_vt_key(row.valid_time, valid_times)
            if vt_key is None:
                continue
            fuel_by_vt.setdefault(vt_key, {})[row.fuel_type or "unknown"] = float(row.total_mw or 0)
        for vt_key, fuel_map in fuel_by_vt.items():
            total = sum(fuel_map.values())
            if total > 0:
                enrichment[vt_key]["fuel_mix_pct"] = {
                    ft: round(mw / total * 100) for ft, mw in sorted(fuel_map.items(), key=lambda x: -x[1])
                }
                enrichment[vt_key]["fuel_mix_mw"] = {ft: round(mw) for ft, mw in fuel_map.items()}
    except Exception as exc:
        logger.debug("Enrichment fuel mix query failed: %s", exc)

    try:
        # ── Weather observations (closest obs within ±30 min of event) ──
        from app.db.models import WeatherObservation
        from datetime import timedelta as _td
        from sqlalchemy import func as _func

        # Build a ±30-min window covering all event valid_times
        if valid_times:
            # valid_times are ISO strings; parse them for the DB query
            parsed_vts = []
            for vt_str in valid_times:
                try:
                    from datetime import datetime as _dt
                    parsed_vts.append(_dt.fromisoformat(vt_str.replace("Z", "+00:00")))
                except Exception:
                    pass
            if parsed_vts:
                window_start = min(parsed_vts) - _td(minutes=35)
                window_end   = max(parsed_vts) + _td(minutes=35)

                wx_result = await session.execute(
                    select(
                        WeatherObservation.region,
                        WeatherObservation.observed_at,
                        WeatherObservation.temperature_c,
                        WeatherObservation.temp_deviation_c,
                        WeatherObservation.humidity_pct,
                        WeatherObservation.wind_speed_kmh,
                        WeatherObservation.cloud_cover_pct,
                    ).where(
                        WeatherObservation.region.in_(regions),
                        WeatherObservation.observed_at >= window_start,
                        WeatherObservation.observed_at <= window_end,
                    ).order_by(WeatherObservation.observed_at)
                )
                wx_rows = wx_result.fetchall()

                # For each event, find the closest weather obs within ±30 min
                for event in events:
                    try:
                        from datetime import datetime as _dt2
                        evt_dt = _dt2.fromisoformat(event["valid_time"].replace("Z", "+00:00"))
                    except Exception:
                        continue
                    evt_region = event["region"]
                    best: Any = None
                    best_delta = _td(minutes=31)
                    for wx in wx_rows:
                        if wx.region != evt_region:
                            continue
                        obs_dt = wx.observed_at
                        if obs_dt.tzinfo is None:
                            from datetime import timezone as _tz2
                            obs_dt = obs_dt.replace(tzinfo=_tz2.utc)
                        delta = abs(evt_dt - obs_dt)
                        if delta < best_delta:
                            best_delta = delta
                            best = wx
                    if best is not None:
                        vt_key = event["valid_time"]
                        enrichment.setdefault(vt_key, {})["weather_obs"] = {
                            "temp_c":        round(best.temperature_c, 1) if best.temperature_c is not None else None,
                            "temp_dev_c":    round(best.temp_deviation_c, 1) if best.temp_deviation_c is not None else None,
                            "humidity_pct":  round(best.humidity_pct) if best.humidity_pct is not None else None,
                            "wind_kmh":      round(best.wind_speed_kmh) if best.wind_speed_kmh is not None else None,
                            "cloud_pct":     round(best.cloud_cover_pct) if best.cloud_cover_pct is not None else None,
                            "obs_age_min":   round(best_delta.total_seconds() / 60),
                        }
    except Exception as exc:
        logger.debug("Enrichment weather obs query failed: %s", exc)

    return enrichment


def _normalise_vt_key(db_vt: Any, known_keys: list[str]) -> str | None:
    """Match a DB datetime to one of the known ISO-string keys.

    The DB may return timezone-aware or naive datetimes; the keys are ISO strings
    from row.valid_time.isoformat() at event-fetch time.  We match by truncating
    to the minute to avoid sub-second drift.
    """
    if db_vt is None:
        return None
    try:
        db_str = db_vt.isoformat() if hasattr(db_vt, "isoformat") else str(db_vt)
        # Exact match first
        if db_str in known_keys:
            return db_str
        # Match by minute (strip seconds and microseconds)
        db_min = db_str[:16]
        for k in known_keys:
            if k[:16] == db_min:
                return k
    except Exception:
        pass
    return None


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
            rows = result.scalars().all()
            events = [_to_dict(r) for r in rows]
            enrichment = await _fetch_enrichment(session, events)
            for evt in events:
                enc = enrichment.get(evt["valid_time"], {})
                if enc:
                    evt["enrichment"] = enc
            return events
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
            if row is None:
                return None
            evt = _to_dict(row)
            enrichment = await _fetch_enrichment(session, [evt])
            enc = enrichment.get(evt["valid_time"], {})
            if enc:
                evt["enrichment"] = enc
            return evt
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
