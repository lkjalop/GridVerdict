"""GET /health and GET /data/status — DB ping, AEMO freshness, data quality."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DBSession
from app.data.cache import get_cache
from config.settings import get_settings

router = APIRouter(tags=["health"])
_settings = get_settings()

_SOURCES = ["AEMO_DISPATCH_PRICE", "AEMO_PREDISPATCH_30MIN"]
_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]


def _to_dt(value) -> datetime | None:
    """Coerce a DB row value to a timezone-aware datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace(" ", "T"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


@router.get("/health")
async def health(db: AsyncSession = DBSession):
    """Basic liveness: DB connectivity + AEMO cache freshness."""
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    cache = get_cache()
    dispatch_age = await cache.age_seconds("dispatch_snapshot")
    aemo_fresh = dispatch_age is not None and dispatch_age < _settings.live_dispatch_max_age_s
    aemo_staleness = int(dispatch_age) if dispatch_age is not None else None

    return {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "aemo_cache": "fresh" if aemo_fresh else "stale",
        "aemo_staleness_seconds": aemo_staleness,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/data/status")
async def data_status(db: AsyncSession = DBSession):
    """Detailed data quality: per-source/region row counts, freshness, scheduler job health.

    Returns a dict with:
      sources: per (source, region) row counts and latest valid_time for last 2h and 24h
      supplementary: unit_dispatch_events, bid_offers, market_driver_events row counts
      backfill_cursors: per-cursor last successful interval + status
      hippograph: per-region in-process node counts
      lnn_trainers: per-region LNN trainer buffer / training state
      scheduler: per-job last_success_at, consecutive_failures, last_error
      cache: per-key age in seconds
    """
    now = datetime.now(timezone.utc)

    # ── Market events (dispatch + predispatch) ─────────────────────────────
    source_stats: list[dict[str, Any]] = []
    try:
        result = await db.execute(text("""
            SELECT source, region,
                   COUNT(*) FILTER (WHERE valid_time >= :cutoff_2h)  AS rows_2h,
                   COUNT(*) FILTER (WHERE valid_time >= :cutoff_24h) AS rows_24h,
                   MAX(valid_time) AS latest_valid_time
            FROM market_events
            WHERE source IN ('AEMO_DISPATCH_PRICE', 'AEMO_PREDISPATCH_30MIN')
              AND region IN ('NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1')
            GROUP BY source, region
            ORDER BY source, region
        """), {
            "cutoff_2h":  _iso(now, hours=2),
            "cutoff_24h": _iso(now, hours=24),
        })
        for row in result.fetchall():
            src, region, rows_2h, rows_24h, latest_vt = row
            vt_dt = _to_dt(latest_vt)
            age_s = int((now - vt_dt).total_seconds()) if vt_dt else None
            rows_2h = rows_2h or 0
            rows_24h = rows_24h or 0
            expected_2h = 24 if src == "AEMO_DISPATCH_PRICE" else 4
            status = (
                "fresh" if age_s is not None and age_s < _settings.live_dispatch_max_age_s
                else "stale" if age_s is not None and age_s < 3600
                else "missing" if rows_24h == 0
                else "degraded"
            )
            source_stats.append({
                "source": src,
                "region": region,
                "rows_last_2h": rows_2h,
                "rows_last_24h": rows_24h,
                "expected_rows_2h": expected_2h,
                "gap_flag": rows_2h < expected_2h // 2,
                "latest_valid_time": vt_dt.isoformat() if vt_dt else None,
                "age_seconds": age_s,
                "status": status,
            })
    except Exception as exc:
        source_stats = [{"error": str(exc)}]

    # ── Supplementary tables ───────────────────────────────────────────────
    supplementary: dict[str, Any] = {}
    try:
        # unit_dispatch_events — per-region
        r = await db.execute(text("""
            SELECT region, COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE valid_time >= :cutoff_24h) AS rows_24h,
                   MAX(valid_time) AS latest
            FROM unit_dispatch_events
            GROUP BY region
        """), {"cutoff_24h": _iso(now, hours=24)})
        supplementary["unit_dispatch_events"] = {
            row[0]: {
                "total": row[1],
                "rows_last_24h": row[2] or 0,
                "latest_valid_time": _to_dt(row[3]).isoformat() if _to_dt(row[3]) else None,
            }
            for row in r.fetchall()
        }
    except Exception as exc:
        supplementary["unit_dispatch_events"] = {"error": str(exc)}

    try:
        # bid_offers — aggregate (no region column guaranteed)
        r = await db.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE settlement_date >= :cutoff_24h) AS rows_24h,
                   MAX(settlement_date) AS latest
            FROM bid_offers
        """), {"cutoff_24h": _iso(now, hours=24)})
        row = r.fetchone()
        latest_dt = _to_dt(row[2]) if row else None
        supplementary["bid_offers"] = {
            "total": row[0] if row else 0,
            "rows_last_24h": (row[1] or 0) if row else 0,
            "latest_settlement_date": latest_dt.isoformat() if latest_dt else None,
        }
    except Exception as exc:
        supplementary["bid_offers"] = {"error": str(exc)}

    try:
        # market_driver_events — per driver_type
        r = await db.execute(text("""
            SELECT driver_type, COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE valid_time >= :cutoff_24h) AS rows_24h,
                   MAX(valid_time) AS latest
            FROM market_driver_events
            GROUP BY driver_type
            ORDER BY total DESC
            LIMIT 10
        """), {"cutoff_24h": _iso(now, hours=24)})
        supplementary["market_driver_events"] = {
            row[0]: {
                "total": row[1],
                "rows_last_24h": row[2] or 0,
                "latest_valid_time": _to_dt(row[3]).isoformat() if _to_dt(row[3]) else None,
            }
            for row in r.fetchall()
        }
    except Exception as exc:
        supplementary["market_driver_events"] = {"error": str(exc)}

    # ── Backfill cursors ───────────────────────────────────────────────────
    backfill_cursors: list[dict[str, Any]] = []
    try:
        r = await db.execute(text("""
            SELECT name, source, status, last_successful_interval,
                   files_completed, files_failed, error, updated_at
            FROM backfill_cursors
            ORDER BY updated_at DESC
        """))
        for row in r.fetchall():
            last_dt = _to_dt(row[3])
            updated_dt = _to_dt(row[7])
            backfill_cursors.append({
                "name": row[0],
                "source": row[1],
                "status": row[2],
                "last_successful_interval": last_dt.isoformat() if last_dt else None,
                "files_completed": row[4] or 0,
                "files_failed": row[5] or 0,
                "error": row[6],
                "updated_at": updated_dt.isoformat() if updated_dt else None,
            })
    except Exception as exc:
        backfill_cursors = [{"error": str(exc)}]

    # ── HippoGraph in-process node counts ─────────────────────────────────
    hippograph: dict[str, Any] = {}
    try:
        from app.engines.hippograph.graph import get_graph
        graph = get_graph()
        for region in _REGIONS:
            nodes = graph.get_region_nodes(region, limit=100_000)
            hippograph[region] = len(nodes)
        hippograph["total"] = sum(hippograph[r] for r in _REGIONS)
    except Exception as exc:
        hippograph = {"error": str(exc)}

    # ── LNN trainer state per region ──────────────────────────────────────
    lnn_trainers: dict[str, Any] = {}
    try:
        from app.engines.forecasting.inference import get_trainer
        for region in _REGIONS:
            trainer = get_trainer(region)
            if trainer is None:
                lnn_trainers[region] = {"available": False, "reason": "not initialised"}
            elif not trainer.is_trained:
                buf = getattr(trainer, "_buffer_count", None) or getattr(trainer, "buffer_len", None) or 0
                lnn_trainers[region] = {
                    "available": False,
                    "buffer_intervals": int(buf),
                    "min_train_required": 288,
                    "torch_version": getattr(trainer, "torch_version", None),
                    "n_features": getattr(trainer, "n_features", None),
                }
            else:
                last_t = getattr(trainer, "last_trained_at", None)
                age_h = None
                if last_t:
                    lt = last_t if last_t.tzinfo else last_t.replace(tzinfo=timezone.utc)
                    age_h = round((now - lt).total_seconds() / 3600, 1)
                lnn_trainers[region] = {
                    "available": True,
                    "trained_on_intervals": getattr(trainer, "training_rows", None),
                    "checkpoint_age_hours": age_h,
                    "torch_version": getattr(trainer, "torch_version", None),
                    "inference_latency_ms": getattr(trainer, "last_inference_latency_ms", None),
                    "rollback_available": getattr(trainer, "_has_backup", False),
                    "n_features": getattr(trainer, "n_features", None),
                }
    except Exception as exc:
        lnn_trainers = {"error": str(exc)}

    # ── Scheduler job health ───────────────────────────────────────────────
    from app.data.scheduler import get_job_states
    scheduler_states = get_job_states()

    # ── Cache age for key data types ───────────────────────────────────────
    cache = get_cache()
    cache_keys = [
        "dispatch_snapshot",
        "nem_news",
        "nem_news_fetched_at",
        *[f"notices_{r}" for r in _REGIONS],
        *[f"notices_{r}_fetched_at" for r in _REGIONS],
        *[f"calibration_{r}" for r in _REGIONS],
    ]
    cache_ages: dict[str, Any] = {}
    for key in cache_keys:
        age = await cache.age_seconds(key)
        cache_ages[key] = round(age, 1) if age is not None else None

    return {
        "timestamp": now.isoformat(),
        "sources": source_stats,
        "supplementary": supplementary,
        "backfill_cursors": backfill_cursors,
        "hippograph": hippograph,
        "lnn_trainers": lnn_trainers,
        "scheduler": scheduler_states,
        "cache_age_seconds": cache_ages,
    }


def _iso(dt: datetime, hours: int) -> datetime:
    from datetime import timedelta
    # asyncpg requires a datetime object, not a string
    return dt - timedelta(hours=hours)
