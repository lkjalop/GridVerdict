"""Background scheduler — AEMO data refresh on a fixed cycle.

Runs inside the FastAPI process using APScheduler's AsyncIOScheduler.
Started in the lifespan context manager; stopped on shutdown.

Jobs:
  dispatch_refresh    — every 5 min, fetches + caches + PERSISTS live prices
  notices_refresh     — every 60 s, caches market notices
  archive_backfill    — once per hour, gap-fill historical market_events table
  predispatch_refresh — every 30 min, ingests PD intervals to DB
  lnn_retrain         — every 1 h, retrains LNN from DB history
  nem_news_refresh    — every 5 min, caches RSS commentary
  weather_refresh     — every 5 min, caches regional weather consensus

Job health is tracked in _job_state and exposed via get_job_states()
for the /api/data/status endpoint.

Leader lock (Redis mode):
  When redis_url is configured, only the instance that holds the Redis leader
  lock runs scheduler jobs. The lock TTL is redis_leader_lock_ttl_s; a
  heartbeat task refreshes it every redis_leader_lock_heartbeat_s. On lock
  expiry, another instance can acquire it (automatic failover). Without Redis,
  each instance behaves as leader (single-process deployment).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import get_settings
from app.engines.spike_detector import SpikeDetector
from app.engines.commentary.engine import CommentaryEngine

logger = logging.getLogger(__name__)
_settings = get_settings()

# Sprint D: singleton spike detector — stateful regime-transition tracking
_spike_detector = SpikeDetector()

# Sprint Q: singleton commentary engine — stateful snapshot comparison per region
_commentary_engine = CommentaryEngine()

_scheduler: AsyncIOScheduler | None = None

# ── Leader lock state ─────────────────────────────────────────────────────────

_LEADER_LOCK_KEY = "gv:scheduler:leader"
_INSTANCE_ID = str(uuid.uuid4())  # unique per process
_is_leader: bool = False
_heartbeat_task: asyncio.Task | None = None
_election_task: asyncio.Task | None = None

# ── Scheduler running-job gauge ───────────────────────────────────────────────

_running_count: int = 0


def _job_enter(job_id: str) -> None:
    global _running_count
    _running_count += 1
    try:
        from app.api.metrics_registry import scheduler_jobs_running
        scheduler_jobs_running.set(_running_count)
    except Exception:
        pass
    logger.debug("Job started: %s (running=%d)", job_id, _running_count)


def _job_exit(job_id: str) -> None:
    global _running_count
    _running_count = max(_running_count - 1, 0)
    try:
        from app.api.metrics_registry import scheduler_jobs_running
        scheduler_jobs_running.set(_running_count)
    except Exception:
        pass
    logger.debug("Job finished: %s (running=%d)", job_id, _running_count)


# ── Leader lock helpers ───────────────────────────────────────────────────────

async def try_acquire_leader_lock(redis: Any) -> bool:
    """Try to acquire the scheduler leader lock. Returns True if acquired."""
    result = await redis.set(
        _LEADER_LOCK_KEY, _INSTANCE_ID,
        nx=True, ex=_settings.redis_leader_lock_ttl_s,
    )
    return result is not None


async def heartbeat_leader_lock(redis: Any) -> bool:
    """Refresh lock TTL only if we still own it. Returns False if leadership lost."""
    current = await redis.get(_LEADER_LOCK_KEY)
    if current == _INSTANCE_ID:
        await redis.expire(_LEADER_LOCK_KEY, _settings.redis_leader_lock_ttl_s)
        return True
    logger.warning(
        "Scheduler %s lost leadership — lock held by %s",
        _INSTANCE_ID[:8], (current or "none")[:8],
    )
    return False


async def release_leader_lock(redis: Any) -> None:
    """Release the lock only if this instance owns it."""
    current = await redis.get(_LEADER_LOCK_KEY)
    if current == _INSTANCE_ID:
        await redis.delete(_LEADER_LOCK_KEY)
        logger.info("Scheduler leader lock released by %s", _INSTANCE_ID[:8])


async def _heartbeat_loop(redis: Any) -> None:
    """Keep the leader lock alive with periodic heartbeats."""
    global _is_leader
    interval = _settings.redis_leader_lock_heartbeat_s
    try:
        while True:
            await asyncio.sleep(interval)
            still_leader = await heartbeat_leader_lock(redis)
            if not still_leader:
                _is_leader = False
                logger.warning("Scheduler stepping down — leader lock lost")
                return
    except asyncio.CancelledError:
        pass


async def _election_loop(redis: Any) -> None:
    """Poll for leadership when this instance is not the current leader."""
    global _is_leader, _heartbeat_task
    try:
        while True:
            await asyncio.sleep(5)
            acquired = await try_acquire_leader_lock(redis)
            if acquired:
                _is_leader = True
                logger.info("Scheduler %s acquired leader lock — starting jobs", _INSTANCE_ID[:8])
                await _start_jobs()
                _heartbeat_task = asyncio.create_task(_heartbeat_loop(redis))
                return
    except asyncio.CancelledError:
        pass


async def _try_publish(event_type: str, payload: dict, region: str | None = None) -> None:
    """Publish to event bus, silently swallowing any error."""
    try:
        from app.data.event_bus import publish
        await publish(event_type, payload, region=region)
    except Exception as exc:
        logger.debug("Event bus publish failed (%s): %s", event_type, exc)

# Per-job health state — read by /api/data/status
_job_state: dict[str, dict[str, Any]] = {}


def get_job_states() -> dict[str, dict[str, Any]]:
    """Return a snapshot of all job health states."""
    return {k: dict(v) for k, v in _job_state.items()}


def _record_success(job_id: str) -> None:
    state = _job_state.setdefault(job_id, {})
    state["last_success_at"] = datetime.now(timezone.utc).isoformat()
    state["consecutive_failures"] = 0
    state["last_error"] = None
    try:
        from app.api.metrics_registry import ingest_success_total
        ingest_success_total.labels(job=job_id).inc()
    except Exception:
        pass


def _record_failure(job_id: str, exc: Exception) -> None:
    state = _job_state.setdefault(job_id, {})
    state["last_failure_at"] = datetime.now(timezone.utc).isoformat()
    state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    state["last_error"] = str(exc)
    if state["consecutive_failures"] >= 3:
        logger.warning(
            "Scheduler job %s has failed %d consecutive times: %s",
            job_id, state["consecutive_failures"], exc,
        )
    try:
        from app.api.metrics_registry import ingest_failure_total
        ingest_failure_total.labels(job=job_id).inc()
    except Exception:
        pass


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


async def _start_jobs() -> None:
    """Register all jobs with APScheduler and start the scheduler."""
    sched = get_scheduler()
    if sched.running:
        return

    sched.add_job(
        _job_dispatch_refresh,
        trigger=IntervalTrigger(seconds=_settings.aemo_dispatch_poll_s),
        id="dispatch_refresh",
        name="AEMO dispatch price refresh + persist",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_notices_refresh,
        trigger=IntervalTrigger(seconds=_settings.aemo_notices_poll_s),
        id="notices_refresh",
        name="AEMO market notices refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_archive_backfill,
        trigger=IntervalTrigger(hours=1),
        id="archive_backfill",
        name="AEMO archive gap-fill",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    if _settings.archive_bulk_backfill_enabled:
        sched.add_job(
            _job_mmsdm_bulk_backfill,
            trigger=DateTrigger(run_date=datetime.now(timezone.utc)),
            id="mmsdm_bulk_backfill",
            name="AEMO MMSDM one-time bulk backfill",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
    # Monthly MMSDM refresh — runs on the 20th of each month at 03:00 UTC
    # Downloads the previous month's MMSDM archive and retrains LNN.
    # 20th gives AEMO ~20 days after month-end to publish files (they typically publish in ~14 days).
    from apscheduler.triggers.cron import CronTrigger
    sched.add_job(
        _job_mmsdm_monthly_refresh,
        trigger=CronTrigger(day=20, hour=3, minute=0, timezone="UTC"),
        id="mmsdm_monthly_refresh",
        name="AEMO MMSDM previous-month archive refresh + LNN retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        _job_predispatch_refresh,
        trigger=IntervalTrigger(minutes=30),
        id="predispatch_refresh",
        name="AEMO predispatch 30-min ahead ingestion",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_lnn_retrain,
        trigger=IntervalTrigger(hours=1),
        id="lnn_retrain",
        name="LNN quantile model retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_nem_news_refresh,
        trigger=IntervalTrigger(seconds=_settings.nem_news_poll_s),
        id="nem_news_refresh",
        name="NEM public RSS news refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_weather_refresh,
        trigger=IntervalTrigger(seconds=_settings.weather_poll_s),
        id="weather_refresh",
        name="NEM regional weather consensus refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        _job_calibration_update,
        trigger=IntervalTrigger(hours=6),
        id="calibration_update",
        name="Fast walk-forward calibration per region",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    sched.add_job(
        _job_commentary_cleanup,
        trigger=IntervalTrigger(hours=24),
        id="commentary_cleanup",
        name="Prune commentary_events older than 7 days",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    sched.start()
    logger.info(
        "Scheduler started — dispatch every %ds, notices every %ds",
        _settings.aemo_dispatch_poll_s,
        _settings.aemo_notices_poll_s,
    )


async def start_scheduler() -> None:
    """Start the scheduler, acquiring the Redis leader lock when Redis is available.

    With Redis: only the instance that acquires the lock starts jobs. Non-leader
    instances enter an election loop and start jobs once the lock becomes free.
    Without Redis (or on lock acquisition failure after timeout): falls back to
    unconditional scheduling for single-process deployments.
    """
    global _is_leader, _heartbeat_task, _election_task

    from app.data.redis_client import get_redis
    redis = await get_redis()

    if redis is not None:
        acquired = await try_acquire_leader_lock(redis)
        if acquired:
            _is_leader = True
            logger.info(
                "Scheduler %s acquired leader lock — starting jobs",
                _INSTANCE_ID[:8],
            )
            await _start_jobs()
            _heartbeat_task = asyncio.create_task(_heartbeat_loop(redis))
        else:
            _is_leader = False
            logger.info(
                "Scheduler %s waiting for leader lock (another instance is leader)",
                _INSTANCE_ID[:8],
            )
            _election_task = asyncio.create_task(_election_loop(redis))
        return

    # No Redis — single-process mode, always leader
    _is_leader = True
    await _start_jobs()


async def stop_scheduler() -> None:
    """Stop gracefully. Release leader lock and cancel background tasks."""
    global _heartbeat_task, _election_task, _is_leader

    # Cancel background tasks
    for task in (_heartbeat_task, _election_task):
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _heartbeat_task = None
    _election_task = None

    # Release Redis lock if we hold it
    try:
        from app.data.redis_client import get_redis
        redis = await get_redis()
        if redis is not None and _is_leader:
            await release_leader_lock(redis)
    except Exception as exc:
        logger.debug("Leader lock release on shutdown failed (non-fatal): %s", exc)

    _is_leader = False

    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        logger.info("Scheduler stopped")


# ── Job implementations ───────────────────────────────────────────────────────

async def _job_dispatch_refresh() -> None:
    """Fetch latest dispatch prices, update cache, AND persist to market_events.

    Persistence turns live operation into durable market memory:
    - HippoGraph graph nodes accumulate across restarts
    - TemporalRAG can reconstruct "what did we know at time T"
    - /api/market/history reflects observations as they happened
    """
    from app.data.aemo_live_client import get_aemo_client
    from app.data.cache import get_cache

    _job_enter("dispatch_refresh")
    try:
        client = get_aemo_client()
        cache = get_cache()
        snapshot = await client.fetch_latest_snapshot()
        await cache.set("dispatch_snapshot", snapshot.to_dict())

        # Persist to DB — idempotent on repeated calls for same interval
        await _persist_dispatch_snapshot(snapshot)

        # Wire interconnector binding + constraint violations → constraints_{region} cache.
        # This is the missing link: CONSTRAINT_ACTIVE detection in commentary/detector.py
        # reads constraints_{region} but nothing was writing it until now.
        _IC_REGION_MAP: dict[str, list[str]] = {
            "V-S-MNSP1": ["SA1", "VIC1"], "HEYWOOD": ["SA1", "VIC1"],
            "N-Q-MNSP1": ["NSW1", "QLD1"], "QNI": ["NSW1", "QLD1"], "TERRANORA": ["NSW1", "QLD1"],
            "N-Q-MNSP2": ["NSW1", "QLD1"],
            "V-N-MNSP1": ["VIC1", "NSW1"],
            "T-V-MNSP1": ["TAS1", "VIC1"], "BASSLINK": ["TAS1", "VIC1"],
        }
        _region_constraints: dict[str, list[dict]] = {r: [] for r in snapshot.regions}
        for _ic in (getattr(snapshot, "interconnector_rows", None) or []):
            _ic_id = (_ic.get("interconnector_id") or "").upper()
            _binding = _ic.get("at_export_limit") or _ic.get("at_import_limit") or (
                (_ic.get("violation_degree") or 0.0) > 0.0
            )
            if _binding:
                _label = {
                    "V-S-MNSP1": "Heywood (VIC↔SA)", "N-Q-MNSP1": "QNI (NSW↔QLD)",
                    "V-N-MNSP1": "VIC–NSW", "T-V-MNSP1": "Basslink (TAS↔VIC)",
                }.get(_ic_id, _ic_id)
                for _r in _IC_REGION_MAP.get(_ic_id, []):
                    if _r in _region_constraints:
                        _region_constraints[_r].append({
                            "constraint_id": _label,
                            "flow_mw": _ic.get("metered_mw_flow"),
                            "limit_mw": _ic.get("export_limit") or _ic.get("import_limit"),
                            "source": "interconnector",
                        })
        for _crow in (getattr(snapshot, "constraint_rows", None) or []):
            _cid = _crow.get("constraint_id") or ""
            _upper = _cid.upper()
            for _pfx, _r in [("N-", "NSW1"), ("V-", "VIC1"), ("Q-", "QLD1"), ("S-", "SA1"), ("T-", "TAS1")]:
                if _upper.startswith(_pfx) and _r in _region_constraints:
                    _region_constraints[_r].append({
                        "constraint_id": _cid,
                        "marginal_value": _crow.get("marginal_value"),
                        "source": "constraint",
                    })
                    break
        for _r, _clist in _region_constraints.items():
            await cache.set(f"constraints_{_r}", _clist, ttl=120)

        # Feed LNN trainer buffer — pass enriched interval dict so v2 features fire
        from app.engines.forecasting.inference import feed_interval
        for region_code, dp in snapshot.regions.items():
            _fcas = getattr(dp, "fcas_prices", None) or {}
            feed_interval(region_code, {
                "region": region_code,
                "price_rrp": dp.price_rrp,
                "demand_mw": dp.demand_mw,
                "availability_mw": dp.availability_mw,
                "valid_time": dp.valid_time.isoformat(),
                # v2 enriched features — 0.0 / None gracefully handled by _extract_features
                "fcas_raise_6s": _fcas.get("raise6sec"),
                # renewable_pct and constraint_count are filled by the scheduler's
                # driver pipeline below; set to None here (resolved post-persist)
                "renewable_pct": None,
                "constraint_count": None,
            })

        # Feed ChronoGraph RegimeClassifier (t-digest + ADWIN) for live percentile ranking
        _region_regimes: dict[str, str] = {}
        try:
            from app.engines.chronograph.regime import get_classifier
            from domain.nem.adapter import _REGIME_THRESHOLDS, classify_regime
            for region_code, dp in snapshot.regions.items():
                thresholds = _REGIME_THRESHOLDS.get(region_code, _REGIME_THRESHOLDS["NSW1"])
                get_classifier(region_code, thresholds).observe(dp.price_rrp, dp.valid_time)
                _region_regimes[region_code] = classify_regime(dp.price_rrp, region_code)
        except Exception as _exc:
            logger.debug("ChronoGraph observe failed: %s", _exc)

        # Sprint D: proactive spike alert detection
        for region_code, dp in snapshot.regions.items():
            try:
                regime = _region_regimes.get(region_code, "normal")
                headroom = max(dp.availability_mw - dp.demand_mw, 0.0)
                alert = _spike_detector.check(
                    region_code, regime, dp.price_rrp, dp.demand_mw, headroom, dp.valid_time,
                )
                if alert:
                    await _try_publish(alert.alert_type, alert.to_dict(), region=region_code)
                    logger.info(
                        "Spike %s: %s %s $%.0f/MWh",
                        alert.alert_type, region_code, regime, dp.price_rrp,
                    )
            except Exception as _exc:
                logger.debug("Spike detection failed for %s: %s", region_code, _exc)

        # Sprint Q: rolling market commentary — detect material changes and generate events
        for region_code, dp in snapshot.regions.items():
            try:
                regime = _region_regimes.get(region_code, "normal")
                headroom = max(dp.availability_mw - dp.demand_mw, 0.0)
                commentary_events = await _commentary_engine.process_tick(
                    region=region_code,
                    price_rrp=dp.price_rrp,
                    demand_mw=dp.demand_mw,
                    headroom_mw=headroom,
                    regime=regime,
                    valid_time=dp.valid_time,
                    cache=cache,
                )
                for evt in commentary_events:
                    await _try_publish("commentary_created", evt.to_dict(), region=region_code)
            except Exception as _exc:
                logger.debug("Commentary engine failed for %s: %s", region_code, _exc)

        # Drift detection: compare actual prices to cached forecast P50s
        # On drift: invalidate forecast cache so next query triggers a full retrain
        try:
            from app.engines.drift_monitor import feed_actual, reset_detector
            for region_code, dp in snapshot.regions.items():
                _fc_cache_key = f"live_forecast_{region_code}"
                _cached_fc = await cache.get(_fc_cache_key)
                if isinstance(_cached_fc, dict) and _cached_fc.get("available"):
                    _primary = _cached_fc.get("primary_model", "")
                    _fc_list = _cached_fc.get("forecasts", [])
                    _fc_entry = next((f for f in _fc_list if f.get("model") == _primary), None)
                    if _fc_entry:
                        _p50_list = _fc_entry.get("p50", [])
                        _p50_first = float(_p50_list[0]) if _p50_list else None
                        drifted = feed_actual(region_code, dp.price_rrp, _p50_first)
                        if drifted:
                            await cache.invalidate(_fc_cache_key)
                            reset_detector(region_code)
                            logger.info(
                                "Drift-driven cache invalidation for %s — retrain triggered on next query",
                                region_code,
                            )
        except Exception as _drift_exc:
            logger.debug("Drift check failed (non-fatal): %s", _drift_exc)

        _record_success("dispatch_refresh")
        logger.debug(
            "Dispatch refresh OK — %d regions, interval %s",
            len(snapshot.regions),
            snapshot.interval.isoformat(),
        )
        await _try_publish("dispatch_updated", {
            "regions": list(snapshot.regions.keys()),
            "interval": snapshot.interval.isoformat(),
        })
        await _try_publish("incident_timeline_updated", {
            "regions": list(snapshot.regions.keys()),
            "interval": snapshot.interval.isoformat(),
        })
    except Exception as exc:
        _record_failure("dispatch_refresh", exc)
        logger.warning("Dispatch refresh failed: %s", exc)
        state = _job_state.get("dispatch_refresh", {})
        if state.get("consecutive_failures", 0) >= 3:
            await _try_publish("scheduler_failure", {
                "job_id": "dispatch_refresh",
                "consecutive_failures": state["consecutive_failures"],
                "error": str(exc),
            })
        await _try_publish("source_stale", {
            "source": "AEMO_DISPATCH",
            "error": str(exc),
        })
    finally:
        _job_exit("dispatch_refresh")


async def _persist_dispatch_snapshot(snapshot: Any) -> None:
    """Upsert each region's DispatchPrice into market_events.

    Uses a deterministic ID (sha256 of source+region+valid_time) so duplicate
    calls for the same interval are safe and idempotent — session.merge()
    updates if the row exists or inserts if it doesn't.
    """
    from app.db.session import db_session
    from app.db.models import MarketEvent

    rows = []
    for region_code, dp in snapshot.regions.items():
        row_id = hashlib.sha256(
            f"dispatch-{region_code}-{dp.valid_time.isoformat()}".encode()
        ).hexdigest()[:36]
        rows.append(MarketEvent(
            id=row_id,
            region=region_code,
            source="AEMO_DISPATCH_PRICE",
            valid_time=dp.valid_time,
            system_time=datetime.now(timezone.utc),
            price_rrp=dp.price_rrp,
            demand_mw=dp.demand_mw,
            availability_mw=dp.availability_mw,
            tenant_id="system",
            raw_ref=(dp.raw_ref or "dispatch-live")[:100],
            data={},
        ))

    if not rows:
        return

    # Collect FCAS rows in parallel
    fcas_rows = []
    for region_code, dp in snapshot.regions.items():
        if dp.fcas_prices and any(v is not None for v in dp.fcas_prices.values()):
            from app.db.models import FcasPriceEvent
            fcas_id = hashlib.sha256(
                f"fcas-{region_code}-{dp.valid_time.isoformat()}".encode()
            ).hexdigest()[:36]
            fcas_rows.append(FcasPriceEvent(
                id=fcas_id,
                source="AEMO_DISPATCH_PRICE",
                region=region_code,
                valid_time=dp.valid_time,
                system_time=datetime.now(timezone.utc),
                raise_6sec_rrp=dp.fcas_prices.get("raise6sec"),
                lower_6sec_rrp=dp.fcas_prices.get("lower6sec"),
                raise_60sec_rrp=dp.fcas_prices.get("raise60sec"),
                lower_60sec_rrp=dp.fcas_prices.get("lower60sec"),
                raise_5min_rrp=dp.fcas_prices.get("raise5min"),
                lower_5min_rrp=dp.fcas_prices.get("lower5min"),
                raise_reg_rrp=dp.fcas_prices.get("raisereg"),
                lower_reg_rrp=dp.fcas_prices.get("lowerreg"),
                raw_ref=dp.raw_ref,
            ))

    # Sprint T: persist DUID-level unit dispatch rows from DISPATCHLOAD
    unit_dispatch_rows_db = []
    if getattr(snapshot, "unit_dispatch_rows", None):
        from app.db.models import UnitDispatchEvent, GeneratorUnit
        from sqlalchemy import select as sa_select
        try:
            async with db_session() as _gen_session:
                # Load generator registry to map DUID → region + fuel_type
                gen_result = await _gen_session.execute(sa_select(GeneratorUnit))
                gen_registry: dict[str, "GeneratorUnit"] = {
                    g.duid: g for g in gen_result.scalars().all()
                }
            for row in snapshot.unit_dispatch_rows:
                duid = row.get("duid", "")
                gen = gen_registry.get(duid)
                if gen is None:
                    continue   # skip DUIDs not in registry yet
                unit_id = hashlib.sha256(
                    f"ud-{duid}-{row['valid_time'].isoformat()}".encode()
                ).hexdigest()[:36]
                unit_dispatch_rows_db.append(UnitDispatchEvent(
                    id=unit_id,
                    source="DISPATCHLOAD",
                    duid=duid,
                    station_name=gen.station_name,
                    participant=gen.participant,
                    region=gen.region,
                    fuel_type=gen.fuel_type,
                    valid_time=row["valid_time"],
                    system_time=row["system_time"],
                    initial_mw=row.get("initialmw"),
                    total_cleared_mw=row.get("totalcleared"),
                    availability_mw=row.get("availability"),
                    ramp_rate=row.get("rampdownrate"),  # store downramp as ramp_rate
                    semi_dispatch_cap=row.get("semi_dispatch_cap"),
                    raw_ref=row.get("raw_ref", snapshot.raw_ref)[:100],
                    data={
                        "rampuprate": row.get("rampuprate"),
                        "dispatchedgeneration": row.get("dispatchedgeneration"),
                        "dispatchedload": row.get("dispatchedload"),
                    },
                ))
        except Exception as _unit_exc:
            logger.debug("Unit dispatch row building failed (non-fatal): %s", _unit_exc)
            unit_dispatch_rows_db = []

    # Sprint T: persist binding constraints from DISPATCHCONSTRAINT
    constraint_rows_db = []
    if getattr(snapshot, "constraint_rows", None):
        from app.db.models import MarketDriverEvent
        for crow in snapshot.constraint_rows:
            cid = crow["constraint_id"]
            c_id = hashlib.sha256(
                f"dc-{cid}-{crow['valid_time'].isoformat()}".encode()
            ).hexdigest()[:36]
            constraint_rows_db.append(MarketDriverEvent(
                id=c_id,
                source="DISPATCHCONSTRAINT",
                driver_type="constraint",
                element_id=cid,
                region=None,   # constraints span regions — filter by element_id prefix
                valid_time=crow["valid_time"],
                system_time=crow["system_time"],
                values={
                    "marginal_value": crow["marginal_value"],
                    "violation_degree": crow["violation_degree"],
                    "rhs": crow.get("rhs"),
                },
                raw_ref=crow.get("raw_ref", snapshot.raw_ref)[:100],
            ))

    try:
        async with db_session() as session:
            for row in rows:
                await session.merge(row)
            for row in fcas_rows:
                await session.merge(row)
            for row in unit_dispatch_rows_db:
                await session.merge(row)
            for row in constraint_rows_db:
                await session.merge(row)
            await session.commit()
        logger.debug(
            "Dispatch persisted: %d price, %d FCAS, %d unit, %d constraint rows",
            len(rows), len(fcas_rows), len(unit_dispatch_rows_db), len(constraint_rows_db),
        )
    except Exception as exc:
        logger.warning("Dispatch persistence failed (non-fatal): %s", exc)


async def _job_notices_refresh() -> None:
    """Fetch and cache active AEMO market notices with timestamp metadata."""
    _job_enter("notices_refresh")
    try:
        import asyncio
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        from app.data.cache import get_cache

        notices_client = AEMOMarketNoticesClient()
        loop = asyncio.get_event_loop()
        notices: list[dict] = await loop.run_in_executor(
            None, notices_client.fetch_active_notices
        )
        cache = get_cache()
        fetched_at = datetime.now(timezone.utc).isoformat()

        from collections import defaultdict
        by_region: dict[str, list] = defaultdict(list)
        for n in notices:
            region = n.get("region")
            if region:
                by_region[region].append(n)
            else:
                for r in ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]:
                    by_region[r].append(n)

        for region, region_notices in by_region.items():
            await cache.set(f"notices_{region}", region_notices)
            # Store fetch timestamp so staleness can be detected in scatter_gather
            await cache.set(f"notices_{region}_fetched_at", fetched_at)

        _record_success("notices_refresh")
        logger.debug("Notices refresh OK — %d notices total", len(notices))
    except Exception as exc:
        _record_failure("notices_refresh", exc)
        logger.debug("Notices refresh failed (non-fatal): %s", exc)
    finally:
        _job_exit("notices_refresh")


async def _job_lnn_retrain() -> None:
    """Trigger LNN retraining for all regions that have enough data."""
    _job_enter("lnn_retrain")
    try:
        import asyncio
        from app.engines.forecasting.inference import maybe_train
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            for region in ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]:
                await loop.run_in_executor(pool, maybe_train, region)
        _record_success("lnn_retrain")
        await _try_publish("forecast_updated", {"source": "lnn_retrain"})
    except Exception as exc:
        _record_failure("lnn_retrain", exc)
        logger.debug("LNN retrain skipped (non-fatal): %s", exc)
        state = _job_state.get("lnn_retrain", {})
        if state.get("consecutive_failures", 0) >= 3:
            await _try_publish("scheduler_failure", {
                "job_id": "lnn_retrain",
                "consecutive_failures": state["consecutive_failures"],
                "error": str(exc),
            })
    finally:
        _job_exit("lnn_retrain")


async def _job_nem_news_refresh() -> None:
    """Fetch public RSS/Atom market commentary into cache."""
    _job_enter("nem_news_refresh")
    try:
        from app.data.cache import get_cache
        from app.mcp.nem_news_client import NEMNewsRSSClient

        items = await NEMNewsRSSClient().fetch_recent(limit=30)
        cache = get_cache()
        await cache.set("nem_news", items)
        await cache.set("nem_news_fetched_at", datetime.now(timezone.utc).isoformat())
        _record_success("nem_news_refresh")
        logger.debug("NEM news RSS refresh OK — %d items", len(items))
    except Exception as exc:
        _record_failure("nem_news_refresh", exc)
        logger.debug("NEM news RSS refresh failed (non-fatal): %s", exc)
    finally:
        _job_exit("nem_news_refresh")


# Monthly mean temperatures (°C) per NEM region — used to compute temp deviation.
# Sourced from BOM climate averages for the capital city of each region.
_SEASONAL_TEMP_NORMS: dict[str, dict[int, float]] = {
    "NSW1": {1: 26, 2: 26, 3: 24, 4: 20, 5: 16, 6: 13, 7: 12, 8: 14, 9: 17, 10: 20, 11: 23, 12: 25},
    "VIC1": {1: 26, 2: 26, 3: 23, 4: 18, 5: 14, 6: 11, 7: 10, 8: 11, 9: 14, 10: 17, 11: 21, 12: 24},
    "QLD1": {1: 30, 2: 30, 3: 28, 4: 26, 5: 23, 6: 20, 7: 19, 8: 21, 9: 24, 10: 28, 11: 30, 12: 31},
    "SA1":  {1: 30, 2: 30, 3: 26, 4: 21, 5: 17, 6: 14, 7: 12, 8: 14, 9: 17, 10: 22, 11: 26, 12: 29},
    "TAS1": {1: 21, 2: 21, 3: 19, 4: 15, 5: 12, 6: 9,  7: 8,  8: 9,  9: 12, 10: 15, 11: 17, 12: 19},
}


async def _job_weather_refresh() -> None:
    """Fetch weather consensus for each NEM region — cache + persist to DB."""
    _job_enter("weather_refresh")
    try:
        import asyncio
        from datetime import timezone as _tz
        from app.data.cache import get_cache
        from app.mcp.weather_client import WeatherConsensusClient
        from app.db.session import db_session
        from app.db.models import WeatherObservation
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        client = WeatherConsensusClient()
        cache = get_cache()
        regions = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
        results = await asyncio.gather(
            *(client.fetch_region_consensus(region) for region in regions),
            return_exceptions=True,
        )
        ok = 0
        db_rows: list[dict] = []
        now = datetime.now(_tz.utc)

        for region, result in zip(regions, results):
            if not isinstance(result, dict):
                continue
            await cache.set(f"weather_{region}", result)
            ok += 1

            # Build DB row — observed_at from consensus timestamp, fallback to now
            consensus = result.get("consensus") or {}
            obs_ts_str = result.get("observed_at") or result.get("timestamp")
            try:
                from datetime import datetime as _dt
                obs_at = _dt.fromisoformat(obs_ts_str.replace("Z", "+00:00")) if obs_ts_str else now
            except Exception:
                obs_at = now

            temp = consensus.get("temperature_c")
            month = obs_at.month
            norm = _SEASONAL_TEMP_NORMS.get(region, {}).get(month)
            deviation = round(temp - norm, 2) if temp is not None and norm is not None else None

            db_rows.append({
                "region": region,
                "observed_at": obs_at,
                "fetched_at": now,
                "temperature_c": temp,
                "temp_deviation_c": deviation,
                "humidity_pct": consensus.get("humidity_pct"),
                "wind_speed_kmh": consensus.get("wind_speed_kmh"),
                "wind_gust_kmh": consensus.get("wind_gust_kmh"),
                "precipitation_mm": consensus.get("precipitation_mm"),
                "cloud_cover_pct": consensus.get("cloud_cover_pct"),
                "source_count": result.get("source_count"),
                "raw_consensus": result,
            })

        if db_rows:
            try:
                async with db_session() as session:
                    stmt = pg_insert(WeatherObservation).values(db_rows)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["region", "observed_at"],
                        set_={
                            "fetched_at": stmt.excluded.fetched_at,
                            "temperature_c": stmt.excluded.temperature_c,
                            "temp_deviation_c": stmt.excluded.temp_deviation_c,
                            "humidity_pct": stmt.excluded.humidity_pct,
                            "wind_speed_kmh": stmt.excluded.wind_speed_kmh,
                            "cloud_cover_pct": stmt.excluded.cloud_cover_pct,
                            "source_count": stmt.excluded.source_count,
                            "raw_consensus": stmt.excluded.raw_consensus,
                        },
                    )
                    await session.execute(stmt)
                    await session.commit()
            except Exception as db_exc:
                logger.debug("Weather obs DB persist failed (non-fatal): %s", db_exc)

        _record_success("weather_refresh")
        logger.debug("Weather refresh OK — %d/%d regions, %d persisted", ok, len(regions), len(db_rows))
    except Exception as exc:
        _record_failure("weather_refresh", exc)
        logger.debug("Weather refresh failed (non-fatal): %s", exc)
    finally:
        _job_exit("weather_refresh")


async def _job_predispatch_refresh() -> None:
    """Fetch AEMO PREDISPATCH run and persist intervals to market_events."""
    _job_enter("predispatch_refresh")
    try:
        from app.data.aemo_live_client import get_aemo_client
        from app.db.session import db_session
        from app.db.models import MarketEvent

        client = get_aemo_client()
        pd_by_region = await client.fetch_predispatch()

        if not pd_by_region:
            logger.debug("Predispatch refresh: no intervals returned")
            _record_success("predispatch_refresh")
            return

        total_inserted = 0
        async with db_session() as session:
            for region, intervals in pd_by_region.items():
                for iv in intervals:
                    row_id = hashlib.sha256(
                        f"pd-{region}-{iv.interval_datetime.isoformat()}".encode()
                    ).hexdigest()[:36]
                    row = MarketEvent(
                        id=row_id,
                        region=region,
                        source="AEMO_PREDISPATCH_30MIN",
                        valid_time=iv.interval_datetime,
                        system_time=datetime.now(timezone.utc),
                        price_rrp=iv.rrp,
                        demand_mw=iv.demand_mw,
                        availability_mw=0.0,
                        tenant_id="system",
                        raw_ref=iv.raw_ref,
                        data={"predispatch": True},
                    )
                    await session.merge(row)
                    total_inserted += 1
            await session.commit()

        _record_success("predispatch_refresh")
        logger.info("Predispatch refresh OK — %d intervals stored", total_inserted)
        await _try_publish("forecast_updated", {"source": "predispatch_refresh", "intervals": total_inserted})
    except Exception as exc:
        _record_failure("predispatch_refresh", exc)
        logger.debug("Predispatch refresh failed (non-fatal): %s", exc)
        state = _job_state.get("predispatch_refresh", {})
        if state.get("consecutive_failures", 0) >= 3:
            await _try_publish("scheduler_failure", {
                "job_id": "predispatch_refresh",
                "consecutive_failures": state["consecutive_failures"],
                "error": str(exc),
            })
    finally:
        _job_exit("predispatch_refresh")


async def _job_archive_backfill() -> None:
    """Gap-fill historical market_events in the DB for HippoGraph indexing."""
    _job_enter("archive_backfill")
    try:
        from app.mcp.aemo_archive import backfill_recent_gaps
        from app.db.session import db_session
        filled = await backfill_recent_gaps(db_session_factory=db_session)
        if filled:
            logger.info("Archive backfill: %d new intervals stored", filled)
        _record_success("archive_backfill")
    except Exception as exc:
        _record_failure("archive_backfill", exc)
        logger.debug("Archive backfill failed (non-fatal): %s", exc)
    finally:
        _job_exit("archive_backfill")


async def _job_mmsdm_bulk_backfill() -> None:
    """One-time MMSDM bulk backfill for multi-year reproducible history."""
    try:
        from app.mcp.aemo_archive import backfill_mmsdm_archive
        from app.db.session import db_session

        tables = [t.strip() for t in _settings.backfill_tables.split(",") if t.strip()]
        counts = await backfill_mmsdm_archive(
            db_session_factory=db_session,
            max_files=_settings.archive_bulk_backfill_max_files,
            tables=tables,
        )
        logger.info(
            "MMSDM bulk backfill complete — %d files ok / %d failed / %d months done",
            counts["files_ok"], counts["files_failed"], counts["months_completed"],
        )
        _record_success("mmsdm_bulk_backfill")
    except Exception as exc:
        _record_failure("mmsdm_bulk_backfill", exc)
        logger.debug("MMSDM bulk backfill failed (non-fatal): %s", exc)


async def _job_mmsdm_monthly_refresh() -> None:
    """Ingest the previous calendar month's MMSDM archive on the 20th of each month.

    AEMO publishes MMSDM monthly ZIP files ~2 weeks after month end.
    Running on the 20th ensures the previous month's data is available.
    This keeps the DB current automatically — no manual backfills needed.

    After ingestion, triggers an LNN retrain so models benefit from new data.
    """
    _job_enter("mmsdm_monthly_refresh")
    try:
        from datetime import timedelta
        from app.mcp.aemo_archive import backfill_mmsdm_archive
        from app.db.session import db_session

        # Target: first day of the previous month
        today = datetime.now(timezone.utc)
        first_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = first_this_month - timedelta(seconds=1)
        last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        tables = [t.strip() for t in _settings.backfill_tables.split(",") if t.strip()]
        counts = await backfill_mmsdm_archive(
            db_session_factory=db_session,
            start_date=last_month_start,
            end_date=last_month_end,
            max_files=None,
            tables=tables,
        )
        logger.info(
            "MMSDM monthly refresh (%s): %d files ok / %d failed / %d price rows",
            last_month_start.strftime("%Y-%m"),
            counts["files_ok"], counts["files_failed"], counts["price_rows"],
        )
        # Trigger LNN retrain so models incorporate the new month's data
        if counts["price_rows"] > 0:
            await _job_lnn_retrain()
        _record_success("mmsdm_monthly_refresh")
    except Exception as exc:
        _record_failure("mmsdm_monthly_refresh", exc)
        logger.warning("MMSDM monthly refresh failed: %s", exc)
    finally:
        _job_exit("mmsdm_monthly_refresh")


async def _job_commentary_cleanup() -> None:
    """Prune commentary_events older than 7 days to control table size."""
    _job_enter("commentary_cleanup")
    try:
        from app.engines.commentary.store import prune_old_events
        deleted = await prune_old_events(days=7)
        if deleted:
            logger.info("Commentary cleanup: %d events pruned", deleted)
        _record_success("commentary_cleanup")
    except Exception as exc:
        _record_failure("commentary_cleanup", exc)
        logger.debug("Commentary cleanup failed (non-fatal): %s", exc)
    finally:
        _job_exit("commentary_cleanup")


async def _job_calibration_update() -> None:
    """Run a fast 3-day walk-forward backtest per region and store calibration scores.

    Uses fast=True (baselines + LEAR only, no GBM) to complete in ~30s per region.
    Results are stored in the in-process harness registry (readable by /api/models/status)
    and written to the cache for the data-status endpoint.
    """
    from app.data.cache import get_cache

    regions = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
    cache = get_cache()
    any_ok = False

    for region in regions:
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            def _run_region(r: str):
                return asyncio.run(_calibrate_region(r))

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = await loop.run_in_executor(pool, _run_region, region)

            if result:
                await cache.set(f"calibration_{region}", result)
                any_ok = True
                logger.debug("Calibration update OK for %s", region)
        except Exception as exc:
            logger.debug("Calibration update failed for %s (non-fatal): %s", region, exc)

    if any_ok:
        _record_success("calibration_update")
        await _try_publish("data_status_changed", {"calibration_updated": True})
    else:
        _record_failure("calibration_update", Exception("no region produced calibration output"))
        state = _job_state.get("calibration_update", {})
        if state.get("consecutive_failures", 0) >= 3:
            await _try_publish("scheduler_failure", {
                "job_id": "calibration_update",
                "consecutive_failures": state["consecutive_failures"],
                "error": "no region produced calibration output",
            })


async def _calibrate_region(region: str) -> dict | None:
    """Run a fast 3-day backtest for one region and return scored results.

    Returns a dict suitable for the /api/models/status calibration field,
    or None when history is insufficient.
    """
    try:
        from app.engines.backtest import run_region_backtest
        from app.engines.forecasting.evaluation.harness import store_eval_result
        from datetime import timezone

        report = await run_region_backtest(
            region=region,
            lookback_days=3,
            horizon_intervals=6,
            step_intervals=12,
            include_lnn=False,
            fast=True,
        )

        calibration_rows = []
        for score in report.scores:
            row: dict = {
                "model": score.model_name,
                "crps": round(score.crps, 4),
                "pinball": round(score.pinball, 4),
                "spike_recall": round(score.spike_recall, 4),
                "calibration_error": round(score.calibration_error, 4),
                "skill_vs_persistence": round(score.skill_vs.get("persistence", 0.0), 4),
                "skill_vs_aemo_predispatch": round(score.skill_vs.get("aemo_predispatch", 0.0), 4),
                "n_origins": report.n_origins,
                "horizon_min": report.horizon_min,
                "per_regime": [r.to_dict() for r in (score.per_regime or [])],
            }
            calibration_rows.append(row)

        store_eval_result(region, calibration_rows)

        from datetime import datetime
        return {
            "region": region,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": 3,
            "horizon_min": report.horizon_min,
            "n_origins": report.n_origins,
            "scores": calibration_rows,
        }
    except Exception as exc:
        logger.debug("Calibration region run failed for %s: %s", region, exc)
        return None
