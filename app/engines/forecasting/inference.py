"""LNN forecast inference — production entry point.

Called by scatter_gather._task_forecast() to get the next-interval
P10/P50/P90 for a given region.  Returns None if the model has not
been trained yet (not enough history).

Trainer instances are module-level singletons so state accumulates
across requests without a database round-trip.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.engines.lnn.trainer import LTCTrainer

logger = logging.getLogger(__name__)

# One trainer per NEM region — warm up as dispatch polls arrive
_trainers: dict[str, LTCTrainer] = {}
_WEIGHTS_DIR = "data/lnn_weights"

# SA1 is highly volatile (frequent negatives + $15k spikes) — larger model needed
_REGION_CONFIG: dict[str, dict] = {
    "SA1": {"hidden_size": 64, "epochs": 120, "lr": 5e-4},
}


def get_trainer(region: str) -> LTCTrainer:
    if region not in _trainers:
        cfg = _REGION_CONFIG.get(region, {})
        _trainers[region] = LTCTrainer(region, weights_dir=_WEIGHTS_DIR, **cfg)
    return _trainers[region]


def feed_interval(region: str, interval: dict[str, Any]) -> None:
    """Accumulate one dispatch interval into the region's training buffer.

    Call this after every successful dispatch poll (from the scheduler or
    scatter_gather).  Once min_samples intervals have been seen, the
    trainer becomes ready; the scheduler triggers train() periodically.
    """
    get_trainer(region).accumulate_interval(interval)


def get_forecast(region: str) -> dict[str, float] | None:
    """Return P10/P50/P90 for the next dispatch interval, or None.

    Uses the trainer's internal history buffer — no data argument needed.
    Returns None until the model has been trained (requires min_samples intervals).
    """
    return get_trainer(region).predict_from_buffer()


def get_multistep_forecast(region: str, steps: int) -> list[dict[str, float]] | None:
    """Autoregressive multi-step rollout for up to `steps` intervals.

    Returns a list of P10/P50/P90 dicts (one per future interval), or None
    if the model is not yet trained or the buffer is too short.
    """
    return get_trainer(region).predict_multistep_from_buffer(steps)


def maybe_train(region: str) -> None:
    """Train the model if it has enough data but hasn't been trained yet.

    Called by the scheduler every ~hour.  Safe to call redundantly —
    is_ready_to_train() returns False once training is complete and the
    buffer has not grown significantly (trainer tracks _is_trained).
    """
    trainer = get_trainer(region)
    if trainer.is_ready_to_train():
        trainer.train()


async def bootstrap_from_db(session: Any, lookback_days: int = 7) -> None:
    """Seed all region trainers from recent DB dispatch history on startup.

    Loads the last `lookback_days` of dispatch price rows from market_events,
    feeds them to each region's trainer buffer, then triggers training.
    Called once from the app lifespan so the LNN is ready immediately
    after a restart rather than waiting 24h for live intervals to accumulate.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    _REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]

    # We need ~600 intervals per region (2 days) minimum to train.
    # Use a subquery to grab the most recent 600 rows per region so the
    # bootstrap works even when the market_events table has gaps.
    _min_needed = 600
    try:
        result = await session.execute(
            text("""
                SELECT region, valid_time, price_rrp, demand_mw, availability_mw
                FROM (
                    SELECT region, valid_time, price_rrp, demand_mw, availability_mw,
                           ROW_NUMBER() OVER (PARTITION BY region ORDER BY valid_time DESC) AS rn
                    FROM market_events
                    WHERE source = 'AEMO_DISPATCH_PRICE'
                      AND region = ANY(:regions)
                ) sub
                WHERE rn <= :limit
                ORDER BY region, valid_time
            """),
            {"regions": _REGIONS, "limit": _min_needed},
        )
        rows = result.fetchall()
    except Exception as exc:
        logger.warning("LNN bootstrap DB query failed: %s", exc)
        return

    by_region: dict[str, list] = {r: [] for r in _REGIONS}
    for row in rows:
        if row.region in by_region:
            by_region[row.region].append(row)

    for region, region_rows in by_region.items():
        if not region_rows:
            continue
        trainer = get_trainer(region)
        if trainer.is_trained:
            logger.info("LNN %s already trained, skipping bootstrap", region)
            continue
        for row in region_rows:
            trainer.accumulate_interval({
                "region": row.region,
                "price_rrp": float(row.price_rrp or 0),
                "demand_mw": float(row.demand_mw or 0),
                "availability_mw": float(row.availability_mw or 0),
                "valid_time": row.valid_time.isoformat() if hasattr(row.valid_time, 'isoformat') else str(row.valid_time),
            })
        logger.info(
            "LNN bootstrap: fed %d intervals to %s trainer (buffer=%d, ready=%s)",
            len(region_rows), region, len(trainer._prices), trainer.is_ready_to_train(),
        )
        if trainer.is_ready_to_train():
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, trainer.train)
            logger.info("LNN %s training started in background", region)
