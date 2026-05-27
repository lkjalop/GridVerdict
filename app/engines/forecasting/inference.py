"""LNN forecast inference — production entry point.

Called by scatter_gather._task_forecast() to get the next-interval
P10/P50/P90 for a given region.  Returns None if the model has not
been trained yet (not enough history).

Trainer instances are module-level singletons so state accumulates
across requests without a database round-trip.
"""
from __future__ import annotations

import logging
from typing import Any

from app.engines.lnn.trainer import LTCTrainer

logger = logging.getLogger(__name__)

# One trainer per NEM region — warm up as dispatch polls arrive
_trainers: dict[str, LTCTrainer] = {}
_WEIGHTS_DIR = "data/lnn_weights"


def get_trainer(region: str) -> LTCTrainer:
    if region not in _trainers:
        _trainers[region] = LTCTrainer(region, weights_dir=_WEIGHTS_DIR)
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


def maybe_train(region: str) -> None:
    """Train the model if it has enough data but hasn't been trained yet.

    Called by the scheduler every ~hour.  Safe to call redundantly —
    is_ready_to_train() returns False once training is complete and the
    buffer has not grown significantly (trainer tracks _is_trained).
    """
    trainer = get_trainer(region)
    if trainer.is_ready_to_train():
        trainer.train()
