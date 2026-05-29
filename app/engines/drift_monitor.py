"""Concept drift detection for live forecast residuals.

Uses River ADWIN to detect when LEAR/QRA forecast errors have significantly
shifted — indicating a regime change that warrants early model retraining
rather than waiting for the next hourly scheduler cycle.

Module-level detector singletons persist across API requests and accumulate
residual history as dispatch intervals arrive every 5 minutes.

Usage (called from scheduler._job_dispatch_refresh):
    from app.engines.drift_monitor import feed_actual, reset_detector

    drifted = feed_actual(region, actual_price, cached_forecast_p50)
    if drifted:
        # invalidate forecast cache → next query triggers full retrain
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_detectors: dict[str, Any] = {}   # region → ADWIN instance


def _get_detector(region: str) -> Any:
    if region not in _detectors:
        try:
            from river.drift import ADWIN
            # delta=0.05 → false-positive rate ≤5%. More conservative than default 0.002
            # to avoid spurious retrains during normal NEM price volatility.
            _detectors[region] = ADWIN(delta=0.05)
        except ImportError:
            _detectors[region] = None
    return _detectors.get(region)


def feed_actual(region: str, actual_price: float, forecast_p50: float | None) -> bool:
    """Feed an actual dispatch price against the predicted P50.

    Returns True when ADWIN detects a significant shift in forecast error,
    indicating concept drift that warrants early model retraining.

    The residual metric is relative absolute error with a $50/MWh floor to
    prevent instability at near-zero prices:
        RAE = |actual - p50| / max(|p50|, 50.0)

    A RAE of 1.0 means |actual - p50| ≈ |p50| — the model is off by 100%.
    ADWIN fires when the rolling mean of RAE has significantly changed.
    """
    if forecast_p50 is None:
        return False

    detector = _get_detector(region)
    if detector is None:
        return False

    rae = abs(actual_price - forecast_p50) / max(abs(forecast_p50), 50.0)

    try:
        detector.update(rae)
        if detector.drift_detected:
            logger.info(
                "Drift detected for %s: actual $%.0f vs forecast P50 $%.0f (RAE %.2f)",
                region, actual_price, forecast_p50, rae,
            )
            return True
    except Exception as exc:
        logger.debug("ADWIN update failed for %s: %s", region, exc)

    return False


def reset_detector(region: str) -> None:
    """Reset the ADWIN detector for a region (called after a forced retrain)."""
    _detectors.pop(region, None)


def detector_stats(region: str) -> dict:
    """Return diagnostic stats for the region's drift detector."""
    det = _get_detector(region)
    if det is None:
        return {"available": False}
    try:
        return {
            "available": True,
            "width": getattr(det, "_window", {}).get("n", 0),
            "drift_detected": det.drift_detected,
        }
    except Exception:
        return {"available": True}
