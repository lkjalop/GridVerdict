"""Calibration of probabilistic forecasts.

Scaffold module. A forecast is calibrated if its stated probabilities are
honest: ~90% of actuals should fall at or below the P90 prediction. Overconfident
forecasters get discounted by reviewers; calibrated ones get trusted. Same
discipline as the JanuSec confidence calibration.
"""
from __future__ import annotations

import numpy as np

from ..types import QuantileForecast


def empirical_coverage(forecast: QuantileForecast, actuals: np.ndarray) -> dict[float, float]:
    """Fraction of actuals at or below each predicted quantile.

    For a well-calibrated forecast, coverage[tau] ~= tau.
    """
    y = actuals.reshape(-1, 1)
    below = y <= forecast.values          # (N, Q)
    cov = below.mean(axis=0)              # (Q,)
    return {float(q): float(c) for q, c in zip(forecast.quantiles, cov)}


def calibration_error(coverage: dict[float, float]) -> float:
    """Mean absolute gap between nominal and empirical coverage. Lower is better."""
    if not coverage:
        return float("nan")
    gaps = [abs(tau - emp) for tau, emp in coverage.items()]
    return float(np.mean(gaps))


def reliability_points(coverage: dict[float, float]) -> list[tuple[float, float]]:
    """(nominal, empirical) pairs for a reliability diagram. The 45-degree line
    is perfect calibration; the harness can hand these straight to the frontend
    chart tool."""
    return sorted((tau, emp) for tau, emp in coverage.items())
