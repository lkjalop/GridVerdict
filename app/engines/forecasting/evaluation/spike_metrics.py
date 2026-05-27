"""Spike-aware evaluation.

Scaffold module. Aggregate error metrics hide the tail because spikes are rare.
For electricity prices the spikes are where the money and the risk live, so we
score extreme-event capture on its own terms: did the forecast flag the spike?
"""
from __future__ import annotations

import numpy as np

from ..types import QuantileForecast


def prob_above_threshold(forecast: QuantileForecast, threshold: float) -> np.ndarray:
    """Estimate P(price > threshold) per target from the quantile grid.

    If the quantile at level tau equals q, then P(price <= q) = tau. We take the
    highest tau whose quantile value is still below the threshold; the implied
    exceedance probability is 1 - tau. Coarse but monotone — fine for a spike flag.
    """
    qs = np.asarray(forecast.quantiles)        # ascending
    vals = forecast.values                     # (N, Q)
    n = vals.shape[0]
    out = np.zeros(n)
    for i in range(n):
        below = vals[i] < threshold
        if below.all():
            out[i] = 1.0 - qs[-1]
        elif (~below).all():
            out[i] = 1.0
        else:
            out[i] = 1.0 - qs[below].max()
    return out


def spike_scores(
    forecast: QuantileForecast,
    actuals: np.ndarray,
    threshold: float,
    prob_cutoff: float = 0.2,
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for spike detection.

    threshold: price level defining a spike (e.g. 300.0 $/MWh).
    prob_cutoff: predict 'spike' when estimated P(price > threshold) >= cutoff.
                 0.2 is deliberately permissive — missing a real spike usually
                 costs more than a false alarm in a dispatch decision.
    """
    actual_spike = actuals > threshold
    pred_spike = prob_above_threshold(forecast, threshold) >= prob_cutoff

    tp = int(np.sum(pred_spike & actual_spike))
    fp = int(np.sum(pred_spike & ~actual_spike))
    fn = int(np.sum(~pred_spike & actual_spike))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1
