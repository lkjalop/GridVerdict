"""Probabilistic forecast scoring: pinball loss, CRPS, and skill scores.

Scaffold module (domain-agnostic). These are proper scoring rules — the
table-stakes credibility metrics. A point-forecast RMSE is deliberately NOT
the headline metric here, because for spiky electricity prices it rewards the
wrong behaviour (predicting the smooth middle and ignoring the tail).
"""
from __future__ import annotations

import numpy as np

from ..types import QuantileForecast


def pinball_loss(forecast: QuantileForecast, actuals: np.ndarray) -> float:
    """Mean pinball (quantile) loss across all quantiles and targets.

    For quantile level tau, prediction q, actual y:
        loss = tau * (y - q)        if y >= q
             = (1 - tau) * (q - y)  otherwise
    Lower is better. Averaging pinball over a quantile grid approximates CRPS.
    """
    qs = np.asarray(forecast.quantiles)            # (Q,)
    preds = forecast.values                        # (N, Q)
    y = actuals.reshape(-1, 1)                      # (N, 1)
    diff = y - preds                               # (N, Q)
    loss = np.where(diff >= 0, qs * diff, (qs - 1.0) * diff)
    return float(loss.mean())


def crps_from_quantiles(forecast: QuantileForecast, actuals: np.ndarray) -> float:
    """CRPS approximated from a quantile grid.

    The mean pinball loss over an evenly-spaced quantile grid converges to CRPS
    as the grid densifies (Gneiting & Raftery, 2007). With only P10/P50/P90 this
    is coarse; pass a denser grid (e.g. 0.05..0.95 step 0.05) when you need a
    publishable CRPS. We multiply by 2 so the scale matches the standard CRPS
    convention for the pinball approximation.
    """
    return 2.0 * pinball_loss(forecast, actuals)


def crps_ensemble(ensemble: np.ndarray, actuals: np.ndarray) -> float:
    """Exact empirical CRPS for a sample/ensemble forecast.

    ensemble: (N, M) — M samples per target. Uses the energy-form estimator
        CRPS = E|X - y| - 0.5 * E|X - X'|
    Use this when a model emits samples rather than fixed quantiles.
    """
    n, m = ensemble.shape
    y = actuals.reshape(-1, 1)
    term1 = np.abs(ensemble - y).mean(axis=1)                      # E|X - y|
    # E|X - X'| via mean absolute pairwise difference per row
    diffs = np.abs(ensemble[:, :, None] - ensemble[:, None, :])    # (N, M, M)
    term2 = diffs.mean(axis=(1, 2))
    return float((term1 - 0.5 * term2).mean())


def skill_score(model_score: float, reference_score: float) -> float:
    """Skill of a model relative to a reference forecast.

        skill = 1 - model_score / reference_score

    Positive => model beats the reference. The headline number is skill_vs_AEMO
    (reference = AEMO pre-dispatch). Negative skill is reported honestly, not hidden.
    """
    if reference_score == 0:
        return float("nan")
    return 1.0 - (model_score / reference_score)
