"""Evaluation harness — orchestrates the whole credibility layer.

Runs each model through identical walk-forward splits, scores with proper metrics,
computes skill against every baseline (headline: AEMO pre-dispatch), and returns
one BacktestReport that the NLP layer narrates and the bitemporal trace records.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from ..types import BacktestReport, ModelScore, QuantileForecast
from ..models.base import ForecastModel
from .calibration import calibration_error, empirical_coverage
from .metrics import crps_from_quantiles, pinball_loss, skill_score
from .spike_metrics import spike_scores
from .walk_forward import walk_forward_splits

# In-process registry: region -> list[dict] calibration result (set by run_backtest callers)
_eval_registry: dict[str, list[dict]] = {}


def store_eval_result(region: str, result: list[dict]) -> None:
    """Store calibration result for /api/models/calibration display."""
    _eval_registry[region] = result


def get_last_eval_result(region: str) -> list[dict] | None:
    """Return the last stored calibration result for a region, or None."""
    return _eval_registry.get(region)


def _accumulate(
    forecasts: list[QuantileForecast],
    actuals: list[np.ndarray],
    all_times: list[Sequence],
) -> tuple[QuantileForecast, np.ndarray]:
    """Concatenate per-origin forecasts and actuals into one aligned pair."""
    qs = forecasts[0].quantiles
    times = [t for ts in all_times for t in ts]
    vals = np.vstack([f.values for f in forecasts])
    ys = np.concatenate(actuals)
    return QuantileForecast(target_times=times, quantiles=qs, values=vals), ys


def run_backtest(
    models: dict[str, ForecastModel],
    X: np.ndarray,
    y: np.ndarray,
    horizon: int,
    step: int,
    min_train: int,
    spike_threshold: float = 300.0,
    window: str = "expanding",
    reference: str = "aemo_predispatch",
    timestamps: Sequence[datetime] | None = None,
) -> BacktestReport:
    """Evaluate every model over the same rolling-origin splits.

    models:     name -> ForecastModel (include baselines so skill can be scored)
    timestamps: optional datetime index aligned to rows of X/y; if None, integer
                indices are used as stand-ins (acceptable for offline evaluation;
                production should pass real datetimes from the AEMO data layer)
    Returns a BacktestReport with per-model CRPS, pinball, spike F1, calibration,
    and skill vs each baseline.
    """
    splits = list(walk_forward_splits(len(y), horizon, step, min_train, window))
    if not splits:
        raise ValueError(
            f"No walk-forward splits produced. "
            f"Check min_train={min_train} + horizon={horizon} <= series_length={len(y)}"
        )

    per_model_fc: dict[str, list[QuantileForecast]] = {n: [] for n in models}
    per_model_y: dict[str, list[np.ndarray]] = {n: [] for n in models}
    per_model_times: dict[str, list[Sequence]] = {n: [] for n in models}

    for sp in splits:
        Xtr = X[sp.train_start:sp.train_end]
        ytr = y[sp.train_start:sp.train_end]
        Xte = X[sp.test_start:sp.test_end]
        yte = y[sp.test_start:sp.test_end]

        # Use real datetimes if provided, else integer indices as stand-ins
        if timestamps is not None:
            target_times = list(timestamps[sp.test_start:sp.test_end])
        else:
            target_times = list(range(sp.test_start, sp.test_end))

        for name, model in models.items():
            model.fit(Xtr, ytr)
            fc = model.predict_quantiles(Xte, target_times)
            per_model_fc[name].append(fc)
            per_model_y[name].append(yte)
            per_model_times[name].append(target_times)

    # Score each model
    raw_crps: dict[str, float] = {}
    aligned: dict[str, tuple[QuantileForecast, np.ndarray]] = {}

    for name in models:
        fc, ys = _accumulate(
            per_model_fc[name], per_model_y[name], per_model_times[name]
        )
        aligned[name] = (fc, ys)
        raw_crps[name] = crps_from_quantiles(fc, ys)

    baseline_names = {
        n for n in models
        if n in ("persistence", "seasonal_naive", "aemo_predispatch")
    }

    scores: list[ModelScore] = []
    for name in models:
        fc, ys = aligned[name]
        cov = empirical_coverage(fc, ys)
        prec, rec, f1 = spike_scores(fc, ys, spike_threshold)

        # Skill vs every OTHER model that is a baseline
        # Positive skill => this model beats the baseline
        skill = {
            base: skill_score(raw_crps[name], raw_crps[base])
            for base in baseline_names
            if base != name and base in raw_crps
        }

        scores.append(ModelScore(
            model_name=name,
            pinball=pinball_loss(fc, ys),
            crps=raw_crps[name],
            spike_precision=prec,
            spike_recall=rec,
            spike_f1=f1,
            calibration_error=calibration_error(cov),
            p50_exceedance_rate=_exceedance_rate(fc, ys, 0.5),
            p90_exceedance_rate=_exceedance_rate(fc, ys, 0.9),
            coverage=cov,
            skill_vs=skill,
        ))

    return BacktestReport(
        horizon_min=horizon * 5,    # 5-min dispatch intervals
        n_origins=len(splits),
        spike_threshold=spike_threshold,
        scores=scores,
    )


def _exceedance_rate(fc: QuantileForecast, ys: np.ndarray, quantile: float) -> float:
    qs = list(fc.quantiles)
    if quantile not in qs:
        return float("nan")
    idx = qs.index(quantile)
    return float(np.mean(ys > fc.values[:, idx]))
