"""Shared data types for the forecasting evaluation harness.

Scaffold module: imports nothing energy-specific. Keep it that way so the
harness reskins to other domains by swapping only the feature builder and
the data adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np


DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class QuantileForecast:
    """A probabilistic forecast: for each target timestamp, a value per quantile.

    Not frozen — np.ndarray is mutable and unhashable; frozen=True would raise
    TypeError at __hash__ time. Immutability is enforced by convention: treat
    instances as read-only after construction.
    """
    target_times: Sequence[datetime]
    quantiles: Sequence[float]
    values: np.ndarray = field(repr=False)   # (n_targets, n_quantiles)

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.target_times), len(self.quantiles)):
            raise ValueError(
                f"values shape {self.values.shape} != "
                f"({len(self.target_times)}, {len(self.quantiles)})"
            )

    def median(self) -> np.ndarray:
        qs = list(self.quantiles)
        idx = qs.index(0.5) if 0.5 in qs else len(qs) // 2
        return self.values[:, idx]


@dataclass
class ForecastWindow:
    """One walk-forward step's ground truth aligned to a forecast."""
    target_times: Sequence[datetime]
    actuals: np.ndarray = field(repr=False)


@dataclass
class ModelScore:
    model_name: str
    pinball: float
    crps: float
    spike_precision: float
    spike_recall: float
    spike_f1: float
    calibration_error: float
    p50_exceedance_rate: float = 0.0
    p90_exceedance_rate: float = 0.0
    coverage: dict[float, float] = field(default_factory=dict)
    skill_vs: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestReport:
    """The single object the NLP layer narrates and the trace records."""
    horizon_min: int
    n_origins: int
    spike_threshold: float
    scores: list[ModelScore] = field(default_factory=list)
    region_breakdown: dict[str, dict] = field(default_factory=dict)
    spike_regime_breakdown: dict[str, dict] = field(default_factory=dict)

    def best_by_crps(self) -> ModelScore | None:
        return min(self.scores, key=lambda s: s.crps) if self.scores else None

    def summary_table(self) -> str:
        def _sk(val: float) -> str:
            return f"{val:+.3f}" if val == val else "  n/a"  # nan-safe

        header = (
            f"{'model':<22} {'CRPS':>7} {'pinball':>8} {'spikeF1':>8} "
            f"{'calib':>7}  {'vs_AEMO':>8} {'vs_persist':>11} {'vs_seasonal':>12}"
        )
        rows = [header]
        for s in sorted(self.scores, key=lambda s: s.crps):
            rows.append(
                f"{s.model_name:<22} {s.crps:7.2f} {s.pinball:8.2f} "
                f"{s.spike_f1:8.3f} {s.calibration_error:7.3f}  "
                f"{_sk(s.skill_vs.get('aemo_predispatch', float('nan'))):>8} "
                f"{_sk(s.skill_vs.get('persistence', float('nan'))):>11} "
                f"{_sk(s.skill_vs.get('seasonal_naive', float('nan'))):>12}"
            )
        return "\n".join(rows)

    def to_dict(self) -> dict:
        """Structured dict for API responses and trace storage."""
        return {
            "horizon_min": self.horizon_min,
            "n_origins": self.n_origins,
            "spike_threshold": self.spike_threshold,
            "scores": [
                {
                    "model": s.model_name,
                    "crps": round(s.crps, 4),
                    "pinball": round(s.pinball, 4),
                    "spike_precision": round(s.spike_precision, 4),
                    "spike_recall": round(s.spike_recall, 4),
                    "spike_f1": round(s.spike_f1, 4),
                    "calibration_error": round(s.calibration_error, 4),
                    "p50_exceedance_rate": round(s.p50_exceedance_rate, 4),
                    "p90_exceedance_rate": round(s.p90_exceedance_rate, 4),
                    "coverage": {str(k): round(v, 4) for k, v in s.coverage.items()},
                    "skill_vs": {k: round(v, 4) for k, v in s.skill_vs.items()},
                    "skill_vs_aemo": round(s.skill_vs.get("aemo_predispatch", float("nan")), 4),
                    "skill_vs_persistence": round(s.skill_vs.get("persistence", float("nan")), 4),
                    "skill_vs_seasonal_naive": round(s.skill_vs.get("seasonal_naive", float("nan")), 4),
                }
                for s in self.scores
            ],
            "region_breakdown": self.region_breakdown,
            "spike_regime_breakdown": self.spike_regime_breakdown,
        }


@dataclass
class NewsItem:
    """A candidate explanatory event from a credible source."""
    timestamp: datetime
    source: str
    credibility_tier: int   # 1 = AEMO Market Notice (gold), 2 = AER/wire press
    title: str
    summary: str
    url: str
    region: str | None = None
