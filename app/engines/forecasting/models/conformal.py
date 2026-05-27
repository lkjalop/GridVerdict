"""Split-conformal prediction interval calibrator (CQR variant).

Conformalized Quantile Regression (Romano et al., 2019):
  Given a calibration set with predicted intervals [p10, p90] and actuals y,
  compute nonconformity scores and find the quantile that inflates the interval
  to achieve marginal (1 - alpha) coverage on held-out data.

Usage:
    cal = ConformalCalibrator(coverage=0.9)
    cal.fit(y_cal, p10_cal, p90_cal)
    p10_adj, p90_adj = cal.adjust(p10_future, p90_future)
    # cal.q_hat is the symmetric expansion applied to both endpoints
"""
from __future__ import annotations

import numpy as np


class ConformalCalibrator:
    """Calibrate a quantile forecaster using split-conformal CQR.

    Nonconformity score per sample:
        s_i = max(p10_i - y_i,  y_i - p90_i)

    Negative scores mean the interval already covered that point.
    The (1-alpha) quantile of these scores is the symmetric inflation q_hat:
        p10_adj = p10 - q_hat
        p90_adj = p90 + q_hat

    Finite-sample guarantee: coverage ≥ (1-alpha) when calibration and test
    points are exchangeable (standard split-conformal assumption).
    """

    def __init__(self, coverage: float = 0.9) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage!r}")
        self.coverage: float = coverage
        self.q_hat: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        y_cal: np.ndarray,
        p10_cal: np.ndarray,
        p90_cal: np.ndarray,
    ) -> "ConformalCalibrator":
        """Compute q_hat from held-out calibration samples.

        Args:
            y_cal:   Actual values, shape (n,)
            p10_cal: Lower predicted quantile, shape (n,)
            p90_cal: Upper predicted quantile, shape (n,)
        """
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        p10_cal = np.asarray(p10_cal, dtype=float).ravel()
        p90_cal = np.asarray(p90_cal, dtype=float).ravel()
        n = len(y_cal)
        if n == 0:
            return self
        scores = np.maximum(p10_cal - y_cal, y_cal - p90_cal)
        # Conformal quantile level: ceil((n+1)(1-alpha)) / n — guarantees coverage.
        alpha = 1.0 - self.coverage
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        self.q_hat = float(np.quantile(scores, level))
        self._fitted = True
        return self

    def adjust(
        self,
        p10: np.ndarray,
        p90: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand [p10, p90] symmetrically by q_hat to achieve target coverage.

        Returns unchanged arrays when not fitted or q_hat ≤ 0 (already covering).
        """
        if not self._fitted or self.q_hat <= 0.0:
            return p10, p90
        return p10 - self.q_hat, p90 + self.q_hat
