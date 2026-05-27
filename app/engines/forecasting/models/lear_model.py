"""LEAR — Lasso Estimated AutoRegression for electricity price forecasting.

Reference: Uniejewski et al. (2019) "Automated Variable Selection and
Shrinkage for Day-Ahead Electricity Price Forecasting"
https://doi.org/10.3390/en12010125

How it works:
  - Builds an extended feature matrix from the standard FEATURE_COLUMNS
    plus explicit autoregressive lags (1, 2, 3, 12, 288, 576 intervals)
  - Fits one Lasso (L1) linear model per quantile
  - Lasso shrinks irrelevant lags to zero, giving automatic variable selection
  - Outputs a proper QuantileForecast with monotone-sorted quantiles

This is the minimum credible econometric baseline: it uses causal price
drivers (lagged prices + demand + supply + calendar) via a model that is
interpretable and reproducible. LNN and GBM are evaluated as improvements
*over* this floor.

The AEMO predispatch column is kept in the feature set but is currently a
proxy (price copied to itself); once real predispatch data is ingested, this
model will benefit automatically.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import numpy as np

from .base import ForecastModel
from .conformal import ConformalCalibrator
from ..types import DEFAULT_QUANTILES, QuantileForecast

logger = logging.getLogger(__name__)

# Lasso alpha (regularisation strength). Empirically tuned on NEM data;
# weak enough to retain all genuine drivers, strong enough to zero out
# correlated lags that add only noise.
_DEFAULT_ALPHA = 0.01

# Autoregressive lags appended to the base feature matrix.
# Units: dispatch intervals (1 interval = 5 min).
#   1   →  5 min (last observed)
#   2   → 10 min
#   3   → 15 min
#  12   →  1 hour
# 288   → 24 hours (same interval yesterday)
# 576   → 48 hours
_AR_LAGS = [1, 2, 3, 12, 288, 576]


def _build_ar_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Append autoregressive price lags to the base feature matrix.

    Uses `y` (the target price vector) rather than the first column of X because
    `y` carries the true observed price, whereas X[:,0] carries the pre-computed
    last_price feature that may have a different lag. During test windows the last
    `max_lag` rows of y are not available; those rows use X[:,0] as a fallback.
    """
    n, d = X.shape
    max_lag = max(_AR_LAGS)
    ar_cols = np.zeros((n, len(_AR_LAGS)), dtype=np.float32)
    for j, lag in enumerate(_AR_LAGS):
        for i in range(n):
            if i >= lag:
                ar_cols[i, j] = y[i - lag]
            else:
                ar_cols[i, j] = X[i, 0]   # fallback: last_price col
    return np.hstack([X, ar_cols])


def _build_ar_features_predict(X: np.ndarray, last_y: np.ndarray) -> np.ndarray:
    """Build AR features for prediction, using the last observed prices.

    `last_y` is the tail of the training y array (at least max_lag values).
    """
    n = len(X)
    max_lag = max(_AR_LAGS)
    ar_cols = np.zeros((n, len(_AR_LAGS)), dtype=np.float32)
    for j, lag in enumerate(_AR_LAGS):
        for i in range(n):
            # Index into last_y: row i of X corresponds to last_y[-n+i-lag ... ]
            idx = -(n - i) - lag
            if abs(idx) <= len(last_y):
                ar_cols[i, j] = last_y[idx]
            else:
                ar_cols[i, j] = X[i, 0]
    return np.hstack([X, ar_cols])


class LEARModel(ForecastModel):
    """Lasso Estimated AutoRegression — proper econometric baseline.

    One Lasso-QR model per quantile; AR lags appended at fit time.
    scikit-learn is the only dependency (lighter than PyTorch / XGBoost).

    Degradation path: if sklearn is unavailable, falls back to Persistence.
    """

    name = "lear"
    quantiles: Sequence[float] = DEFAULT_QUANTILES

    def __init__(
        self,
        alpha: float = _DEFAULT_ALPHA,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        max_iter: int = 2000,
    ):
        self.alpha = alpha
        self.quantiles = quantiles
        self.max_iter = max_iter
        self._models: list | None = None
        self._scaler = None
        self._last_y: np.ndarray | None = None
        self._n_features: int = 0
        # Per-regime conformal calibrators {regime_key → ConformalCalibrator}
        self._regime_calibrators: dict[str, ConformalCalibrator] = {}
        self._feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "LEARModel":
        try:
            from sklearn.linear_model import QuantileRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("sklearn not available — LEAR fit skipped, using persistence fallback")
            self._models = None
            self._last_y = y.copy()
            return self

        X_ar = _build_ar_features(X.astype(np.float64), y.astype(np.float64))
        self._n_features = X_ar.shape[1]
        self._last_y = y.copy()

        # Build feature name list for contribution summary
        ar_names = [f"ar_lag_{lag}" for lag in _AR_LAGS]
        base_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        self._feature_names = base_names + ar_names

        # Scale features so HiGHS LP works on numerically stable inputs.
        # Without scaling, NSW1 price range (-878..+16600) causes slow convergence.
        scaler = StandardScaler()
        active = np.std(X_ar, axis=0) > 1e-8
        X_scaled = X_ar.copy()
        if active.any():
            scaler.fit(X_ar[:, active])
            X_scaled[:, active] = scaler.transform(X_ar[:, active])
        self._scaler = (scaler, active)

        self._models = []
        for q in self.quantiles:
            qr = QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs",
                                   fit_intercept=True)
            try:
                qr.fit(X_scaled, y)
                self._models.append(qr)
            except Exception as exc:
                logger.warning("LEAR quantile %s fit failed: %s", q, exc)
                self._models.append(None)

        return self

    def fit_regime_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        regimes: np.ndarray,
        coverage: float = 0.9,
    ) -> "LEARModel":
        """Fit per-regime conformal calibrators on a held-out calibration set.

        ``regimes`` is a string array of the same length as y_cal, with values
        such as ``"spike"``, ``"normal"``, ``"negative"``.  Each regime gets its
        own ConformalCalibrator; unknown regimes fall back to the global one.
        """
        if not self._models or self._last_y is None:
            return self

        cal_dummy_times = [None] * len(X_cal)
        fc_cal = self.predict_quantiles(X_cal, cal_dummy_times)  # type: ignore[arg-type]
        q_idx = {float(q): i for i, q in enumerate(fc_cal.quantiles)}
        if 0.1 not in q_idx or 0.9 not in q_idx:
            return self

        p10 = fc_cal.values[:, q_idx[0.1]]
        p90 = fc_cal.values[:, q_idx[0.9]]

        self._regime_calibrators = {}
        for regime in set(regimes):
            mask = regimes == regime
            if mask.sum() < 5:
                continue
            cal = ConformalCalibrator(coverage=coverage)
            try:
                cal.fit(y_cal[mask], p10[mask], p90[mask])
                self._regime_calibrators[regime] = cal
            except Exception as exc:
                logger.debug("Regime conformal fit failed for %s: %s", regime, exc)

        # Always fit a global fallback calibrator too
        global_cal = ConformalCalibrator(coverage=coverage)
        try:
            global_cal.fit(y_cal, p10, p90)
            self._regime_calibrators["__global__"] = global_cal
        except Exception as exc:
            logger.debug("Global conformal fit failed: %s", exc)

        return self

    def get_conformal_q_hat(self, regime: str = "__global__") -> float:
        """Return q_hat for the given regime (falls back to global if not found)."""
        cal = self._regime_calibrators.get(regime) or self._regime_calibrators.get("__global__")
        return cal.q_hat if cal is not None else 0.0

    def feature_importances(self, top_k: int = 10) -> list[dict]:
        """Return top-k features by mean absolute coefficient across all quantile models.

        Only meaningful after fit() with sklearn available.  Returns an empty list
        if the model has not been fitted or sklearn is unavailable.
        """
        if not self._models:
            return []
        coef_rows = []
        for qr in self._models:
            if qr is not None and hasattr(qr, "coef_"):
                coef_rows.append(np.abs(qr.coef_))
        if not coef_rows:
            return []
        mean_abs_coef = np.mean(np.stack(coef_rows), axis=0)
        indices = np.argsort(mean_abs_coef)[::-1][:top_k]
        names = self._feature_names
        return [
            {
                "feature": names[i] if i < len(names) else f"feat_{i}",
                "mean_abs_coef": round(float(mean_abs_coef[i]), 6),
                "rank": rank + 1,
            }
            for rank, i in enumerate(indices)
        ]

    def predict_quantiles(
        self, X: np.ndarray, target_times: Sequence[datetime]
    ) -> QuantileForecast:
        if not self._models or self._last_y is None:
            # Fallback: persistence spread
            from .baselines import _spread
            point = X[:, 0]
            values = _spread(point, self.quantiles, 0.3)
            return self._as_forecast(target_times, values)

        X_ar = _build_ar_features_predict(
            X.astype(np.float64), self._last_y.astype(np.float64)
        )
        X_scaled = X_ar.copy()
        if self._scaler is not None:
            scaler, active = self._scaler
            if active.any():
                X_scaled[:, active] = scaler.transform(X_ar[:, active])

        preds = []
        for qr in self._models:
            if qr is None:
                preds.append(X_ar[:, 0])   # fallback to last_price col
            else:
                preds.append(qr.predict(X_scaled))

        values = np.column_stack(preds)
        return self._as_forecast(target_times, values)
