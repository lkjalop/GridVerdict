"""Gradient-boosting quantile forecaster (LightGBM).

The "boring but rigorous" battery — usually the credibility winner, because a
well-evaluated GBM beats an exotic poorly-evaluated net in the eyes of anyone who
knows forecasting. One model per quantile via the pinball objective.

Requires: lightgbm. Install with `pip install lightgbm`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from ..types import QuantileForecast, DEFAULT_QUANTILES
from .base import ForecastModel

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # keep the module importable without the dep
    _HAS_LGB = False


class GBMQuantileModel(ForecastModel):
    """One LightGBM regressor per quantile level (objective='quantile')."""

    name = "gbm_quantile"

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ):
        if not _HAS_LGB:
            raise ImportError("lightgbm not installed; `pip install lightgbm`")
        self.quantiles = quantiles
        self._params = dict(
            objective="quantile",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            verbose=-1,
        )
        self._models: dict[float, "lgb.LGBMRegressor"] = {}

    def fit(self, X, y):
        self._models = {}
        for tau in self.quantiles:
            m = lgb.LGBMRegressor(alpha=tau, **self._params)
            m.fit(X, y)
            self._models[tau] = m
        return self

    def predict_quantiles(self, X, target_times):
        cols = [self._models[tau].predict(X) for tau in self.quantiles]
        values = np.column_stack(cols)  # (N, Q)
        return self._as_forecast(target_times, values)
