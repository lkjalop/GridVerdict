"""Sklearn-based gradient-boosting quantile forecaster.

Uses sklearn.ensemble.GradientBoostingRegressor with loss='quantile' and
alpha=tau for each quantile level.  Always available (sklearn is already a
QRA dependency); acts as a reliable GBM component when lightgbm is absent.

Pinball (quantile) loss is native to sklearn GBM since 0.18.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import numpy as np

from ..types import DEFAULT_QUANTILES, QuantileForecast
from .base import ForecastModel

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SKLEARN_GBM = True
except ImportError:
    _HAS_SKLEARN_GBM = False


class SklearnGBMQuantileModel(ForecastModel):
    """One GradientBoostingRegressor per quantile level (loss='quantile').

    Lighter than LightGBM (fewer trees, sklearn deps only), faster to train
    on the short live-forecast windows (~288–4032 rows).  Used as the default
    GBM component inside QRA when lightgbm is not installed.
    """

    name = "gbm_sklearn"

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.10,
        subsample: float = 0.80,
        min_samples_leaf: int = 5,
    ):
        if not _HAS_SKLEARN_GBM:
            raise ImportError("scikit-learn not installed")
        self.quantiles = list(quantiles)
        self._params = dict(
            loss="quantile",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
        )
        self._models: dict[float, GradientBoostingRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnGBMQuantileModel":
        self._models = {}
        for tau in self.quantiles:
            m = GradientBoostingRegressor(alpha=tau, **self._params)
            m.fit(X, y)
            self._models[tau] = m
        return self

    def predict_quantiles(
        self,
        X: np.ndarray,
        target_times: Sequence[datetime],
    ) -> QuantileForecast:
        if not self._models:
            raise RuntimeError("Model has not been fitted")
        cols = [self._models[tau].predict(X) for tau in self.quantiles]
        values = np.column_stack(cols)  # (N, Q)
        return QuantileForecast(
            quantiles=self.quantiles,
            values=values,
            target_times=list(target_times),
        )
