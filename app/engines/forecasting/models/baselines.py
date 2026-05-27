"""Baseline forecasters — the bar every real model must clear or honestly concede to.

The AEMO passthrough is the killer benchmark: beating the regulator's own free
forecast is the credible claim; not beating it is reported honestly and the value
pivots to explainability.

`baselines.py` is partly skin (the AEMO passthrough is energy-specific); persistence
and seasonal-naive are domain-agnostic.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from ..types import QuantileForecast, DEFAULT_QUANTILES
from .base import ForecastModel


def _spread(point: np.ndarray, quantiles: Sequence[float], rel_width: float) -> np.ndarray:
    """Turn a point forecast into quantiles via a symmetric multiplicative band.

    Crude but honest: baselines are not meant to be sharp, just unbiased. The
    band lets them be scored by the same probabilistic metrics as real models.
    """
    qs = np.asarray(quantiles)
    # map tau in (0,1) to a signed multiplier around 1.0
    mult = 1.0 + rel_width * (qs - 0.5) * 2.0
    return point.reshape(-1, 1) * mult.reshape(1, -1)


class PersistenceModel(ForecastModel):
    """Next price = last observed price. The trivial floor."""

    name = "persistence"

    def __init__(self, last_value_col: int = 0, rel_width: float = 0.3,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.last_value_col = last_value_col
        self.rel_width = rel_width
        self.quantiles = quantiles

    def fit(self, X, y):
        return self

    def predict_quantiles(self, X, target_times):
        point = X[:, self.last_value_col]
        return self._as_forecast(target_times, _spread(point, self.quantiles, self.rel_width))


class SeasonalNaiveModel(ForecastModel):
    """Price = value one season ago (e.g. same interval yesterday)."""

    name = "seasonal_naive"

    def __init__(self, season_col: int, rel_width: float = 0.3,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.season_col = season_col  # feature column holding the lagged seasonal value
        self.rel_width = rel_width
        self.quantiles = quantiles

    def fit(self, X, y):
        return self

    def predict_quantiles(self, X, target_times):
        point = X[:, self.season_col]
        return self._as_forecast(target_times, _spread(point, self.quantiles, self.rel_width))


class AEMOPredispatchModel(ForecastModel):
    """Passthrough of AEMO's published pre-dispatch forecast (skin/energy-specific).

    The official forecast is supplied as a feature column (the data layer attaches
    it). This is the reference the headline skill score is computed against.
    """

    name = "aemo_predispatch"

    def __init__(self, predispatch_col: int, rel_width: float = 0.25,
                 quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.predispatch_col = predispatch_col
        self.rel_width = rel_width
        self.quantiles = quantiles

    def fit(self, X, y):
        return self

    def predict_quantiles(self, X, target_times):
        point = X[:, self.predispatch_col]
        return self._as_forecast(target_times, _spread(point, self.quantiles, self.rel_width))
