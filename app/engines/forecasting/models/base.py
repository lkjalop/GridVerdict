"""ForecastModel interface — the battery socket.

Scaffold module. Every forecaster (baseline, GBM, LNN, future TFT) implements
this so the harness is model-agnostic: it asks for fit/predict, never names a
model. This is the same "interchangeable batteries" discipline as the LLM tier
config — swapping a model must require no change to the harness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

import numpy as np

from ..types import QuantileForecast, DEFAULT_QUANTILES


class ForecastModel(ABC):
    """Abstract probabilistic forecaster."""

    name: str = "unnamed"
    quantiles: Sequence[float] = DEFAULT_QUANTILES

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecastModel":
        """Train on features X (n, d) and targets y (n,). Returns self.

        Implementations must use ONLY the rows handed in — the walk-forward
        harness guarantees these strictly precede the test window.
        """

    @abstractmethod
    def predict_quantiles(
        self, X: np.ndarray, target_times: Sequence[datetime]
    ) -> QuantileForecast:
        """Predict the configured quantiles for each row of X."""

    def _as_forecast(
        self, target_times: Sequence[datetime], values: np.ndarray
    ) -> QuantileForecast:
        """Helper: enforce monotone non-crossing quantiles, then box up."""
        values = np.sort(values, axis=1)  # prevent quantile crossing
        return QuantileForecast(
            target_times=target_times, quantiles=tuple(self.quantiles), values=values
        )
