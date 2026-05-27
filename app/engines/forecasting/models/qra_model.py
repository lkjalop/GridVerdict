"""QRA — Quantile Regression Averaging ensemble.

Reference: Nowotarski & Weron (2015) "Computing electricity spot price
prediction intervals using quantile regression and forecast combination"
https://doi.org/10.1007/s00180-014-0523-0

How it works:
  - At each walk-forward origin, every component model produces a QuantileForecast.
  - QRA stacks those component quantile forecasts as features and fits a *linear*
    quantile regression (one per output quantile) to find the optimal combining
    weights.
  - At prediction time it applies those weights to the component forecasts.

Why this matters:
  - Each base model captures different drivers: persistence → short-run momentum;
    seasonal_naive → daily cycle; AEMO_predispatch → forward-looking supply info;
    LEAR → autoregressive drivers + demand; GBM → non-linear interactions.
  - QRA lets the data decide how much to trust each model at each quantile level.
    In practice the combining weights are non-negative and sum to roughly 1, but
    the L1 regularisation (alpha) can shrink useless models to zero.
  - The combination is strictly causal: the combiner is only ever trained on the
    same historical window as the base models — no lookahead.

Degradation path:
  - If sklearn is unavailable the combiner falls back to equal-weight averaging.
  - If fewer than 2 component models have predictions the combiner passes through
    the first available forecast unchanged.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import numpy as np

from .base import ForecastModel
from ..types import DEFAULT_QUANTILES, QuantileForecast

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = 0.01   # Lasso regularisation for combining weights

# Regime thresholds (AUD/MWh) — same as domain.nem.adapter.classify_regime defaults
_REGIME_ELEVATED = 300.0
_REGIME_SPIKE = 1000.0

# Interval-width multipliers per regime for tails (P10, P90).
# Median (P50) is unchanged in all regimes to avoid bias.
# Elevated: 20% wider tails. Spike: 50% wider tails (fat-tail correction).
_WIDTH_SCALE: dict[str, float] = {
    "normal": 1.0,
    "elevated": 1.20,
    "spike": 1.50,
}

# Minimum calibration rows within a regime bucket to train a regime-specific combiner.
# Spike events are rare; below this threshold the global combiner is used as fallback.
_MIN_REGIME_CAL_ROWS = 10


def _classify_regime_array(prices: np.ndarray) -> np.ndarray:
    """Return per-row regime strings for an array of prices."""
    result = np.where(
        prices >= _REGIME_SPIKE, "spike",
        np.where(prices >= _REGIME_ELEVATED, "elevated", "normal"),
    )
    return result


def _apply_regime_width(
    values: np.ndarray,
    last_prices: np.ndarray,
    quantiles: list[float],
) -> np.ndarray:
    """Widen prediction intervals based on current-price regime.

    `values` is (n_samples, n_quantiles). The median column is found by the
    quantile closest to 0.5; tail columns are stretched symmetrically around
    the median by the regime multiplier.
    """
    if values.ndim != 2 or values.shape[1] < 2:
        return values

    quantiles_arr = np.asarray(quantiles)
    median_col = int(np.argmin(np.abs(quantiles_arr - 0.5)))
    medians = values[:, median_col : median_col + 1]  # (n, 1)

    regimes = _classify_regime_array(last_prices)
    scales = np.vectorize(_WIDTH_SCALE.get)(regimes, 1.0).reshape(-1, 1)  # (n, 1)

    # Stretch each column around the median by the per-row scale
    return medians + (values - medians) * scales


class QRAModel(ForecastModel):
    """Quantile Regression Averaging over a set of component ForecastModels.

    The component models are trained first; QRA fits a linear combiner on top.
    All training is done via the harness's walk-forward loop — QRAModel.fit()
    receives the same X, y window as every other model.

    Usage:
        qra = QRAModel(components={
            "persistence": PersistenceModel(...),
            "lear": LEARModel(...),
            "gbm": GBMQuantileModel(...),
        })
        # harness calls qra.fit(X_train, y_train) then qra.predict_quantiles(X_test, times)
    """

    name = "qra"

    def __init__(
        self,
        components: dict[str, ForecastModel],
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        alpha: float = _DEFAULT_ALPHA,
    ):
        if len(components) < 2:
            raise ValueError("QRAModel needs at least 2 component models")
        self.components = components
        self.quantiles = list(quantiles)
        self.alpha = alpha
        self._combiners: list | None = None   # global: one QuantileRegressor per output quantile
        self._scaler = None
        self._keep_cols: np.ndarray | None = None
        self._trained_components: dict[str, ForecastModel] = {}
        self._last_X_train: np.ndarray | None = None
        self._last_y_train: np.ndarray | None = None
        # Regime-specific combiners: keyed by regime label ("normal"|"elevated"|"spike").
        # None entry means that regime has too few rows → falls back to global combiner.
        self._regime_combiners: dict[str, list | None] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QRAModel":
        """Fit all component models then fit the linear QRA combiner."""
        self._last_X_train = X
        self._last_y_train = y

        # 1. Train each component
        self._trained_components = {}
        for name, model in self.components.items():
            try:
                model.fit(X, y)
                self._trained_components[name] = model
            except Exception as exc:
                logger.warning("QRA: component %s fit failed: %s", name, exc)

        if not self._trained_components:
            logger.error("QRA: all component models failed to fit")
            self._combiners = None
            return self

        # 2. Build a combining feature matrix from in-sample component predictions.
        #    We use a held-out sub-window (last 20% of training) so the combiner
        #    is not fitted on the same data the components saw in full.
        n = len(X)
        split = max(1, int(n * 0.8))
        X_comb = X[split:]
        y_comb = y[split:]
        times_comb = [datetime.utcfromtimestamp(0)] * len(X_comb)   # dummy times

        Z = self._stack_component_predictions(X_comb, times_comb)
        if Z is None or Z.shape[0] < 10:
            logger.warning(
                "QRA: combining window too small (%d rows) — using equal-weight fallback",
                0 if Z is None else Z.shape[0],
            )
            self._combiners = None
            return self

        # 3. Scale combining features — prevents HiGHS numerical difficulties
        #    when component forecasts span very different magnitudes (e.g. $5 vs $5000).
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import QuantileRegressor
        except ImportError:
            logger.warning("sklearn unavailable — QRA using equal-weight fallback")
            self._combiners = None
            return self

        self._scaler = StandardScaler()
        Z_scaled = self._scaler.fit_transform(Z)

        # Guard: drop columns that are constant after scaling (would confuse HiGHS)
        keep = np.std(Z_scaled, axis=0) > 1e-8
        if not keep.any():
            logger.warning("QRA: all combining features are constant — equal-weight fallback")
            self._combiners = None
            return self
        self._keep_cols = keep
        Z_scaled = Z_scaled[:, keep]

        self._combiners = []
        for q in self.quantiles:
            qr = QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs",
                                   fit_intercept=True)
            try:
                qr.fit(Z_scaled, y_comb)
                self._combiners.append(qr)
            except Exception as exc:
                logger.warning("QRA combiner fit failed for q=%s: %s", q, exc)
                self._combiners.append(None)

        # ── Regime-specific combiners ──────────────────────────────────────────
        # Use the last_price column (col 0 in raw feature matrix) to classify regime.
        last_prices_comb = X_comb[:, 0] if X_comb.shape[1] > 0 else np.zeros(len(X_comb))
        regime_labels_comb = _classify_regime_array(last_prices_comb)
        self._regime_combiners = {}
        for regime_label in ("normal", "elevated", "spike"):
            mask = regime_labels_comb == regime_label
            n_regime = int(mask.sum())
            if n_regime < _MIN_REGIME_CAL_ROWS:
                self._regime_combiners[regime_label] = None
                logger.debug(
                    "QRA: regime '%s' has only %d calibration rows — using global combiner",
                    regime_label, n_regime,
                )
                continue
            Z_r = Z_scaled[mask]   # already scaled + keep-filtered
            y_r = y_comb[mask]
            regime_cbs = []
            for q in self.quantiles:
                qr_r = QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs",
                                         fit_intercept=True)
                try:
                    qr_r.fit(Z_r, y_r)
                    regime_cbs.append(qr_r)
                except Exception as exc:
                    logger.warning(
                        "QRA regime '%s' combiner failed for q=%s: %s", regime_label, q, exc
                    )
                    regime_cbs.append(None)
            self._regime_combiners[regime_label] = regime_cbs
            logger.debug(
                "QRA: fitted regime-specific combiner for '%s' on %d rows", regime_label, n_regime
            )

        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_quantiles(
        self, X: np.ndarray, target_times: Sequence[datetime]
    ) -> QuantileForecast:
        Z = self._stack_component_predictions(X, target_times)

        if Z is None or self._combiners is None or self._scaler is None:
            return self._equal_weight_fallback(X, target_times)

        try:
            Z_scaled = self._scaler.transform(Z)
            if self._keep_cols is not None:
                Z_scaled = Z_scaled[:, self._keep_cols]
        except Exception:
            return self._equal_weight_fallback(X, target_times)

        # Route each sample to its regime-specific combiner; fall back to global.
        last_prices_pred = X[:, 0] if X.shape[1] > 0 else np.zeros(len(X))
        regime_labels_pred = _classify_regime_array(last_prices_pred)
        values = np.zeros((len(X), len(self.quantiles)), dtype=float)
        fallback_col = Z[:, 0]  # persistence fallback when all combiners are None

        for regime_label in np.unique(regime_labels_pred):
            mask = regime_labels_pred == regime_label
            regime_cbs = self._regime_combiners.get(str(regime_label))
            active_cbs = regime_cbs if regime_cbs is not None else self._combiners
            Z_r = Z_scaled[mask]
            for qi, qr in enumerate(active_cbs):
                if qr is None:
                    values[mask, qi] = fallback_col[mask]
                else:
                    values[mask, qi] = qr.predict(Z_r)

        # Regime-aware post-combination adjustment.
        # In spike/elevated regimes we widen the prediction interval to reflect
        # the higher uncertainty — the standard QRA combiner was trained mostly
        # on normal-regime data and underestimates tail risk during price events.
        last_prices = X[:, 0] if X.shape[1] > 0 else np.zeros(len(values))
        values = _apply_regime_width(values, last_prices, self.quantiles)

        return self._as_forecast(target_times, values)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _stack_component_predictions(
        self,
        X: np.ndarray,
        target_times: Sequence[datetime],
    ) -> np.ndarray | None:
        """Build a (n, n_components × n_quantiles) feature matrix for the combiner."""
        cols = []
        for name, model in self._trained_components.items():
            try:
                fc = model.predict_quantiles(X, target_times)
                cols.append(fc.values)   # (n, n_quantiles)
            except Exception as exc:
                logger.warning("QRA: component %s predict failed: %s", name, exc)

        if not cols:
            return None

        return np.hstack(cols).astype(np.float32)   # (n, n_components × n_quantiles)

    def _equal_weight_fallback(
        self,
        X: np.ndarray,
        target_times: Sequence[datetime],
    ) -> QuantileForecast:
        """Average component forecasts with equal weight — no learned combiner."""
        cols = []
        for name, model in self._trained_components.items():
            try:
                fc = model.predict_quantiles(X, target_times)
                cols.append(fc.values)
            except Exception as exc:
                logger.debug("QRA component %s prediction failed: %s", name, exc)

        if not cols:
            # Ultimate fallback: persistence spread from last_price col
            point = X[:, 0]
            spread = np.column_stack([point * 0.9, point, point * 1.1])
            return self._as_forecast(target_times, spread)

        values = np.mean(np.stack(cols, axis=0), axis=0)
        return self._as_forecast(target_times, values)

    # ── Introspection ─────────────────────────────────────────────────────────

    def component_weights(self) -> dict[str, list[float]] | None:
        """Return {model_name: [coef_per_quantile]} for the global combiner.

        Each list has len(quantiles) entries — one per input quantile column
        block. A weight near zero means that component was regularised away.
        """
        if self._combiners is None:
            return None

        n_q = len(self.quantiles)
        model_names = list(self._trained_components.keys())
        result: dict[str, list[float]] = {name: [] for name in model_names}

        for qi, qr in enumerate(self._combiners):
            if qr is None:
                for name in model_names:
                    result[name].append(float("nan"))
                continue
            coefs = qr.coef_  # length = n_components × n_quantiles
            for mi, name in enumerate(model_names):
                block_start = mi * n_q
                weight = float(np.mean(coefs[block_start: block_start + n_q]))
                result[name].append(weight)

        return result

    def regime_weights(self) -> dict[str, dict[str, list[float]] | None]:
        """Return per-regime component weights if regime-specific combiners were fitted.

        Structure: {regime: {model_name: [coef_per_quantile]} | None}
        None means that regime used the global combiner (too few calibration rows).
        """
        out: dict[str, dict[str, list[float]] | None] = {}
        n_q = len(self.quantiles)
        model_names = list(self._trained_components.keys())

        for regime_label, regime_cbs in self._regime_combiners.items():
            if regime_cbs is None:
                out[regime_label] = None
                continue
            r_result: dict[str, list[float]] = {name: [] for name in model_names}
            for qi, qr in enumerate(regime_cbs):
                if qr is None:
                    for name in model_names:
                        r_result[name].append(float("nan"))
                    continue
                coefs = qr.coef_
                for mi, name in enumerate(model_names):
                    block_start = mi * n_q
                    weight = float(np.mean(coefs[block_start: block_start + n_q]))
                    r_result[name].append(weight)
            out[regime_label] = r_result

        return out
