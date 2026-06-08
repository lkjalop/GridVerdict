"""Live probabilistic forecast service for the cockpit chart.

This trains only on persisted market history and returns only real model
outputs. If a model cannot fit, it is reported as unavailable rather than
fabricated. The route consumer can still render historical prices without
forecast bands.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.engines.backtest import _fetch_history
from app.engines.forecasting.features.market_features import (
    build_features,
    COL_LAST_PRICE,
    COL_SEASONAL,
    COL_AEMO,
    COL_TEMP_C,
    COL_WIND_KMH,
)
from app.engines.forecasting.models.conformal import ConformalCalibrator
from app.engines.forecasting.models.meta_ensemble import blend_forecasts
from app.engines.forecasting.models.baselines import (
    AEMOPredispatchModel,
    PersistenceModel,
    SeasonalNaiveModel,
)
from app.engines.forecasting.models.lear_model import LEARModel
from app.engines.forecasting.models.qra_model import QRAModel
from app.engines.forecasting.models.sklearn_gbm_model import SklearnGBMQuantileModel
from app.engines.forecasting.model_registry import register_model, make_training_ref
from config.settings import get_settings

try:
    from app.engines.forecasting.models.lnn_model import LNNQuantileModel
    _HAS_LNN = True
except ImportError:
    _HAS_LNN = False

try:
    from app.engines.forecasting.models.tcn_model import TCNQuantileModel
    _HAS_TCN = True
except ImportError:
    _HAS_TCN = False

logger = logging.getLogger(__name__)

_MIN_TRAIN_INTERVALS = 288
_DEFAULT_HORIZON_INTERVALS = 48   # 4-hour ahead forecast (48 × 5 min)
_TRAIN_HORIZON = 6                 # training alignment: 30-min-ahead targets (best-calibrated range)


async def run_live_forecast(
    region: str,
    lookback_days: int = 14,
    horizon_intervals: int = _DEFAULT_HORIZON_INTERVALS,
    weather_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train LEAR/QRA on current stored history and forecast the next intervals."""
    region = region.upper()
    anchor = await _latest_dispatch_time(region)
    if anchor is None:
        return _unavailable(region, "no persisted AEMO_DISPATCH_PRICE rows")
    series, effective_lookback_days = await _fetch_sufficient_history(
        region, lookback_days, anchor, horizon_intervals
    )

    def _run() -> dict[str, Any]:
        return _run_sync(
            region,
            effective_lookback_days,
            horizon_intervals,
            anchor,
            series,
            weather_context,
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


async def _latest_dispatch_time(region: str) -> datetime | None:
    try:
        from sqlalchemy import text
        from app.db.session import db_session

        async with db_session() as session:
            result = await session.execute(
                text("""
                    SELECT MAX(valid_time)
                    FROM market_events
                    WHERE region = :region
                      AND source = 'AEMO_DISPATCH_PRICE'
                """),
                {"region": region},
            )
            value = result.scalar()
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as exc:
        logger.warning("Latest dispatch lookup failed for %s: %s", region, exc)
        return None


_MIN_CONFORMAL_CAL_SIZE = 20  # minimum calibration set rows to fit ConformalCalibrator


def _run_sync(
    region: str,
    lookback_days: int,
    horizon_intervals: int,
    anchor: datetime,
    series: list[dict[str, Any]] | None = None,
    weather_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_lookback_days = lookback_days
    if series is None:
        maybe_series = _fetch_history(region, lookback_days, end_date=anchor)
        series = asyncio.run(maybe_series) if inspect.isawaitable(maybe_series) else maybe_series

    if len(series) < _MIN_TRAIN_INTERVALS + horizon_intervals + 1:
        return _unavailable(
            region,
            f"insufficient persisted history: {len(series)} intervals, need at least "
            f"{_MIN_TRAIN_INTERVALS + horizon_intervals + 1}",
            anchor,
        )

    target_times = [r["valid_time"] for r in series]
    X_raw, y_raw = build_features(series, target_times)
    if len(X_raw) < _MIN_TRAIN_INTERVALS + horizon_intervals + 1:
        return _unavailable(region, "feature matrix too short after joins", anchor)

    # Training uses _TRAIN_HORIZON (30-min-ahead targets) regardless of forecast horizon.
    # This preserves model calibration quality; uncertainty widening handles longer horizons.
    h = min(horizon_intervals, _TRAIN_HORIZON)
    X_train = X_raw[:-h]
    y_train = y_raw[h:]
    if len(X_train) < _MIN_TRAIN_INTERVALS:
        return _unavailable(region, "not enough aligned train rows", anchor)

    X_future = _future_feature_rows(X_raw[-1], horizon_intervals, weather_context)
    future_times = [anchor + timedelta(minutes=5 * (i + 1)) for i in range(horizon_intervals)]

    # 80/20 chronological split: fit on first 80%, calibrate on last 20%.
    cal_size = max(_MIN_CONFORMAL_CAL_SIZE, len(X_train) // 5)
    fit_size = len(X_train) - cal_size
    if fit_size >= _MIN_TRAIN_INTERVALS:
        X_fit, y_fit = X_train[:fit_size], y_train[:fit_size]
        X_cal, y_cal = X_train[fit_size:], y_train[fit_size:]
    else:
        # Not enough data for a calibration split — train on all, skip conformal.
        X_fit, y_fit = X_train, y_train
        X_cal, y_cal = np.empty((0, X_train.shape[1])), np.empty((0,))

    qra_components: dict = {
        "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
        "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
        "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
        "lear": LEARModel(),
    }
    try:
        qra_components["gbm"] = SklearnGBMQuantileModel()
    except Exception as exc:
        logger.debug("sklearn GBM unavailable for QRA: %s", exc)

    models = {
        "lear": LEARModel(),
        "qra": QRAModel(components=qra_components),
    }
    settings = get_settings()
    if settings.enable_experimental_sequence_forecasters and _HAS_LNN:
        try:
            models["lnn"] = LNNQuantileModel(epochs=30)
        except Exception as exc:
            logger.debug("LNN init failed (torch/ncps unavailable): %s", exc)
    if settings.enable_experimental_sequence_forecasters and _HAS_TCN:
        try:
            models["tcn"] = TCNQuantileModel(epochs=30)
        except Exception as exc:
            logger.debug("TCN init failed (torch unavailable): %s", exc)

    train_start = target_times[0] if target_times else anchor
    train_ref_base = make_training_ref(region, train_start, anchor, fit_size)

    # Derive current regime for regime-specific q_hat selection (E1)
    _current_price = float(X_future[0, COL_LAST_PRICE]) if X_future.shape[0] > 0 else 0.0
    try:
        from domain.nem.adapter import classify_regime as _classify_regime
        _current_regime = _classify_regime(_current_price, region)
    except Exception:
        _current_regime = "__global__"

    # Fit regime-conditional conformal calibration once for all models (E1)
    _regimes_cal: np.ndarray | None = None
    if len(X_cal) >= _MIN_CONFORMAL_CAL_SIZE:
        try:
            from domain.nem.adapter import _REGIME_THRESHOLDS
            _thresholds = _REGIME_THRESHOLDS.get(region, _REGIME_THRESHOLDS.get("NSW1", {}))
            _ext = float(_thresholds.get("extreme", 1000))
            _spk = float(_thresholds.get("spike", 300))
            _elv = float(_thresholds.get("elevated", 100))
            def _p_to_r(p: float) -> str:
                if p >= _ext: return "extreme"
                if p >= _spk: return "spike"
                if p >= _elv: return "elevated"
                return "normal"
            _regimes_cal = np.array([_p_to_r(float(p)) for p in y_cal])
        except Exception as exc:
            logger.debug("Regime label derivation failed (non-fatal): %s", exc)

    forecasts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name, model in models.items():
        try:
            model.fit(X_fit, y_fit)

            # Regime-conditional conformal calibration for LEAR (E1)
            if _regimes_cal is not None and hasattr(model, "fit_regime_conformal"):
                try:
                    model.fit_regime_conformal(X_cal, y_cal, _regimes_cal)
                except Exception as exc:
                    logger.debug("Regime conformal fit failed for %s/%s: %s", region, name, exc)

            register_model(name, "1.0.0", train_ref_base, {"cal_size": cal_size, "fit_size": fit_size})
            fc = model.predict_quantiles(X_future, future_times)

            q_hat = 0.0
            calibrated = False
            if len(X_cal) >= _MIN_CONFORMAL_CAL_SIZE:
                try:
                    cal_dummy_times = [anchor] * len(X_cal)
                    fc_cal = model.predict_quantiles(X_cal, cal_dummy_times)
                    q_idx_cal = {float(q): i for i, q in enumerate(fc_cal.quantiles)}
                    p10_cal = fc_cal.values[:, q_idx_cal[0.1]]
                    p90_cal = fc_cal.values[:, q_idx_cal[0.9]]
                    calibrator = ConformalCalibrator(coverage=0.9)
                    calibrator.fit(y_cal, p10_cal, p90_cal)
                    # Use regime-specific q_hat when available (E1)
                    if hasattr(model, "get_conformal_q_hat"):
                        _regime_qhat = model.get_conformal_q_hat(_current_regime)
                        q_hat = _regime_qhat if _regime_qhat > 0 else calibrator.q_hat
                    else:
                        q_hat = calibrator.q_hat
                    calibrated = True
                except Exception as exc:
                    logger.debug("Conformal calibration failed for %s/%s: %s", region, name, exc)

            spike_probs: dict | None = None
            spike_probs_series: dict | None = None
            if hasattr(model, "predict_spike_probs"):
                try:
                    raw = model.predict_spike_probs(X_future)
                    # First-interval scalars (for answer text / downstream logic)
                    spike_probs = {k: round(float(v[0]), 4) for k, v in raw.items()}
                    # Full per-interval arrays (for chart overlay)
                    spike_probs_series = {
                        k: [round(float(p), 4) for p in v] for k, v in raw.items()
                    }
                except Exception as exc:
                    logger.debug("Spike prob prediction failed for %s/%s: %s", region, name, exc)

            # Per-sample feature attribution — LEAR uses linear coef×x; LNN uses IG
            feature_attributions: list[tuple[str, float]] = []
            if hasattr(model, "feature_attribution") and X_future.shape[0] > 0:
                try:
                    feature_attributions = model.feature_attribution(X_future[0], top_n=5)
                except Exception as exc:
                    logger.debug("LEAR feature attribution failed for %s/%s: %s", region, name, exc)
            elif hasattr(model, "integrated_gradients") and X_future.shape[0] > 0:
                try:
                    feature_attributions = model.integrated_gradients(X_future[0])
                except Exception as exc:
                    logger.debug("LNN IG attribution failed for %s/%s: %s", region, name, exc)

            forecasts.append(
                _forecast_to_dict(name, fc, "trained on persisted dispatch history",
                                  calibrated=calibrated, q_hat=q_hat,
                                  spike_probs=spike_probs, spike_probs_series=spike_probs_series,
                                  feature_attributions=feature_attributions)
            )
        except Exception as exc:
            errors.append({"model": name, "error": str(exc)})

    # Multi-step LNN rollout when available (replaces single-step _lnn_forecast)
    lnn = _lnn_multistep_forecast(region, future_times)
    if lnn is None:
        lnn = _lnn_forecast(region, future_times[:1])
    if lnn:
        forecasts.append(lnn)
    else:
        errors.append({"model": "lnn_ltc", "error": "trained LNN weights/buffer unavailable"})

    # Widen P10/P90 uncertainty bands beyond the 30-min training horizon
    if horizon_intervals > _TRAIN_HORIZON:
        forecasts = _apply_horizon_widening(forecasts, horizon_intervals, _TRAIN_HORIZON)

    if not forecasts:
        return _unavailable(region, "no forecast model produced output", anchor, errors)

    # Regime-aware meta-ensemble blend (Sprint I)
    try:
        from domain.nem.adapter import classify_regime
        current_price = float(X_future[0, COL_LAST_PRICE]) if X_future.shape[0] > 0 else 0.0
        current_regime = classify_regime(current_price, region)
    except Exception:
        current_regime = "normal"
    meta_fc = blend_forecasts(forecasts, current_regime)
    if meta_fc:
        forecasts.append(meta_fc)
        register_model("meta_ensemble", "1.0.0", train_ref_base, {"regime": current_regime})

    _model_names = {f["model"] for f in forecasts}
    primary = (
        "meta_ensemble" if "meta_ensemble" in _model_names
        else "lnn_ltc"    if "lnn_ltc"    in _model_names   # LNN is primary when trained
        else "lnn_cfc"    if "lnn_cfc"    in _model_names
        else "qra"        if "qra"        in _model_names   # QRA as statistical fallback
        else forecasts[0]["model"]
    )
    return {
        "region": region,
        "available": True,
        "as_of": anchor.isoformat(),
        "horizon_intervals": horizon_intervals,
        "interval_minutes": 5,
        "training_lookback_days": effective_lookback_days,
        "training_intervals": len(series),
        "primary_model": primary,
        "forecasts": forecasts,
        "errors": errors,
        "caveat": (
            f"Live chart forecast is trained from {len(series)} persisted public dispatch intervals "
            f"over a {effective_lookback_days}-day lookback. "
            "It is decision-support only and must be checked against evidence and missing-data flags."
        ),
    }


async def _fetch_sufficient_history(
    region: str,
    requested_lookback_days: int,
    anchor: datetime,
    horizon_intervals: int,
) -> tuple[list[dict[str, Any]], int]:
    """Expand lookback when recent DB history is sparse.

    Demo and local Docker datasets may have real multi-year AEMO history but a
    gap near "now". Forecasting should use persisted history honestly instead
    of declaring unavailable only because the default 14-day window is thin.
    """
    required = _MIN_TRAIN_INTERVALS + horizon_intervals + 1
    candidates = []
    for days in [requested_lookback_days, 30, 90, 180, 365, 730]:
        if days not in candidates:
            candidates.append(days)

    best: list[dict[str, Any]] = []
    best_days = requested_lookback_days
    for days in candidates:
        series = await _fetch_history(region, days, end_date=anchor)
        if len(series) > len(best):
            best = series
            best_days = days
        if len(series) >= required:
            return series, days
    return best, best_days


def _future_feature_rows(
    last_row: np.ndarray,
    horizon_intervals: int,
    weather_context: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build future feature rows advancing time-of-day and day-of-week per step.

    TOD sin/cos is advanced by one 5-minute interval per row so that predictions
    at hour 4 use 4am features, not repeated midnight features. DOW rolls over at
    day boundaries. All other features hold their last-observed values — the model
    does not invent future drivers.
    """
    from app.engines.forecasting.features.market_features import FEATURE_COLUMNS
    COL_TOD_SIN = FEATURE_COLUMNS.index("tod_sin")
    COL_TOD_COS = FEATURE_COLUMNS.index("tod_cos")
    COL_DOW_IDX = FEATURE_COLUMNS.index("dow")

    _INTERVAL_RAD = 2.0 * np.pi * 5.0 / (24.0 * 60.0)  # radians per 5-min interval

    rows = np.tile(last_row, (horizon_intervals, 1)).astype(float)

    # Reconstruct current TOD angle from sin/cos and advance per step
    last_sin = float(last_row[COL_TOD_SIN])
    last_cos = float(last_row[COL_TOD_COS])
    current_angle = np.arctan2(last_sin, last_cos)
    last_dow = float(last_row[COL_DOW_IDX])

    for i in range(horizon_intervals):
        step = i + 1
        angle = current_angle + step * _INTERVAL_RAD
        rows[i, COL_TOD_SIN] = np.sin(angle)
        rows[i, COL_TOD_COS] = np.cos(angle)
        rows[i, COL_DOW_IDX] = (last_dow + step // 288) % 7  # roll DOW every 288 intervals

    if weather_context:
        temp_c = weather_context.get("temp_c")
        wind_kmh = weather_context.get("wind_kmh")
        if temp_c is not None:
            rows[:, COL_TEMP_C] = float(temp_c)
        if wind_kmh is not None:
            rows[:, COL_WIND_KMH] = float(wind_kmh)
    return rows


def _apply_horizon_widening(
    forecasts: list[dict[str, Any]],
    horizon_intervals: int,
    train_h: int,
) -> list[dict[str, Any]]:
    """Widen P10/P90 uncertainty bands for intervals beyond the training horizon.

    Uses √(h / train_h) scaling — uncertainty grows with the square root of
    relative horizon distance, analogous to Brownian motion. Applied only to
    probabilistic ensemble/model outputs; baselines are left unchanged.
    """
    primary_names = {"meta_ensemble", "qra", "lear", "lnn_cfc", "lnn_ltc", "gbm"}
    result = []
    for fc in forecasts:
        if fc.get("model") not in primary_names:
            result.append(fc)
            continue

        p10_list = list(fc.get("p10") or [])
        p50_list = list(fc.get("p50") or [])
        p90_list = list(fc.get("p90") or [])
        n = len(p50_list)
        if n == 0:
            result.append(fc)
            continue

        new_p10, new_p90 = [], []
        base_half = (
            abs(p90_list[0] - p10_list[0]) / 2.0
            if p10_list and p90_list
            else abs(p50_list[0]) * 0.15
        )

        for i in range(n):
            h = i + 1
            p50 = float(p50_list[i])
            factor = np.sqrt(max(h, train_h) / train_h)
            raw_half = (
                abs(float(p90_list[i]) - float(p10_list[i])) / 2.0
                if i < len(p10_list) and i < len(p90_list)
                else base_half
            )
            widened_half = raw_half * factor
            new_p10.append(round(p50 - widened_half, 2))
            new_p90.append(round(p50 + widened_half, 2))

        fc_out = dict(fc)
        fc_out["p10"] = new_p10
        fc_out["p90"] = new_p90
        fc_out["horizon_widened"] = True
        existing_caveat = fc.get("caveat") or "trained on persisted dispatch history"
        fc_out["caveat"] = (
            existing_caveat
            + f" Uncertainty bands widen beyond {train_h * 5}-min training horizon (√h scaling)."
        )
        result.append(fc_out)

    return result


def _lnn_multistep_forecast(
    region: str, target_times: list[datetime]
) -> dict[str, Any] | None:
    """Multi-step LTC autoregressive rollout for the full forecast horizon."""
    try:
        from app.engines.forecasting.inference import get_trainer
        trainer = get_trainer(region)
        if trainer is None or not trainer.is_trained:
            return None
        steps = len(target_times)
        results = trainer.predict_multistep_from_buffer(steps)
        if not results:
            return None
        register_model("lnn_ltc", "1.0.0", f"{region}:live-ltc-multistep", {"source": "ltc_trainer", "steps": steps})
        return {
            "model": "lnn_ltc",
            "target_times": [t.isoformat() for t in target_times],
            "p10": [round(float(r["p10"]), 2) for r in results],
            "p50": [round(float(r["p50"]), 2) for r in results],
            "p90": [round(float(r["p90"]), 2) for r in results],
            "quantiles": [0.1, 0.5, 0.9],
            "caveat": f"LNN-LTC multi-step autoregressive rollout ({steps} intervals); uncertainty accumulates with horizon.",
        }
    except Exception as exc:
        logger.debug("LNN multi-step forecast unavailable for %s: %s", region, exc)
        return None


def _lnn_forecast(region: str, target_times: list[datetime]) -> dict[str, Any] | None:
    try:
        from app.engines.forecasting.inference import get_forecast

        result = get_forecast(region)
        if not result:
            return None
        register_model("lnn_ltc", "1.0.0", f"{region}:live-ltc-weights", {"source": "ltc_trainer"})
        values = [[float(result["p10"]), float(result["p50"]), float(result["p90"])]]
        return {
            "model": "lnn_ltc",
            "target_times": [t.isoformat() for t in target_times],
            "p10": [values[0][0]],
            "p50": [values[0][1]],
            "p90": [values[0][2]],
            "quantiles": [0.1, 0.5, 0.9],
            "caveat": "LNN output loaded from the live LTC trainer; omitted when weights are unavailable.",
        }
    except Exception as exc:
        logger.debug("LNN live forecast unavailable for %s: %s", region, exc)
        return None


def _forecast_to_dict(
    model_name: str,
    fc,
    caveat: str,
    calibrated: bool = False,
    q_hat: float = 0.0,
    spike_probs: dict | None = None,
    spike_probs_series: dict | None = None,
    feature_attributions: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    q_idx = {float(q): i for i, q in enumerate(fc.quantiles)}
    p10_raw = fc.values[:, q_idx[0.1]]
    p50_raw = fc.values[:, q_idx[0.5]]
    p90_raw = fc.values[:, q_idx[0.9]]

    if calibrated and q_hat > 0.0:
        p10_out = p10_raw - q_hat
        p90_out = p90_raw + q_hat
    else:
        p10_out = p10_raw
        p90_out = p90_raw

    d: dict[str, Any] = {
        "model": model_name,
        "target_times": [t.isoformat() for t in fc.target_times],
        "p10": [round(float(v), 2) for v in p10_out],
        "p50": [round(float(v), 2) for v in p50_raw],
        "p90": [round(float(v), 2) for v in p90_out],
        "quantiles": [0.1, 0.5, 0.9],
        "caveat": caveat,
    }
    if calibrated:
        d["calibrated"] = True
        d["conformal_coverage"] = 0.9
    if spike_probs is not None:
        d["spike_probs"] = spike_probs
    if spike_probs_series is not None:
        d["spike_probs_series"] = spike_probs_series
    if feature_attributions:
        d["feature_attributions"] = [
            {"feature": name, "contribution": round(val, 2)}
            for name, val in feature_attributions
        ]
    return d


def _unavailable(
    region: str,
    reason: str,
    anchor: datetime | None = None,
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "region": region,
        "available": False,
        "as_of": anchor.isoformat() if anchor else datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "forecasts": [],
        "errors": errors or [],
    }
