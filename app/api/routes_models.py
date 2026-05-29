"""Model calibration and status routes.

GET /api/models/status?region=NSW1
  Returns LEAR, QRA (with component list), LNN, and GBM calibration status:
  - last_trained_at
  - training_rows
  - metrics: CRPS, MAE, pinball loss per quantile
  - component_models: list with per-model metrics (QRA only)
  - available: bool
  - notes / caveats

The model state lives in the LNN trainer (for LNN) and is inferred from the
evaluation harness results stored in-process.  When no live training has run,
reports "not_trained" status rather than 500.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query

from app.api.auth import TokenPayload
from app.api.deps import get_current_user
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])

_SUPPORTED_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]


@router.get("/status")
async def get_model_status(
    region: str = Query(default="NSW1"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return calibration status for all forecast models for the given region.

    Model states are populated by the background scheduler's LNN retrain job
    and by live_forecast calls.  When no training has run, each model reports
    available=False with a reason.
    """
    region = region.upper()

    # LNN state (from in-process trainer registry)
    lnn_status = _get_lnn_status(region)

    # Live forecast model state (LEAR + QRA + GBM components)
    forecast_status = await _get_forecast_status(region)

    calibration = await _get_cached_calibration(region)

    try:
        import torch as _torch
        _torch_available = True
        _torch_version = _torch.__version__
    except ImportError:
        _torch_available = False
        _torch_version = None

    settings = get_settings()

    return {
        "region": region,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "torch_available": _torch_available,
        "torch_version": _torch_version,
        "sequence_forecasters_enabled": settings.enable_experimental_sequence_forecasters,
        "models": {
            "lnn": lnn_status,
            **forecast_status,
        },
        "calibration": calibration,
        "caveat": (
            "Model calibration metrics are computed on held-out walk-forward splits "
            "from persisted dispatch history.  CRPS and MAE reflect in-sample "
            "evaluation — out-of-sample performance may differ. "
            "All models are decision-support only."
        ),
    }


@router.get("/calibration")
async def get_calibration_detail(
    region: str = Query(default="NSW1"),
    user: TokenPayload = Depends(get_current_user),
):
    """Return detailed per-quantile calibration plots data for frontend rendering.

    Returns quantile coverage rates (observed fraction below each quantile)
    so the frontend can plot a calibration curve.
    """
    region = region.upper()
    detail = await _get_calibration_detail(region)
    return {
        "region": region,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "calibration": detail,
    }


# ── Model state readers ───────────────────────────────────────────────────────

def _get_lnn_status(region: str) -> dict:
    try:
        from app.engines.forecasting.inference import get_trainer
        from datetime import datetime, timezone
        trainer = get_trainer(region)
        if trainer is None or not trainer.is_trained:
            required = int(getattr(trainer, "min_samples", 288) or 288) if trainer else 288
            price_count = len(getattr(trainer, "_prices", []) or []) if trainer else 0
            feature_count = len(getattr(trainer, "_buffer", []) or []) if trainer else 0
            buffer_count = max(price_count, feature_count - 1 if feature_count else 0, 0)
            try:
                import torch  # noqa: F401
                torch_available = True
            except ImportError:
                torch_available = False
            if not trainer:
                reason = "LNN trainer is unavailable in this runtime."
            elif not torch_available:
                reason = (
                    f"Torch is not installed in this runtime; LNN has "
                    f"{buffer_count}/{required} target intervals buffered but cannot train."
                )
            elif buffer_count < required:
                reason = (
                    f"Not yet trained - {buffer_count}/{required} dispatch intervals buffered "
                    "(288 intervals is about 24 h of 5-minute data)."
                )
            else:
                reason = (
                    f"Ready to train ({buffer_count} intervals buffered), but no successful "
                    "LNN checkpoint is available yet."
                )
            payload = {
                "available": False,
                "model": "lnn",
                "model_type": "point_forecast",
                "reason": "Not yet trained — needs 288 dispatch intervals (~24 h of data).",
                "trained_on_intervals": 0,
                "buffer_intervals": int(buffer_count),
                "required_intervals": required,
                "torch_available": torch_available,
                "checkpoint_age_hours": None,
                "notes": (
                    "LNN (Liquid Time-constant Network): single-step deterministic point "
                    "forecast. Not probabilistic — does not produce P10/P90 intervals. "
                    "Training runs automatically after 288 intervals are ingested."
                ),
            }
            payload["reason"] = reason
            return payload
        metrics = trainer.last_metrics or {}
        last_trained = trainer.last_trained_at
        checkpoint_age_hours: float | None = None
        if last_trained:
            now = datetime.now(timezone.utc)
            if last_trained.tzinfo is None:
                last_trained = last_trained.replace(tzinfo=timezone.utc)
            checkpoint_age_hours = round((now - last_trained).total_seconds() / 3600, 1)

        return {
            "available": True,
            "model": "lnn",
            "model_type": "point_forecast",
            "last_trained_at": trainer.last_trained_at.isoformat() if trainer.last_trained_at else None,
            "trained_on_intervals": trainer.training_rows,
            "checkpoint_age_hours": checkpoint_age_hours,
            "hidden_size": trainer.hidden_size,
            "metrics": {
                "mae": round(metrics.get("mae", 0.0), 3),
                "rmse": round(metrics.get("rmse", 0.0), 3),
            },
            "notes": (
                "LNN (Liquid Time-constant Network): single-step deterministic point "
                "forecast. Not probabilistic — does not produce P10/P90 prediction "
                "intervals. Use LEAR/QRA for uncertainty quantification."
            ),
        }
    except Exception as exc:
        logger.debug("LNN status read failed: %s", exc)
        return {"available": False, "reason": str(exc), "model": "lnn", "model_type": "point_forecast"}


async def _get_forecast_status(region: str) -> dict:
    """Run a minimal live_forecast call to report model availability."""
    try:
        from app.engines.forecasting.live_forecast import run_live_forecast
        result = await run_live_forecast(region, lookback_days=7, horizon_intervals=1)

        if not result.get("available"):
            return {
                "lear": {"available": False, "reason": result.get("reason", "insufficient data"), "model": "lear"},
                "qra":  {"available": False, "reason": result.get("reason", "insufficient data"), "model": "qra"},
                "gbm":  {"available": False, "reason": "qra unavailable", "model": "gbm_sklearn"},
            }

        out: dict = {}
        for fc in result.get("forecasts", []):
            name = fc.get("model", "")
            out[name] = {
                "available": True,
                "model": name,
                "as_of": result.get("as_of"),
                "horizon_intervals": result.get("horizon_intervals"),
                "caveat": fc.get("caveat", ""),
            }

        # QRA component detail
        if "qra" in out:
            out["qra"]["components"] = _qra_component_names()

        # Errors
        for err in result.get("errors", []):
            m = err.get("model", "unknown")
            if m not in out:
                out[m] = {"available": False, "reason": err.get("error", ""), "model": m}

        return out
    except Exception as exc:
        logger.debug("Forecast status read failed: %s", exc)
        return {
            "lear": {"available": False, "reason": str(exc), "model": "lear"},
            "qra":  {"available": False, "reason": str(exc), "model": "qra"},
        }


async def _get_cached_calibration(region: str) -> dict | None:
    """Return the most recent calibration result for a region.

    Checks in order:
      1. In-process evaluation registry (written by the scheduler calibration job)
      2. MarketCache (populated by the same scheduler job as a backup)
      3. Returns a structured null with a reason so the frontend can show a clear state
    """
    # 1. In-process registry
    try:
        from app.engines.forecasting.evaluation.harness import get_last_eval_result
        result = get_last_eval_result(region)
        if result:
            return {"scores": result, "source": "in_process_registry"}
    except Exception:
        pass

    # 2. MarketCache
    try:
        from app.data.cache import get_cache
        cached = await get_cache().get(f"calibration_{region}")
        if cached:
            return {"scores": cached, "source": "market_cache"}
    except Exception:
        pass

    return {"scores": None, "source": "not_available", "reason": "Calibration job has not run yet for this region."}


def _qra_component_names() -> list[str]:
    try:
        from app.engines.forecasting.models.sklearn_gbm_model import SklearnGBMQuantileModel
        has_gbm = True
    except ImportError:
        has_gbm = False
    base = ["persistence", "seasonal_naive", "aemo_predispatch", "lear"]
    return base + (["gbm"] if has_gbm else [])


async def _get_calibration_detail(region: str) -> list[dict]:
    """Return quantile coverage rates from walk-forward evaluation if available."""
    try:
        from app.engines.forecasting.evaluation.harness import get_last_eval_result
        result = get_last_eval_result(region)
        if result is None:
            return [{"note": "No walk-forward evaluation has run yet for this region."}]
        return result
    except Exception as exc:
        logger.debug("Calibration detail read failed: %s", exc)
        return [{"note": f"Calibration data unavailable: {exc}"}]
