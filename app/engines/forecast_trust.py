"""Forecast Trust Panel — per-model accuracy and availability summary.

Aggregates model_registry provenance + backtest harness evaluation results
into a single "trust pack" for each active forecast model. The endpoint is
intentionally read-only and non-cached; it reflects the current in-process
state of the model registry and last backtest evaluation.

Fields per model:
  model               — model identifier
  version             — semver string from registry
  last_trained        — ISO timestamp when model was last fit
  training_window     — human-readable training data window from training_data_ref
  training_data_ref   — full provenance reference string
  crps                — Continuous Ranked Probability Score (lower is better)
  pinball_loss        — mean pinball loss across quantiles (lower is better)
  calibration_error   — mean absolute coverage gap (lower is better)
  skill_vs_persistence — CRPS skill score vs naive persistence baseline (higher is better)
  skill_vs_predispatch — CRPS skill score vs AEMO 30-min predispatch (higher is better)
  spike_recall        — fraction of spike intervals the model correctly detected
  availability        — True if the model participated in the last forecast run
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "metric not available"
_NA = None


def build_forecast_trust(region: str) -> dict[str, Any]:
    """Return per-model trust metrics for *region*."""
    models = _collect_models()
    eval_rows = _load_eval(region)
    eval_by_name = {r["model"]: r for r in eval_rows} if eval_rows else {}

    entries = []
    for m in models:
        name = m["model_name"]
        eval_data = eval_by_name.get(name, {})

        entries.append({
            "model": name,
            "version": m.get("version"),
            "last_trained": m.get("registered_at"),
            "training_window": _parse_window(m.get("training_data_ref")),
            "training_data_ref": m.get("training_data_ref"),
            "crps": eval_data.get("crps", _NA),
            "pinball_loss": eval_data.get("pinball", _NA),
            "calibration_error": eval_data.get("calibration_error", _NA),
            "skill_vs_persistence": _skill(eval_data, "persistence"),
            "skill_vs_predispatch": _skill(eval_data, "aemo_predispatch"),
            "spike_recall": eval_data.get("spike_recall", _NA),
            "availability": name in eval_by_name or m.get("registered_at") not in (None, "static"),
        })

    return {
        "region": region,
        "models": entries,
        "eval_source": "last_backtest" if eval_rows else "not_yet_evaluated",
        "note": (
            "Metrics are from the most recent in-process backtest run. "
            "Restart or trigger a retrain to refresh."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_models() -> list[dict[str, Any]]:
    """Return all forecast models from the registry (static + dynamic)."""
    try:
        from app.engines.forecasting.model_registry import get_all_models
        return [
            m for m in get_all_models()
            if m.get("model_name") not in ("bess-policy", "fleet-policy")
        ]
    except Exception as exc:
        logger.debug("Model registry unavailable: %s", exc)
        return []


def _load_eval(region: str) -> list[dict] | None:
    """Return last backtest evaluation rows for this region, if any."""
    try:
        from app.engines.forecasting.evaluation.harness import get_last_eval_result
        result = get_last_eval_result(region)
        if result is None:
            return None
        # Result may be list[dict] (from to_dict()) or a BacktestReport
        if isinstance(result, list):
            return result
        if hasattr(result, "to_dict"):
            return result.to_dict().get("scores", [])
        return None
    except Exception as exc:
        logger.debug("Eval registry unavailable: %s", exc)
        return None


def _skill(eval_data: dict, baseline: str) -> float | None:
    """Extract skill score vs a named baseline from eval_data."""
    skill_vs = eval_data.get("skill_vs") or {}
    if isinstance(skill_vs, dict):
        v = skill_vs.get(baseline)
        return round(v, 4) if v is not None else None
    return None


def _parse_window(training_data_ref: str | None) -> str | None:
    """Extract a human-readable window from training_data_ref.

    Format: "REGION:start/end:n=NNN:sha=HHH" → "start to end (NNN rows)"
    """
    if not training_data_ref:
        return None
    try:
        parts = training_data_ref.split(":")
        if len(parts) >= 3:
            window_part = parts[1]       # "start/end"
            n_part = parts[2]            # "n=NNN"
            if "/" in window_part:
                start, end = window_part.split("/", 1)
                n = n_part.replace("n=", "")
                return f"{start} to {end} ({n} rows)"
        return training_data_ref
    except Exception:
        return training_data_ref
