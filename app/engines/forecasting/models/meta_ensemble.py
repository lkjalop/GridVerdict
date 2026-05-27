"""Regime-aware meta-ensemble blender for NEM price forecasting.

Combines P10/P50/P90 outputs from LEAR, QRA, and LNN using static
regime-conditional weights. The blend is deterministic and explainable:
the weight table is hardcoded and the active weights are returned with
each output so the narrative layer can reference exact blend ratios.

Regime weight rationale:
  normal  — QRA dominates (best-calibrated on normal-market data)
  elevated — balanced; LNN gets equal weight to LEAR (more volatility robustness)
  spike   — LNN weight increases (CfC robustness to distribution shift at extremes)
  extreme — LNN leads; QRA and LEAR equally split the rest
"""
from __future__ import annotations

import numpy as np

# Regime-conditional weights per model.
# Only models present in the forecast list contribute; weights are re-normalised
# to sum to 1.0 when a model is missing.
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "normal": {
        "qra":              0.50,
        "lear":             0.30,
        "experimental_lnn": 0.20,
    },
    "elevated": {
        "qra":              0.40,
        "lear":             0.30,
        "experimental_lnn": 0.30,
    },
    "spike": {
        "qra":              0.30,
        "lear":             0.30,
        "experimental_lnn": 0.40,
    },
    "extreme": {
        "qra":              0.25,
        "lear":             0.25,
        "experimental_lnn": 0.50,
    },
}

_KNOWN_REGIMES = frozenset(_REGIME_WEIGHTS)


def blend_forecasts(
    forecasts: list[dict],
    price_regime: str = "normal",
) -> dict | None:
    """Blend individual model forecasts into a single meta-ensemble output.

    Args:
        forecasts:    List of forecast dicts from run_live_forecast(). Each must
                      have keys "model", "p10", "p50", "p90", "target_times".
        price_regime: Current price regime label (normal|elevated|spike|extreme).
                      Unknown labels fall back to the "normal" weight table.

    Returns:
        A forecast dict with model="meta_ensemble" and "blend_weights",
        or None when no component model produced output.
    """
    regime = price_regime if price_regime in _KNOWN_REGIMES else "normal"
    weights_template = _REGIME_WEIGHTS[regime]

    # Index available forecasts by model name (skip entries without p50)
    by_model: dict[str, dict] = {}
    for fc in forecasts:
        model = fc.get("model", "")
        if model and fc.get("p50"):
            by_model[model] = fc

    # Determine which template models are available and compute active weights
    active_weights: dict[str, float] = {}
    for model_name, w in weights_template.items():
        if model_name in by_model:
            active_weights[model_name] = w

    if not active_weights:
        return None

    # Re-normalise so weights always sum to 1.0
    total = sum(active_weights.values())
    active_weights = {k: round(v / total, 4) for k, v in active_weights.items()}

    # Use shortest horizon to avoid index errors
    horizon = min(len(by_model[m]["p50"]) for m in active_weights)
    if horizon == 0:
        return None

    # Weighted blend per horizon step
    p10_blend = np.zeros(horizon)
    p50_blend = np.zeros(horizon)
    p90_blend = np.zeros(horizon)

    for model_name, w in active_weights.items():
        fc = by_model[model_name]
        p10_blend += w * np.asarray(fc["p10"][:horizon], dtype=float)
        p50_blend += w * np.asarray(fc["p50"][:horizon], dtype=float)
        p90_blend += w * np.asarray(fc["p90"][:horizon], dtype=float)

    # Inherit target_times from the highest-priority available model
    for preferred in ("qra", "lear", "experimental_lnn"):
        if preferred in by_model:
            target_times = by_model[preferred]["target_times"][:horizon]
            break
    else:
        target_times = by_model[next(iter(active_weights))]["target_times"][:horizon]

    # Propagate calibration flag only when all active components are calibrated
    all_calibrated = all(
        by_model[m].get("calibrated", False) for m in active_weights
    )

    weight_str = ", ".join(f"{m}: {w:.2f}" for m, w in active_weights.items())
    result: dict = {
        "model": "meta_ensemble",
        "target_times": target_times,
        "p10": [round(float(v), 2) for v in p10_blend],
        "p50": [round(float(v), 2) for v in p50_blend],
        "p90": [round(float(v), 2) for v in p90_blend],
        "quantiles": [0.1, 0.5, 0.9],
        "regime": regime,
        "blend_weights": active_weights,
        "component_models": list(active_weights.keys()),
        "caveat": (
            f"Regime-aware meta-ensemble ({regime} weights). "
            f"Component blend: {weight_str}."
        ),
    }
    if all_calibrated:
        result["calibrated"] = True
        result["conformal_coverage"] = 0.9

    return result
