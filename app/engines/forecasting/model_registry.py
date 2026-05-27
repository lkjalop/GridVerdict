"""ISO/IEC 42001:2023 model registry — version tracking and training data provenance.

Each active forecast or decision model registers itself here with a version string
and (optionally) a reference to the training data window used for the most recent
fit. These are written to DecisionAuditLog rows to satisfy:

  ISO/IEC 42001:2023 §8.4  — AI system documentation
  ISO/IEC 42001:2023 §9.1  — Monitoring and measurement
  ISO/IEC 42001:2023 §6.1  — AI risk treatment

Usage:
    from app.engines.forecasting.model_registry import register_model, get_model_info

    register_model(
        "LEAR",
        version="1.0.0",
        training_data_ref="NSW1:2024-04-01/2024-05-01:n=8928",
        metadata={"architecture": "quantile_linear", "quantiles": [0.1, 0.5, 0.9]},
    )
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

# In-process registry — reset on server restart (intentional: models re-register on each fit).
_registry: dict[str, dict[str, Any]] = {}

# Built-in entries for rule-based components that don't train (always present).
_STATIC_ENTRIES: dict[str, dict[str, Any]] = {
    "bess-policy": {
        "model_name": "bess-policy",
        "version": "1.0.0",
        "architecture": "rule_based",
        "training_approach": "deterministic",
        "training_data_ref": "no_training_data:rule_based_policy",
        "description": "BESS dispatch policy — rule-based economics engine (compute_economics + evaluate).",
        "known_limitations": [
            "Does not learn from historical dispatch outcomes",
            "FCAS opportunity value is approximated; requires live FCAS price feed for accuracy",
        ],
        "human_oversight": "simulation_only=True hardcoded; all outputs require operator validation",
        "registered_at": "static",
    },
    "fleet-policy": {
        "model_name": "fleet-policy",
        "version": "1.0.0",
        "architecture": "rule_based",
        "training_approach": "deterministic",
        "training_data_ref": "no_training_data:rule_based_policy",
        "description": "Fleet dispatch coordinator — rule-based export cap allocation by net value ranking.",
        "known_limitations": [
            "Does not model inter-asset thermal or network constraints",
            "Ranking by net_value may suboptimally allocate assets with similar economics",
        ],
        "human_oversight": "simulation_only=True hardcoded; all outputs require operator validation",
        "registered_at": "static",
    },
}


def register_model(
    model_name: str,
    version: str,
    training_data_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register or update a model's version and training data provenance."""
    _registry[model_name] = {
        "model_name": model_name,
        "version": version,
        "training_data_ref": training_data_ref,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }


def get_model_info(model_name: str) -> dict[str, Any] | None:
    """Return registry entry for a model (dynamic or static)."""
    return _registry.get(model_name) or _STATIC_ENTRIES.get(model_name)


def get_active_version(model_name: str) -> str | None:
    info = get_model_info(model_name)
    return info.get("version") if info else None


def get_training_data_ref(model_name: str) -> str | None:
    info = get_model_info(model_name)
    return info.get("training_data_ref") if info else None


def get_all_models() -> list[dict[str, Any]]:
    """Return all registered models (dynamic registrations + static built-ins)."""
    merged: dict[str, dict] = {**_STATIC_ENTRIES, **_registry}
    return sorted(merged.values(), key=lambda m: m["model_name"])


def compute_data_hash(data: Any) -> str:
    """Compute a stable 16-char SHA-256 prefix for training data provenance.

    Pass a dict describing the training window (region, start, end, n_rows)
    and receive a stable hash suitable for storing in training_data_ref.
    """
    content = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(content).hexdigest()[:16]


def make_training_ref(region: str, start_dt: Any, end_dt: Any, n_rows: int) -> str:
    """Produce a human-readable + hash training data reference string.

    Format: "{region}:{start_date}/{end_date}:n={n}:sha={hash}"
    """
    start_s = str(start_dt)[:10] if start_dt else "?"
    end_s = str(end_dt)[:10] if end_dt else "?"
    h = compute_data_hash({"region": region, "start": start_s, "end": end_s, "n": n_rows})
    return f"{region}:{start_s}/{end_s}:n={n_rows}:sha={h}"
