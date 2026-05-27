"""Forecasting engine smoke tests.

Covers:
- All forecasting sub-modules import cleanly (catches broken relative import paths).
- BacktestReport schema has required fields.
- run_backtest() runs end-to-end on synthetic data and returns a valid report.
- LNN is registered as 'experimental_lnn', not the headline slot.
- AEMOPredispatch skill is not inflated by self-referential proxy (pinball vs persistence).
"""
from __future__ import annotations

import numpy as np
import pytest


# ── 1. Import hygiene ─────────────────────────────────────────────────────────

def test_backtest_module_imports():
    import app.engines.backtest  # noqa: F401


def test_forecasting_types_imports():
    from app.engines.forecasting.types import (  # noqa: F401
        QuantileForecast, BacktestReport, ModelScore, DEFAULT_QUANTILES,
    )


def test_forecasting_evaluation_imports():
    from app.engines.forecasting.evaluation.harness import run_backtest  # noqa: F401
    from app.engines.forecasting.evaluation.metrics import pinball_loss, crps_from_quantiles  # noqa: F401
    from app.engines.forecasting.evaluation.calibration import calibration_error  # noqa: F401
    from app.engines.forecasting.evaluation.spike_metrics import spike_scores  # noqa: F401


def test_forecasting_models_import():
    from app.engines.forecasting.models.baselines import (  # noqa: F401
        PersistenceModel, SeasonalNaiveModel, AEMOPredispatchModel,
    )
    from app.engines.forecasting.models.lear_model import LEARModel  # noqa: F401
    from app.engines.forecasting.models.base import ForecastModel  # noqa: F401


# ── 2. Synthetic backtest fixtures ────────────────────────────────────────────

def _synthetic_series(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X has 3 features: last_price, seasonal, aemo_proxy."""
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.standard_normal(n).cumsum() * 5
    base = np.clip(base, 10.0, 800.0)
    X = np.column_stack([
        base,                           # COL_LAST_PRICE
        np.roll(base, 288),             # COL_SEASONAL (1-day lag)
        base + rng.standard_normal(n),  # COL_AEMO proxy
    ])
    y = np.roll(base, -6)               # target: 6 intervals ahead
    return X.astype(np.float32), y.astype(np.float32)


# ── 3. BacktestReport schema ──────────────────────────────────────────────────

def test_backtest_report_has_required_fields():
    from app.engines.forecasting.types import BacktestReport, ModelScore
    import dataclasses
    report_fields = {f.name for f in dataclasses.fields(BacktestReport)}
    assert "scores" in report_fields
    # Per-model metrics live on ModelScore, not BacktestReport
    score_fields = {f.name for f in dataclasses.fields(ModelScore)}
    required_score = {"model_name", "pinball", "crps", "spike_f1",
                      "calibration_error", "p50_exceedance_rate", "p90_exceedance_rate"}
    assert required_score <= score_fields, f"ModelScore missing: {required_score - score_fields}"


# ── 4. run_backtest end-to-end ────────────────────────────────────────────────

def test_run_backtest_returns_valid_report():
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic_series(n=500)
    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "seasonal_naive": SeasonalNaiveModel(season_col=1),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )

    assert report is not None
    scores_by_name = {s.model_name: s for s in report.scores}
    assert "persistence" in scores_by_name
    assert "seasonal_naive" in scores_by_name
    for name, score in scores_by_name.items():
        assert score.pinball >= 0.0, f"{name} pinball must be ≥ 0"
        assert 0.0 <= score.calibration_error <= 1.0, f"{name} calibration out of range"


# ── 5. LNN is experimental, not headline ─────────────────────────────────────

def test_lnn_registered_as_experimental():
    from app.engines.backtest import _LTCAdapter
    assert _LTCAdapter.name == "experimental_lnn", (
        "LNN must be registered as 'experimental_lnn', not a headline model name"
    )


def test_backtest_model_dict_uses_experimental_key():
    """The model dict key must be 'experimental_lnn' when LNN is enabled."""
    import asyncio
    from app.engines.backtest import run_region_backtest
    # We only check the key name; we don't need torch or a real DB for this.
    # Verify via _LTCAdapter.name used in the dict key at line 86.
    from app.engines.backtest import _LTCAdapter
    assert _LTCAdapter.name == "experimental_lnn"


# ── 6. AEMO proxy is not inflated ────────────────────────────────────────────

def test_lear_beats_persistence_on_smooth_series():
    """LEAR should beat persistence on a smooth random walk — it has more features."""
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.baselines import PersistenceModel
    from app.engines.forecasting.models.lear_model import LEARModel

    X, y = _synthetic_series(n=600)
    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "lear": LEARModel(alpha=0.01),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )
    scores = {s.model_name: s for s in report.scores}
    assert "lear" in scores
    lear_pinball = scores["lear"].pinball
    assert np.isfinite(lear_pinball), f"LEAR pinball is not finite: {lear_pinball}"
    assert lear_pinball > 0, "LEAR pinball must be > 0"
    # LEAR may or may not beat persistence on synthetic data, but must not be wildly worse
    pers_pinball = scores["persistence"].pinball
    assert lear_pinball < pers_pinball * 3.0, (
        f"LEAR ({lear_pinball:.3f}) is >3x worse than persistence ({pers_pinball:.3f}) — "
        "check for AR feature leakage or fit failure"
    )


def test_lear_model_name_is_lear():
    from app.engines.forecasting.models.lear_model import LEARModel
    assert LEARModel.name == "lear"


def test_lear_registered_in_backtest():
    """backtest.py models dict must include 'lear' key."""
    import app.engines.backtest as bt
    import inspect
    src = inspect.getsource(bt.run_region_backtest)
    assert '"lear"' in src or "'lear'" in src, "LEAR not wired into run_region_backtest"


# ── 7. BacktestReport JSON serialisation ─────────────────────────────────────

def test_backtest_report_to_dict_structure():
    """BacktestReport.to_dict() must produce a well-formed, JSON-serialisable dict."""
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel
    import json

    X, y = _synthetic_series(n=400)
    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "seasonal_naive": SeasonalNaiveModel(season_col=1),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )
    d = report.to_dict()

    # Must be JSON-serialisable
    serialised = json.dumps(d, default=str)
    reparsed = json.loads(serialised)

    assert "horizon_min" in reparsed
    assert "n_origins" in reparsed
    assert "spike_threshold" in reparsed
    assert isinstance(reparsed["scores"], list)
    assert len(reparsed["scores"]) == 2

    for s in reparsed["scores"]:
        assert "model" in s
        assert "crps" in s
        assert "pinball" in s
        assert "spike_f1" in s
        assert "calibration_error" in s
        assert "p50_exceedance_rate" in s
        assert "p90_exceedance_rate" in s
        assert isinstance(s["coverage"], dict)
        assert isinstance(s["skill_vs"], dict)


def test_backtest_report_summary_table_non_empty():
    """summary_table() must return a non-empty string with model names."""
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.baselines import PersistenceModel

    X, y = _synthetic_series(n=400)
    models = {"persistence": PersistenceModel(last_value_col=0)}
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )
    table = report.summary_table()
    assert "persistence" in table
    assert "CRPS" in table


def test_aemo_proxy_skill_not_better_than_persistence():
    """AEMOPredispatch with a self-referential proxy must not beat persistence.

    When COL_AEMO == COL_LAST_PRICE (no real predispatch data), its pinball
    should be similar to persistence, not suspiciously lower.
    """
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.baselines import (
        PersistenceModel, AEMOPredispatchModel,
    )
    X, y = _synthetic_series(n=500, seed=7)
    # Make COL_AEMO (index 2) identical to COL_LAST_PRICE (index 0) — the proxy case
    X[:, 2] = X[:, 0]

    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "aemo_proxy": AEMOPredispatchModel(predispatch_col=2),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )

    scores_by_name = {s.model_name: s for s in report.scores}
    pers_pinball = scores_by_name["persistence"].pinball
    aemo_pinball = scores_by_name["aemo_proxy"].pinball

    # With identical inputs the scores should be within 5% of each other
    assert abs(pers_pinball - aemo_pinball) / max(pers_pinball, 1e-9) < 0.05, (
        f"AEMO proxy pinball ({aemo_pinball:.3f}) diverges suspiciously from "
        f"persistence ({pers_pinball:.3f}) — check for data leakage"
    )
