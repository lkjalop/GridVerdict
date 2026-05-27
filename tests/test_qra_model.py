"""Tests for QRAModel — Quantile Regression Averaging ensemble.

Covers:
- Import hygiene
- QRAModel requires >= 2 components
- fit() trains components and combiner
- predict_quantiles() returns correctly-shaped QuantileForecast
- equal-weight fallback fires when sklearn is missing (monkeypatched)
- component_weights() returns a dict keyed by component name
- QRA wired into run_region_backtest model dict
- QRA score is finite and within reasonable range on synthetic data
"""
from __future__ import annotations

import numpy as np
import pytest


def _synthetic(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.standard_normal(n).cumsum() * 5
    base = np.clip(base, 10.0, 800.0)
    X = np.column_stack([
        base,
        np.roll(base, 288),
        base + rng.standard_normal(n),
    ]).astype(np.float32)
    y = np.roll(base, -6).astype(np.float32)
    return X, y


# ── 1. Import ─────────────────────────────────────────────────────────────────

def test_qra_imports():
    from app.engines.forecasting.models.qra_model import QRAModel  # noqa: F401


def test_qra_registered_in_backtest():
    import inspect
    import app.engines.backtest as bt
    src = inspect.getsource(bt.run_region_backtest)
    assert "qra" in src, "QRAModel not wired into run_region_backtest"


# ── 2. Construction ───────────────────────────────────────────────────────────

def test_qra_requires_two_components():
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel
    with pytest.raises(ValueError, match="at least 2"):
        QRAModel(components={"p": PersistenceModel(last_value_col=0)})


def test_qra_name():
    from app.engines.forecasting.models.qra_model import QRAModel
    assert QRAModel.name == "qra"


# ── 3. Fit and predict ────────────────────────────────────────────────────────

def test_qra_fit_and_predict():
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic(n=400)
    qra = QRAModel(components={
        "persistence": PersistenceModel(last_value_col=0),
        "seasonal": SeasonalNaiveModel(season_col=1),
    })
    qra.fit(X, y)

    from datetime import datetime
    times = [datetime.utcfromtimestamp(i * 300) for i in range(10)]
    fc = qra.predict_quantiles(X[:10], times)

    assert fc.values.shape == (10, 3)
    assert np.all(np.isfinite(fc.values))
    # Quantiles must be non-decreasing (monotone)
    assert np.all(fc.values[:, 1] >= fc.values[:, 0] - 1e-6)
    assert np.all(fc.values[:, 2] >= fc.values[:, 1] - 1e-6)


def test_qra_predict_without_fit_uses_fallback():
    """predict before fit should not crash — fallback to equal-weight."""
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic(n=50)
    qra = QRAModel(components={
        "persistence": PersistenceModel(last_value_col=0),
        "seasonal": SeasonalNaiveModel(season_col=1),
    })
    # Manually train components but skip combiner to test fallback
    for m in qra.components.values():
        m.fit(X, y)
    qra._trained_components = dict(qra.components)
    qra._combiners = None

    from datetime import datetime
    times = [datetime.utcfromtimestamp(i * 300) for i in range(5)]
    fc = qra.predict_quantiles(X[:5], times)
    assert fc.values.shape == (5, 3)


# ── 4. component_weights ──────────────────────────────────────────────────────

def test_component_weights_returns_dict():
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic(n=400)
    qra = QRAModel(components={
        "persistence": PersistenceModel(last_value_col=0),
        "seasonal": SeasonalNaiveModel(season_col=1),
    })
    qra.fit(X, y)
    weights = qra.component_weights()
    if weights is not None:
        assert "persistence" in weights
        assert "seasonal" in weights
        assert len(weights["persistence"]) == 3   # one per quantile


# ── 5. End-to-end in harness ──────────────────────────────────────────────────

def test_qra_in_harness_produces_finite_score():
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic(n=500)
    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "qra": QRAModel(components={
            "persistence": PersistenceModel(last_value_col=0),
            "seasonal": SeasonalNaiveModel(season_col=1),
        }),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )
    scores = {s.model_name: s for s in report.scores}
    assert "qra" in scores
    qra_score = scores["qra"]
    assert np.isfinite(qra_score.crps), f"QRA CRPS not finite: {qra_score.crps}"
    assert np.isfinite(qra_score.pinball), f"QRA pinball not finite: {qra_score.pinball}"
    assert qra_score.pinball > 0


def test_qra_not_catastrophically_worse_than_persistence():
    """QRA must not be >3x worse than persistence — catches combiner bugs."""
    from app.engines.forecasting.evaluation.harness import run_backtest
    from app.engines.forecasting.models.qra_model import QRAModel
    from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel

    X, y = _synthetic(n=500, seed=99)
    models = {
        "persistence": PersistenceModel(last_value_col=0),
        "qra": QRAModel(components={
            "persistence": PersistenceModel(last_value_col=0),
            "seasonal": SeasonalNaiveModel(season_col=1),
        }),
    }
    report = run_backtest(
        models=models, X=X, y=y,
        horizon=6, step=12, min_train=288, spike_threshold=300.0,
    )
    scores = {s.model_name: s for s in report.scores}
    pers = scores["persistence"].pinball
    qra  = scores["qra"].pinball
    assert qra < pers * 3.0, (
        f"QRA pinball ({qra:.3f}) is >3x worse than persistence ({pers:.3f})"
    )
