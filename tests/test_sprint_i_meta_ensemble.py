"""Sprint I: Regime-aware meta-ensemble tests.

Tests cover:
  - blend_forecasts: returns None when no forecasts provided
  - blend_forecasts: weights normalised to 1.0 when a model is absent
  - blend_forecasts: output shape matches horizon of shortest component
  - blend_forecasts: normal regime weights match _REGIME_WEIGHTS["normal"]
  - blend_forecasts: spike regime gives LNN higher weight than normal regime
  - blend_forecasts: unknown regime falls back to "normal"
  - blend_forecasts: P10 ≤ P50 ≤ P90 in blended output
  - blend_forecasts: model="meta_ensemble" in output
  - blend_forecasts: blend_weights present and positive
  - blend_forecasts: calibrated=True propagated only when all components calibrated
  - blend_forecasts: target_times inherited from highest-priority model
  - live_forecast: primary model becomes "meta_ensemble" when blend succeeds
  - live_forecast: meta_ensemble present in forecasts list
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.forecasting.models.meta_ensemble import blend_forecasts, _REGIME_WEIGHTS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fc(model: str, n: int = 3, base: float = 100.0,
        calibrated: bool = False) -> dict:
    """Build a synthetic forecast dict for model `model`."""
    p50 = [base + i for i in range(n)]
    p10 = [v - 10.0 for v in p50]
    p90 = [v + 10.0 for v in p50]
    d = {
        "model": model,
        "target_times": [f"2024-01-01T{i:02d}:00:00" for i in range(n)],
        "p10": p10, "p50": p50, "p90": p90,
        "quantiles": [0.1, 0.5, 0.9],
    }
    if calibrated:
        d["calibrated"] = True
    return d


_ALL_MODELS = ["qra", "lear", "experimental_lnn"]


# ── blend_forecasts: basic correctness ────────────────────────────────────────

class TestBlendForecastsBasic:
    def test_returns_none_when_no_forecasts(self):
        assert blend_forecasts([]) is None

    def test_returns_none_when_no_p50(self):
        fc = {"model": "qra"}  # missing p50
        assert blend_forecasts([fc]) is None

    def test_model_name_is_meta_ensemble(self):
        result = blend_forecasts([_fc("qra"), _fc("lear")])
        assert result is not None
        assert result["model"] == "meta_ensemble"

    def test_output_has_required_keys(self):
        result = blend_forecasts([_fc("qra"), _fc("lear"), _fc("experimental_lnn")])
        assert result is not None
        for k in ("p10", "p50", "p90", "target_times", "blend_weights",
                  "component_models", "regime", "caveat"):
            assert k in result, f"missing key: {k}"

    def test_horizon_matches_shortest_component(self):
        fc_short = _fc("qra", n=2)
        fc_long = _fc("lear", n=5)
        result = blend_forecasts([fc_short, fc_long])
        assert len(result["p50"]) == 2

    def test_p10_le_p50_le_p90(self):
        result = blend_forecasts([_fc("qra"), _fc("lear"), _fc("experimental_lnn")])
        p10 = np.array(result["p10"])
        p50 = np.array(result["p50"])
        p90 = np.array(result["p90"])
        assert np.all(p10 <= p50 + 1e-4)
        assert np.all(p50 <= p90 + 1e-4)

    def test_single_model_weight_becomes_1(self):
        result = blend_forecasts([_fc("qra")])
        assert result is not None
        assert result["blend_weights"]["qra"] == pytest.approx(1.0)

    def test_blend_weights_sum_to_1(self):
        result = blend_forecasts([_fc("qra"), _fc("lear"), _fc("experimental_lnn")])
        total = sum(result["blend_weights"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_blend_weights_all_positive(self):
        result = blend_forecasts([_fc("qra"), _fc("lear"), _fc("experimental_lnn")])
        for w in result["blend_weights"].values():
            assert w > 0.0


# ── blend_forecasts: regime-conditional weights ────────────────────────────────

class TestBlendForecastsRegimeWeights:
    def test_normal_regime_qra_has_highest_weight(self):
        all_fcs = [_fc(m) for m in _ALL_MODELS]
        result = blend_forecasts(all_fcs, "normal")
        weights = result["blend_weights"]
        assert weights["qra"] >= weights["lear"]
        assert weights["qra"] >= weights["experimental_lnn"]

    def test_spike_regime_lnn_has_highest_weight(self):
        all_fcs = [_fc(m) for m in _ALL_MODELS]
        result = blend_forecasts(all_fcs, "spike")
        weights = result["blend_weights"]
        assert weights["experimental_lnn"] >= weights["qra"]
        assert weights["experimental_lnn"] >= weights["lear"]

    def test_extreme_regime_lnn_has_highest_weight(self):
        all_fcs = [_fc(m) for m in _ALL_MODELS]
        result = blend_forecasts(all_fcs, "extreme")
        weights = result["blend_weights"]
        assert weights["experimental_lnn"] >= weights["qra"]

    def test_unknown_regime_falls_back_to_normal(self):
        all_fcs = [_fc(m) for m in _ALL_MODELS]
        result_unknown = blend_forecasts(all_fcs, "unknown_regime")
        result_normal = blend_forecasts(all_fcs, "normal")
        assert result_unknown["regime"] == "normal"
        assert result_unknown["blend_weights"] == result_normal["blend_weights"]

    def test_regime_stored_in_output(self):
        result = blend_forecasts([_fc("qra"), _fc("lear")], "elevated")
        assert result["regime"] == "elevated"

    def test_lnn_higher_weight_in_spike_than_normal(self):
        all_fcs = [_fc(m) for m in _ALL_MODELS]
        w_normal = blend_forecasts(all_fcs, "normal")["blend_weights"]["experimental_lnn"]
        w_spike = blend_forecasts(all_fcs, "spike")["blend_weights"]["experimental_lnn"]
        assert w_spike > w_normal

    def test_absent_model_weights_renormalized(self):
        """Without LNN, QRA + LEAR weights are rescaled to sum to 1."""
        result = blend_forecasts([_fc("qra"), _fc("lear")], "normal")
        weights = result["blend_weights"]
        assert "experimental_lnn" not in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
        # QRA should still have higher weight than LEAR in normal regime
        assert weights["qra"] > weights["lear"]


# ── blend_forecasts: calibration propagation ──────────────────────────────────

class TestBlendForecastsCalibration:
    def test_calibrated_true_when_all_components_calibrated(self):
        all_fcs = [_fc(m, calibrated=True) for m in _ALL_MODELS]
        result = blend_forecasts(all_fcs, "normal")
        assert result.get("calibrated") is True
        assert result.get("conformal_coverage") == 0.9

    def test_calibrated_absent_when_one_component_not_calibrated(self):
        fcs = [_fc("qra", calibrated=True), _fc("lear", calibrated=False)]
        result = blend_forecasts(fcs, "normal")
        assert result.get("calibrated") is not True

    def test_calibrated_absent_when_no_calibration_info(self):
        result = blend_forecasts([_fc("qra"), _fc("lear")])
        assert result.get("calibrated") is not True


# ── blend_forecasts: arithmetic correctness ───────────────────────────────────

class TestBlendForecastsArithmetic:
    def test_weighted_average_is_correct(self):
        """With equal weights, blend should be the arithmetic mean."""
        # Two models with equal weight (when only two models from same-weight template)
        # Use custom equal-contribution: one model from spike template
        # Actually easier: mock two models that aren't in the weight template
        # → both get weight 0 from template → blend returns None
        # Better: use only "qra" and "lear" with a regime where they sum clearly
        # In "normal": qra=0.5, lear=0.3, so with only these two:
        # normalized: qra=0.5/(0.5+0.3)=0.625, lear=0.3/(0.5+0.3)=0.375
        fc_qra = _fc("qra", n=2, base=100.0)
        fc_lear = _fc("lear", n=2, base=200.0)
        result = blend_forecasts([fc_qra, fc_lear], "normal")
        w_qra = 0.5 / (0.5 + 0.3)  # ≈ 0.625
        w_lear = 0.3 / (0.5 + 0.3)  # ≈ 0.375
        expected_p50_0 = w_qra * 100.0 + w_lear * 200.0
        assert result["p50"][0] == pytest.approx(expected_p50_0, abs=0.1)

    def test_target_times_from_qra_when_available(self):
        """QRA has priority for target_times inheritance."""
        fc_qra = _fc("qra", n=3)
        fc_qra["target_times"] = ["qra-t1", "qra-t2", "qra-t3"]
        fc_lear = _fc("lear", n=3)
        fc_lear["target_times"] = ["lear-t1", "lear-t2", "lear-t3"]
        result = blend_forecasts([fc_lear, fc_qra], "normal")
        assert result["target_times"] == ["qra-t1", "qra-t2", "qra-t3"]


# ── live_forecast integration ─────────────────────────────────────────────────

class TestLiveForecastMetaEnsemble:
    """Verify meta_ensemble appears in live_forecast output when models are present."""

    def _make_series(self, n: int) -> list[dict]:
        from datetime import datetime, timezone, timedelta
        base = datetime(2024, 3, 1, tzinfo=timezone.utc)
        return [
            {
                "valid_time": base + timedelta(minutes=5 * i),
                "price": 80.0 + (i % 12) * 2.0,
                "last_price": 80.0 + (i % 12) * 2.0,
                "seasonal_price": 78.0,
                "aemo_predispatch": 79.0,
                "demand": 7500.0,
                "demand_forecast": 7600.0,
                "available_gen": 10000.0,
                "interconnector_room": 500.0,
                "renewable_frac": 0.3,
                "roll_vol_12": 5.0,
                "notice_lor_active": 0.0,
            }
            for i in range(n)
        ]

    def test_meta_ensemble_in_forecasts_list(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from app.engines.forecasting.live_forecast import _run_sync

        anchor = datetime(2024, 3, 10, tzinfo=timezone.utc)
        series = self._make_series(400)
        with patch("app.engines.forecasting.live_forecast._fetch_history") as mock_fetch:
            mock_fetch.return_value = series
            result = _run_sync("NSW1", lookback_days=14, horizon_intervals=6, anchor=anchor)

        if result.get("available"):
            model_names = [f["model"] for f in result.get("forecasts", [])]
            assert "meta_ensemble" in model_names, (
                f"meta_ensemble missing from forecasts: {model_names}"
            )

    def test_primary_model_is_meta_ensemble(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from app.engines.forecasting.live_forecast import _run_sync

        anchor = datetime(2024, 3, 10, tzinfo=timezone.utc)
        series = self._make_series(400)
        with patch("app.engines.forecasting.live_forecast._fetch_history") as mock_fetch:
            mock_fetch.return_value = series
            result = _run_sync("NSW1", lookback_days=14, horizon_intervals=6, anchor=anchor)

        if result.get("available"):
            assert result.get("primary_model") == "meta_ensemble"
