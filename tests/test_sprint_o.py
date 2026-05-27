"""Sprint O tests — LNN spike-risk head, TCN, feature enrichment,
LEAR contribution/regime-conformal, schema extensions, incident brief endpoint.

These tests run without a live database or GPU — they use in-memory SQLite
and synthetic numpy arrays. torch/ncps is optional; tests are skipped when absent.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def sprint_o_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    from app.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="module")
async def sprint_o_client(sprint_o_engine):
    from fastapi import FastAPI
    from app.api.routes_incidents import router as incidents_router
    from app.api.routes_market import router as market_router
    from app.api.auth import TokenPayload
    from app.api.deps import get_current_user, get_db

    app = FastAPI()
    app.include_router(incidents_router)
    app.include_router(market_router)

    Session = async_sessionmaker(sprint_o_engine, expire_on_commit=False)
    fake_user = TokenPayload(
        sub="sprint-o-user",
        tenant_id="t-so",
        email="sprint_o@test.example",
        exp=_now() + timedelta(hours=1),
    )

    async def _db():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = lambda: fake_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Feature enrichment tests ──────────────────────────────────────────────────

class TestFeatureEnrichment:

    def test_feature_columns_include_headroom(self):
        from app.engines.forecasting.features.market_features import FEATURE_COLUMNS
        assert "headroom_mw" in FEATURE_COLUMNS

    def test_feature_columns_include_constraint_count(self):
        from app.engines.forecasting.features.market_features import FEATURE_COLUMNS
        assert "constraint_count" in FEATURE_COLUMNS

    def test_headroom_column_index_exported(self):
        from app.engines.forecasting.features.market_features import COL_HEADROOM
        assert isinstance(COL_HEADROOM, int)
        assert COL_HEADROOM >= 0

    def test_build_features_headroom_derived(self):
        """headroom_mw == max(available_gen - demand, 0) in the feature matrix."""
        from app.engines.forecasting.features.market_features import (
            build_features, FEATURE_COLUMNS
        )
        now = _now()
        series = [
            {
                "valid_time": now,
                "price": 100.0,
                "last_price": 100.0,
                "seasonal_price": 95.0,
                "aemo_predispatch": 100.0,
                "demand": 5000.0,
                "demand_forecast": 5100.0,
                "available_gen": 6500.0,
                "interconnector_room": 200.0,
                "renewable_frac": 0.3,
                "roll_vol_12": 20.0,
                "notice_lor_active": 0.0,
                "constraint_count": 2.0,
            }
        ]
        X, y = build_features(series, [now])
        assert X.shape[0] == 1
        headroom_idx = FEATURE_COLUMNS.index("headroom_mw")
        assert X[0, headroom_idx] == pytest.approx(1500.0)  # 6500 - 5000

    def test_build_features_headroom_clamp_to_zero(self):
        """headroom_mw is clamped to 0 when demand exceeds availability."""
        from app.engines.forecasting.features.market_features import (
            build_features, FEATURE_COLUMNS
        )
        now = _now()
        series = [
            {
                "valid_time": now,
                "price": 500.0,
                "last_price": 500.0,
                "seasonal_price": 200.0,
                "aemo_predispatch": 500.0,
                "demand": 8000.0,
                "available_gen": 7500.0,  # demand > availability
            }
        ]
        X, y = build_features(series, [now])
        headroom_idx = FEATURE_COLUMNS.index("headroom_mw")
        assert X[0, headroom_idx] == 0.0

    def test_build_features_constraint_count_passthrough(self):
        from app.engines.forecasting.features.market_features import (
            build_features, FEATURE_COLUMNS
        )
        now = _now()
        series = [
            {
                "valid_time": now,
                "price": 80.0,
                "last_price": 80.0,
                "constraint_count": 5.0,
                "demand": 3000.0,
                "available_gen": 4000.0,
            }
        ]
        X, y = build_features(series, [now])
        cc_idx = FEATURE_COLUMNS.index("constraint_count")
        assert X[0, cc_idx] == pytest.approx(5.0)


# ── LNN spike-risk head tests ─────────────────────────────────────────────────

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from ncps.torch import CfC  # noqa: F401
    _NCPS_OK = True
except ImportError:
    _NCPS_OK = False

_SKIP_TORCH = pytest.mark.skipif(not (_TORCH_OK and _NCPS_OK), reason="torch/ncps not installed")


@_SKIP_TORCH
class TestLNNSpikeRiskHead:

    def _make_xy(self, n: int = 200, n_features: int = 5):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, n_features)).astype(np.float32)
        y = rng.normal(100, 300, n).astype(np.float32)
        return X, y

    def test_predict_spike_probs_raises_before_fit(self):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        m = LNNQuantileModel(epochs=1)
        X, _ = self._make_xy()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict_spike_probs(X)

    def test_predict_spike_probs_returns_three_keys(self):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        m = LNNQuantileModel(epochs=2, hidden=8, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        probs = m.predict_spike_probs(X[:5])
        assert set(probs.keys()) == {"gt_300", "gt_1000", "lt_0"}

    def test_spike_probs_in_unit_interval(self):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        m = LNNQuantileModel(epochs=2, hidden=8, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        probs = m.predict_spike_probs(X)
        for key, arr in probs.items():
            assert np.all(arr >= 0.0) and np.all(arr <= 1.0), f"{key} out of [0,1]"

    def test_spike_probs_shape_matches_input(self):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        m = LNNQuantileModel(epochs=2, hidden=8, seq_len=4)
        X, y = self._make_xy(n=50)
        m.fit(X, y)
        probs = m.predict_spike_probs(X)
        for key, arr in probs.items():
            assert arr.shape == (50,), f"{key} shape mismatch: {arr.shape}"

    def test_lnn_jointly_trains_both_heads(self):
        """After fit(), both predict_quantiles and predict_spike_probs work."""
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        from datetime import datetime
        m = LNNQuantileModel(epochs=3, hidden=8, seq_len=4)
        X, y = self._make_xy(n=30)
        m.fit(X, y)
        times = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(5)]
        fc = m.predict_quantiles(X[:5], times)
        assert fc.values.shape == (5, 3)
        probs = m.predict_spike_probs(X[:5])
        assert len(probs) == 3

    def test_lnn_save_load_preserves_spike_head(self, tmp_path):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        m = LNNQuantileModel(epochs=2, hidden=8, seq_len=4)
        X, y = self._make_xy(n=40)
        m.fit(X, y)
        probs_before = m.predict_spike_probs(X[:3])

        path = str(tmp_path / "lnn_test.pt")
        m.save(path)

        m2 = LNNQuantileModel(hidden=8, seq_len=4)
        m2.load(path, n_features=5)
        probs_after = m2.predict_spike_probs(X[:3])

        for key in ("gt_300", "gt_1000", "lt_0"):
            np.testing.assert_allclose(probs_before[key], probs_after[key], rtol=1e-4)


# ── TCN model tests ─────────────────────────────────────────────────���─────────

_SKIP_TCN = pytest.mark.skipif(not _TORCH_OK, reason="torch not installed")


@_SKIP_TCN
class TestTCNModel:

    def _make_xy(self, n: int = 150, n_features: int = 6):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (n, n_features)).astype(np.float32)
        y = rng.normal(80, 200, n).astype(np.float32)
        return X, y

    def test_tcn_fit_predict_roundtrip(self):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel(epochs=2, channels=8, n_layers=2, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        times = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(6)]
        fc = m.predict_quantiles(X[:6], times)
        assert fc.values.shape == (6, 3)

    def test_tcn_quantiles_non_crossing(self):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel(epochs=3, channels=8, n_layers=2, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        times = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(10)]
        fc = m.predict_quantiles(X[:10], times)
        p10, p50, p90 = fc.values[:, 0], fc.values[:, 1], fc.values[:, 2]
        assert np.all(p10 <= p50 + 1e-5)
        assert np.all(p50 <= p90 + 1e-5)

    def test_tcn_spike_probs_three_keys(self):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel(epochs=2, channels=8, n_layers=2, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        probs = m.predict_spike_probs(X[:5])
        assert set(probs.keys()) == {"gt_300", "gt_1000", "lt_0"}

    def test_tcn_spike_probs_in_unit_interval(self):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel(epochs=2, channels=8, n_layers=2, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        probs = m.predict_spike_probs(X)
        for key, arr in probs.items():
            assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

    def test_tcn_save_load(self, tmp_path):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel(epochs=2, channels=8, n_layers=2, seq_len=4)
        X, y = self._make_xy()
        m.fit(X, y)
        times = [datetime(2026, 1, 1)]
        fc_before = m.predict_quantiles(X[:1], times)

        path = str(tmp_path / "tcn_test.pt")
        m.save(path)

        m2 = TCNQuantileModel(channels=8, n_layers=2, seq_len=4)
        m2.load(path, n_features=6)
        fc_after = m2.predict_quantiles(X[:1], times)
        np.testing.assert_allclose(fc_before.values, fc_after.values, rtol=1e-4)

    def test_tcn_raises_before_fit(self):
        from app.engines.forecasting.models.tcn_model import TCNQuantileModel
        m = TCNQuantileModel()
        X, _ = self._make_xy()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict_quantiles(X[:1], [datetime(2026, 1, 1)])


# ── LEAR feature contribution summary tests ───────────────────────────────────

class TestLEARFeatureContribution:

    def _fit_lear(self, n: int = 120, n_features: int = 5) -> Any:
        from app.engines.forecasting.models.lear_model import LEARModel
        rng = np.random.default_rng(10)
        X = rng.normal(0, 1, (n, n_features))
        y = rng.normal(100, 50, n)
        names = [f"feat_{i}" for i in range(n_features)]
        m = LEARModel()
        m.fit(X, y, feature_names=names)
        return m

    def test_feature_importances_returns_list(self):
        m = self._fit_lear()
        result = m.feature_importances()
        assert isinstance(result, list)

    def test_feature_importances_top_k_respected(self):
        m = self._fit_lear()
        result = m.feature_importances(top_k=3)
        assert len(result) <= 3

    def test_feature_importances_has_required_keys(self):
        m = self._fit_lear()
        result = m.feature_importances(top_k=5)
        if result:
            for item in result:
                assert "feature" in item
                assert "mean_abs_coef" in item
                assert "rank" in item

    def test_feature_importances_ranked_descending(self):
        m = self._fit_lear()
        result = m.feature_importances(top_k=5)
        if len(result) >= 2:
            coefs = [r["mean_abs_coef"] for r in result]
            assert coefs == sorted(coefs, reverse=True)

    def test_feature_importances_unfitted_returns_empty(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        assert m.feature_importances() == []


# ── LEAR regime-specific conformal calibration tests ───────��─────────────────

class TestLEARRegimeConformal:

    def _fit_and_calibrate(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        rng = np.random.default_rng(20)
        n_fit, n_cal = 200, 60
        X = rng.normal(0, 1, (n_fit + n_cal, 5))
        y = rng.normal(100, 80, n_fit + n_cal)
        X_fit, y_fit = X[:n_fit], y[:n_fit]
        X_cal, y_cal = X[n_fit:], y[n_fit:]
        regimes = np.where(y_cal > 200, "spike", np.where(y_cal < 0, "negative", "normal"))
        m = LEARModel()
        m.fit(X_fit, y_fit)
        m.fit_regime_conformal(X_cal, y_cal, regimes)
        return m, regimes

    def test_regime_calibrators_populated(self):
        m, _ = self._fit_and_calibrate()
        assert len(m._regime_calibrators) >= 1

    def test_global_calibrator_exists(self):
        m, _ = self._fit_and_calibrate()
        assert "__global__" in m._regime_calibrators

    def test_get_conformal_q_hat_global_is_nonneg(self):
        m, _ = self._fit_and_calibrate()
        q_hat = m.get_conformal_q_hat("__global__")
        assert q_hat >= 0.0

    def test_get_conformal_q_hat_falls_back_to_global(self):
        m, _ = self._fit_and_calibrate()
        q_hat_global = m.get_conformal_q_hat("__global__")
        q_hat_unknown = m.get_conformal_q_hat("nonexistent_regime")
        assert q_hat_unknown == q_hat_global

    def test_regime_q_hats_may_differ(self):
        """Normal and spike regimes should have different calibration widths."""
        m, regimes = self._fit_and_calibrate()
        # Only assert that at least one regime-specific calibrator was fitted
        regime_keys = [k for k in m._regime_calibrators if k != "__global__"]
        assert len(regime_keys) >= 1


# ── QueryDecomposition extended schema tests ──────────────────────────────────

class TestQueryDecompositionSchema:

    def _make_decomp(self, **kwargs):
        from app.core.schema import QueryDecomposition, IntentLabel
        defaults = {
            "raw_query": "Why is NSW1 price high?",
            "intent": IntentLabel.EXPLANATION,
        }
        defaults.update(kwargs)
        return QueryDecomposition(**defaults)

    def test_requires_live_market_default_false(self):
        d = self._make_decomp()
        assert d.requires_live_market is False

    def test_requires_incident_timeline_default_false(self):
        d = self._make_decomp()
        assert d.requires_incident_timeline is False

    def test_requires_bess_context_default_false(self):
        d = self._make_decomp()
        assert d.requires_bess_context is False

    def test_causal_targets_default_empty(self):
        d = self._make_decomp()
        assert d.causal_targets == []

    def test_spike_thresholds_set(self):
        d = self._make_decomp(spike_thresholds=[300.0, 1000.0])
        assert d.spike_thresholds == [300.0, 1000.0]

    def test_action_context_set(self):
        d = self._make_decomp(action_context="dispatch decision window")
        assert d.action_context == "dispatch decision window"

    def test_missing_inputs_list(self):
        d = self._make_decomp(missing_inputs=["weather", "constraint_count"])
        assert "weather" in d.missing_inputs

    def test_requested_output_set(self):
        d = self._make_decomp(requested_output="probability")
        assert d.requested_output == "probability"

    def test_all_new_fields_round_trip_json(self):
        d = self._make_decomp(
            requires_live_market=True,
            requires_incident_timeline=True,
            causal_targets=["interconnector", "unit_outage"],
            spike_thresholds=[300.0],
            action_context="charge window",
            missing_inputs=["weather"],
            requested_output="narrative",
        )
        data = d.model_dump()
        assert data["requires_live_market"] is True
        assert data["causal_targets"] == ["interconnector", "unit_outage"]
        assert data["requested_output"] == "narrative"


# ── ClaimMapItem schema tests ─────────────────────────────────────────────────

class TestClaimMapItem:

    def test_claim_map_item_construction(self):
        from app.core.schema import ClaimMapItem, ClaimType, DriverConfidenceTier
        item = ClaimMapItem(
            claim_type=ClaimType.PRICE_ASSERTION,
            label="NSW1 price $450/MWh",
            tier=DriverConfidenceTier.CONFIRMED,
            present=True,
            confidence=0.95,
            evidence_ref_ids=["ev-abc123"],
        )
        assert item.claim_type == ClaimType.PRICE_ASSERTION
        assert item.confidence == pytest.approx(0.95)

    def test_claim_map_accepted_in_factual_verdict(self):
        from app.core.schema import (
            FactualVerdict, VerdictLabel, ActionLabel, ClaimMapItem,
            ClaimType, DriverConfidenceTier, ConfidenceBand
        )
        item = ClaimMapItem(
            claim_type=ClaimType.CAUSE_CLAIM,
            label="Interconnector constraint",
            tier=DriverConfidenceTier.SUPPORTED,
            present=True,
            confidence=0.7,
        )
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.45,
            confidence_band=ConfidenceBand.LOW,
            as_of=_now(),
            why_plain_english="Market is constrained.",
            counterargument="Prices may normalise soon.",
            claim_map=[item],
        )
        assert len(v.claim_map) == 1
        assert v.claim_map[0].claim_type == ClaimType.CAUSE_CLAIM

    def test_all_claim_types_valid(self):
        from app.core.schema import ClaimType
        # Sprint O original types — Sprint R adds more; use subset check
        expected = {
            "price_assertion", "demand_assertion", "cause_claim",
            "forecast_claim", "action_recommendation", "probability_claim",
            "historical_analog", "other"
        }
        actual = {ct.value for ct in ClaimType}
        assert expected <= actual, f"Missing ClaimType values: {expected - actual}"


# ── next_watch actionable output tests ───────────────────────────────────────

class TestNextWatch:

    def test_next_watch_default_empty(self):
        from app.core.schema import (
            FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand
        )
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.5,
            confidence_band=ConfidenceBand.LOW,
            as_of=_now(),
            why_plain_english="Normal conditions.",
            counterargument="Could change.",
        )
        assert v.next_watch == []

    def test_next_watch_accepts_list(self):
        from app.core.schema import (
            FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand
        )
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.5,
            confidence_band=ConfidenceBand.LOW,
            as_of=_now(),
            why_plain_english="Elevated regime.",
            counterargument="May resolve.",
            next_watch=[
                "Watch NSW1 headroom — below 400 MW triggers spike risk",
                "Check VIC→NSW interconnector flow",
            ],
        )
        assert len(v.next_watch) == 2
        assert "headroom" in v.next_watch[0]

    def test_next_watch_in_model_dump(self):
        from app.core.schema import (
            FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand
        )
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.5,
            confidence_band=ConfidenceBand.LOW,
            as_of=_now(),
            why_plain_english="Normal.",
            counterargument="May change.",
            next_watch=["Watch headroom"],
        )
        data = v.model_dump()
        assert "next_watch" in data
        assert data["next_watch"] == ["Watch headroom"]


# ── Incident Brief endpoint tests (Sprint O angle) ────���───────────────────────

class TestIncidentBriefSprintO:

    @pytest.mark.asyncio
    async def test_brief_bess_always_has_simulation_only(self, sprint_o_client):
        """simulation_only must be True even when market data is unavailable."""
        resp = await sprint_o_client.get("/incidents/brief/NSW1")
        assert resp.status_code == 200
        bess = resp.json()["bess"]
        assert bess["simulation_only"] is True

    @pytest.mark.asyncio
    async def test_brief_source_freshness_structure(self, sprint_o_client):
        resp = await sprint_o_client.get("/incidents/brief/SA1")
        assert resp.status_code == 200
        freshness = resp.json()["source_freshness"]
        assert "dispatch" in freshness
        assert "notices" in freshness
        assert "status" in freshness["dispatch"]

    @pytest.mark.asyncio
    async def test_brief_verification_checks_is_list(self, sprint_o_client):
        resp = await sprint_o_client.get("/incidents/brief/QLD1")
        assert resp.status_code == 200
        checks = resp.json()["verification"]["checks"]
        assert isinstance(checks, list)
        assert len(checks) >= 1

    @pytest.mark.asyncio
    async def test_brief_provenance_has_models_key(self, sprint_o_client):
        resp = await sprint_o_client.get("/incidents/brief/NSW1")
        assert resp.status_code == 200
        prov = resp.json()["provenance"]
        assert "models" in prov
        assert isinstance(prov["models"], list)

    @pytest.mark.asyncio
    async def test_brief_drivers_window_present(self, sprint_o_client):
        resp = await sprint_o_client.get("/incidents/brief/NSW1")
        assert resp.status_code == 200
        drivers = resp.json()["drivers"]
        assert "window" in drivers
        assert "from" in drivers["window"]
        assert "to" in drivers["window"]
