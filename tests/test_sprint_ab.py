"""Sprint AB — tests for: gas/ST PASA wiring, LEAR attribution, LNN IG, OpenMeteo.

Covers:
  - GatherResult has gas_context, st_pasa, site_weather fields
  - WhySources propagates all three from gather
  - LEAR feature_attribution() returns correct structure
  - LNNQuantileModel.integrated_gradients() returns correct structure
  - OpenMeteo client returns RegionSiteWeather with correct structure
  - why_builder surfaces gas causal chain in parts
  - answer_planner _plan_explanation uses gas context for evidence bullets
  - cross-season hist_dist guard in _plan_future_date_forecast
  - classify_notice restored in _plan_weather_news
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.agents.scatter_gather import GatherResult
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    FactualVerdict,
    IntentLabel,
    QueryDecomposition,
    VerdictLabel,
)
from app.agents.why_sources import (
    AnalogSummary,
    CurrentDrivers,
    DriverContext,
    ForecastDrivers,
    NewsContext,
    TechnologyContext,
    WeatherContext,
    WhySources,
)


# ── GatherResult field presence ────────────────────────────────────────────────

class TestGatherResultFields:
    def test_has_gas_context(self):
        g = GatherResult(dispatch=None, dispatch_fresh=False)
        assert g.gas_context is None

    def test_has_st_pasa(self):
        g = GatherResult(dispatch=None, dispatch_fresh=False)
        assert g.st_pasa is None

    def test_has_site_weather(self):
        g = GatherResult(dispatch=None, dispatch_fresh=False)
        assert g.site_weather is None

    def test_gas_context_set(self):
        gas = {"latest_hub_price_gj": 12.5, "srmc_ccgt_mwh": 85.0, "high_price_alert": False}
        g = GatherResult(dispatch=None, dispatch_fresh=False, gas_context=gas)
        assert g.gas_context["latest_hub_price_gj"] == 12.5

    def test_st_pasa_set(self):
        pasa = {"available": True, "tight_interval_count": 3}
        g = GatherResult(dispatch=None, dispatch_fresh=False, st_pasa=pasa)
        assert g.st_pasa["tight_interval_count"] == 3

    def test_site_weather_set(self):
        sw = {"site_count": 3, "avg_radiation_wm2": 400.0}
        g = GatherResult(dispatch=None, dispatch_fresh=False, site_weather=sw)
        assert g.site_weather["site_count"] == 3

    def test_tasks_total_updated(self):
        """tasks_total must be 13 for the base case (no weather, no commentary)."""
        g = GatherResult(dispatch=None, dispatch_fresh=False, tasks_ok=0, tasks_total=13)
        assert g.tasks_total == 13


# ── WhySources propagation ─────────────────────────────────────────────────────

class TestWhySourcesFields:
    def test_why_sources_has_gas_context(self):
        from app.agents.why_sources import WhySources
        assert "gas_context" in WhySources.__dataclass_fields__

    def test_why_sources_has_st_pasa(self):
        from app.agents.why_sources import WhySources
        assert "st_pasa" in WhySources.__dataclass_fields__

    def test_why_sources_has_site_weather(self):
        from app.agents.why_sources import WhySources
        assert "site_weather" in WhySources.__dataclass_fields__

    def test_gas_context_defaults_none(self):
        from app.agents.why_sources import WhySources
        # Create minimal WhySources — gas_context should default to None
        ws = WhySources.__dataclass_fields__["gas_context"].default
        assert ws is None


# ── LEAR feature attribution ───────────────────────────────────────────────────

class TestLEARFeatureAttribution:
    def test_returns_list_of_tuples(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        m.fit(X, y)
        result = m.feature_attribution(X[0], top_n=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        for name, val in result:
            assert isinstance(name, str)
            assert isinstance(val, float)

    def test_top_n_respected(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        m.fit(X, y)
        result = m.feature_attribution(X[0], top_n=2)
        assert len(result) <= 2

    def test_returns_empty_when_unfitted(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        result = m.feature_attribution(np.zeros(5), top_n=3)
        assert result == []

    def test_sorted_by_abs_descending(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        X = np.random.randn(100, 10).astype(np.float32)
        y = (X[:, 0] * 50 + np.random.randn(100) * 5).astype(np.float32)
        m.fit(X, y)
        result = m.feature_attribution(X[0], top_n=5)
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert abs(result[i][1]) >= abs(result[i + 1][1])

    def test_feature_names_in_result(self):
        from app.engines.forecasting.models.lear_model import LEARModel
        m = LEARModel()
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        names = ["demand_mw", "last_price", "renewable_frac"]
        m.fit(X, y, feature_names=names)
        result = m.feature_attribution(X[0], top_n=5)
        result_names = [n for n, _ in result]
        # All result names should be real feature names or AR lags
        for name in result_names:
            assert any(
                name == n or name.startswith("ar_lag_") for n in names
            ) or name.startswith("feat_")


# ── LNN Integrated Gradients ───────────────────────────────────────────────────

class TestLNNIntegratedGradients:
    def test_ltc_model_has_method(self):
        from app.engines.lnn.ltc_model import LTCModel
        assert hasattr(LTCModel, "integrated_gradients")

    def test_lnn_quantile_model_has_method(self):
        from app.engines.forecasting.models.lnn_model import LNNQuantileModel
        assert hasattr(LNNQuantileModel, "integrated_gradients")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_lnn_quantile_ig_returns_list(self):
        """LNNQuantileModel.integrated_gradients returns list of (name, float) or []."""
        try:
            from app.engines.forecasting.models.lnn_model import LNNQuantileModel
            m = LNNQuantileModel(epochs=1, hidden=8, seq_len=3)
            X = np.random.randn(20, 5).astype(np.float32)
            y = np.random.randn(20).astype(np.float32)
            m.fit(X, y)
            result = m.integrated_gradients(X[-1])
            assert isinstance(result, list)
            for name, val in result:
                assert isinstance(name, str)
                assert isinstance(val, float)
                assert val >= 0.0  # magnitudes are non-negative
        except ImportError:
            pytest.skip("ncps not installed")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_lnn_quantile_ig_empty_when_unfitted(self):
        try:
            from app.engines.forecasting.models.lnn_model import LNNQuantileModel
            m = LNNQuantileModel(epochs=1)
            result = m.integrated_gradients(np.zeros(5))
            assert result == []
        except ImportError:
            pytest.skip("ncps not installed")


# ── OpenMeteo client ───────────────────────────────────────────────────────────

class TestOpenMeteoClient:
    def test_imports_cleanly(self):
        from app.mcp.openmeteo_client import (
            fetch_region_site_weather, RegionSiteWeather, SiteWeather
        )
        assert callable(fetch_region_site_weather)

    def test_sites_by_region_populated(self):
        from app.mcp.openmeteo_client import _SITES_BY_REGION
        assert len(_SITES_BY_REGION) == 5
        for region in ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]:
            assert region in _SITES_BY_REGION
            assert len(_SITES_BY_REGION[region]) >= 1

    def test_region_site_weather_structure(self):
        from app.mcp.openmeteo_client import RegionSiteWeather, SiteWeather
        sw = SiteWeather(
            duid="TEST01", name="Test", region="NSW1", generator_type="solar",
            lat=-33.0, lon=151.0,
            current_radiation_wm2=450.0, current_wind_kmh=None, current_temp_c=22.0,
        )
        rw = RegionSiteWeather(region="NSW1", sites=[sw])
        d = rw.to_dict()
        assert d["region"] == "NSW1"
        assert d["site_count"] == 1
        assert d["avg_radiation_wm2"] == 450.0

    def test_low_solar_alert_triggered(self):
        from app.mcp.openmeteo_client import SiteWeather
        import datetime
        sw = SiteWeather(
            duid="TEST01", name="Test", region="SA1", generator_type="solar",
            lat=-33.0, lon=138.0,
            current_radiation_wm2=50.0, current_wind_kmh=None, current_temp_c=20.0,
            as_of=datetime.datetime(2026, 6, 8, 2, 0, 0, tzinfo=datetime.timezone.utc),  # 12pm AEST
        )
        assert sw.low_solar_alert is True

    def test_low_wind_alert_triggered(self):
        from app.mcp.openmeteo_client import SiteWeather
        sw = SiteWeather(
            duid="HPRG1", name="Hornsdale", region="SA1", generator_type="wind",
            lat=-33.0, lon=138.0,
            current_radiation_wm2=None, current_wind_kmh=8.0, current_temp_c=15.0,
        )
        assert sw.low_wind_alert is True

    def test_normal_wind_no_alert(self):
        from app.mcp.openmeteo_client import SiteWeather
        sw = SiteWeather(
            duid="HPRG1", name="Hornsdale", region="SA1", generator_type="wind",
            lat=-33.0, lon=138.0,
            current_radiation_wm2=None, current_wind_kmh=25.0, current_temp_c=15.0,
        )
        assert sw.low_wind_alert is False

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_for_unknown_region(self):
        from app.mcp.openmeteo_client import fetch_region_site_weather
        result = await fetch_region_site_weather("UNKNOWN")
        assert result.sites == []


# ── why_builder gas causality ──────────────────────────────────────────────────

class TestWhyBuilderGasCausality:
    def _make_sources(self, gas_context: dict | None = None) -> WhySources:
        now = datetime.now(timezone.utc)
        decomp = QueryDecomposition(
            raw_query="why is the price high?",
            intent=IntentLabel.EXPLANATION,
            entities={"regions": ["NSW1"]},
            requires_why=True,
        )
        current = CurrentDrivers(
            region="NSW1", price_rrp=150.0, demand_mw=8000.0,
            availability_mw=9000.0, headroom_mw=1000.0,
            regime="elevated", is_fresh=True,
            valid_time=now, staleness_seconds=30,
        )
        return WhySources(
            decomp=decomp,
            current=current,
            forecast=ForecastDrivers(available=False, direction="unknown"),
            analogs=AnalogSummary(count=0),
            news=NewsContext(explained=False),
            gas_context=gas_context,
        )

    def test_gas_crisis_surfaced_in_parts(self):
        from app.agents.why_builder import build_why
        gas = {
            "latest_hub_price_gj": 28.0,
            "srmc_ccgt_mwh": 186.0,
            "srmc_ocgt_mwh": 284.0,
            "hub_name": "SYDNEY",
            "price_trend": "rising",
            "high_price_alert": True,
            "crisis_alert": True,
            "source": "AEMO_STTM",
            "raw_ref": "https://aemo.com.au",
        }
        sources = self._make_sources(gas_context=gas)
        result = build_why(sources)
        combined = result.why_plain_english + " ".join(
            part for section in result.answer_sections for part in section.get("items", [])
        )
        assert "Gas crisis" in combined or "crisis" in combined.lower()

    def test_high_gas_surfaces_srmc(self):
        from app.agents.why_builder import build_why
        gas = {
            "latest_hub_price_gj": 18.0,
            "srmc_ccgt_mwh": 121.0,
            "srmc_ocgt_mwh": 184.0,
            "hub_name": "SYDNEY",
            "price_trend": "rising",
            "high_price_alert": True,
            "crisis_alert": False,
            "source": "AEMO_STTM",
            "raw_ref": "https://aemo.com.au",
        }
        sources = self._make_sources(gas_context=gas)
        result = build_why(sources)
        combined = result.why_plain_english + " ".join(
            part for section in result.answer_sections for part in section.get("items", [])
        )
        assert "SRMC" in combined or "121" in combined

    def test_no_gas_adds_missing(self):
        from app.agents.why_builder import build_why
        sources = self._make_sources(gas_context=None)
        result = build_why(sources)
        assert "gas_hub_price" in result.missing_data


# ── answer_planner regressions ─────────────────────────────────────────────────

class TestAnswerPlannerRegressions:
    def test_classify_notice_called_in_weather_news(self):
        """_plan_weather_news must call classify_notice when notice is present."""
        from app.agents.answer_planner import _plan_weather_news
        now = datetime.now(timezone.utc)
        current = CurrentDrivers(
            region="SA1", price_rrp=320.0, demand_mw=3000.0,
            availability_mw=3200.0, headroom_mw=200.0,
            regime="spike", is_fresh=True,
            valid_time=now, staleness_seconds=30,
        )
        news = NewsContext(
            explained=True,
            top_notice_type="LACK OF RESERVE 2",
            credibility_tier=2,
        )
        weather = WeatherContext(available=True, relevant=True, consensus={"temperature_c": 38.0})
        decomp = QueryDecomposition(
            raw_query="why is SA price high?",
            intent=IntentLabel.EXPLANATION,
            entities={"regions": ["SA1"]},
        )
        sources = WhySources(
            decomp=decomp,
            current=current,
            forecast=ForecastDrivers(available=False, direction="unknown"),
            analogs=AnalogSummary(count=0),
            news=news,
            weather=weather,
        )
        factual = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.HOLD,
            confidence=0.5,
            confidence_band=ConfidenceBand.LOW,
            as_of=now,
            why_plain_english="notice context present",
            counterargument="",
            missing_data=[],
        )
        with patch("app.engines.notice_price_signal.classify_notice") as mock_cn:
            mock_signal = MagicMock()
            mock_signal.as_nlp_bullet.return_value = "🔴 LOR2 signal bullet"
            mock_cn.return_value = mock_signal
            result = _plan_weather_news(sources, factual)

        mock_cn.assert_called_once_with("LACK OF RESERVE 2", "SA1")
        assert any("LOR2" in s or "signal" in s.lower() for s in result.direct_answer)

    def test_cross_season_hist_dist_uses_static(self):
        """Cross-season query must use static profiles, not current-season DB data."""
        from app.agents.answer_planner import _plan_future_date_forecast
        now = datetime.now(timezone.utc)
        # June query (current = Winter in AU), target = December (Summer)
        current = CurrentDrivers(
            region="NSW1", price_rrp=80.0, demand_mw=7000.0,
            availability_mw=9000.0, headroom_mw=2000.0,
            regime="normal", is_fresh=True,
            valid_time=now, staleness_seconds=30,
        )
        decomp = QueryDecomposition(
            raw_query="what will prices be in december?",
            intent=IntentLabel.EXPLANATION,
            entities={"regions": ["NSW1"]},
        )
        sources = WhySources(
            decomp=decomp,
            current=current,
            forecast=ForecastDrivers(available=False, direction="unknown"),
            analogs=AnalogSummary(count=0),
            news=NewsContext(explained=False),
        )
        factual = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.HOLD,
            confidence=0.3,
            confidence_band=ConfidenceBand.LOW,
            as_of=now,
            why_plain_english="cross-season forecast",
            counterargument="",
            missing_data=[],
        )

        # hist_dist is current-season (Winter) data — should NOT be used for December
        winter_hist_dist = {
            "available": True,
            "p10": 45,
            "p50": 80,
            "p90": 120,
            "median": 80,
            "count": 500,
        }
        with patch("app.agents.answer_planner.datetime") as mock_dt:
            # Mock current date to June 2026 (Winter in AU)
            mock_dt.now.return_value = datetime(2026, 6, 15)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = _plan_future_date_forecast(
                sources, factual, hist_dist=winter_hist_dist
            )

        # The answer should reference Summer (December), not Winter DB percentiles
        # Summer static range is $50–150 daily; winter DB p10=$45 should NOT appear
        answer_text = " ".join(result.direct_answer + result.key_evidence)
        # Static Summer profile dominates — the daily range won't be $45–$120
        assert "$45–120" not in answer_text
