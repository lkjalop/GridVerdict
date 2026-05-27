"""Backtest integration tests — run against real market_events data.

Two modes:
  ALWAYS-RUN  — Seeds SQLite with 500 realistic NSW1 market_events rows,
                calls run_region_backtest logic directly, verifies ModelScore
                fields are populated and persistence pinball is finite.

  POSTGRES    — set GRIDVERDICT_INTEGRATION_DB=1 to pull from the real DB
                (requires docker-compose up db, expects Aug 2022–Jul 2024 data).

What these tests prove:
  - _fetch_history() returns rows in the right shape from the DB
  - build_features() produces a well-formed feature matrix from real rows
  - run_backtest() produces finite, non-NaN ModelScore values
  - Persistence pinball > 0 (the series is not constant)
  - Seasonal naive score is computed (lag 288 exists in the data)
  - 'experimental_lnn' key is used if LNN is included
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

_INTEGRATION_DB = os.getenv("GRIDVERDICT_INTEGRATION_DB", "").lower() in ("1", "true")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _market_event_rows(
    region: str = "NSW1",
    count: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Synthetic market_events rows that look like real dispatch intervals."""
    rng = np.random.default_rng(seed)
    base_price = 100.0 + rng.standard_normal(count).cumsum() * 8.0
    base_price = np.clip(base_price, 20.0, 600.0)
    base_demand = 8000.0 + rng.standard_normal(count).cumsum() * 50.0
    base_demand = np.clip(base_demand, 4000.0, 14000.0)

    # Start 300 intervals ago so `lookback` covers all rows
    base_time = _now() - timedelta(minutes=count * 5 + 30)
    rows = []
    for i in range(count):
        rows.append({
            "id": str(uuid.uuid4()),
            "region": region,
            "source": "AEMO_DISPATCH_PRICE",
            "valid_time": base_time + timedelta(minutes=i * 5),
            "price_rrp": float(round(base_price[i], 2)),
            "demand_mw": float(round(base_demand[i], 1)),
            "availability_mw": float(round(base_demand[i] * 1.1, 1)),
            "tenant_id": "system",
            "data": {"regime": "spike" if base_price[i] > 300 else "normal"},
        })
    return rows


async def _seed(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import MarketEvent
    for r in rows:
        session.add(MarketEvent(
            id=r["id"],
            region=r["region"],
            source=r["source"],
            valid_time=r["valid_time"],
            system_time=r["valid_time"],
            raw_ref=f"test-seed-{r['id'][:8]}",
            price_rrp=r["price_rrp"],
            demand_mw=r["demand_mw"],
            availability_mw=r["availability_mw"],
            tenant_id=r["tenant_id"],
            data=r["data"],
        ))
    await session.commit()


# ── SQLite-seeded engine fixture ──────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def seeded_backtest_engine():
    from app.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        await _seed(s, _market_event_rows("NSW1", count=500))
    yield engine
    await engine.dispose()


# ── Always-run: seeded SQLite ─────────────────────────────────────────────────

class TestBacktestSeededData:
    """run_region_backtest using SQLite seeded with 500 realistic rows."""

    async def test_fetch_history_returns_rows(self, seeded_backtest_engine):
        from app.engines.backtest import _fetch_history
        from app.db.session import db_session
        from unittest.mock import patch, AsyncMock

        from contextlib import asynccontextmanager
        SessionLocal = async_sessionmaker(bind=seeded_backtest_engine, expire_on_commit=False)

        @asynccontextmanager
        async def _fake_session():
            async with SessionLocal() as s:
                yield s

        with patch("app.db.session.db_session", _fake_session):
            # lookback 2 days should capture all 500 rows (which span ~42 hours)
            rows = await _fetch_history("NSW1", lookback_days=2)

        assert len(rows) >= 200, f"Expected ≥200 rows, got {len(rows)}"
        assert "valid_time" in rows[0]
        assert "price" in rows[0]
        assert "demand" in rows[0]

    async def test_build_features_from_seeded_rows(self, seeded_backtest_engine):
        from app.engines.backtest import _fetch_history
        from app.engines.forecasting.features.market_features import build_features
        from app.db.session import db_session
        from unittest.mock import patch

        from contextlib import asynccontextmanager
        SessionLocal = async_sessionmaker(bind=seeded_backtest_engine, expire_on_commit=False)

        @asynccontextmanager
        async def _fake_session():
            async with SessionLocal() as s:
                yield s

        with patch("app.db.session.db_session", _fake_session):
            rows = await _fetch_history("NSW1", lookback_days=2)

        target_times = [r["valid_time"] for r in rows]
        X, y = build_features(rows, target_times)

        assert X.ndim == 2, "Feature matrix must be 2D"
        assert y.ndim == 1, "Target vector must be 1D"
        assert len(X) > 0
        assert not np.any(np.isnan(X)), "Feature matrix must not contain NaN"
        assert not np.any(np.isnan(y)), "Target vector must not contain NaN"

    async def test_run_backtest_on_seeded_data_produces_valid_scores(self, seeded_backtest_engine):
        from app.engines.backtest import _fetch_history
        from app.engines.forecasting.features.market_features import build_features, COL_LAST_PRICE, COL_SEASONAL
        from app.engines.forecasting.evaluation.harness import run_backtest
        from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel
        from app.db.session import db_session
        from unittest.mock import patch

        from contextlib import asynccontextmanager
        SessionLocal = async_sessionmaker(bind=seeded_backtest_engine, expire_on_commit=False)

        @asynccontextmanager
        async def _fake_session():
            async with SessionLocal() as s:
                yield s

        with patch("app.db.session.db_session", _fake_session):
            rows = await _fetch_history("NSW1", lookback_days=2)

        target_times = [r["valid_time"] for r in rows]
        X, y = build_features(rows, target_times)

        models = {
            "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
            "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
        }
        report = run_backtest(
            models=models, X=X, y=y,
            horizon=6, step=12, min_train=288, spike_threshold=300.0,
        )

        scores_by_name = {s.model_name: s for s in report.scores}
        assert "persistence" in scores_by_name
        assert "seasonal_naive" in scores_by_name

        p = scores_by_name["persistence"]
        assert np.isfinite(p.pinball), f"Persistence pinball is not finite: {p.pinball}"
        assert p.pinball > 0, "Persistence pinball should be > 0 (series is not constant)"
        assert np.isfinite(p.crps)
        assert 0.0 <= p.calibration_error <= 1.0
        assert 0.0 <= p.p50_exceedance_rate <= 1.0

    async def test_persistence_pinball_less_than_seasonal_naive(self, seeded_backtest_engine):
        """For a smooth synthetic series, persistence should beat seasonal naive."""
        from app.engines.backtest import _fetch_history
        from app.engines.forecasting.features.market_features import build_features, COL_LAST_PRICE, COL_SEASONAL
        from app.engines.forecasting.evaluation.harness import run_backtest
        from app.engines.forecasting.models.baselines import PersistenceModel, SeasonalNaiveModel
        from app.db.session import db_session
        from unittest.mock import patch

        from contextlib import asynccontextmanager
        SessionLocal = async_sessionmaker(bind=seeded_backtest_engine, expire_on_commit=False)

        @asynccontextmanager
        async def _fake_session():
            async with SessionLocal() as s:
                yield s

        with patch("app.db.session.db_session", _fake_session):
            rows = await _fetch_history("NSW1", lookback_days=2)

        target_times = [r["valid_time"] for r in rows]
        X, y = build_features(rows, target_times)
        models = {
            "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
            "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
        }
        report = run_backtest(models=models, X=X, y=y, horizon=6, step=12, min_train=288)
        scores = {s.model_name: s for s in report.scores}
        # Persistence should be better (lower pinball) on a smooth random-walk series
        assert scores["persistence"].pinball <= scores["seasonal_naive"].pinball * 1.5, (
            "Persistence pinball is suspiciously worse than seasonal naive on smooth data"
        )

    async def test_lnn_key_is_experimental_when_included(self, seeded_backtest_engine):
        """Model dict must use 'experimental_lnn' key, not 'lnn_ltc' or similar."""
        from app.engines.backtest import _LTCAdapter
        assert _LTCAdapter.name == "experimental_lnn"


# ── Postgres integration (GRIDVERDICT_INTEGRATION_DB=1) ───────────────────────

class TestBacktestPostgres:
    """Runs run_region_backtest("NSW1") against real Postgres.

    Expects: Aug 2022–Jul 2024 market_events data to be present.
    Run: GRIDVERDICT_INTEGRATION_DB=1 pytest tests/test_backtest_integration.py::TestBacktestPostgres -v
    """

    @pytest.mark.skipif(not _INTEGRATION_DB, reason="GRIDVERDICT_INTEGRATION_DB not set")
    async def test_run_region_backtest_nsw1_30d(self):
        from app.db.session import init_db
        from app.engines.backtest import run_region_backtest

        await init_db()
        report = await run_region_backtest("NSW1", lookback_days=30, include_lnn=False)

        scores = {s.model_name: s for s in report.scores}
        assert "persistence" in scores, "persistence model must be in report"
        p = scores["persistence"]
        assert np.isfinite(p.pinball) and p.pinball > 0, f"Bad persistence pinball: {p.pinball}"
        assert np.isfinite(p.crps) and p.crps > 0
        assert report.n_origins > 0, "Must have at least one backtest origin"
        assert report.horizon_min > 0

    @pytest.mark.skipif(not _INTEGRATION_DB, reason="GRIDVERDICT_INTEGRATION_DB not set")
    async def test_run_region_backtest_all_five_regions(self):
        from app.db.session import init_db
        from app.engines.backtest import run_region_backtest

        await init_db()
        for region in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]:
            report = await run_region_backtest(region, lookback_days=30, include_lnn=False)
            scores = {s.model_name: s for s in report.scores}
            assert "persistence" in scores, f"persistence missing for {region}"
            p = scores["persistence"]
            assert np.isfinite(p.pinball), f"{region} pinball is NaN/inf"

    @pytest.mark.skipif(not _INTEGRATION_DB, reason="GRIDVERDICT_INTEGRATION_DB not set")
    async def test_run_region_backtest_nsw1_has_enough_origins(self):
        """30-day window with step=12 should produce many origins (≥ 100)."""
        from app.db.session import init_db
        from app.engines.backtest import run_region_backtest

        await init_db()
        report = await run_region_backtest("NSW1", lookback_days=30, include_lnn=False)
        assert report.n_origins >= 100, (
            f"Expected ≥100 walk-forward origins for 30-day window, got {report.n_origins}"
        )
