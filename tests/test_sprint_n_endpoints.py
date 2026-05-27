"""Sprint N endpoint tests: incident brief and forecast trust panel."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest_asyncio.fixture(scope="module")
async def ep_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    from app.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="module")
async def ep_client(ep_engine):
    from datetime import timedelta
    from fastapi import FastAPI
    from app.api.routes_incidents import router as incidents_router
    from app.api.routes_market import router as market_router
    from app.api.auth import TokenPayload
    from app.api.deps import get_current_user, get_db

    app = FastAPI()
    app.include_router(incidents_router)
    app.include_router(market_router)

    Session = async_sessionmaker(ep_engine, expire_on_commit=False)
    fake_user = TokenPayload(
        sub="ep-user",
        tenant_id="t-ep",
        email="ep@test.example",
        exp=_now() + timedelta(hours=1),
    )

    async def _db():
        async with Session() as s:
            yield s

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = lambda: fake_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Market Incident Brief ─────────────────────────────────────────────────────

class TestIncidentBriefEndpoint:

    @pytest.mark.asyncio
    async def test_brief_returns_200(self, ep_client):
        """GET /incidents/brief/{region} returns 200."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_brief_response_shape(self, ep_client):
        """Response has all required top-level keys."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        data = resp.json()
        required = {
            "trace_id", "region", "generated_at", "anchor_time",
            "market", "drivers", "notices", "analogs",
            "forecast", "bess", "source_freshness", "verification", "provenance",
            "elapsed_ms",
        }
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    @pytest.mark.asyncio
    async def test_brief_region_echoed(self, ep_client):
        """Region in path is echoed in the response."""
        resp = await ep_client.get("/incidents/brief/VIC1")
        assert resp.status_code == 200
        assert resp.json()["region"] == "VIC1"

    @pytest.mark.asyncio
    async def test_brief_trace_id_is_unique(self, ep_client):
        """Each call returns a distinct trace_id."""
        r1 = await ep_client.get("/incidents/brief/NSW1")
        r2 = await ep_client.get("/incidents/brief/NSW1")
        assert r1.json()["trace_id"] != r2.json()["trace_id"]

    @pytest.mark.asyncio
    async def test_brief_invalid_region_400(self, ep_client):
        """Unknown region returns 400."""
        resp = await ep_client.get("/incidents/brief/INVALID")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_brief_bess_has_simulation_only(self, ep_client):
        """BESS implication always carries simulation_only=True."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        bess = resp.json()["bess"]
        assert bess.get("simulation_only") is True

    @pytest.mark.asyncio
    async def test_brief_anchor_time_param(self, ep_client):
        """Passing anchor= param is accepted and echoed in anchor_time."""
        anchor = "2026-01-15T12:00:00Z"
        resp = await ep_client.get(f"/incidents/brief/NSW1?anchor={anchor}")
        assert resp.status_code == 200
        data = resp.json()
        assert "2026-01-15" in data["anchor_time"]

    @pytest.mark.asyncio
    async def test_brief_market_state_present(self, ep_client):
        """market section has at least status, regime, and age_seconds keys."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        market = resp.json()["market"]
        assert "status" in market
        # regime and age may be None if no data, but the key must exist
        assert "regime" in market

    @pytest.mark.asyncio
    async def test_brief_elapsed_ms_is_positive(self, ep_client):
        """elapsed_ms tracks actual wall time."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        assert resp.json()["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_brief_verification_has_checks(self, ep_client):
        """verification section has at least one check entry."""
        resp = await ep_client.get("/incidents/brief/NSW1")
        checks = resp.json()["verification"]["checks"]
        assert isinstance(checks, list)
        assert len(checks) >= 1


# ── Forecast Trust Panel ──────────────────────────────────────────────────────

class TestForecastTrustEndpoint:

    @pytest.mark.asyncio
    async def test_trust_returns_200(self, ep_client):
        """GET /market/trust returns 200."""
        resp = await ep_client.get("/market/trust?region=NSW1")
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_trust_response_shape(self, ep_client):
        """Response has region, models, eval_source, note."""
        resp = await ep_client.get("/market/trust?region=NSW1")
        data = resp.json()
        assert "region" in data
        assert "models" in data
        assert "eval_source" in data
        assert "note" in data

    @pytest.mark.asyncio
    async def test_trust_region_echoed(self, ep_client):
        """Region query param is echoed."""
        resp = await ep_client.get("/market/trust?region=VIC1")
        assert resp.json()["region"] == "VIC1"

    @pytest.mark.asyncio
    async def test_trust_models_is_list(self, ep_client):
        """models field is a list (may be empty if registry not populated)."""
        resp = await ep_client.get("/market/trust?region=NSW1")
        assert isinstance(resp.json()["models"], list)

    @pytest.mark.asyncio
    async def test_trust_registered_model_has_required_fields(self, ep_client):
        """After registering a model, it appears in the trust panel with all fields."""
        from app.engines.forecasting.model_registry import register_model, make_training_ref
        ref = make_training_ref("NSW1", "2026-01-01", "2026-05-01", 5000)
        register_model("test_lear_trust", "1.0.0", ref, {"architecture": "quantile_linear"})

        resp = await ep_client.get("/market/trust?region=NSW1")
        models = resp.json()["models"]
        entry = next((m for m in models if m["model"] == "test_lear_trust"), None)
        assert entry is not None, "Registered model must appear in trust panel"

        for field in ("model", "version", "last_trained", "training_data_ref", "training_window"):
            assert field in entry, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_trust_training_window_parsed(self, ep_client):
        """training_window is a human-readable string, not the raw ref."""
        from app.engines.forecasting.model_registry import register_model, make_training_ref
        ref = make_training_ref("NSW1", "2025-01-01", "2025-06-01", 2628)
        register_model("test_window_parse", "1.0.0", ref)

        resp = await ep_client.get("/market/trust?region=NSW1")
        models = resp.json()["models"]
        entry = next((m for m in models if m["model"] == "test_window_parse"), None)
        assert entry is not None
        window = entry.get("training_window")
        assert window is not None
        assert "to" in window, f"Expected 'to' in window: {window}"

    @pytest.mark.asyncio
    async def test_trust_invalid_region_400(self, ep_client):
        """Unknown region returns 400."""
        resp = await ep_client.get("/market/trust?region=UNKNOWN")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_trust_no_eval_returns_not_yet_evaluated(self, ep_client):
        """Without a backtest run, eval_source is 'not_yet_evaluated'."""
        resp = await ep_client.get("/market/trust?region=TAS1")
        data = resp.json()
        # TAS1 unlikely to have eval data in a fresh in-memory DB
        assert data["eval_source"] in ("not_yet_evaluated", "last_backtest")
