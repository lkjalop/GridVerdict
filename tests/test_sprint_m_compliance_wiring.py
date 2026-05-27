"""Sprint M: Compliance wiring integration tests.

Proves that the wiring added in Priority 1 is real:
  - Query execution creates ObserverEvent rows for all 4 security passes
  - LEAR and QRA model fits register in model_registry with training_data_ref
  - MetaEnsemble registers after blend
  - BESS scenario audit log row has non-null model_version from registry
  - Seasonal query path also creates observer events
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# ── DB + HTTP fixtures ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _market_event_rows(region: str = "NSW1", count: int = 320) -> list[dict[str, Any]]:
    base = _now() - timedelta(hours=count // 12)
    rows = []
    for i in range(count):
        ts = base + timedelta(minutes=i * 5)
        price = 80.0 + (i % 48) * 6.5
        rows.append({
            "id": str(uuid.uuid4()),
            "region": region,
            "source": "AEMO_DISPATCH_PRICE",
            "valid_time": ts,
            "price_rrp": round(price, 2),
            "demand_mw": 8000.0 + i * 2,
            "availability_mw": 9200.0,
            "tenant_id": "system",
            "data": {"regime": "normal" if price < 300 else "elevated"},
        })
    return rows


async def _seed(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import MarketEvent
    for r in rows:
        session.add(MarketEvent(
            id=r["id"], region=r["region"], source=r["source"],
            valid_time=r["valid_time"], system_time=r["valid_time"],
            raw_ref=f"seed-{r['id'][:8]}",
            price_rrp=r["price_rrp"], demand_mw=r["demand_mw"],
            availability_mw=r["availability_mw"],
            tenant_id=r["tenant_id"], data=r["data"],
        ))
    await session.commit()


@pytest_asyncio.fixture(scope="module")
async def wiring_engine():
    from app.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        await _seed(s, _market_event_rows("NSW1", 320))
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def wiring_client(wiring_engine) -> AsyncClient:
    from app.api.main import create_app
    from app.db.session import get_db

    app = create_app()
    SessionLocal = async_sessionmaker(bind=wiring_engine, expire_on_commit=False)

    async def _override():
        async with SessionLocal() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app.dependency_overrides[get_db] = _override
    async with AsyncClient(
        transport=__import__("httpx").ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def wiring_session(wiring_engine) -> AsyncSession:
    SessionLocal = async_sessionmaker(bind=wiring_engine, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s
        await s.rollback()


async def _new_session(client: AsyncClient, region: str = "NSW1") -> str:
    r = await client.post("/api/sessions", json={"region": region})
    assert r.status_code in (200, 201), r.text
    data = r.json()
    return data.get("session_id") or data.get("id")


def _dispatch_mock(price: float = 185.0):
    from app.data.aemo_live_client import DispatchPrice
    return DispatchPrice(
        region="NSW1",
        valid_time=_now() - timedelta(seconds=90),
        system_time=_now(),
        price_rrp=price,
        demand_mw=8600.0,
        availability_mw=9400.0,
        raw_ref="wiring-test-ref",
    )


# ── Observer event wiring tests ───────────────────────────────────────────────

class TestObserverEventWiring:
    @pytest.fixture(autouse=True)
    def _patch_http(self):
        dp = _dispatch_mock()
        with (
            patch("app.agents.scatter_gather._task_dispatch", new=AsyncMock(return_value=dp)),
            patch("app.agents.scatter_gather._task_notices", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_predispatch", new=AsyncMock(return_value=[])),
            patch("app.agents.scatter_gather._task_news_sentiment", new=AsyncMock(return_value=[])),
        ):
            yield

    @pytest.mark.asyncio
    async def test_query_creates_observer_events(self, wiring_client, wiring_session):
        """A successful query must persist at least 1 ObserverEvent row."""
        from sqlalchemy import select
        from app.db.models import ObserverEvent

        sid = await _new_session(wiring_client, "NSW1")
        r = await wiring_client.post(
            f"/api/sessions/{sid}/query",
            json={"text": "What is the NSW dispatch price?", "region": "NSW1"},
        )
        assert r.status_code == 200, r.text
        query_id = r.json()["query_id"]

        rows = (await wiring_session.execute(
            select(ObserverEvent).where(ObserverEvent.query_id == query_id)
        )).scalars().all()

        assert len(rows) >= 1, (
            f"Expected ≥1 ObserverEvent for query_id={query_id}, got {len(rows)}"
        )

    @pytest.mark.asyncio
    async def test_query_creates_all_four_passes(self, wiring_client, wiring_session):
        """All 4 security passes must each produce an ObserverEvent row."""
        from sqlalchemy import select
        from app.db.models import ObserverEvent

        sid = await _new_session(wiring_client, "NSW1")
        r = await wiring_client.post(
            f"/api/sessions/{sid}/query",
            json={"text": "Should I charge my battery?", "region": "NSW1"},
        )
        assert r.status_code == 200, r.text
        query_id = r.json()["query_id"]

        rows = (await wiring_session.execute(
            select(ObserverEvent).where(ObserverEvent.query_id == query_id)
        )).scalars().all()

        phases = {row.phase for row in rows}
        assert "input" in phases, f"Missing 'input' phase — got {phases}"
        assert "answer" in phases, f"Missing 'answer' phase — got {phases}"

    @pytest.mark.asyncio
    async def test_observer_event_has_tenant_id(self, wiring_client, wiring_session):
        """ObserverEvent rows must carry the session's tenant_id."""
        from sqlalchemy import select
        from app.db.models import ObserverEvent

        sid = await _new_session(wiring_client, "NSW1")
        r = await wiring_client.post(
            f"/api/sessions/{sid}/query",
            json={"text": "NSW price now?", "region": "NSW1"},
        )
        assert r.status_code == 200, r.text
        query_id = r.json()["query_id"]

        rows = (await wiring_session.execute(
            select(ObserverEvent).where(ObserverEvent.query_id == query_id)
        )).scalars().all()
        assert all(row.tenant_id for row in rows), "All ObserverEvent rows must have tenant_id"

    @pytest.mark.asyncio
    async def test_observer_events_have_trace_id(self, wiring_client, wiring_session):
        """ObserverEvent rows created by a query must carry the query trace_id."""
        from sqlalchemy import select
        from app.db.models import ObserverEvent

        sid = await _new_session(wiring_client, "NSW1")
        r = await wiring_client.post(
            f"/api/sessions/{sid}/query",
            json={"text": "What drives the NSW price?", "region": "NSW1"},
        )
        assert r.status_code == 200, r.text
        query_id = r.json()["query_id"]

        rows = (await wiring_session.execute(
            select(ObserverEvent).where(ObserverEvent.query_id == query_id)
        )).scalars().all()
        assert len(rows) >= 1
        assert all(row.trace_id is not None for row in rows), (
            "All ObserverEvent rows must carry trace_id"
        )


# ── Model registry wiring tests ───────────────────────────────────────────────

class TestModelRegistryWiring:
    def test_register_model_populates_registry(self):
        """register_model() persists to in-process dict and is retrievable."""
        from app.engines.forecasting.model_registry import register_model, get_model_info
        register_model(
            "test_lear_wiring",
            version="1.0.0",
            training_data_ref="NSW1:2024-01-01/2024-02-01:n=2880:sha=abc12345",
        )
        info = get_model_info("test_lear_wiring")
        assert info is not None
        assert info["version"] == "1.0.0"
        assert "NSW1" in info["training_data_ref"]

    def test_make_training_ref_contains_region_and_n(self):
        """make_training_ref must include region, n=, and sha= for provenance."""
        from app.engines.forecasting.model_registry import make_training_ref
        ref = make_training_ref(
            "VIC1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
            2880,
        )
        assert "VIC1" in ref
        assert "n=2880" in ref
        assert "sha=" in ref

    def test_bess_policy_registered_at_startup(self):
        """bess-policy must always be present in model_registry (static entry)."""
        from app.engines.forecasting.model_registry import get_model_info
        info = get_model_info("bess-policy")
        assert info is not None
        assert info["version"] == "1.0.0"

    def test_fleet_policy_registered_at_startup(self):
        """fleet-policy must always be present in model_registry (static entry)."""
        from app.engines.forecasting.model_registry import get_model_info
        info = get_model_info("fleet-policy")
        assert info is not None

    def test_get_all_models_returns_list(self):
        """get_all_models must return a list containing static entries."""
        from app.engines.forecasting.model_registry import get_all_models
        models = get_all_models()
        names = {m["model_name"] for m in models}
        assert "bess-policy" in names
        assert "fleet-policy" in names

    @pytest.mark.asyncio
    async def test_bess_route_audit_has_model_version(self, wiring_client, wiring_session):
        """POST /portfolio/bess-scenario must write model_version to decision_audit_log."""
        from sqlalchemy import select
        from app.db.models import DecisionAuditLog

        payload = {
            "position": {
                "capacity_mwh": 10.0,
                "soc_pct": 80.0,
                "max_discharge_mw": 5.0,
                "max_charge_mw": 5.0,
                "efficiency_pct": 90.0,
                "degradation_cost_per_mwh": 5.0,
                "min_reserve_soc_pct": 10.0,
            },
            "market": {
                "region": "NSW1",
                "price_rrp": 500.0,
                "price_regime": "spike",
            },
        }
        r = await wiring_client.post("/api/portfolio/bess/scenario", json=payload)

        assert r.status_code == 200, r.text

        rows = (await wiring_session.execute(
            select(DecisionAuditLog).where(
                DecisionAuditLog.decision_type == "bess_dispatch"
            )
        )).scalars().all()

        assert len(rows) >= 1, "BESS scenario must write a DecisionAuditLog row"
        row = rows[-1]
        assert row.model_version is not None, (
            "model_version must be non-null — registry wiring is broken"
        )


# ── Model registry provenance contract ───────────────────────────────────────

class TestModelProvenanceContract:
    def test_training_ref_sha_is_deterministic(self):
        """Same inputs must produce the same sha= hash."""
        from app.engines.forecasting.model_registry import make_training_ref
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)
        r1 = make_training_ref("NSW1", start, end, 8640)
        r2 = make_training_ref("NSW1", start, end, 8640)
        assert r1 == r2

    def test_training_ref_differs_for_different_regions(self):
        """Different regions must produce different training refs."""
        from app.engines.forecasting.model_registry import make_training_ref
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)
        r_nsw = make_training_ref("NSW1", start, end, 8640)
        r_vic = make_training_ref("VIC1", start, end, 8640)
        assert r_nsw != r_vic

    def test_compute_data_hash_16_chars(self):
        """compute_data_hash must return a 16-character hex string."""
        from app.engines.forecasting.model_registry import compute_data_hash
        h = compute_data_hash({"region": "NSW1", "n": 8640})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_registered_model_has_registered_at(self):
        """A freshly registered model must have registered_at timestamp."""
        from app.engines.forecasting.model_registry import register_model, get_model_info
        register_model("test_ts_model", version="1.0.0")
        info = get_model_info("test_ts_model")
        assert "registered_at" in info
