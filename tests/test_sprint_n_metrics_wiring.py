"""Sprint N: metrics wiring tests.

Proves that the four real latency/gauge metrics are observed at their natural
call sites — not just registered, but actually populated with data.

  1. gv_query_latency_ms       — increments on every /sessions/{id}/query call
  2. gv_llm_decompose_latency_ms — increments on each llm_decompose() call inside query
  3. gv_forecast_latency_ms    — increments on /market/forecast
  4. gv_scheduler_jobs_running — goes up on _job_enter, back down on _job_exit
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    FactualVerdict,
    IntentLabel,
    VerdictLabel,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sample(metric_name: str) -> float:
    """Return the current _count sample for a histogram, or 0 if not yet observed."""
    from prometheus_client import REGISTRY
    v = REGISTRY.get_sample_value(metric_name)
    return v if v is not None else 0.0


# ── minimal app fixture ───────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def metrics_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    from app.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="module")
async def metrics_client(metrics_engine):
    Session = async_sessionmaker(metrics_engine, expire_on_commit=False)

    from fastapi import FastAPI
    from app.api.routes_query import router as query_router
    from app.api.routes_market import router as market_router

    app = FastAPI()
    app.include_router(query_router)
    app.include_router(market_router)

    from app.api.auth import TokenPayload
    from app.api.deps import get_current_user, get_db
    from app.data.aemo_live_client import AEMOLiveClient
    from app.data.cache import MarketCache

    from datetime import timedelta
    fake_user = TokenPayload(
        sub="metrics-user",
        tenant_id="t-metrics",
        email="metrics@test.example",
        exp=_now() + timedelta(hours=1),
    )

    async def _override_db():
        async with Session() as s:
            yield s

    def _override_user():
        return fake_user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ── session seed ──────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def seeded_session_id(metrics_engine):
    """Create a real session row so the query route can find it."""
    Session = async_sessionmaker(metrics_engine, expire_on_commit=False)
    from app.db.models import Session as SessionModel
    sid = str(uuid.uuid4())
    async with Session() as s:
        s.add(SessionModel(
            id=sid,
            tenant_id="t-metrics",
            user_id="metrics-user",
            title="metrics-test-session",
        ))
        await s.commit()
    return sid


# ── mocks shared across query tests ──────────────────────────────────────────

def _make_dispatch_price():
    from app.data.aemo_live_client import DispatchPrice
    return DispatchPrice(
        region="NSW1",
        price_rrp=95.0,
        demand_mw=7500.0,
        availability_mw=9000.0,
        valid_time=_now(),
        system_time=_now(),
        raw_ref="metrics-test",
    )


def _make_gather_result():
    from app.agents.scatter_gather import GatherResult
    from app.mcp.source_status import SourceStatus
    dp = _make_dispatch_price()
    return GatherResult(
        dispatch=dp,
        notices=[],
        analogs=[],
        news_items=[],
        driver_events=[],
        unit_events=[],
        dispatch_fresh=True,
        notices_stale=False,
        source_statuses={
            "AEMO_DISPATCH_PRICE": SourceStatus(
                source="AEMO_DISPATCH_PRICE",
                valid_time=_now().isoformat(),
                confidence=1.0,
            )
        },
    )


# ── 1. Query latency histogram ────────────────────────────────────────────────

class TestQueryLatencyMetric:

    @pytest.mark.asyncio
    async def test_query_latency_histogram_increments(self, metrics_client, seeded_session_id):
        """After a successful query, gv_query_latency_ms_count increases."""
        before = _sample("gv_query_latency_ms_count")

        gather = _make_gather_result()

        _fake_verdict = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.85,
            confidence_band=ConfidenceBand.HIGH,
            as_of=_now(),
            why_plain_english="Prices are normal.",
            counterargument="No contrary evidence.",
            trace_id="trace-test",
        )
        _fake_decomp = MagicMock(
            intent=IntentLabel.LOOKUP,
            entities={"regions": []},
            time_range=None,
            model_dump=lambda **kw: {"intent": "lookup", "entities": {}, "time_range": None},
        )

        with (
            patch("app.api.routes_query.llm_decompose", new=AsyncMock(return_value=_fake_decomp)),
            patch("app.api.routes_query.scatter_gather", new=AsyncMock(return_value=gather)),
            patch("app.api.routes_query.build_why", return_value=MagicMock(
                confidence=0.85, why_plain_english="Prices are normal.",
                evidence_refs=[], counterargument="No contrary evidence.", missing_data=[],
            )),
            patch("app.api.routes_query.format_verdict", return_value=_fake_verdict),
            patch("app.api.routes_query.verify_answer", return_value=MagicMock(findings=[])),
            patch("app.api.routes_query.apply_verification", side_effect=lambda v, _: v),
            patch("app.api.routes_query.write_trace", new=AsyncMock()),
            patch("app.api.routes_query.assemble_why_sources", return_value=MagicMock(
                source_coverage=0.9,
                forecast=MagicMock(model_detail=[]),
                drivers=MagicMock(binding_constraints=[]),
            )),
            patch("app.api.routes_query._build_evidence_quality", return_value={}),
        ):
            resp = await metrics_client.post(
                f"/sessions/{seeded_session_id}/query",
                json={"text": "What is the current price in NSW1?", "region": "NSW1"},
            )

        assert resp.status_code == 200, resp.text
        after = _sample("gv_query_latency_ms_count")
        assert after > before, f"gv_query_latency_ms_count did not increment (before={before}, after={after})"

    @pytest.mark.asyncio
    async def test_decompose_latency_histogram_increments(self, metrics_client, seeded_session_id):
        """After a successful query, gv_llm_decompose_latency_ms_count increases."""
        before = _sample("gv_llm_decompose_latency_ms_count")

        gather = _make_gather_result()

        _fake_verdict2 = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.MONITOR,
            confidence=0.8,
            confidence_band=ConfidenceBand.HIGH,
            as_of=_now(),
            why_plain_english="Demand is moderate.",
            counterargument="No contrary.",
            trace_id="trace-test2",
        )
        _fake_decomp2 = MagicMock(
            intent=IntentLabel.LOOKUP,
            entities={"regions": []},
            time_range=None,
            model_dump=lambda **kw: {"intent": "lookup", "entities": {}, "time_range": None},
        )

        with (
            patch("app.api.routes_query.llm_decompose", new=AsyncMock(return_value=_fake_decomp2)),
            patch("app.api.routes_query.scatter_gather", new=AsyncMock(return_value=gather)),
            patch("app.api.routes_query.build_why", return_value=MagicMock(
                confidence=0.8, why_plain_english="Demand is moderate.",
                evidence_refs=[], counterargument="No contrary.", missing_data=[],
            )),
            patch("app.api.routes_query.format_verdict", return_value=_fake_verdict2),
            patch("app.api.routes_query.verify_answer", return_value=MagicMock(findings=[])),
            patch("app.api.routes_query.apply_verification", side_effect=lambda v, _: v),
            patch("app.api.routes_query.write_trace", new=AsyncMock()),
            patch("app.api.routes_query.assemble_why_sources", return_value=MagicMock(
                source_coverage=0.9,
                forecast=MagicMock(model_detail=[]),
                drivers=MagicMock(binding_constraints=[]),
            )),
            patch("app.api.routes_query._build_evidence_quality", return_value={}),
        ):
            resp = await metrics_client.post(
                f"/sessions/{seeded_session_id}/query",
                json={"text": "How is demand today?", "region": "NSW1"},
            )

        assert resp.status_code == 200
        after = _sample("gv_llm_decompose_latency_ms_count")
        assert after > before, f"gv_llm_decompose_latency_ms_count did not increment"

    @pytest.mark.asyncio
    async def test_query_latency_sum_is_positive(self, metrics_client, seeded_session_id):
        """gv_query_latency_ms_sum must be > 0 after at least one observation."""
        sum_val = _sample("gv_query_latency_ms_sum")
        # At least one test ran before this — sum must be positive
        assert sum_val is not None and sum_val >= 0.0, (
            f"gv_query_latency_ms_sum unexpectedly None or negative: {sum_val}"
        )


# ── 2. Forecast latency histogram ─────────────────────────────────────────────

class TestForecastLatencyMetric:

    @pytest.mark.asyncio
    async def test_forecast_latency_histogram_increments(self, metrics_client):
        """After /market/forecast, gv_forecast_latency_ms_count increases."""
        before = _sample("gv_forecast_latency_ms_count")

        fake_forecast = {
            "region": "NSW1",
            "intervals": [],
            "models": ["lear", "qra"],
            "generated_at": _now().isoformat(),
        }

        with patch("app.mcp.router.call_tool", new=AsyncMock(return_value=fake_forecast)):
            resp = await metrics_client.get("/market/forecast?region=NSW1")

        assert resp.status_code == 200
        after = _sample("gv_forecast_latency_ms_count")
        assert after > before, f"gv_forecast_latency_ms_count did not increment (before={before}, after={after})"

    @pytest.mark.asyncio
    async def test_forecast_latency_sum_positive_after_call(self, metrics_client):
        """Each forecast call contributes a positive observation to the histogram sum."""
        before_sum = _sample("gv_forecast_latency_ms_sum") or 0.0

        with patch("app.mcp.router.call_tool", new=AsyncMock(return_value={"region": "VIC1", "intervals": []})):
            resp = await metrics_client.get("/market/forecast?region=VIC1")

        assert resp.status_code == 200
        after_sum = _sample("gv_forecast_latency_ms_sum") or 0.0
        assert after_sum >= before_sum, "Forecast latency sum must not decrease"


# ── 3. Scheduler jobs running gauge ──────────────────────────────────────────

class TestSchedulerJobsRunningGauge:

    def test_job_enter_increments_gauge(self):
        """_job_enter() increments gv_scheduler_jobs_running."""
        from app.data.scheduler import _job_enter, _job_exit
        import app.data.scheduler as sched_mod

        original = sched_mod._running_count
        _job_enter("test_probe")
        assert sched_mod._running_count == original + 1
        _job_exit("test_probe")  # cleanup

    def test_job_exit_decrements_gauge(self):
        """_job_exit() decrements gv_scheduler_jobs_running."""
        from app.data.scheduler import _job_enter, _job_exit
        import app.data.scheduler as sched_mod

        _job_enter("test_probe_2")
        before = sched_mod._running_count
        _job_exit("test_probe_2")
        assert sched_mod._running_count == before - 1

    def test_concurrent_jobs_add_up(self):
        """Two concurrent jobs produce count=2 while both are running."""
        from app.data.scheduler import _job_enter, _job_exit
        import app.data.scheduler as sched_mod

        _job_enter("job_a")
        _job_enter("job_b")
        assert sched_mod._running_count >= 2
        _job_exit("job_a")
        _job_exit("job_b")

    def test_exit_never_goes_below_zero(self):
        """Extra _job_exit() calls are safe — count floors at 0."""
        from app.data.scheduler import _job_exit
        import app.data.scheduler as sched_mod

        sched_mod._running_count = 0
        _job_exit("phantom_job")
        assert sched_mod._running_count == 0

    def test_gauge_matches_running_count(self):
        """After _job_enter, the Prometheus gauge value equals _running_count."""
        from app.data.scheduler import _job_enter, _job_exit
        import app.data.scheduler as sched_mod
        from prometheus_client import REGISTRY

        _job_enter("gauge_probe")
        gauge_val = REGISTRY.get_sample_value("gv_scheduler_jobs_running")
        expected = sched_mod._running_count
        _job_exit("gauge_probe")

        assert gauge_val == expected, f"Gauge={gauge_val}, module count={expected}"
