"""Predispatch ingestion pipeline tests.

Covers:
- _job_predispatch_refresh() stores AEMO_PREDISPATCH_30MIN rows in the DB
- Rows use deterministic SHA-256 IDs so re-runs are idempotent (no duplicates)
- _row_to_series_dict() joins predispatch to dispatch by 30-min boundary
- _fetch_history() returns predispatch-joined rows when DB has PD data
"""
from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# ── SQLite fixture ────────────────────────────────────────────────────────────

@pytest.fixture
async def mem_engine():
    from app.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(mem_engine):
    return async_sessionmaker(bind=mem_engine, expire_on_commit=False)


# ── 1. _row_to_series_dict predispatch join ───────────────────────────────────

def test_row_to_series_dict_uses_predispatch_lookup():
    """When pd_lookup has a matching 30-min key, aemo_predispatch != price."""
    from app.engines.backtest import _row_to_series_dict

    vt = datetime(2024, 7, 15, 10, 20, 0, tzinfo=timezone.utc)
    row = (vt, 150.0, 8000.0, 9000.0)

    # PD lookup keyed at :00 because minute=20 < 30
    pd_lookup = {
        datetime(2024, 7, 15, 10, 0, 0, tzinfo=timezone.utc): 175.0
    }
    d = _row_to_series_dict(row, "NSW1", pd_lookup)
    assert d["aemo_predispatch"] == 175.0
    assert d["last_price"] == 150.0


def test_row_to_series_dict_marks_missing_predispatch_for_lag_fill():
    """When pd_lookup is None, aemo_predispatch stays None until non-leaking lag fill."""
    from app.engines.backtest import _row_to_series_dict

    vt = datetime(2024, 7, 15, 10, 20, 0, tzinfo=timezone.utc)
    row = (vt, 150.0, 8000.0, 9000.0)
    d = _row_to_series_dict(row, "NSW1", None)
    assert d["aemo_predispatch"] is None


def test_row_to_series_dict_rounds_to_half_hour_boundary():
    """Minutes 0-29 → :00 key; minutes 30-59 → :30 key."""
    from app.engines.backtest import _row_to_series_dict

    # Minute 35 → rounded to :30
    vt = datetime(2024, 7, 15, 14, 35, 0, tzinfo=timezone.utc)
    row = (vt, 100.0, 0.0, 0.0)
    pd_lookup = {
        datetime(2024, 7, 15, 14, 30, 0, tzinfo=timezone.utc): 200.0,
        datetime(2024, 7, 15, 14, 0, 0, tzinfo=timezone.utc): 999.0,
    }
    d = _row_to_series_dict(row, "NSW1", pd_lookup)
    assert d["aemo_predispatch"] == 200.0


# ── 2. Idempotent row ID ──────────────────────────────────────────────────────

def test_predispatch_row_id_is_deterministic():
    """The SHA-256-based row ID must be the same on every call for the same input."""
    region = "NSW1"
    ts = datetime(2024, 7, 15, 10, 30, 0, tzinfo=timezone.utc)
    key = f"pd-{region}-{ts.isoformat()}"
    id1 = hashlib.sha256(key.encode()).hexdigest()[:36]
    id2 = hashlib.sha256(key.encode()).hexdigest()[:36]
    assert id1 == id2


# ── 3. _job_predispatch_refresh stores rows ───────────────────────────────────

@pytest.mark.asyncio
async def test_predispatch_job_stores_rows(session_factory):
    """_job_predispatch_refresh should insert AEMO_PREDISPATCH_30MIN rows."""
    from app.data.aemo_live_client import PredispatchInterval
    from app.db.models import MarketEvent

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    fake_interval = PredispatchInterval(
        region="NSW1",
        interval_datetime=now + timedelta(minutes=30),
        rrp=185.0,
        demand_mw=8500.0,
        raw_ref="test-raw-ref-abc",
    )
    fake_pd = {"NSW1": [fake_interval]}

    mock_client = AsyncMock()
    mock_client.fetch_predispatch = AsyncMock(return_value=fake_pd)

    @asynccontextmanager
    async def fake_db_session():
        async with session_factory() as s:
            yield s

    with (
        patch("app.data.aemo_live_client.get_aemo_client", return_value=mock_client),
        patch("app.db.session.db_session", fake_db_session),
    ):
        from app.data.scheduler import _job_predispatch_refresh
        await _job_predispatch_refresh()

    # Verify the row was stored
    async with session_factory() as s:
        from sqlalchemy import text
        result = await s.execute(
            text("SELECT COUNT(*) AS cnt FROM market_events WHERE source='AEMO_PREDISPATCH_30MIN'")
        )
        cnt = result.scalar()
    assert cnt == 1, f"Expected 1 predispatch row, got {cnt}"


@pytest.mark.asyncio
async def test_predispatch_job_is_idempotent(session_factory):
    """Running the job twice must not duplicate rows (merge/upsert)."""
    from app.data.aemo_live_client import PredispatchInterval

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    fake_interval = PredispatchInterval(
        region="NSW1",
        interval_datetime=now + timedelta(minutes=30),
        rrp=185.0,
        demand_mw=8500.0,
        raw_ref="test-raw-ref-idem",
    )
    fake_pd = {"NSW1": [fake_interval]}
    mock_client = AsyncMock()
    mock_client.fetch_predispatch = AsyncMock(return_value=fake_pd)

    @asynccontextmanager
    async def fake_db_session():
        async with session_factory() as s:
            yield s

    with (
        patch("app.data.aemo_live_client.get_aemo_client", return_value=mock_client),
        patch("app.db.session.db_session", fake_db_session),
    ):
        from app.data.scheduler import _job_predispatch_refresh
        await _job_predispatch_refresh()
        await _job_predispatch_refresh()   # second run — same data

    async with session_factory() as s:
        from sqlalchemy import text
        result = await s.execute(
            text("SELECT COUNT(*) AS cnt FROM market_events WHERE source='AEMO_PREDISPATCH_30MIN'")
        )
        cnt = result.scalar()
    assert cnt == 1, f"Idempotency failed: expected 1 row, got {cnt}"


# ── 4. _fetch_history returns predispatch-enriched series ─────────────────────

@pytest.mark.asyncio
async def test_fetch_history_joins_predispatch(session_factory):
    """When PD rows exist in DB, _fetch_history should return aemo_predispatch != price."""
    from app.db.models import MarketEvent
    from app.engines.backtest import _fetch_history

    base = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    dispatch_rows = []
    pd_rows = []

    for i in range(300):
        vt = base + timedelta(minutes=i * 5)
        price = 100.0 + i * 0.1
        row_id = str(uuid.uuid4())
        dispatch_rows.append(MarketEvent(
            id=row_id,
            region="NSW1",
            source="AEMO_DISPATCH_PRICE",
            valid_time=vt,
            system_time=vt,
            raw_ref=f"disp-{row_id[:8]}",
            price_rrp=price,
            demand_mw=8000.0,
            availability_mw=9000.0,
            tenant_id="system",
            data={},
        ))

    # Add one PD row at 00:30 with a distinctly different price
    pd_vt = base + timedelta(minutes=30)
    pd_row_id = hashlib.sha256(f"pd-NSW1-{pd_vt.isoformat()}".encode()).hexdigest()[:36]
    pd_rows.append(MarketEvent(
        id=pd_row_id,
        region="NSW1",
        source="AEMO_PREDISPATCH_30MIN",
        valid_time=pd_vt,
        system_time=pd_vt,
        raw_ref="pd-test-ref",
        price_rrp=999.0,
        demand_mw=0.0,
        availability_mw=0.0,
        tenant_id="system",
        data={"predispatch": True},
    ))

    async with session_factory() as s:
        for r in dispatch_rows + pd_rows:
            s.add(r)
        await s.commit()

    @asynccontextmanager
    async def fake_db_session():
        async with session_factory() as s:
            yield s

    with patch("app.db.session.db_session", fake_db_session):
        series = await _fetch_history("NSW1", lookback_days=30,
                                      end_date=base + timedelta(days=2))

    assert len(series) > 0
    # SQLite strips tzinfo on retrieval — compare naive timestamps
    pd_vt_naive = pd_vt.replace(tzinfo=None)
    matching = [
        r for r in series
        if r["valid_time"].replace(tzinfo=None) == pd_vt_naive
    ]
    assert matching, (
        f"No series row at pd_vt={pd_vt_naive}. "
        f"Available times (first 5): {[r['valid_time'] for r in series[:5]]}"
    )
    assert matching[0]["aemo_predispatch"] == 999.0, (
        f"Expected 999.0 from PD lookup, got {matching[0]['aemo_predispatch']}"
    )
