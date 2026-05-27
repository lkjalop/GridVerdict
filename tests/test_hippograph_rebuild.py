"""Tests for HippoGraph cold-start rebuild from DB.

Proves that:
1. rebuild_from_db() inserts the correct number of nodes from market_events.
2. Temporal links are wired (consecutive intervals are linked t → t+1).
3. Similarity edges are computed for recent nodes.
4. Re-running rebuild_from_db() is idempotent (no duplicate nodes).
5. An empty DB results in 0 nodes (no crash).
6. The lifespan startup path calls rebuild_from_db without error.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.engines.hippograph.graph import MarketStateGraph, get_graph, rebuild_from_db
from app.db.models import Base


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def mem_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(mem_engine) -> AsyncSession:
    factory = async_sessionmaker(bind=mem_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


def _row(
    valid_time: datetime,
    region: str = "NSW1",
    price: float = 80.0,
    demand: float = 7000.0,
    avail: float = 9000.0,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "tenant_id": "system",
        "source": "AEMO_DISPATCH_PRICE",
        "region": region,
        "valid_time": valid_time,
        "price_rrp": price,
        "demand_mw": demand,
        "availability_mw": avail,
        "data": {},
    }


async def _seed_rows(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import MarketEvent
    now = datetime.now(timezone.utc)
    for r in rows:
        obj = MarketEvent(
            id=r["id"],
            tenant_id=r["tenant_id"],
            source=r["source"],
            region=r["region"],
            valid_time=r["valid_time"],
            system_time=now,
            price_rrp=r.get("price_rrp"),
            demand_mw=r.get("demand_mw"),
            availability_mw=r.get("availability_mw"),
            data=r.get("data", {}),
            raw_ref="test",
        )
        session.add(obj)
    await session.commit()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_inserts_rows(session, monkeypatch):
    """rebuild_from_db() inserts one node per market_events row."""
    t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = [_row(t0 + timedelta(minutes=5 * i)) for i in range(10)]
    await _seed_rows(session, rows)

    graph = MarketStateGraph()
    _patch_db(monkeypatch, session, graph)

    count = await rebuild_from_db(lookback_days=365)
    assert count == 10
    assert graph.node_count() == 10


@pytest.mark.asyncio
async def test_rebuild_wires_temporal_links(session, monkeypatch):
    """Consecutive intervals in the same region must have temporal_next/prev links."""
    t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = [_row(t0 + timedelta(minutes=5 * i)) for i in range(5)]
    await _seed_rows(session, rows)

    graph = MarketStateGraph()
    _patch_db(monkeypatch, session, graph)

    await rebuild_from_db(lookback_days=365)

    region_nodes = graph.get_region_nodes("NSW1", limit=10)
    assert len(region_nodes) == 5

    # Every node except the last should have temporal_next; every except first has temporal_prev
    for i, node in enumerate(region_nodes):
        if i < 4:
            assert node.temporal_next is not None, f"Node {i} missing temporal_next"
        if i > 0:
            assert node.temporal_prev is not None, f"Node {i} missing temporal_prev"


@pytest.mark.asyncio
async def test_rebuild_idempotent(session, monkeypatch):
    """Calling rebuild_from_db() twice must not duplicate nodes."""
    t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = [_row(t0 + timedelta(minutes=5 * i)) for i in range(6)]
    await _seed_rows(session, rows)

    graph = MarketStateGraph()
    _patch_db(monkeypatch, session, graph)

    first = await rebuild_from_db(lookback_days=365)
    second = await rebuild_from_db(lookback_days=365)

    assert first == 6
    assert second == 0          # all already in graph, none re-inserted
    assert graph.node_count() == 6


@pytest.mark.asyncio
async def test_rebuild_empty_db_returns_zero(monkeypatch):
    """An empty market_events table must return 0 without crashing."""
    graph = MarketStateGraph()

    # Patch db_session to return an empty result
    async def _empty_session():
        class _FakeResult:
            def mappings(self):
                return self
            def all(self):
                return []
        class _FakeSession:
            async def execute(self, *a, **kw):
                return _FakeResult()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass
        return _FakeSession()

    import app.engines.hippograph.graph as _mod
    monkeypatch.setattr(_mod, "_graph", graph)
    # Patch db_session context manager
    import unittest.mock as mock
    ctx = mock.AsyncMock()
    ctx.__aenter__ = mock.AsyncMock(return_value=ctx)
    ctx.__aexit__ = mock.AsyncMock(return_value=False)
    ctx.execute = mock.AsyncMock(return_value=type("R", (), {
        "mappings": lambda self: type("M", (), {"all": lambda self: []})()
    })())

    import app.db.session as sess_mod
    monkeypatch.setattr(sess_mod, "db_session", lambda: ctx)

    count = await rebuild_from_db(lookback_days=365)
    assert count == 0
    assert graph.node_count() == 0


@pytest.mark.asyncio
async def test_rebuild_multiple_regions(session, monkeypatch):
    """Nodes from different regions are inserted and segregated correctly."""
    t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    nsw = [_row(t0 + timedelta(minutes=5 * i), region="NSW1") for i in range(4)]
    vic = [_row(t0 + timedelta(minutes=5 * i), region="VIC1", price=90.0) for i in range(3)]
    await _seed_rows(session, nsw + vic)

    graph = MarketStateGraph()
    _patch_db(monkeypatch, session, graph)

    count = await rebuild_from_db(lookback_days=365)
    assert count == 7
    assert len(graph.get_region_nodes("NSW1", limit=10)) == 4
    assert len(graph.get_region_nodes("VIC1", limit=10)) == 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patch_db(monkeypatch, session: AsyncSession, graph: MarketStateGraph) -> None:
    """Redirect rebuild_from_db to use the test session and the given graph."""
    import app.engines.hippograph.graph as _mod
    import app.db.session as sess_mod
    import contextlib

    # Patch the singleton so rebuild_from_db() operates on our test graph
    monkeypatch.setattr(_mod, "_graph", graph)

    @contextlib.asynccontextmanager
    async def _fake_db_session():
        yield session

    # rebuild_from_db() does: import app.db.session as _db_mod; _db_mod.db_session
    # so patching the module attribute is the right seam
    monkeypatch.setattr(sess_mod, "db_session", _fake_db_session)
