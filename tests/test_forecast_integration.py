"""Integration tests for GET /api/market/forecast.

Proves that LEAR and QRA appear in the response when enough persisted
market_events rows exist, and that the response structure is correct.

Design notes:
  - Uses a file-based SQLite (via tmp_path) because run_live_forecast()
    calls asyncio.run(_fetch_history(...)) in a thread-pool executor,
    creating a new event loop that cannot share an in-memory SQLite
    connection. A file-based DB allows multiple connections to share data.
  - Patches app.db.session.db_session (module attribute) so both the
    main test event loop and the executor event loop resolve to the same
    file DB.
  - lookback_days=3 with 310 seeded rows ensures the model has more than
    the 295-interval minimum (288 min-train + 6 horizon + 1).
  - GRIDVERDICT_DEV_NO_AUTH=true is already set in conftest.py so no auth
    token is needed.
"""
from __future__ import annotations

import contextlib
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base


# ── Row generator ─────────────────────────────────────────────────────────────

def _make_dispatch_rows(region: str, count: int) -> list[dict[str, Any]]:
    """Generate `count` realistic AEMO_DISPATCH_PRICE rows ending ~now."""
    anchor = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for i in range(count):
        ts = anchor - timedelta(minutes=5 * (count - i))
        # Realistic price variation: daily sine wave + some noise
        hour_of_day = ts.hour + ts.minute / 60.0
        price = 80.0 + 40.0 * math.sin(math.pi * hour_of_day / 12) + 5.0 * ((i * 13) % 7 - 3)
        price = max(price, 30.0)
        rows.append({
            "id": str(uuid.uuid4()),
            "region": region,
            "source": "AEMO_DISPATCH_PRICE",
            "valid_time": ts,
            "system_time": ts,
            "price_rrp": round(price, 2),
            "demand_mw": 7500.0 + 500.0 * math.sin(math.pi * hour_of_day / 12),
            "availability_mw": 9200.0,
            "tenant_id": "system",
            "data": {},
            "raw_ref": f"test-{i}",
        })
    return rows


async def _seed_rows(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import MarketEvent
    for r in rows:
        me = MarketEvent(
            id=r["id"],
            region=r["region"],
            source=r["source"],
            valid_time=r["valid_time"],
            system_time=r["system_time"],
            price_rrp=r["price_rrp"],
            demand_mw=r["demand_mw"],
            availability_mw=r["availability_mw"],
            tenant_id=r["tenant_id"],
            data=r.get("data", {}),
            raw_ref=r.get("raw_ref", "test"),
        )
        session.add(me)
    await session.commit()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_forecast_lear_and_qra_present_when_history_exists(tmp_path, monkeypatch):
    """LEAR and QRA appear in forecasts list when >= 295 intervals are stored."""
    db_file = tmp_path / "forecast_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    # Build the file-based test DB
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as s:
        await _seed_rows(s, _make_dispatch_rows("NSW1", 310))
    await engine.dispose()

    # Patch db_session so both the test event loop and the executor event loop
    # (created by asyncio.run inside _run_sync) connect to the same file DB.
    _engine2 = create_async_engine(db_url, echo=False)
    _factory2 = async_sessionmaker(bind=_engine2, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def _patched_db():
        async with _factory2() as s:
            yield s

    import app.db.session as sess_mod
    monkeypatch.setattr(sess_mod, "db_session", _patched_db)

    from app.api.main import create_app
    from app.db.session import get_db

    app = create_app()

    async def _override_get_db():
        async with _factory2() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/market/forecast?region=NSW1&lookback_days=3")

    await _engine2.dispose()

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:400]}"
    data = r.json()
    assert data.get("available") is True, (
        f"Forecast should be available with 310 rows, got: {data}"
    )

    model_names = {f["model"] for f in data.get("forecasts", [])}
    assert "lear" in model_names, f"LEAR must appear in forecasts, got: {model_names}"
    assert "qra" in model_names, f"QRA must appear in forecasts, got: {model_names}"


@pytest.mark.asyncio
async def test_forecast_unavailable_when_no_history(tmp_path, monkeypatch):
    """Without persisted rows the forecast response reports available=False."""
    db_file = tmp_path / "empty_forecast.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    _engine2 = create_async_engine(db_url, echo=False)
    _factory2 = async_sessionmaker(bind=_engine2, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def _patched_db():
        async with _factory2() as s:
            yield s

    import app.db.session as sess_mod
    monkeypatch.setattr(sess_mod, "db_session", _patched_db)

    from app.api.main import create_app
    from app.db.session import get_db

    app = create_app()

    async def _override_get_db():
        async with _factory2() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/market/forecast?region=NSW1&lookback_days=3")

    await _engine2.dispose()

    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("available") is False, (
        f"Empty DB should yield available=False, got: {data}"
    )


@pytest.mark.asyncio
async def test_forecast_response_structure(tmp_path, monkeypatch):
    """Response schema: region, available, forecasts, primary_model, caveat."""
    db_file = tmp_path / "schema_forecast.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as s:
        await _seed_rows(s, _make_dispatch_rows("NSW1", 310))
    await engine.dispose()

    _engine2 = create_async_engine(db_url, echo=False)
    _factory2 = async_sessionmaker(bind=_engine2, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def _patched_db():
        async with _factory2() as s:
            yield s

    import app.db.session as sess_mod
    monkeypatch.setattr(sess_mod, "db_session", _patched_db)

    from app.api.main import create_app
    from app.db.session import get_db

    app = create_app()

    async def _override_get_db():
        async with _factory2() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/market/forecast?region=NSW1&lookback_days=3&horizon_intervals=6")

    await _engine2.dispose()

    assert r.status_code == 200
    data = r.json()
    assert "region" in data
    assert "available" in data
    assert "caveat" in data

    if data.get("available"):
        assert "forecasts" in data
        assert "primary_model" in data
        assert "horizon_intervals" in data
        assert data["region"] == "NSW1"

        for fc in data["forecasts"]:
            assert "model" in fc
            assert "p10" in fc and "p50" in fc and "p90" in fc, (
                f"Forecast must have p10/p50/p90 bands, got: {list(fc.keys())}"
            )
            assert len(fc["p50"]) > 0, "p50 array must be non-empty"
            assert "target_times" in fc, "Forecast must include target_times"


@pytest.mark.asyncio
async def test_forecast_invalid_region_returns_400(tmp_path):
    """GET /api/market/forecast with an unknown region must return 400."""
    from app.api.main import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/market/forecast?region=INVALID")
    assert r.status_code == 400
