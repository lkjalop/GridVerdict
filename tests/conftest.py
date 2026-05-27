"""Shared pytest fixtures for GridVerdict tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Use an in-memory SQLite for unit tests (async compatible via aiosqlite)
# For integration tests that need PostGIS, override DATABASE_URL in env.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GRIDVERDICT_DEV_NO_AUTH", "true")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")
os.environ.setdefault("DECOMPOSER_BACKEND", "rule_based")

FIXTURES = Path(__file__).parent / "fixtures"


# ── DB fixtures ───────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    from app.db.models import Base
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    SessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
        await session.rollback()


# ── HTTP client ───────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    from app.api.main import create_app
    from app.db.session import get_db
    from sqlalchemy.ext.asyncio import AsyncSession

    app = create_app()

    # Override DB dependency to use test engine
    SessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    async def override_get_db():
        async with SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Fixture loaders ───────────────────────────────────────────────────

def load_fixture(path: str) -> dict | list:
    return json.loads((FIXTURES / path).read_text())


def sample_answer(name: str) -> dict:
    return load_fixture(f"sample_answers/{name}.json")


def sample_market(name: str) -> dict:
    return load_fixture(f"sample_market_states/{name}.json")


def injection_payloads() -> list[dict]:
    return load_fixture("injection_payloads/payloads.json")


@pytest.fixture(name="load_fixture")
def load_fixture_fixture():
    """Pytest fixture wrapper — returns the load_fixture callable."""
    return load_fixture


# ── Shared market snapshot mock ───────────────────────────────────────

class MockDispatchPrice:
    def __init__(self, region="NSW1", price=80.0, demand=7500.0, avail=9000.0):
        self.region = region
        self.price_rrp = price
        self.demand_mw = demand
        self.availability_mw = avail
        self.valid_time = datetime.now(timezone.utc)
        self.system_time = datetime.now(timezone.utc)
        self.raw_ref = "test-fixture-ref"
