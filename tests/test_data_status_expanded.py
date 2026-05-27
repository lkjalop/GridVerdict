"""Tests for expanded /api/data/status endpoint.

Verifies that the expanded data status response includes all new keys
added in the last sprint:
  - supplementary: unit_dispatch_events, bid_offers, market_driver_events
  - backfill_cursors: list
  - hippograph: per-region + total node counts
  - lnn_trainers: per-region LNN state
  - scheduler: job health states
  - cache_age_seconds: covers calibration_{region} keys
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]


@pytest.mark.asyncio
async def test_data_status_returns_200(client: AsyncClient):
    r = await client.get("/api/data/status")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_data_status_has_timestamp(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "timestamp" in data
    assert isinstance(data["timestamp"], str)


@pytest.mark.asyncio
async def test_data_status_has_sources(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_data_status_has_supplementary(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "supplementary" in data
    supp = data["supplementary"]
    assert isinstance(supp, dict)
    # All three supplementary tables should be represented
    assert "unit_dispatch_events" in supp
    assert "bid_offers" in supp
    assert "market_driver_events" in supp


@pytest.mark.asyncio
async def test_data_status_bid_offers_has_total(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    bo = data["supplementary"]["bid_offers"]
    # Either has "total" key or an "error" key (table may not exist in test DB)
    assert "total" in bo or "error" in bo


@pytest.mark.asyncio
async def test_data_status_has_backfill_cursors(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "backfill_cursors" in data
    assert isinstance(data["backfill_cursors"], list)


@pytest.mark.asyncio
async def test_data_status_has_hippograph(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "hippograph" in data
    hg = data["hippograph"]
    assert isinstance(hg, dict)
    # Either has region keys or an error key
    assert "total" in hg or "error" in hg


@pytest.mark.asyncio
async def test_data_status_hippograph_has_region_keys_or_error(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    hg = data["hippograph"]
    if "error" not in hg:
        for region in _REGIONS:
            assert region in hg, f"hippograph missing key {region}"


@pytest.mark.asyncio
async def test_data_status_has_lnn_trainers(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "lnn_trainers" in data
    lt = data["lnn_trainers"]
    assert isinstance(lt, dict)


@pytest.mark.asyncio
async def test_data_status_lnn_trainers_has_region_keys_or_error(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    lt = data["lnn_trainers"]
    if "error" not in lt:
        for region in _REGIONS:
            assert region in lt, f"lnn_trainers missing key {region}"


@pytest.mark.asyncio
async def test_data_status_lnn_trainer_entries_have_available(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    lt = data["lnn_trainers"]
    if "error" not in lt:
        for region, val in lt.items():
            assert "available" in val, f"lnn_trainers[{region}] missing 'available' key"


@pytest.mark.asyncio
async def test_data_status_has_scheduler(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "scheduler" in data
    assert isinstance(data["scheduler"], (dict, list))


@pytest.mark.asyncio
async def test_data_status_has_cache_age_seconds(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    assert "cache_age_seconds" in data
    ca = data["cache_age_seconds"]
    assert isinstance(ca, dict)


@pytest.mark.asyncio
async def test_data_status_cache_includes_dispatch_snapshot(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    ca = data["cache_age_seconds"]
    assert "dispatch_snapshot" in ca


@pytest.mark.asyncio
async def test_data_status_cache_includes_calibration_keys(client: AsyncClient):
    r = await client.get("/api/data/status")
    data = r.json()
    ca = data["cache_age_seconds"]
    for region in _REGIONS:
        key = f"calibration_{region}"
        assert key in ca, f"cache_age_seconds missing key {key}"


@pytest.mark.asyncio
async def test_data_status_no_crash_when_tables_empty(client: AsyncClient):
    """SQLite in-memory DB has no rows — response must succeed, not 500."""
    r = await client.get("/api/data/status")
    assert r.status_code == 200
    data = r.json()
    # sources may be empty list or error item — must not be absent
    assert "sources" in data
