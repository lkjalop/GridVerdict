"""Tests for historical price distribution module."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.engines.historical_price import classify_vs_history, get_historical_price_distribution


# ── classify_vs_history ───────────────────────────────────────────────────────

_DIST = {
    "available": True,
    "p25": 50.0,
    "median": 100.0,
    "p75": 150.0,
    "p90": 300.0,
    "mean": 110.0,
    "count": 100,
    "period_label": "last 12 months",
    "hour_window": 2,
    "lookback_days": 365,
}


def test_classify_cheap_at_p25_boundary():
    assert classify_vs_history(50.0, _DIST) == "cheap"


def test_classify_cheap_below_p25():
    assert classify_vs_history(20.0, _DIST) == "cheap"


def test_classify_normal_between_p25_and_median():
    assert classify_vs_history(75.0, _DIST) == "normal"


def test_classify_elevated_between_median_and_p75():
    assert classify_vs_history(125.0, _DIST) == "elevated"


def test_classify_high_between_p75_and_p90():
    assert classify_vs_history(200.0, _DIST) == "high"


def test_classify_spike_above_p90():
    assert classify_vs_history(500.0, _DIST) == "spike"


def test_classify_unknown_when_not_available():
    dist = {**_DIST, "available": False, "median": None}
    assert classify_vs_history(100.0, dist) == "unknown"


def test_classify_unknown_when_missing_median():
    dist = {**_DIST, "median": None}
    assert classify_vs_history(100.0, dist) == "unknown"


# ── get_historical_price_distribution — happy path ───────────────────────────

def _make_db_row(p25=50.0, median=100.0, p75=150.0, p90=300.0, mean=110.0, cnt=80):
    row = MagicMock()
    row.p25 = p25
    row.median = median
    row.p75 = p75
    row.p90 = p90
    row.mean = mean
    row.cnt = cnt
    return row


@pytest.mark.asyncio
async def test_returns_percentiles_when_enough_data():
    db = AsyncMock()
    row = _make_db_row()
    db.execute = AsyncMock(return_value=MagicMock(fetchone=lambda: row))

    anchor = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    result = await get_historical_price_distribution(db, "NSW1", anchor)

    assert result["available"] is True
    assert result["median"] == 100.0
    assert result["p25"] == 50.0
    assert result["p75"] == 150.0
    assert result["p90"] == 300.0
    assert result["count"] == 80
    assert result["period_label"] == "last 12 months"


@pytest.mark.asyncio
async def test_returns_unavailable_when_fewer_than_5_rows():
    db = AsyncMock()
    row = _make_db_row(cnt=3)
    db.execute = AsyncMock(return_value=MagicMock(fetchone=lambda: row))

    anchor = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    result = await get_historical_price_distribution(db, "NSW1", anchor)

    assert result["available"] is False
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_returns_unavailable_when_no_row():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(fetchone=lambda: None))

    anchor = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    result = await get_historical_price_distribution(db, "NSW1", anchor)

    assert result["available"] is False


@pytest.mark.asyncio
async def test_returns_unavailable_on_db_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("DB connection error"))

    anchor = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    result = await get_historical_price_distribution(db, "NSW1", anchor)

    assert result["available"] is False
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_period_label_last_quarter():
    db = AsyncMock()
    row = _make_db_row()
    db.execute = AsyncMock(return_value=MagicMock(fetchone=lambda: row))

    anchor = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    result = await get_historical_price_distribution(db, "NSW1", anchor, period="last_quarter")

    assert result["period_label"] == "last 90 days"
    assert result["lookback_days"] == 90


@pytest.mark.asyncio
async def test_naive_anchor_gets_utc_timezone():
    db = AsyncMock()
    row = _make_db_row()
    db.execute = AsyncMock(return_value=MagicMock(fetchone=lambda: row))

    anchor_naive = datetime(2026, 5, 27, 14, 0)  # no tzinfo
    result = await get_historical_price_distribution(db, "NSW1", anchor_naive)

    assert result["available"] is True
