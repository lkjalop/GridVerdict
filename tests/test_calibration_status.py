"""Tests for calibration cache integration in /api/models/status.

Verifies:
  - /api/models/status returns 200
  - calibration key present in response (null or populated)
  - After injecting into the eval registry, calibration.scores is accessible
  - Calibration null comes with a source hint
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient

_FAKE_SCORES = [
    {
        "model": "qra",
        "crps": 18.5,
        "mae": 32.1,
        "pinball_loss": 9.2,
        "spike_recall": 0.71,
        "calibration_error": 0.04,
        "skill_vs_persistence": 0.18,
        "skill_vs_aemo_predispatch": 0.05,
    }
]


@pytest.mark.asyncio
async def test_model_status_returns_200(client: AsyncClient):
    r = await client.get("/api/models/status?region=NSW1")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_model_status_has_calibration_key(client: AsyncClient):
    r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    assert "calibration" in data


@pytest.mark.asyncio
async def test_model_status_calibration_null_when_no_eval(client: AsyncClient):
    """When no backtest has been run, calibration should return a structured null."""
    from unittest.mock import AsyncMock as _AsyncMock
    with patch(
        "app.api.routes_models._get_cached_calibration",
        new=_AsyncMock(return_value={"scores": None, "source": "not_available", "reason": "not run"}),
    ):
        r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    # calibration key present, scores is null
    assert "calibration" in data
    assert data["calibration"]["scores"] is None
    assert data["calibration"]["source"] == "not_available"


@pytest.mark.asyncio
async def test_model_status_calibration_populated_when_eval_exists(client: AsyncClient):
    """After storing a fake eval result, calibration.scores must appear."""
    fake_result = _FAKE_SCORES
    with patch(
        "app.engines.forecasting.evaluation.harness.get_last_eval_result",
        return_value=fake_result,
    ):
        r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    assert data["calibration"] is not None
    assert "scores" in data["calibration"]
    assert data["calibration"]["source"] == "in_process_registry"


@pytest.mark.asyncio
async def test_calibration_scores_structure(client: AsyncClient):
    """scores should be the list returned by the eval harness."""
    with patch(
        "app.engines.forecasting.evaluation.harness.get_last_eval_result",
        return_value=_FAKE_SCORES,
    ):
        r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    scores = data["calibration"]["scores"]
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert scores[0]["model"] == "qra"
    assert scores[0]["crps"] == 18.5


@pytest.mark.asyncio
async def test_model_status_has_models_key(client: AsyncClient):
    r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    assert "models" in data


@pytest.mark.asyncio
async def test_model_status_has_caveat(client: AsyncClient):
    r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    assert "caveat" in data
    assert isinstance(data["caveat"], str)
    assert len(data["caveat"]) > 0


@pytest.mark.asyncio
async def test_model_status_has_region(client: AsyncClient):
    r = await client.get("/api/models/status?region=VIC1")
    data = r.json()
    assert data["region"] == "VIC1"


@pytest.mark.asyncio
async def test_model_status_lnn_key_in_models(client: AsyncClient):
    r = await client.get("/api/models/status?region=NSW1")
    data = r.json()
    assert "lnn" in data["models"]
    lnn = data["models"]["lnn"]
    assert "available" in lnn


@pytest.mark.asyncio
async def test_calibration_registry_injection():
    """Unit test: store_eval_result + get_last_eval_result round-trips correctly."""
    from app.engines.forecasting.evaluation.harness import store_eval_result, get_last_eval_result

    store_eval_result("NSW1", _FAKE_SCORES)
    result = get_last_eval_result("NSW1")
    assert result == _FAKE_SCORES


@pytest.mark.asyncio
async def test_calibration_registry_returns_none_for_unknown_region():
    from app.engines.forecasting.evaluation.harness import get_last_eval_result

    result = get_last_eval_result("UNKNOWN99")
    assert result is None
