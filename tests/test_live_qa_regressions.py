"""Regression tests from live browser QA findings."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.api.routes_market import _db_datetime_param
from app.engines.incident_brief import _forecast_state, _model_provenance


def test_db_datetime_param_uses_sqlite_compatible_utc_naive_string():
    dt = datetime(2026, 5, 26, 11, 25, 0, tzinfo=timezone.utc)
    with patch("config.settings.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "sqlite+aiosqlite:///:memory:"
        assert _db_datetime_param(dt) == "2026-05-26 11:25:00"


def test_db_datetime_param_returns_datetime_for_postgres():
    dt = datetime(2026, 5, 26, 11, 25, 0, tzinfo=timezone.utc)
    with patch("config.settings.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "postgresql+asyncpg://gv:pw@localhost/gridverdict"
        value = _db_datetime_param(dt)

    assert isinstance(value, datetime)
    assert value.tzinfo is None
    assert value == datetime(2026, 5, 26, 11, 25, 0)


@pytest.mark.asyncio
async def test_incident_brief_forecast_empty_models_is_unavailable():
    async def fake_call_tool(*args, **kwargs):
        return {"models": [], "intervals": [], "generated_at": None}

    with patch("app.mcp.router.call_tool", fake_call_tool):
        state = await _forecast_state("NSW1")

    assert state["available"] is False
    assert state["models"] == []
    assert state["intervals"] == []
    assert "no model intervals" in state["note"].lower()


def test_model_provenance_accepts_model_registry_model_name_key():
    with patch(
        "app.engines.forecasting.model_registry.get_all_models",
        return_value=[{
            "model_name": "LEAR",
            "version": "1.0.0",
            "training_data_ref": "NSW1:window:n=290:sha=abc",
            "registered_at": "2026-05-26T00:00:00+00:00",
        }],
    ):
        prov = _model_provenance()

    assert prov["models"][0]["name"] == "LEAR"
