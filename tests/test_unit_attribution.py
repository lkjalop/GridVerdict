from __future__ import annotations

import pytest

from app.engines.unit_attribution import summarise_unit_dispatch


def test_summarise_unit_dispatch_groups_by_fuel_and_caveats():
    summary = summarise_unit_dispatch([
        {
            "duid": "ERARING1",
            "fuel_type": "coal",
            "initial_mw": 450.0,
            "total_cleared_mw": 510.0,
            "availability_mw": 650.0,
            "raw_ref": "coal-ref",
        },
        {
            "duid": "POAT220",
            "fuel_type": "hydro",
            "initial_mw": 90.0,
            "total_cleared_mw": 140.0,
            "availability_mw": 200.0,
            "raw_ref": "hydro-ref",
        },
        {
            "duid": "WIND1",
            "fuel_type": "wind",
            "initial_mw": 20.0,
            "total_cleared_mw": 10.0,
            "availability_mw": 60.0,
            "semi_dispatch_cap": 1.0,
            "raw_ref": "wind-ref",
        },
    ])

    assert summary["has_unit_evidence"] is True
    assert summary["by_fuel"]["coal"]["delta_mw"] == pytest.approx(60.0)
    assert summary["by_fuel"]["hydro"]["total_cleared_mw"] == pytest.approx(140.0)
    assert summary["by_fuel"]["wind"]["semi_dispatch_cap_count"] == 1
    assert "hydro_water_storage" in summary["caveats"]
    assert "coal_outage_commitment" in summary["caveats"]
    assert "renewable_forecast_actual" in summary["caveats"]


def test_summarise_unit_dispatch_missing_unit_evidence():
    summary = summarise_unit_dispatch([])
    assert summary["has_unit_evidence"] is False
    assert summary["caveats"] == ["unit_dispatch_events"]
