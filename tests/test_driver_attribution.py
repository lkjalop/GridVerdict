from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.engines.driver_attribution import rank_driver_rows, summarise_driver_events


def test_summarise_driver_events_finds_binding_constraint_and_tight_interconnector():
    rows = [
        {
            "driver_type": "constraint",
            "element_id": "N^^Q_NIL",
            "values": {"marginal_value": 12.5, "violation_degree": 0.0},
        },
        {
            "driver_type": "interconnector",
            "element_id": "V-S-MNSP1",
            "values": {"mw_flow": 295.0, "export_limit": 300.0},
        },
    ]
    summary = summarise_driver_events(rows)
    assert summary["has_confirmed_driver"]
    assert summary["binding_constraints"][0]["element_id"] == "N^^Q_NIL"
    assert summary["tight_interconnectors"][0]["element_id"] == "V-S-MNSP1"


def test_rank_driver_rows_prioritises_marginal_value():
    rows = [
        {
            "source": "AEMO_DISPATCHCONSTRAINT",
            "driver_type": "constraint",
            "element_id": "LOW",
            "region": "NSW1",
            "valid_time": datetime.now(timezone.utc),
            "values": {"marginal_value": 1.0},
            "raw_ref": "a",
        },
        {
            "source": "AEMO_DISPATCHCONSTRAINT",
            "driver_type": "constraint",
            "element_id": "HIGH",
            "region": "NSW1",
            "valid_time": datetime.now(timezone.utc),
            "values": {"marginal_value": 50.0},
            "raw_ref": "b",
        },
    ]
    ranked = rank_driver_rows("NSW1", rows)
    assert ranked[0]["element_id"] == "HIGH"
