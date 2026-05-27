"""Sprint F: BESS fleet multi-asset coordination tests.

Tests cover:
  - FleetCoordinator: single asset produces a valid FleetDispatchPlan
  - FleetCoordinator: multiple assets — totals are sum of dispatching assets only
  - FleetCoordinator: fleet_export_limit_mw caps total dispatch MW
  - FleetCoordinator: export limit allocation by net value rank (highest first)
  - FleetCoordinator: assets held when limit exhausted get HOLD action
  - FleetCoordinator: non-dispatching assets (HOLD, CHARGE, AVOID) excluded from totals
  - FleetCoordinator: fleet_confidence is weakest among dispatching assets
  - FleetCoordinator: empty dispatching assets → insufficient_data confidence
  - Schema: FleetScenarioRequest rejects empty assets list
  - Route: POST /portfolio/bess/fleet/scenario returns 200 with valid plan
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.portfolio.schema import (
    BessPosition,
    ContractType,
    DispatchAction,
    FleetAsset,
    FleetScenarioRequest,
    MarketSnapshot,
)
from app.portfolio.fleet_coordinator import evaluate_fleet, _weakest_confidence


# ── Helpers ────────────────────────────────────────────────────────────────────

def _position(
    capacity_mwh: float = 2.0,
    soc_pct: float = 80.0,
    max_discharge_mw: float = 1.0,
    max_charge_mw: float = 1.0,
    degradation: float = 5.0,
    min_reserve: float = 10.0,
    fcas: bool = False,
) -> BessPosition:
    return BessPosition(
        capacity_mwh=capacity_mwh,
        soc_pct=soc_pct,
        max_discharge_mw=max_discharge_mw,
        max_charge_mw=max_charge_mw,
        efficiency_pct=90.0,
        degradation_cost_per_mwh=degradation,
        min_reserve_soc_pct=min_reserve,
        fcas_enabled=fcas,
    )


def _market(
    price_rrp: float = 500.0,
    regime: str = "spike",
    evidence: str = "supported",
) -> MarketSnapshot:
    return MarketSnapshot(
        region="NSW1",
        price_rrp=price_rrp,
        price_regime=regime,
        evidence_quality=evidence,
        forecast_direction="stable",
    )


def _asset(
    asset_id: str = "BESS_01",
    **pos_kwargs,
) -> FleetAsset:
    return FleetAsset(asset_id=asset_id, position=_position(**pos_kwargs))


def _request(
    assets: list[FleetAsset],
    market: MarketSnapshot | None = None,
    limit_mw: float | None = None,
) -> FleetScenarioRequest:
    return FleetScenarioRequest(
        assets=assets,
        market=market or _market(),
        fleet_export_limit_mw=limit_mw,
    )


# ── evaluate_fleet: basic correctness ─────────────────────────────────────────

class TestEvaluateFleetBasic:
    def test_single_asset_returns_plan(self):
        plan = evaluate_fleet(_request([_asset("A1")]))
        assert plan.market_region == "NSW1"
        assert len(plan.assets) == 1
        assert plan.assets[0].asset_id == "A1"

    def test_total_dispatch_is_sum_of_dispatching_assets(self):
        # Two spike-price assets with plenty of SOC — both should dispatch
        assets = [_asset("A1", max_discharge_mw=2.0), _asset("A2", max_discharge_mw=3.0)]
        plan = evaluate_fleet(_request(assets))
        dispatching = [a for a in plan.assets if a.action in (
            DispatchAction.DISPATCH_FULL, DispatchAction.DISPATCH_PARTIAL
        )]
        expected_mw = sum(a.dispatch_mw for a in dispatching)
        assert plan.total_dispatch_mw == pytest.approx(expected_mw, abs=0.01)

    def test_total_net_value_excludes_non_dispatching(self):
        # One spike asset (dispatches) + one low-SOC asset (holds)
        spike_asset = _asset("SPIKE", soc_pct=80.0, max_discharge_mw=1.0)
        depleted = _asset("EMPTY", soc_pct=5.0, min_reserve=10.0)  # SOC below reserve
        plan = evaluate_fleet(_request([spike_asset, depleted]))

        dispatching_net = sum(
            a.net_value for a in plan.assets
            if a.action in (DispatchAction.DISPATCH_FULL, DispatchAction.DISPATCH_PARTIAL)
        )
        assert plan.total_net_value == pytest.approx(dispatching_net, abs=0.01)

    def test_no_limit_applies_all_assets_at_full_dispatch_mw(self):
        assets = [_asset("A", max_discharge_mw=5.0), _asset("B", max_discharge_mw=3.0)]
        plan = evaluate_fleet(_request(assets))
        assert not plan.fleet_export_limit_applied

    def test_plan_has_caveat(self):
        plan = evaluate_fleet(_request([_asset()]))
        assert "simulation" in plan.caveat.lower()

    def test_plan_stores_market_fields(self):
        m = _market(price_rrp=250.0, regime="elevated")
        plan = evaluate_fleet(_request([_asset()], market=m))
        assert plan.market_price_rrp == 250.0
        assert plan.market_regime == "elevated"


# ── evaluate_fleet: export cap ────────────────────────────────────────────────

class TestFleetExportCap:
    def test_total_dispatch_does_not_exceed_limit(self):
        # Three 2 MW assets, limit 3 MW
        assets = [_asset(f"A{i}", max_discharge_mw=2.0) for i in range(3)]
        plan = evaluate_fleet(_request(assets, limit_mw=3.0))
        assert plan.total_dispatch_mw <= 3.01  # allow fp tolerance
        assert plan.fleet_export_limit_applied

    def test_highest_net_value_asset_dispatched_first(self):
        """When limit forces a choice, highest net-value asset gets priority."""
        # Asset A: high degradation cost (low net value)
        # Asset B: low degradation cost (high net value)
        # Asset B should get MW first
        a_lo = FleetAsset(asset_id="A_lo", position=_position(
            degradation=50.0, max_discharge_mw=2.0, soc_pct=80.0
        ))
        b_hi = FleetAsset(asset_id="B_hi", position=_position(
            degradation=2.0, max_discharge_mw=2.0, soc_pct=80.0
        ))
        plan = evaluate_fleet(_request([a_lo, b_hi], limit_mw=2.0))

        b_result = next(a for a in plan.assets if a.asset_id == "B_hi")
        a_result = next(a for a in plan.assets if a.asset_id == "A_lo")

        # B (higher net value) should be dispatching; A held or reduced
        assert b_result.action in (
            DispatchAction.DISPATCH_FULL, DispatchAction.DISPATCH_PARTIAL
        )
        assert b_result.dispatch_mw > 0

    def test_assets_held_when_limit_exhausted(self):
        # Limit 1 MW, two 2 MW assets — at least one must be held to respect cap
        assets = [_asset(f"A{i}", max_discharge_mw=2.0) for i in range(2)]
        plan = evaluate_fleet(_request(assets, limit_mw=1.0))
        held = [a for a in plan.assets if a.action == DispatchAction.HOLD]
        assert len(held) >= 1

    def test_held_by_limit_asset_has_zero_dispatch_mw(self):
        assets = [_asset("A", max_discharge_mw=5.0), _asset("B", max_discharge_mw=5.0)]
        plan = evaluate_fleet(_request(assets, limit_mw=3.0))
        # Second asset (if held) should have dispatch_mw = 0
        for a in plan.assets:
            if a.action == DispatchAction.HOLD and "exhausted" in " ".join(a.why).lower():
                assert a.dispatch_mw == 0.0

    def test_limit_larger_than_total_capacity_unchanged(self):
        # Limit 100 MW >> total asset MW → no cap effect
        assets = [_asset(f"A{i}", max_discharge_mw=2.0) for i in range(3)]
        plan_no_limit = evaluate_fleet(_request(assets))
        plan_with_limit = evaluate_fleet(_request(assets, limit_mw=100.0))
        # Total dispatch should be the same (all assets unaffected)
        assert plan_no_limit.total_dispatch_mw == pytest.approx(
            plan_with_limit.total_dispatch_mw, abs=0.01
        )


# ── fleet_confidence ──────────────────────────────────────────────────────────

class TestFleetConfidence:
    def test_all_supported_gives_supported(self):
        assert _weakest_confidence(["supported", "supported"]) == "supported"

    def test_one_low_confidence_downgrades(self):
        assert _weakest_confidence(["supported", "low_confidence"]) == "low_confidence"

    def test_one_insufficient_data_downgrades_to_worst(self):
        assert _weakest_confidence(["supported", "insufficient_data"]) == "insufficient_data"

    def test_empty_gives_insufficient_data(self):
        assert _weakest_confidence([]) == "insufficient_data"

    def test_fleet_confidence_matches_weakest(self):
        # Mix: one "supported", one dispatching with insufficient_data evidence
        confirmed_asset = _asset("A", max_discharge_mw=1.0)
        uncertain_market = _market(price_rrp=600.0, regime="spike", evidence="insufficient")
        # Both assets share same market — both get insufficient_data confidence
        plan = evaluate_fleet(FleetScenarioRequest(
            assets=[confirmed_asset, _asset("B", max_discharge_mw=1.0)],
            market=uncertain_market,
        ))
        dispatching = [a for a in plan.assets if a.action in (
            DispatchAction.DISPATCH_FULL, DispatchAction.DISPATCH_PARTIAL
        )]
        if dispatching:
            # All dispatching assets have the same insufficient evidence → worst
            assert plan.fleet_confidence in ("insufficient_data", "low_confidence")


# ── Schema validation ──────────────────────────────────────────────────────────

class TestFleetSchema:
    def test_empty_assets_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FleetScenarioRequest(assets=[], market=_market())

    def test_negative_fleet_limit_raises_validation_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FleetScenarioRequest(
                assets=[_asset()], market=_market(), fleet_export_limit_mw=-1.0
            )

    def test_asset_id_is_required(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FleetAsset(position=_position())

    def test_plan_fields_present(self):
        plan = evaluate_fleet(_request([_asset()]))
        assert hasattr(plan, "total_dispatch_mw")
        assert hasattr(plan, "total_expected_revenue")
        assert hasattr(plan, "total_degradation_cost")
        assert hasattr(plan, "total_net_value")
        assert hasattr(plan, "fleet_confidence")
        assert hasattr(plan, "fleet_export_limit_applied")


# ── API route ─────────────────────────────────────────────────────────────────

def _fleet_payload(limit_mw: float | None = None) -> dict:
    pos = {
        "capacity_mwh": 2.0,
        "soc_pct": 75.0,
        "max_discharge_mw": 1.0,
        "max_charge_mw": 1.0,
        "efficiency_pct": 90.0,
        "degradation_cost_per_mwh": 5.0,
        "min_reserve_soc_pct": 10.0,
    }
    market = {
        "region": "NSW1",
        "price_rrp": 450.0,
        "price_regime": "spike",
        "evidence_quality": "supported",
    }
    body: dict = {
        "assets": [
            {"asset_id": "BESS_01", "position": pos},
            {"asset_id": "BESS_02", "position": {**pos, "soc_pct": 60.0}},
        ],
        "market": market,
    }
    if limit_mw is not None:
        body["fleet_export_limit_mw"] = limit_mw
    return body


@pytest.mark.asyncio
async def test_fleet_route_returns_200(client):
    resp = await client.post("/api/portfolio/bess/fleet/scenario",
                             json=_fleet_payload())
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fleet_route_response_has_expected_keys(client):
    resp = await client.post("/api/portfolio/bess/fleet/scenario",
                             json=_fleet_payload())
    data = resp.json()
    for key in ("market_region", "total_dispatch_mw", "total_net_value",
                "fleet_confidence", "assets", "caveat"):
        assert key in data, f"missing key: {key}"


@pytest.mark.asyncio
async def test_fleet_route_with_export_limit(client):
    resp = await client.post("/api/portfolio/bess/fleet/scenario",
                             json=_fleet_payload(limit_mw=0.8))
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_dispatch_mw"] <= 0.81
    assert data["fleet_export_limit_applied"] is True


@pytest.mark.asyncio
async def test_fleet_route_empty_assets_returns_422(client):
    payload = _fleet_payload()
    payload["assets"] = []
    resp = await client.post("/api/portfolio/bess/fleet/scenario", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_fleet_assets_count_matches_input(client):
    resp = await client.post("/api/portfolio/bess/fleet/scenario",
                             json=_fleet_payload())
    data = resp.json()
    assert len(data["assets"]) == 2
