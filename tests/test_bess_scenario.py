"""Tests for BESS Portfolio Scenario Engine.

Covers:
  - BessPosition schema validation
  - MarketSnapshot schema defaults
  - BessEconomics calculations (compute_economics)
  - dispatch_policy: each action branch
  - Missing-before-action checklist
  - Confidence mapping from evidence quality
  - HTTP route: POST /api/portfolio/bess/scenario
  - HTTP route: GET /api/portfolio/bess/market-prefill
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.portfolio.schema import (
    BessPosition,
    BessScenarioRequest,
    ContractType,
    DispatchAction,
    MarketSnapshot,
)
from app.portfolio.bess_engine import (
    compute_economics,
    compute_charge_cost,
    fcas_market_is_tight,
    headroom_is_compressed,
)
from app.portfolio.dispatch_policy import evaluate, _confidence_from_evidence, _missing_before_action


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _position(
    capacity_mwh=100.0,
    soc_pct=62.0,
    max_discharge_mw=50.0,
    max_charge_mw=25.0,
    efficiency_pct=88.0,
    degradation_cost_per_mwh=8.0,
    min_reserve_soc_pct=20.0,
    contract_type=ContractType.MERCHANT,
    fcas_enabled=False,
    risk_limit_dollar=None,
    site_export_limit_mw=None,
) -> BessPosition:
    return BessPosition(
        capacity_mwh=capacity_mwh,
        soc_pct=soc_pct,
        max_discharge_mw=max_discharge_mw,
        max_charge_mw=max_charge_mw,
        efficiency_pct=efficiency_pct,
        degradation_cost_per_mwh=degradation_cost_per_mwh,
        min_reserve_soc_pct=min_reserve_soc_pct,
        contract_type=contract_type,
        fcas_enabled=fcas_enabled,
        risk_limit_dollar=risk_limit_dollar,
        site_export_limit_mw=site_export_limit_mw,
    )


def _market(
    region="NSW1",
    price_rrp=500.0,
    price_regime="spike",
    headroom_mw=400.0,
    forecast_direction="stable",
    fcas_raise_6sec_rrp=None,
    fcas_raise_reg_rrp=None,
    rebid_evidence_tier=None,
    outage_evidence_tier=None,
    evidence_quality="supported",
) -> MarketSnapshot:
    return MarketSnapshot(
        region=region,
        price_rrp=price_rrp,
        price_regime=price_regime,
        headroom_mw=headroom_mw,
        forecast_direction=forecast_direction,
        fcas_raise_6sec_rrp=fcas_raise_6sec_rrp,
        fcas_raise_reg_rrp=fcas_raise_reg_rrp,
        rebid_evidence_tier=rebid_evidence_tier,
        outage_evidence_tier=outage_evidence_tier,
        evidence_quality=evidence_quality,
    )


# ── BessPosition validation ───────────────────────────────────────────────────

def test_position_valid():
    p = _position()
    assert p.capacity_mwh == 100.0
    assert p.soc_pct == 62.0


def test_position_rejects_zero_capacity():
    with pytest.raises(Exception):
        BessPosition(
            capacity_mwh=0,
            soc_pct=50,
            max_discharge_mw=50,
            max_charge_mw=25,
            efficiency_pct=88,
            degradation_cost_per_mwh=8,
            min_reserve_soc_pct=20,
        )


def test_position_rejects_soc_over_100():
    with pytest.raises(Exception):
        _position(soc_pct=101.0)


def test_position_rejects_min_reserve_at_100():
    with pytest.raises(Exception):
        _position(min_reserve_soc_pct=100.0)


def test_position_site_export_limit_optional():
    p = _position(site_export_limit_mw=None)
    assert p.site_export_limit_mw is None


# ── compute_economics ─────────────────────────────────────────────────────────

def test_economics_available_energy():
    # soc=62, reserve=20 → usable=42% of 100MWh = 42 MWh
    p = _position(soc_pct=62, min_reserve_soc_pct=20, capacity_mwh=100)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.available_energy_mwh == pytest.approx(42.0)


def test_economics_dispatch_mw_capped_by_power_rating():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50, capacity_mwh=1000)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.dispatch_mw == pytest.approx(50.0)


def test_economics_dispatch_mw_capped_by_energy():
    # Only 1 MWh available; max_discharge=50 MW; interval=5min=0.0833h
    # At 50MW for 5min → 4.167 MWh needed, but only 1 MWh available
    # → dispatch_mw = 1 MWh / 0.0833h = 12 MW
    p = _position(soc_pct=21, min_reserve_soc_pct=20, capacity_mwh=100, max_discharge_mw=50)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    # available = 1% * 100 = 1 MWh
    assert e.available_energy_mwh == pytest.approx(1.0)
    assert e.dispatch_mw < 50.0  # energy-constrained


def test_economics_revenue():
    # 50 MW × 5/60 h × $500/MWh = $208.33
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50, capacity_mwh=1000)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.expected_revenue == pytest.approx(50 * (5/60) * 500, rel=1e-3)


def test_economics_degradation_cost():
    # 50 MW × 5/60 h × $8/MWh = $3.33
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50,
                  capacity_mwh=1000, degradation_cost_per_mwh=8)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.degradation_cost == pytest.approx(50 * (5/60) * 8, rel=1e-3)


def test_economics_net_value():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50,
                  capacity_mwh=1000, degradation_cost_per_mwh=8)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.net_expected_value == pytest.approx(e.expected_revenue - e.degradation_cost, rel=1e-3)


def test_economics_site_export_limit_caps_dispatch():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50,
                  capacity_mwh=1000, site_export_limit_mw=30)
    m = _market(price_rrp=500)
    e = compute_economics(p, m)
    assert e.dispatch_mw == pytest.approx(30.0)


def test_economics_fcas_opportunity_value_zero_when_disabled():
    p = _position(fcas_enabled=False)
    m = _market(fcas_raise_6sec_rrp=200.0)
    e = compute_economics(p, m)
    assert e.fcas_opportunity_value == pytest.approx(0.0)


def test_economics_fcas_opportunity_value_when_enabled():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50,
                  capacity_mwh=1000, fcas_enabled=True)
    m = _market(fcas_raise_6sec_rrp=200.0)
    e = compute_economics(p, m)
    assert e.fcas_opportunity_value > 0


def test_economics_zero_available_energy():
    # soc == min_reserve → available = 0
    p = _position(soc_pct=20, min_reserve_soc_pct=20)
    m = _market()
    e = compute_economics(p, m)
    assert e.available_energy_mwh == pytest.approx(0.0)
    assert e.dispatch_mw == pytest.approx(0.0)


def test_compute_charge_cost():
    p = _position(max_charge_mw=25, efficiency_pct=88)
    m = _market(price_rrp=100)
    cost = compute_charge_cost(p, m)
    # 25 MW × 5/60 h × $100/MWh
    assert cost == pytest.approx(25 * (5/60) * 100, rel=1e-3)


def test_headroom_compressed_below_500():
    m = _market(headroom_mw=400)
    assert headroom_is_compressed(m) is True


def test_headroom_not_compressed_above_500():
    m = _market(headroom_mw=600)
    assert headroom_is_compressed(m) is False


def test_fcas_tight_above_threshold():
    m = _market(fcas_raise_6sec_rrp=150.0)
    assert fcas_market_is_tight(m) is True


def test_fcas_not_tight_below_threshold():
    m = _market(fcas_raise_6sec_rrp=50.0)
    assert fcas_market_is_tight(m) is False


# ── dispatch_policy ───────────────────────────────────────────────────────────

def test_policy_avoid_when_evidence_insufficient():
    p = _position()
    m = _market(price_regime="normal", evidence_quality="insufficient")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.AVOID_INSUFFICIENT_DATA
    assert result.confidence == "insufficient_data"


def test_policy_extreme_price_still_dispatches_with_insufficient_evidence():
    # Rule 0 only fires when NOT extreme/spike
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000)
    m = _market(price_regime="extreme", price_rrp=14500, evidence_quality="insufficient")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.DISPATCH_FULL


def test_policy_hold_when_empty():
    p = _position(soc_pct=20, min_reserve_soc_pct=20)  # available=0
    m = _market(price_regime="spike", evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.HOLD


def test_policy_reserve_fcas_when_fcas_price_high():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000, fcas_enabled=True)
    m = _market(price_regime="elevated", fcas_raise_6sec_rrp=250.0, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.RESERVE_FCAS


def test_policy_reserve_fcas_requires_fcas_enabled():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000, fcas_enabled=False)
    m = _market(price_regime="elevated", fcas_raise_6sec_rrp=250.0, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action != DispatchAction.RESERVE_FCAS


def test_policy_dispatch_full_on_extreme():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50, capacity_mwh=1000)
    m = _market(price_regime="extreme", price_rrp=14500, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.DISPATCH_FULL


def test_policy_dispatch_full_on_spike():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, max_discharge_mw=50, capacity_mwh=1000)
    m = _market(price_regime="spike", price_rrp=500, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.DISPATCH_FULL


def test_policy_dispatch_partial_on_elevated():
    p = _position(soc_pct=80, min_reserve_soc_pct=20, max_discharge_mw=50, capacity_mwh=100)
    m = _market(price_regime="elevated", price_rrp=350, forecast_direction="stable",
                evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.DISPATCH_PARTIAL


def test_policy_hold_when_forecast_rising_and_elevated():
    p = _position(soc_pct=80, min_reserve_soc_pct=20, max_discharge_mw=50, capacity_mwh=100)
    m = _market(price_regime="elevated", price_rrp=350, forecast_direction="rising",
                evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    # Rising forecast → hold (rule 4 skips elevated+rising)
    assert result.action == DispatchAction.HOLD


def test_policy_charge_when_low_soc_and_normal_price():
    p = _position(soc_pct=40, min_reserve_soc_pct=20, max_discharge_mw=50, capacity_mwh=100)
    m = _market(price_regime="normal", price_rrp=50, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.CHARGE


def test_policy_hold_when_normal_price_and_high_soc():
    p = _position(soc_pct=90, min_reserve_soc_pct=20)
    m = _market(price_regime="normal", price_rrp=50, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    # No charge (SOC > LOW_SOC_CHARGE_THRESHOLD=60 is actually False here since 90>60)
    # Actually 90 >= 60 → charge condition fails → hold
    assert result.action == DispatchAction.HOLD


def test_policy_hold_when_net_value_negative():
    # degradation_cost_per_mwh=500, price=50 → net is very negative
    p = _position(soc_pct=80, degradation_cost_per_mwh=500)
    m = _market(price_regime="elevated", price_rrp=50, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert result.action == DispatchAction.HOLD


# ── Confidence mapping ────────────────────────────────────────────────────────

def test_confidence_confirmed():
    assert _confidence_from_evidence("confirmed") == "supported"


def test_confidence_supported():
    assert _confidence_from_evidence("supported") == "supported"


def test_confidence_plausible():
    assert _confidence_from_evidence("plausible") == "low_confidence"


def test_confidence_insufficient():
    assert _confidence_from_evidence("insufficient") == "insufficient_data"


# ── Missing before action ─────────────────────────────────────────────────────

def test_missing_includes_soc_telemetry_always():
    p = _position()
    m = _market()
    items = _missing_before_action(p, m)
    assert any("SOC telemetry" in i for i in items)


def test_missing_export_limit_when_none():
    p = _position(site_export_limit_mw=None)
    m = _market()
    items = _missing_before_action(p, m)
    assert any("export limit" in i for i in items)


def test_missing_no_export_limit_when_provided():
    p = _position(site_export_limit_mw=50.0)
    m = _market()
    items = _missing_before_action(p, m)
    assert not any("export limit" in i for i in items)


def test_missing_fcas_enablement_when_fcas_enabled():
    p = _position(fcas_enabled=True)
    m = _market()
    items = _missing_before_action(p, m)
    assert any("FCAS enablement" in i for i in items)


def test_missing_no_fcas_when_disabled():
    p = _position(fcas_enabled=False)
    m = _market()
    items = _missing_before_action(p, m)
    assert not any("FCAS enablement" in i for i in items)


def test_missing_forecast_when_direction_none():
    p = _position()
    m = _market(forecast_direction=None)
    items = _missing_before_action(p, m)
    assert any("forecast" in i for i in items)


def test_missing_outage_when_tier_none():
    p = _position()
    m = _market(outage_evidence_tier=None)
    items = _missing_before_action(p, m)
    assert any("outage" in i for i in items)


def test_missing_contract_obligations_for_ppa():
    p = _position(contract_type=ContractType.PPA)
    m = _market()
    items = _missing_before_action(p, m)
    assert any("contract obligations" in i for i in items)


def test_no_contract_obligations_for_merchant():
    p = _position(contract_type=ContractType.MERCHANT)
    m = _market()
    items = _missing_before_action(p, m)
    assert not any("contract obligations" in i for i in items)


# ── Why list content ──────────────────────────────────────────────────────────

def test_why_includes_price_line():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000)
    m = _market(price_rrp=500, price_regime="spike")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert any("500.00" in w or "price" in w.lower() for w in result.why)


def test_why_includes_economics_line():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000)
    m = _market()
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert any("net" in w.lower() or "revenue" in w.lower() for w in result.why)


def test_why_includes_headroom():
    p = _position(soc_pct=100, min_reserve_soc_pct=0, capacity_mwh=1000)
    m = _market(headroom_mw=300)
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert any("headroom" in w.lower() for w in result.why)


# ── Risk flags ────────────────────────────────────────────────────────────────

def test_risk_flag_negative_net_value():
    p = _position(soc_pct=80, degradation_cost_per_mwh=500, risk_limit_dollar=0)
    m = _market(price_rrp=50, evidence_quality="supported")
    e = compute_economics(p, m)
    result = evaluate(p, m, e)
    assert any("negative" in f.lower() or "risk limit" in f.lower() for f in result.risk_flags)


# ── HTTP routes ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bess_scenario_route_returns_200(client: AsyncClient):
    payload = {
        "position": {
            "capacity_mwh": 100,
            "soc_pct": 62,
            "max_discharge_mw": 50,
            "max_charge_mw": 25,
            "efficiency_pct": 88,
            "degradation_cost_per_mwh": 8,
            "min_reserve_soc_pct": 20,
            "contract_type": "merchant",
            "fcas_enabled": False,
        },
        "market": {
            "region": "NSW1",
            "price_rrp": 500,
            "price_regime": "spike",
            "evidence_quality": "supported",
        },
    }
    r = await client.post("/api/portfolio/bess/scenario", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert "economics" in data
    assert "why" in data
    assert "missing_before_action" in data


@pytest.mark.asyncio
async def test_bess_scenario_dispatch_full(client: AsyncClient):
    payload = {
        "position": {
            "capacity_mwh": 1000,
            "soc_pct": 100,
            "max_discharge_mw": 50,
            "max_charge_mw": 25,
            "efficiency_pct": 88,
            "degradation_cost_per_mwh": 8,
            "min_reserve_soc_pct": 0,
        },
        "market": {
            "region": "NSW1",
            "price_rrp": 14500,
            "price_regime": "extreme",
            "evidence_quality": "supported",
        },
    }
    r = await client.post("/api/portfolio/bess/scenario", json=payload)
    assert r.status_code == 200
    assert r.json()["action"] == "dispatch_full"


@pytest.mark.asyncio
async def test_bess_scenario_avoid_insufficient_evidence(client: AsyncClient):
    payload = {
        "position": {
            "capacity_mwh": 100,
            "soc_pct": 62,
            "max_discharge_mw": 50,
            "max_charge_mw": 25,
            "efficiency_pct": 88,
            "degradation_cost_per_mwh": 8,
            "min_reserve_soc_pct": 20,
        },
        "market": {
            "region": "NSW1",
            "price_rrp": 150,
            "price_regime": "elevated",
            "evidence_quality": "insufficient",
        },
    }
    r = await client.post("/api/portfolio/bess/scenario", json=payload)
    assert r.status_code == 200
    assert r.json()["action"] == "avoid_insufficient_data"


@pytest.mark.asyncio
async def test_bess_scenario_validation_error_on_bad_soc(client: AsyncClient):
    payload = {
        "position": {
            "capacity_mwh": 100,
            "soc_pct": 150,           # invalid
            "max_discharge_mw": 50,
            "max_charge_mw": 25,
            "efficiency_pct": 88,
            "degradation_cost_per_mwh": 8,
            "min_reserve_soc_pct": 20,
        },
        "market": {
            "region": "NSW1",
            "price_rrp": 500,
            "price_regime": "spike",
            "evidence_quality": "supported",
        },
    }
    r = await client.post("/api/portfolio/bess/scenario", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_bess_market_prefill_valid_region(client: AsyncClient):
    r = await client.get("/api/portfolio/bess/market-prefill?region=NSW1")
    assert r.status_code == 200
    data = r.json()
    assert data["region"] == "NSW1"
    assert "price_rrp" in data
    assert "evidence_quality" in data


@pytest.mark.asyncio
async def test_bess_market_prefill_invalid_region(client: AsyncClient):
    r = await client.get("/api/portfolio/bess/market-prefill?region=INVALID")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_bess_scenario_caveat_present(client: AsyncClient):
    payload = {
        "position": {
            "capacity_mwh": 100,
            "soc_pct": 62,
            "max_discharge_mw": 50,
            "max_charge_mw": 25,
            "efficiency_pct": 88,
            "degradation_cost_per_mwh": 8,
            "min_reserve_soc_pct": 20,
        },
        "market": {
            "region": "NSW1",
            "price_rrp": 500,
            "price_regime": "spike",
            "evidence_quality": "supported",
        },
    }
    r = await client.post("/api/portfolio/bess/scenario", json=payload)
    assert "caveat" in r.json()
    assert "Simulation" in r.json()["caveat"]
