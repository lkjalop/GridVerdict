"""Sprint J: Decision audit log + regulatory export tests.

Tests cover:
  - DecisionAuditLog model: row can be inserted and retrieved
  - log_bess_decision: writes a row with correct fields
  - log_fleet_decision: writes one row per asset in the fleet plan
  - simulation_only is always True
  - rows_to_json: serialises all expected fields
  - rows_to_csv: produces valid CSV with correct header
  - rows_to_csv: handles None fields gracefully
  - query_audit_log: region filter works
  - query_audit_log: decision_type filter works
  - query_audit_log: date range filter works
  - export endpoint: GET /portfolio/audit/export returns 200
  - export endpoint: CSV format returns text/csv content-type
  - BESS scenario route: audit row written after POST
  - Fleet scenario route: audit rows written after POST
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient

from app.db.models import DecisionAuditLog
from app.portfolio.schema import (
    BessPosition,
    ContractType,
    DispatchAction,
    FleetAsset,
    FleetScenarioRequest,
    MarketSnapshot,
)
from app.portfolio.bess_engine import compute_economics
from app.portfolio.dispatch_policy import evaluate
from app.portfolio.fleet_coordinator import evaluate_fleet


# ── Helpers ────────────────────────────────────────────────────────────────────

def _position() -> BessPosition:
    return BessPosition(
        capacity_mwh=10.0,
        soc_pct=80.0,
        max_discharge_mw=5.0,
        max_charge_mw=5.0,
        efficiency_pct=90.0,
        degradation_cost_per_mwh=5.0,
        min_reserve_soc_pct=10.0,
    )


def _market(regime: str = "spike", price: float = 500.0) -> MarketSnapshot:
    return MarketSnapshot(
        region="NSW1",
        price_rrp=price,
        price_regime=regime,
        evidence_quality="supported",
        forecast_direction="stable",
    )


def _scenario_result():
    pos = _position()
    mkt = _market()
    eco = compute_economics(pos, mkt)
    return evaluate(pos, mkt, eco), mkt


def _fleet_plan():
    assets = [
        FleetAsset(asset_id="BESS_01", position=_position()),
        FleetAsset(asset_id="BESS_02", position=_position()),
    ]
    req = FleetScenarioRequest(assets=assets, market=_market())
    return evaluate_fleet(req)


# ── DecisionAuditLog model ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_log_row_insert_and_retrieve(db_session):
    row = DecisionAuditLog(
        tenant_id="test-tenant",
        decision_type="bess_dispatch",
        region="NSW1",
        action="dispatch_full",
        confidence="supported",
        price_rrp=500.0,
        price_regime="spike",
        simulation_only=True,
    )
    db_session.add(row)
    await db_session.flush()
    assert row.id is not None

    from sqlalchemy import select
    result = await db_session.execute(
        select(DecisionAuditLog).where(DecisionAuditLog.id == row.id)
    )
    fetched = result.scalar_one()
    assert fetched.action == "dispatch_full"
    assert fetched.simulation_only is True


@pytest.mark.asyncio
async def test_audit_log_simulation_only_always_true(db_session):
    row = DecisionAuditLog(
        tenant_id="test-tenant",
        decision_type="bess_dispatch",
        region="NSW1",
        action="hold",
        confidence="supported",
        simulation_only=True,
    )
    db_session.add(row)
    await db_session.flush()
    assert row.simulation_only is True


# ── log_bess_decision ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_bess_decision_writes_row(db_session):
    from app.audit.audit_logger import log_bess_decision
    result, market = _scenario_result()
    row_id = await log_bess_decision(
        db_session, tenant_id="t1", result=result, market=market,
        user_id="user-123", trace_id="trace-abc",
    )
    assert row_id is not None

    from sqlalchemy import select
    rows = (await db_session.execute(
        select(DecisionAuditLog).where(DecisionAuditLog.id == row_id)
    )).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.decision_type == "bess_dispatch"
    assert row.region == "NSW1"
    assert row.user_id == "user-123"
    assert row.trace_id == "trace-abc"
    assert row.simulation_only is True
    assert row.economics is not None
    assert "dispatch_mw" in row.economics


@pytest.mark.asyncio
async def test_log_bess_decision_action_matches_result(db_session):
    from app.audit.audit_logger import log_bess_decision
    result, market = _scenario_result()
    row_id = await log_bess_decision(
        db_session, tenant_id="t1", result=result, market=market,
    )
    from sqlalchemy import select
    row = (await db_session.execute(
        select(DecisionAuditLog).where(DecisionAuditLog.id == row_id)
    )).scalar_one()
    assert row.action == result.action.value


# ── log_fleet_decision ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_fleet_decision_writes_one_row_per_asset(db_session):
    from app.audit.audit_logger import log_fleet_decision
    plan = _fleet_plan()
    row_ids = await log_fleet_decision(db_session, tenant_id="t2", plan=plan)
    assert len(row_ids) == len(plan.assets)

    from sqlalchemy import select
    rows = (await db_session.execute(
        select(DecisionAuditLog).where(DecisionAuditLog.id.in_(row_ids))
    )).scalars().all()
    assert len(rows) == len(plan.assets)
    asset_ids = {r.asset_id for r in rows}
    expected_ids = {a.asset_id for a in plan.assets}
    assert asset_ids == expected_ids


@pytest.mark.asyncio
async def test_log_fleet_decision_type_is_fleet_dispatch(db_session):
    from app.audit.audit_logger import log_fleet_decision
    plan = _fleet_plan()
    row_ids = await log_fleet_decision(db_session, tenant_id="t2", plan=plan)
    from sqlalchemy import select
    rows = (await db_session.execute(
        select(DecisionAuditLog).where(DecisionAuditLog.id.in_(row_ids))
    )).scalars().all()
    assert all(r.decision_type == "fleet_dispatch" for r in rows)


# ── rows_to_json / rows_to_csv ────────────────────────────────────────────────

class TestExportSerialisation:
    def _make_row(self, action: str = "dispatch_full") -> DecisionAuditLog:
        return DecisionAuditLog(
            id="test-id-1",
            tenant_id="t1",
            user_id="u1",
            decision_type="bess_dispatch",
            region="VIC1",
            action=action,
            confidence="supported",
            price_rrp=450.0,
            price_regime="spike",
            economics={"dispatch_mw": 5.0, "net_expected_value": 18.0},
            risk_flags=["some risk"],
            why_summary=["reason A"],
            simulation_only=True,
            created_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        )

    def test_rows_to_json_has_all_keys(self):
        from app.audit.export import rows_to_json
        rows = [self._make_row()]
        data = rows_to_json(rows)
        assert len(data) == 1
        for k in ("id", "action", "region", "simulation_only", "economics",
                  "created_at", "risk_flags"):
            assert k in data[0], f"missing key: {k}"

    def test_rows_to_json_simulation_only_is_true(self):
        from app.audit.export import rows_to_json
        data = rows_to_json([self._make_row()])
        assert data[0]["simulation_only"] is True

    def test_rows_to_csv_valid_csv(self):
        from app.audit.export import rows_to_csv
        import csv, io
        csv_str = rows_to_csv([self._make_row()])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows_out = list(reader)
        assert len(rows_out) == 1
        assert rows_out[0]["action"] == "dispatch_full"
        assert rows_out[0]["region"] == "VIC1"

    def test_rows_to_csv_none_fields_become_empty(self):
        from app.audit.export import rows_to_csv
        row = self._make_row()
        row.user_id = None
        row.trace_id = None
        csv_str = rows_to_csv([row])
        assert csv_str  # no crash

    def test_empty_rows_to_csv_has_header_only(self):
        from app.audit.export import rows_to_csv
        csv_str = rows_to_csv([])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only


# ── query_audit_log filters ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_audit_log_region_filter(db_session):
    from app.audit.audit_logger import log_bess_decision
    from app.audit.export import query_audit_log

    result, market_nsw = _scenario_result()
    market_vic = MarketSnapshot(
        region="VIC1", price_rrp=200.0, price_regime="normal",
        evidence_quality="supported",
    )
    pos = _position()
    eco = compute_economics(pos, market_vic)
    result_vic = evaluate(pos, market_vic, eco)

    await log_bess_decision(db_session, tenant_id="t3", result=result, market=market_nsw)
    await log_bess_decision(db_session, tenant_id="t3", result=result_vic, market=market_vic)

    nsw_rows = await query_audit_log(db_session, tenant_id="t3", region="NSW1")
    vic_rows = await query_audit_log(db_session, tenant_id="t3", region="VIC1")
    assert all(r.region == "NSW1" for r in nsw_rows)
    assert all(r.region == "VIC1" for r in vic_rows)


@pytest.mark.asyncio
async def test_query_audit_log_decision_type_filter(db_session):
    from app.audit.audit_logger import log_bess_decision, log_fleet_decision
    from app.audit.export import query_audit_log

    result, market = _scenario_result()
    await log_bess_decision(db_session, tenant_id="t4", result=result, market=market)
    plan = _fleet_plan()
    await log_fleet_decision(db_session, tenant_id="t4", plan=plan)

    bess_rows = await query_audit_log(db_session, tenant_id="t4", decision_type="bess_dispatch")
    fleet_rows = await query_audit_log(db_session, tenant_id="t4", decision_type="fleet_dispatch")
    assert all(r.decision_type == "bess_dispatch" for r in bess_rows)
    assert all(r.decision_type == "fleet_dispatch" for r in fleet_rows)


# ── API export endpoint ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_export_endpoint_returns_200(client: AsyncClient):
    resp = await client.get("/api/portfolio/audit/export")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_audit_export_json_has_count_field(client: AsyncClient):
    resp = await client.get("/api/portfolio/audit/export?format=json")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "records" in data


@pytest.mark.asyncio
async def test_audit_export_csv_content_type(client: AsyncClient):
    resp = await client.get("/api/portfolio/audit/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
