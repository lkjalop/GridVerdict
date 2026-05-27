"""Tests for FCAS price ingestion, rebid engine, enhanced incident timeline,
and claim verifier rule 6.

Covers:
  - FcasPriceEvent DB model fields
  - parse_mmsdm_fcas_content — extracts 8 FCAS columns from DISPATCHPRICE CSV
  - rebid_engine.detect_rebids — extraction + dataclass shape
  - incident_timeline _add_fcas_events, _add_outage_events, enhanced _add_rebid_events
  - claim_verifier rule 6: rebid/outage language → supported+ claim_tier required
"""
from __future__ import annotations

import gzip
import io
import zipfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.aemo_archive import parse_mmsdm_fcas_content
from app.engines.rebid_engine import (
    RebidEvent,
    _low_band_mw,
    _prices_by_period,
    _severity,
    detect_rebids,
)
from app.agents.claim_verifier import ClaimFinding, verify_answer, apply_verification
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    EvidenceRefSchema,
    FactualVerdict,
    VerdictLabel,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_DISPATCHPRICE_CSV = (
    "C,NEMDispatchIS,Public,,DispatchIS,20260523,0430\n"
    "I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,"
    "RRP,RAISE6SECRRP,RAISE60SECRRP,RAISE5MINRRP,RAISEREGRRP,"
    "LOWER6SECRRP,LOWER60SECRRP,LOWER5MINRRP,LOWERREGRRP,TOTALDEMAND,AVAILABLEGENERATION\n"
    "D,DISPATCH,PRICE,4,2026/05/23 04:30:00,1,NSW1,1,0,"
    "347.50,85.00,42.00,18.50,12.00,5.00,3.50,2.00,1.50,8000.0,9500.0\n"
    "D,DISPATCH,PRICE,4,2026/05/23 04:30:00,1,VIC1,1,0,"
    "290.00,120.00,60.00,25.00,15.00,8.00,4.00,3.00,2.00,6000.0,7200.0\n"
    "D,DISPATCH,PRICE,4,2026/05/23 04:30:00,1,QLD1,1,0,"
    "150.00,,,,,,,,,5000.0,5800.0\n"  # QLD1 has no FCAS prices
)

_BASE_TIME = datetime(2026, 5, 22, 18, 30, tzinfo=timezone.utc)  # 2026/05/23 04:30 AEST → UTC
_NOW = datetime(2026, 5, 25, 14, 0, 0, tzinfo=timezone.utc)

_DUMMY_REF = EvidenceRefSchema(
    source="AEMO_DISPATCH_PRICE", region="NSW1", interval=_NOW,
    field="price_rrp", value=500.0, raw_ref="test",
)


def _make_factual(
    text: str,
    claim_tiers=None,
    driver_tiers=None,
    verdict: VerdictLabel = VerdictLabel.LOW_CONFIDENCE,
    confidence: float = 0.65,
    refs=None,
) -> FactualVerdict:
    return FactualVerdict(
        verdict=verdict,
        action=ActionLabel.MONITOR,
        confidence=confidence,
        confidence_band=ConfidenceBand.MEDIUM,
        as_of=_NOW,
        why_plain_english=text,
        claim_tiers=claim_tiers or [],
        driver_tiers=driver_tiers or [],
        evidence_refs=refs if refs is not None else [],
        counterargument="No additional counterargument.",
    )


def _make_supported(text: str, claim_tiers=None, driver_tiers=None) -> FactualVerdict:
    """FactualVerdict with SUPPORTED verdict — requires a dummy evidence_ref."""
    return _make_factual(
        text,
        claim_tiers=claim_tiers,
        driver_tiers=driver_tiers,
        verdict=VerdictLabel.SUPPORTED,
        confidence=0.80,
        refs=[_DUMMY_REF],
    )


# ── FcasPriceEvent model ──────────────────────────────────────────────────────

def test_fcas_price_event_model_has_8_fcas_fields():
    from app.db.models import FcasPriceEvent
    columns = {c.name for c in FcasPriceEvent.__table__.columns}
    for field in [
        "raise_6sec_rrp", "raise_60sec_rrp", "raise_5min_rrp", "raise_reg_rrp",
        "lower_6sec_rrp", "lower_60sec_rrp", "lower_5min_rrp", "lower_reg_rrp",
    ]:
        assert field in columns, f"Missing column: {field}"


def test_fcas_price_event_model_has_unique_constraint():
    from app.db.models import FcasPriceEvent
    from sqlalchemy import UniqueConstraint
    constraints = {
        type(c).__name__: c
        for c in FcasPriceEvent.__table__.constraints
    }
    uc_names = [
        c.name for c in FcasPriceEvent.__table__.constraints
        if isinstance(c, UniqueConstraint)
    ]
    assert "uq_fcas_price_region_valid_time" in uc_names


# ── parse_mmsdm_fcas_content ──────────────────────────────────────────────────

def test_parse_fcas_plain_csv_returns_rows():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    # NSW1 and VIC1 have FCAS values; QLD1 has all-None FCAS → skipped
    regions = {r["region"] for r in rows}
    assert "NSW1" in regions
    assert "VIC1" in regions


def test_parse_fcas_correct_raise_6sec():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    nsw = next(r for r in rows if r["region"] == "NSW1")
    assert nsw["raise_6sec_rrp"] == pytest.approx(85.0)


def test_parse_fcas_correct_lower_reg():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    nsw = next(r for r in rows if r["region"] == "NSW1")
    assert nsw["lower_reg_rrp"] == pytest.approx(1.50)


def test_parse_fcas_all_8_fields_present():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    nsw = next(r for r in rows if r["region"] == "NSW1")
    for field in [
        "raise_6sec_rrp", "raise_60sec_rrp", "raise_5min_rrp", "raise_reg_rrp",
        "lower_6sec_rrp", "lower_60sec_rrp", "lower_5min_rrp", "lower_reg_rrp",
    ]:
        assert field in nsw, f"Missing field: {field}"


def test_parse_fcas_skips_row_with_all_none_fcas():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    regions = {r["region"] for r in rows}
    assert "QLD1" not in regions


def test_parse_fcas_gzip_content():
    rows = parse_mmsdm_fcas_content(
        gzip.compress(_DISPATCHPRICE_CSV.encode()), raw_ref="gz"
    )
    assert len(rows) >= 1


def test_parse_fcas_zip_content():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("PUBLIC_DVD_DISPATCHPRICE_202605010000.CSV", _DISPATCHPRICE_CSV)
    rows = parse_mmsdm_fcas_content(buf.getvalue(), raw_ref="zip")
    assert {r["region"] for r in rows} >= {"NSW1", "VIC1"}


def test_parse_fcas_source_field():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    assert all(r["source"] == "AEMO_DISPATCH_PRICE" for r in rows)


def test_parse_fcas_valid_time_parsed():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    nsw = next(r for r in rows if r["region"] == "NSW1")
    assert nsw["valid_time"] == _BASE_TIME


def test_parse_fcas_vic_raise_60sec():
    rows = parse_mmsdm_fcas_content(_DISPATCHPRICE_CSV.encode(), raw_ref="test")
    vic = next(r for r in rows if r["region"] == "VIC1")
    assert vic["raise_60sec_rrp"] == pytest.approx(60.0)


# ── rebid_engine helpers ───────────────────────────────────────────────────────

def test_low_band_mw_sums_cheap_bands():
    price_bands = {"1": 50, "2": 100, "3": 500}
    avail_bands = {"1": 200, "2": 150, "3": 100}
    result = _low_band_mw(price_bands, avail_bands)
    assert result == pytest.approx(350.0)  # bands 1 + 2 (≤300)


def test_low_band_mw_excludes_expensive_bands():
    price_bands = {"1": 1000}
    avail_bands = {"1": 500}
    assert _low_band_mw(price_bands, avail_bands) == pytest.approx(0.0)


def test_severity_high():
    # withdrawal_mw=400, spot=5000 → score=4+5=9 → high
    assert _severity(400.0, 5000.0) == "high"


def test_severity_medium():
    assert _severity(150.0, 500.0) == "medium"


def test_severity_low():
    assert _severity(60.0, None) == "low"


def test_prices_by_period_maps_correctly():
    base_dt = datetime(2026, 5, 23, 0, 0, tzinfo=timezone.utc)
    # 00:30 → period 2, price 300
    vt = datetime(2026, 5, 23, 0, 30, tzinfo=timezone.utc)
    result = _prices_by_period([(vt, 300.0)], base_dt)
    assert result[2] == pytest.approx(300.0)


def test_prices_by_period_averages_within_period():
    base_dt = datetime(2026, 5, 23, 0, 0, tzinfo=timezone.utc)
    vt1 = datetime(2026, 5, 23, 0, 5, tzinfo=timezone.utc)
    vt2 = datetime(2026, 5, 23, 0, 10, tzinfo=timezone.utc)
    result = _prices_by_period([(vt1, 100.0), (vt2, 200.0)], base_dt)
    assert result[1] == pytest.approx(150.0)


# ── detect_rebids (async, mocked session) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_detect_rebids_empty_when_no_day_data():
    session = AsyncMock()
    # day query returns no rows
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(fetchall=lambda: []),  # day_res
        ]
    )
    result = await detect_rebids(session, "NSW1", datetime(2026, 5, 23, tzinfo=timezone.utc))
    assert result == []


@pytest.mark.asyncio
async def test_detect_rebids_returns_rebid_event_dataclass():
    session = AsyncMock()
    base = datetime(2026, 5, 23, tzinfo=timezone.utc)

    day_row = ("DUID1", "ENERGY", 300.0, base, {"1": 50}, {"1": 200}, "coal", "Station A")
    intra_row = ("DUID1", 10, 100.0, base + timedelta(hours=6), {"1": 50}, {"1": 50})
    price_row = (base + timedelta(hours=4, minutes=30), 500.0)

    day_mock = MagicMock()
    day_mock.fetchall.return_value = [day_row]
    intra_mock = MagicMock()
    intra_mock.fetchall.return_value = [intra_row]
    price_mock = MagicMock()
    price_mock.fetchall.return_value = [price_row]

    session.execute = AsyncMock(side_effect=[day_mock, intra_mock, price_mock])

    events = await detect_rebids(session, "NSW1", base, threshold_mw=50.0, min_price=100.0)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, RebidEvent)
    assert ev.duid == "DUID1"
    assert ev.withdrawal_mw == pytest.approx(200.0)
    assert ev.rebid_flag in ("strategic", "availability")
    assert ev.severity in ("high", "medium", "low")


@pytest.mark.asyncio
async def test_detect_rebids_skips_below_threshold():
    session = AsyncMock()
    base = datetime(2026, 5, 23, tzinfo=timezone.utc)

    day_row = ("DUID1", "ENERGY", 100.0, base, {}, {}, "coal", "S1")
    intra_row = ("DUID1", 5, 80.0, base, {}, {})  # withdrawal=20 < threshold=50

    day_mock = MagicMock()
    day_mock.fetchall.return_value = [day_row]
    intra_mock = MagicMock()
    intra_mock.fetchall.return_value = [intra_row]
    price_mock = MagicMock()
    price_mock.fetchall.return_value = []

    session.execute = AsyncMock(side_effect=[day_mock, intra_mock, price_mock])
    events = await detect_rebids(session, "NSW1", base, threshold_mw=50.0, min_price=0.0)
    assert events == []


@pytest.mark.asyncio
async def test_detect_rebids_skips_below_min_price():
    session = AsyncMock()
    base = datetime(2026, 5, 23, tzinfo=timezone.utc)

    day_row = ("DUID1", "ENERGY", 300.0, base, {}, {}, "coal", "S1")
    intra_row = ("DUID1", 1, 100.0, base, {}, {})  # withdrawal=200 MW
    price_row = (base, 50.0)  # spot=50 < min_price=100

    day_mock = MagicMock()
    day_mock.fetchall.return_value = [day_row]
    intra_mock = MagicMock()
    intra_mock.fetchall.return_value = [intra_row]
    price_mock = MagicMock()
    price_mock.fetchall.return_value = [price_row]

    session.execute = AsyncMock(side_effect=[day_mock, intra_mock, price_mock])
    events = await detect_rebids(session, "NSW1", base, threshold_mw=50.0, min_price=100.0)
    assert events == []


@pytest.mark.asyncio
async def test_detect_rebids_sorted_by_withdrawal_desc():
    session = AsyncMock()
    base = datetime(2026, 5, 23, tzinfo=timezone.utc)

    day_rows = [
        ("DUID1", "ENERGY", 300.0, base, {}, {}, "coal", "S1"),
        ("DUID2", "ENERGY", 500.0, base, {}, {}, "gas", "S2"),
    ]
    intra_rows = [
        ("DUID1", 5, 100.0, base, {}, {}),  # withdrawal=200
        ("DUID2", 5, 100.0, base, {}, {}),  # withdrawal=400
    ]
    day_mock = MagicMock()
    day_mock.fetchall.return_value = day_rows
    intra_mock = MagicMock()
    intra_mock.fetchall.return_value = intra_rows
    price_mock = MagicMock()
    price_mock.fetchall.return_value = [(base, 500.0)]

    session.execute = AsyncMock(side_effect=[day_mock, intra_mock, price_mock])
    events = await detect_rebids(session, "NSW1", base, threshold_mw=50.0, min_price=0.0)
    assert len(events) == 2
    assert events[0].withdrawal_mw >= events[1].withdrawal_mw


# ── Claim verifier rule 6 ─────────────────────────────────────────────────────

def test_rule6_passes_with_no_outage_language():
    f = _make_factual("Prices rose due to low wind output.")
    result = verify_answer(f)
    assert not any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_triggers_on_rebid_language():
    f = _make_factual("The price spike was driven by a strategic rebid from AGL.")
    result = verify_answer(f)
    assert any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_triggers_on_forced_outage():
    f = _make_factual("Bayswater had a forced outage reducing capacity by 600 MW.")
    result = verify_answer(f)
    assert any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_triggers_on_unit_tripped():
    f = _make_factual("A unit tripped at 14:30, removing 660 MW from the market.")
    result = verify_answer(f)
    assert any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_triggers_on_availability_withdrawal():
    f = _make_factual("An availability withdrawal of 300 MW was submitted at 13:00.")
    result = verify_answer(f)
    assert any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_does_not_trigger_with_supported_rebid_tier():
    f = _make_factual(
        "The price spike was driven by a strategic rebid.",
        claim_tiers=[{"label": "DUID1 rebid", "tier": "supported", "category": "rebid", "evidence_ref_ids": ["ev-1"]}],
    )
    result = verify_answer(f)
    assert not any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_does_not_trigger_with_confirmed_outage_tier():
    f = _make_factual(
        "A forced outage removed 600 MW.",
        claim_tiers=[{"label": "Bayswater outage", "tier": "confirmed", "category": "outage", "evidence_ref_ids": ["ev-2"]}],
    )
    result = verify_answer(f)
    assert not any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_does_not_trigger_with_confirmed_unit_dispatch_tier():
    f = _make_factual(
        "A unit tripped at peak.",
        claim_tiers=[{"label": "trip", "tier": "confirmed", "category": "unit_dispatch", "evidence_ref_ids": ["ev-3"]}],
    )
    result = verify_answer(f)
    assert not any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_plausible_tier_still_triggers():
    """Only supported/confirmed tier satisfies Rule 6 — plausible is insufficient."""
    f = _make_factual(
        "A rebid caused the price to spike.",
        claim_tiers=[{"label": "rebid", "tier": "plausible", "category": "rebid", "evidence_ref_ids": []}],
    )
    result = verify_answer(f)
    assert any(r.rule == "outage_rebid_without_evidence_tier" for r in result.findings)


def test_rule6_downgrade_severity():
    f = _make_factual("A forced outage removed 600 MW from the market.")
    result = verify_answer(f)
    r6 = next(r for r in result.findings if r.rule == "outage_rebid_without_evidence_tier")
    assert r6.severity == "downgrade"


def test_rule6_apply_downgrades_supported_verdict():
    # Start with SUPPORTED verdict — rule 6 should downgrade to LOW_CONFIDENCE
    f = _make_supported("A unit trip at Callide occurred during the event.")
    result = verify_answer(f)
    corrected = apply_verification(f, result)
    assert corrected.verdict == VerdictLabel.LOW_CONFIDENCE


def test_rule6_apply_caps_confidence():
    f = _make_supported("A unit trip at Callide occurred during the event.")
    result = verify_answer(f)
    corrected = apply_verification(f, result)
    assert corrected.confidence <= 0.55


def test_rule6_appends_counterargument():
    f = _make_factual("A forced outage occurred near the price event.")
    result = verify_answer(f)
    corrected = apply_verification(f, result)
    assert "[Verifier:" in (corrected.counterargument or "")
    assert "rebid or outage" in (corrected.counterargument or "").lower()
