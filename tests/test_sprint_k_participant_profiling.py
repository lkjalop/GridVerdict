"""Sprint K: Participant behavior profiling tests.

Tests cover:
  - profile_participant: returns low_activity + data_available=False with no bid data
  - profile_participant: classifies "habitual" when ≥5 strategic rebids
  - profile_participant: classifies "occasional" when 2–4 rebids, low strategic fraction
  - profile_participant: classifies "low_activity" when < 2 qualifying rebids
  - profile_participant: respects threshold_mw (small withdrawals not counted)
  - profile_participant: data_available=True when day offers exist but no qualifying rebids
  - _classify_tier: unit tests for tier logic
  - elevate_rebid_tier_if_habitual: upgrades tier to 'confirmed' for habitual participant
  - elevate_rebid_tier_if_habitual: unchanged when participant not habitual
  - elevate_rebid_tier_if_habitual: 'confirmed' tier is never downgraded
  - elevate_rebid_tier_if_habitual: None tier returned unchanged when no data
  - GET /api/market/participants/{duid}/profile: returns 200
  - GET /api/market/participants/{duid}/profile: response has required fields
  - GET /api/market/participants/{duid}/profile: invalid region returns 400
  - GET /api/market/participants/{duid}/profile: behavioral_tier field present
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import date, datetime, timedelta, timezone

from httpx import AsyncClient

from app.db.models import BidOffer


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _base_date(offset_days: int = 0) -> date:
    """Return a recent date offset from today to stay within the default window."""
    return (datetime.now(timezone.utc) - timedelta(days=5 + offset_days)).date()


def _day_offer(
    duid: str,
    d: date,
    max_avail: float = 500.0,
    region: str = "NSW1",
    strategic: bool = True,
) -> BidOffer:
    """BIDDAYOFFER row.  strategic=True puts capacity in cheap bands."""
    price_bands = {"1": 50.0, "2": 200.0, "3": 500.0}
    avail_bands = (
        {"1": 300.0, "2": 150.0, "3": 50.0}
        if strategic
        else {"1": 0.0, "2": 0.0, "3": max_avail}
    )
    return BidOffer(
        source="BIDDAYOFFER",
        duid=duid,
        region=region,
        bid_type="ENERGY",
        settlement_date=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
        period_id=None,
        max_avail_mw=max_avail,
        price_bands=price_bands,
        avail_bands=avail_bands,
        raw_ref=f"test-day-{duid}-{d.isoformat()}",
        data={},
    )


def _intra_offer(
    duid: str,
    d: date,
    period_id: int = 1,
    max_avail: float = 200.0,
    region: str = "NSW1",
    strategic: bool = True,
) -> BidOffer:
    """BIDPEROFFER row that withdraws ≥50 MW from the day offer.
    strategic=True withdraws cheap-band capacity as well.
    """
    price_bands = {"1": 50.0, "2": 200.0, "3": 500.0}
    avail_bands = (
        {"1": 0.0, "2": 50.0, "3": 50.0}   # cheap bands emptied → strategic
        if strategic
        else {"1": 300.0, "2": 0.0, "3": 0.0}  # cheap bands maintained → NOT strategic
    )
    return BidOffer(
        source="BIDPEROFFER",
        duid=duid,
        region=region,
        bid_type="ENERGY",
        settlement_date=datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
        period_id=period_id,
        max_avail_mw=max_avail,
        price_bands=price_bands,
        avail_bands=avail_bands,
        raw_ref=f"test-intra-{duid}-{d.isoformat()}-{period_id}",
        data={},
    )


# ── _classify_tier unit tests (no DB needed) ──────────────────────────────────

class TestClassifyTier:
    def test_habitual_five_strategic(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(5, 0.50) == "habitual"

    def test_habitual_boundary(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(5, 0.40) == "habitual"

    def test_not_habitual_too_few_rebids(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(4, 0.90) == "occasional"

    def test_not_habitual_low_strategic_fraction(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(10, 0.39) == "occasional"

    def test_occasional_two_rebids(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(2, 0.0) == "occasional"

    def test_low_activity_one_rebid(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(1, 1.0) == "low_activity"

    def test_low_activity_zero_rebids(self):
        from app.engines.participant_profiler import _classify_tier
        assert _classify_tier(0, 0.0) == "low_activity"


# ── profile_participant: no data ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_no_data_returns_low_activity(db_session):
    from app.engines.participant_profiler import profile_participant
    profile = await profile_participant(db_session, "NODUID_XYZ", region="NSW1")
    assert profile.behavioral_tier == "low_activity"
    assert profile.data_available is False
    assert profile.rebid_count == 0


@pytest.mark.asyncio
async def test_profile_no_data_duid_is_uppercased(db_session):
    from app.engines.participant_profiler import profile_participant
    profile = await profile_participant(db_session, "lowercase_duid")
    assert profile.duid == "LOWERCASE_DUID"


# ── profile_participant: habitual ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_habitual_six_strategic_rebids(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_HAB_01"
    for i in range(6):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d))
        db_session.add(_intra_offer(duid, d, strategic=True))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.behavioral_tier == "habitual"
    assert profile.rebid_count == 6
    assert profile.strategic_fraction >= 0.40
    assert profile.data_available is True


@pytest.mark.asyncio
async def test_profile_habitual_counts_days_correctly(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_HAB_02"
    for i in range(5):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d))
        db_session.add(_intra_offer(duid, d, period_id=1, strategic=True))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1")
    assert profile.days_with_rebids == 5


# ── profile_participant: occasional ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_occasional_two_non_strategic_rebids(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_OCC_01"
    for i in range(2):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d, strategic=False))
        db_session.add(_intra_offer(duid, d, max_avail=200.0, strategic=False))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.behavioral_tier == "occasional"
    assert profile.rebid_count == 2
    assert profile.strategic_fraction < 0.40


@pytest.mark.asyncio
async def test_profile_three_rebids_low_strategic_is_occasional(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_OCC_02"
    for i in range(3):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d, strategic=False))
        db_session.add(_intra_offer(duid, d, max_avail=200.0, strategic=False))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.behavioral_tier == "occasional"


# ── profile_participant: low_activity ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_one_rebid_is_low_activity(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_LOW_01"
    d = _base_date()
    db_session.add(_day_offer(duid, d))
    db_session.add(_intra_offer(duid, d, strategic=True))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.behavioral_tier == "low_activity"
    assert profile.rebid_count == 1


# ── profile_participant: threshold / filter logic ─────────────────────────────

@pytest.mark.asyncio
async def test_profile_small_withdrawal_not_counted(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_THR_01"
    d = _base_date()
    # day max_avail=500, intra max_avail=470 → withdrawal=30 < threshold_mw=50
    db_session.add(_day_offer(duid, d, max_avail=500.0))
    db_session.add(_intra_offer(duid, d, max_avail=470.0, strategic=True))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.rebid_count == 0
    assert profile.behavioral_tier == "low_activity"


@pytest.mark.asyncio
async def test_profile_data_available_true_when_day_offers_exist(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_THR_02"
    d = _base_date()
    # Only day offer, no intra offer → no rebids but data_available=True
    db_session.add(_day_offer(duid, d))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1")
    assert profile.data_available is True
    assert profile.rebid_count == 0
    assert profile.behavioral_tier == "low_activity"


# ── profile_participant: statistics ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_avg_mw_withdrawn_correct(db_session):
    from app.engines.participant_profiler import profile_participant

    duid = "PROF_STAT_01"
    # Day=500, intra=200 → withdrawal=300 for all 6 days
    for i in range(6):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d, max_avail=500.0))
        db_session.add(_intra_offer(duid, d, max_avail=200.0, strategic=True))
    await db_session.flush()

    profile = await profile_participant(db_session, duid, region="NSW1", threshold_mw=50.0)
    assert profile.avg_mw_withdrawn == pytest.approx(300.0, abs=1.0)
    assert profile.max_mw_withdrawn == pytest.approx(300.0, abs=1.0)


# ── elevate_rebid_tier_if_habitual ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_elevate_tier_habitual_upgrades_to_confirmed(db_session):
    from app.engines.why_builder import elevate_rebid_tier_if_habitual

    duid = "WHY_HAB_01"
    for i in range(6):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d))
        db_session.add(_intra_offer(duid, d, strategic=True))
    await db_session.flush()

    result = await elevate_rebid_tier_if_habitual(
        db_session, duid=duid, region="NSW1", rebid_evidence_tier="plausible"
    )
    assert result == "confirmed"


@pytest.mark.asyncio
async def test_elevate_tier_occasional_does_not_upgrade(db_session):
    from app.engines.why_builder import elevate_rebid_tier_if_habitual

    duid = "WHY_OCC_01"
    for i in range(2):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d, strategic=False))
        db_session.add(_intra_offer(duid, d, max_avail=200.0, strategic=False))
    await db_session.flush()

    result = await elevate_rebid_tier_if_habitual(
        db_session, duid=duid, region="NSW1", rebid_evidence_tier="plausible"
    )
    assert result == "plausible"


@pytest.mark.asyncio
async def test_elevate_tier_already_confirmed_unchanged(db_session):
    from app.engines.why_builder import elevate_rebid_tier_if_habitual

    # Even if participant is habitual, 'confirmed' is not changed
    duid = "WHY_CONF_01"
    for i in range(6):
        d = _base_date(i)
        db_session.add(_day_offer(duid, d))
        db_session.add(_intra_offer(duid, d, strategic=True))
    await db_session.flush()

    result = await elevate_rebid_tier_if_habitual(
        db_session, duid=duid, region="NSW1", rebid_evidence_tier="confirmed"
    )
    assert result == "confirmed"


@pytest.mark.asyncio
async def test_elevate_tier_none_input_returned_when_no_data(db_session):
    from app.engines.why_builder import elevate_rebid_tier_if_habitual

    result = await elevate_rebid_tier_if_habitual(
        db_session, duid="NODUID_WHY_99", region="NSW1", rebid_evidence_tier=None
    )
    assert result is None


@pytest.mark.asyncio
async def test_elevate_tier_unconfirmed_returned_when_no_data(db_session):
    from app.engines.why_builder import elevate_rebid_tier_if_habitual

    result = await elevate_rebid_tier_if_habitual(
        db_session, duid="NODUID_WHY_88", region="NSW1", rebid_evidence_tier="unconfirmed"
    )
    assert result == "unconfirmed"


# ── API endpoint ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_participant_profile_endpoint_returns_200(client: AsyncClient):
    resp = await client.get("/api/market/participants/TESTDUID/profile")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_participant_profile_endpoint_has_required_fields(client: AsyncClient):
    resp = await client.get("/api/market/participants/TESTDUID/profile")
    data = resp.json()
    for key in ("duid", "behavioral_tier", "data_available"):
        assert key in data, f"missing key: {key}"


@pytest.mark.asyncio
async def test_participant_profile_endpoint_duid_uppercased(client: AsyncClient):
    resp = await client.get("/api/market/participants/lowercase_duid/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["duid"] == "LOWERCASE_DUID"


@pytest.mark.asyncio
async def test_participant_profile_endpoint_invalid_region_returns_400(client: AsyncClient):
    resp = await client.get("/api/market/participants/TESTDUID/profile?region=INVALID")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_participant_profile_endpoint_no_data_is_low_activity(client: AsyncClient):
    resp = await client.get("/api/market/participants/UNKNOWN_DUID_XYZ/profile?region=NSW1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["behavioral_tier"] == "low_activity"
    assert data["data_available"] is False


@pytest.mark.asyncio
async def test_participant_profile_endpoint_window_days_param(client: AsyncClient):
    resp = await client.get("/api/market/participants/TESTDUID/profile?window_days=60")
    assert resp.status_code == 200
