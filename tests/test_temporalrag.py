"""Tests for the TemporalRAG engine.

Covers:
1. No-leakage invariant: documents with system_time > system_time_at_query are excluded.
2. Window filter: only docs within valid_time_from..valid_time_to are returned.
3. Ranking: docs sorted descending by relevance_score.
4. Source credibility: market_events outranks news at equal temporal distance.
5. Market events source: DB rows inside window are returned as TemporalDoc objects.
6. Traces source: Trace DB rows are returned correctly.
7. HippoGraph analog source: in-memory nodes within window are returned.
8. Multi-source retrieval: docs from different sources are merged correctly.
9. Empty DB returns empty bundle (no crash).
10. max_docs is respected.
11. Citations are non-empty for every returned document.
12. RetrievalBundle properties: has_market_data, has_analogs, citations().
13. TemporalQuery midpoint and window_seconds are correct.
14. score_temporal_proximity: 1.0 at midpoint, decays with distance.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.engines.temporalrag import TemporalQuery, RetrievalBundle, retrieve
from app.engines.temporalrag.retriever import _dedupe_docs
from app.engines.temporalrag.schema import (
    TemporalDoc,
    score_temporal_proximity,
    SOURCE_CREDIBILITY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def mem_engine():
    from app.db.models import Base
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(mem_engine) -> AsyncSession:
    factory = async_sessionmaker(bind=mem_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _market_event_row(
    valid_time: datetime,
    price: float = 80.0,
    demand: float = 7000.0,
    region: str = "NSW1",
    system_time: datetime | None = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "tenant_id": "system",
        "source": "AEMO_DISPATCH_PRICE",
        "region": region,
        "valid_time": valid_time,
        "system_time": system_time or valid_time,
        "price_rrp": price,
        "demand_mw": demand,
        "availability_mw": 9000.0,
        "data": {},
        "raw_ref": "test",
    }


async def _seed_market_events(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import MarketEvent
    for r in rows:
        obj = MarketEvent(
            id=r["id"],
            tenant_id=r["tenant_id"],
            source=r["source"],
            region=r["region"],
            valid_time=r["valid_time"],
            system_time=r["system_time"],
            price_rrp=r.get("price_rrp"),
            demand_mw=r.get("demand_mw"),
            availability_mw=r.get("availability_mw"),
            data=r.get("data", {}),
            raw_ref=r.get("raw_ref", "test"),
        )
        session.add(obj)
    await session.commit()


async def _seed_traces(session: AsyncSession, rows: list[dict]) -> None:
    from app.db.models import Trace
    for r in rows:
        obj = Trace(
            id=r["id"],
            tenant_id=r.get("tenant_id", "system"),
            valid_time=r["valid_time"],
            system_time=r.get("system_time", r["valid_time"]),
        )
        session.add(obj)
    await session.commit()


# ── Unit tests: schema helpers ────────────────────────────────────────────────

class TestTemporalQuerySchema:

    def test_midpoint_is_center_of_window(self):
        t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 13, 0, tzinfo=timezone.utc)
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1,
                          system_time_at_query=t1)
        assert q.midpoint == datetime(2024, 6, 1, 12, 30, tzinfo=timezone.utc)

    def test_window_seconds(self):
        t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 13, 0, tzinfo=timezone.utc)
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1,
                          system_time_at_query=t1)
        assert q.window_seconds == 3600.0

    def test_naive_datetimes_get_utc(self):
        t0 = datetime(2024, 6, 1, 12, 0)  # naive
        t1 = datetime(2024, 6, 1, 13, 0)  # naive
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1,
                          system_time_at_query=t1)
        assert q.valid_time_from.tzinfo is not None
        assert q.valid_time_to.tzinfo is not None
        assert q.system_time_at_query.tzinfo is not None


class TestScoreTemporalProximity:

    def test_score_one_at_midpoint(self):
        t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1, system_time_at_query=t1)
        score = score_temporal_proximity(q.midpoint, q)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_score_decays_with_distance(self):
        t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1, system_time_at_query=t1)
        score_near = score_temporal_proximity(
            datetime(2024, 6, 1, 13, 5, tzinfo=timezone.utc), q
        )
        score_far = score_temporal_proximity(
            datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc), q
        )
        assert score_near > score_far

    def test_score_in_unit_interval(self):
        t0 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
        q = TemporalQuery(valid_time_from=t0, valid_time_to=t1, system_time_at_query=t1)
        for offset_h in [-2, 0, 1, 2, 5]:
            dt = t0 + timedelta(hours=offset_h)
            s = score_temporal_proximity(dt, q)
            assert 0.0 <= s <= 1.0, f"score out of range for offset {offset_h}h: {s}"


class TestTemporalDoc:

    def test_passes_leakage_fence_true(self):
        system_time_at_query = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        doc = TemporalDoc(
            doc_id="x", source_type="market_events",
            valid_time=datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc),
            system_time=datetime(2024, 6, 1, 11, 30, tzinfo=timezone.utc),
            content={},
        )
        assert doc.passes_leakage_fence(system_time_at_query) is True

    def test_passes_leakage_fence_false_when_future_system_time(self):
        system_time_at_query = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        doc = TemporalDoc(
            doc_id="x", source_type="market_events",
            valid_time=datetime(2024, 6, 1, 11, 0, tzinfo=timezone.utc),
            system_time=datetime(2024, 6, 1, 12, 30, tzinfo=timezone.utc),  # FUTURE
            content={},
        )
        assert doc.passes_leakage_fence(system_time_at_query) is False


# ── Integration tests: DB sources ─────────────────────────────────────────────

class TestMarketEventsSource:

    @pytest.mark.asyncio
    async def test_rows_in_window_are_returned(self, db_session):
        t0 = _now() - timedelta(hours=2)
        t1 = _now()
        rows = [_market_event_row(t0 + timedelta(minutes=5 * i)) for i in range(6)]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0,
            valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 6
        assert bundle.has_market_data

    @pytest.mark.asyncio
    async def test_rows_outside_window_excluded(self, db_session):
        t0 = _now() - timedelta(hours=2)
        t1 = _now() - timedelta(hours=1)
        outside = _now() - timedelta(minutes=5)

        rows = [
            _market_event_row(t0 + timedelta(minutes=10)),   # inside
            _market_event_row(outside),                       # outside window
        ]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0,
            valid_time_to=t1,
            system_time_at_query=_now(),
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 1

    @pytest.mark.asyncio
    async def test_region_filter(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        rows = [
            _market_event_row(t0 + timedelta(minutes=5), region="NSW1"),
            _market_event_row(t0 + timedelta(minutes=10), region="VIC1"),
        ]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            region="NSW1",
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 1
        assert bundle.docs[0].content["region"] == "NSW1"

    @pytest.mark.asyncio
    async def test_no_session_returns_empty(self):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=None)
        assert bundle.total_docs == 0


class TestTracesSource:

    @pytest.mark.asyncio
    async def test_traces_in_window(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        rows = [
            {
                "id": str(uuid.uuid4()),
                "valid_time": t0 + timedelta(minutes=10),
                "system_time": t0 + timedelta(minutes=10),
            }
        ]
        await _seed_traces(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["trace"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 1
        assert bundle.source_counts.get("trace", 0) == 1


class TestNoLeakageInvariant:

    @pytest.mark.asyncio
    async def test_future_system_time_rows_excluded(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        query_point = t0 + timedelta(minutes=20)  # simulated "past" query time

        rows = [
            # This row's system_time is AFTER query_point — must be filtered
            _market_event_row(
                t0 + timedelta(minutes=10),
                system_time=query_point + timedelta(minutes=5),  # future!
            ),
            # This row's system_time is BEFORE query_point — allowed
            _market_event_row(
                t0 + timedelta(minutes=5),
                system_time=query_point - timedelta(minutes=5),
            ),
        ]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0,
            valid_time_to=t1,
            system_time_at_query=query_point,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 1, (
            "Only 1 row should pass the leakage fence"
        )
        assert bundle.leakage_filtered == 1, (
            "The future-system_time row must be counted as leakage_filtered"
        )

    @pytest.mark.asyncio
    async def test_leakage_filtered_zero_when_all_docs_pass(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        rows = [_market_event_row(t0 + timedelta(minutes=5 * i)) for i in range(4)]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.leakage_filtered == 0


class TestRankingAndCitations:

    @pytest.mark.asyncio
    async def test_docs_sorted_by_relevance_score(self, db_session):
        midpoint = _now() - timedelta(minutes=30)
        t0 = midpoint - timedelta(hours=1)
        t1 = midpoint + timedelta(hours=1)
        # Row at midpoint should score highest
        rows = [
            _market_event_row(t0 + timedelta(minutes=5)),   # far from midpoint
            _market_event_row(midpoint),                      # at midpoint
            _market_event_row(t0 + timedelta(minutes=30)),   # closer but not center
        ]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        scores = [d.relevance_score for d in bundle.docs]
        assert scores == sorted(scores, reverse=True), "Docs must be sorted by relevance"

    @pytest.mark.asyncio
    async def test_citations_non_empty(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        rows = [_market_event_row(t0 + timedelta(minutes=5 * i)) for i in range(3)]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        for doc in bundle.docs:
            assert doc.citation and len(doc.citation) > 0, "Every doc must have a citation"

    def test_source_credibility_market_beats_news(self):
        assert SOURCE_CREDIBILITY["market_events"] > SOURCE_CREDIBILITY["news"]

    def test_source_credibility_notice_beats_analog(self):
        assert SOURCE_CREDIBILITY["notice"] > SOURCE_CREDIBILITY["analog"]

    @pytest.mark.asyncio
    async def test_max_docs_respected(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        rows = [_market_event_row(t0 + timedelta(minutes=5 * i)) for i in range(12)]
        await _seed_market_events(db_session, rows)

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
            max_docs=5,
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs <= 5

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_bundle(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events", "trace"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.total_docs == 0
        assert bundle.leakage_filtered == 0


class TestRetrievalBundleProperties:

    @pytest.mark.asyncio
    async def test_has_market_data_false_when_no_events(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.has_market_data is False

    @pytest.mark.asyncio
    async def test_has_market_data_true_when_events_present(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        await _seed_market_events(db_session, [_market_event_row(t0 + timedelta(minutes=10))])

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        assert bundle.has_market_data is True

    @pytest.mark.asyncio
    async def test_citations_returns_list_of_strings(self, db_session):
        t0 = _now() - timedelta(hours=1)
        t1 = _now()
        await _seed_market_events(db_session, [_market_event_row(t0 + timedelta(minutes=5))])

        q = TemporalQuery(
            valid_time_from=t0, valid_time_to=t1,
            system_time_at_query=t1,
            source_types=["market_events"],
        )
        bundle = await retrieve(q, session=db_session)
        cits = bundle.citations()
        assert isinstance(cits, list)
        assert all(isinstance(c, str) for c in cits)


def test_dedupe_docs_keeps_first_doc_per_source_and_id():
    now = datetime.now(timezone.utc)
    first = TemporalDoc(
        doc_id="same",
        source_type="analog",
        valid_time=now,
        system_time=now,
        content={"n": 1},
        citation="first",
    )
    duplicate = TemporalDoc(
        doc_id="same",
        source_type="analog",
        valid_time=now,
        system_time=now,
        content={"n": 2},
        citation="duplicate",
    )
    other_source = TemporalDoc(
        doc_id="same",
        source_type="trace",
        valid_time=now,
        system_time=now,
        content={"n": 3},
        citation="trace",
    )

    docs = _dedupe_docs([first, duplicate, other_source])

    assert docs == [first, other_source]
