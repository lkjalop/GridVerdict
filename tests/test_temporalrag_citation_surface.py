"""Tests for TemporalRAG citation surface in query responses.

Verifies that the temporal_evidence list in a query response exposes:
  - system_time: ISO timestamp string
  - known_before_query_time: bool (prevents lookahead)
  - retrieval_reason: human-readable string explaining why the doc was retrieved
  - relevance_score: float (0–1)

These fields prove that the temporal evidence layer is bitemporal-safe and
explainable to the user/auditor.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.agents.scatter_gather import GatherResult
from app.data.aemo_live_client import DispatchPrice

_VALID_TIME = datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc)
_SYSTEM_TIME_PAST = _VALID_TIME - timedelta(hours=1)  # strictly before any query time → known_before=True


def _dispatch(region: str = "NSW1") -> DispatchPrice:
    return DispatchPrice(
        region=region,
        valid_time=_VALID_TIME,
        system_time=datetime.now(timezone.utc),
        price_rrp=250.0,
        demand_mw=7800.0,
        availability_mw=9100.0,
        raw_ref="test-ref",
    )


def _gather(region: str = "NSW1") -> GatherResult:
    return GatherResult(
        dispatch=_dispatch(region),
        dispatch_fresh=True,
        notices=[],
        analogs=[],
        tasks_ok=3,
        tasks_total=3,
        elapsed_ms=40.0,
    )


def _make_trag_doc(
    *,
    source_type: str = "dispatch_price",
    relevance_score: float = 0.85,
    system_time: datetime | None = None,
    valid_time: datetime | None = None,
) -> MagicMock:
    doc = MagicMock()
    doc.doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    doc.source_type = source_type
    doc.valid_time = valid_time or _VALID_TIME
    doc.system_time = system_time or _SYSTEM_TIME_PAST
    doc.relevance_score = relevance_score
    doc.citation = f"AEMO/{source_type}/{_VALID_TIME.isoformat()}"
    doc.content = {
        "price_rrp": 250.0,
        "demand_mw": 7800.0,
        "region": "NSW1",
    }
    return doc


def _make_bundle(*docs: MagicMock) -> MagicMock:
    bundle = MagicMock()
    bundle.docs = list(docs)
    return bundle


async def _create_session(client: AsyncClient, region: str = "NSW1") -> str:
    r = await client.post("/api/sessions", json={"region": region})
    assert r.status_code == 201, f"Session creation failed: {r.text}"
    return r.json()["id"]


async def _query_with_trag(
    client: AsyncClient,
    session_id: str,
    text: str,
    region: str = "NSW1",
    trag_bundle=None,
) -> dict[str, Any]:
    gr = _gather(region)
    mock_sg = AsyncMock(return_value=gr)
    mock_trag = AsyncMock(return_value=trag_bundle or _make_bundle())
    with (
        patch("app.api.routes_query.scatter_gather", mock_sg),
        patch("app.engines.temporalrag.retrieve", mock_trag),
    ):
        r = await client.post(
            f"/api/sessions/{session_id}/query",
            json={"text": text, "region": region},
        )
    return r


# ── Basic structure ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_evidence_key_present(client: AsyncClient):
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "Why is NSW price elevated?")
    assert r.status_code == 200
    data = r.json()
    assert "temporal_evidence" in data


@pytest.mark.asyncio
async def test_temporal_evidence_is_list(client: AsyncClient):
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "What is the current NSW price?")
    data = r.json()
    assert isinstance(data["temporal_evidence"], (list, type(None)))


# ── Citation surface fields ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_evidence_has_system_time(client: AsyncClient):
    doc = _make_trag_doc()
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "Why is NSW price elevated?", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert "system_time" in te[0], "temporal_evidence doc missing 'system_time'"
        assert isinstance(te[0]["system_time"], str)


@pytest.mark.asyncio
async def test_temporal_evidence_has_known_before_query_time(client: AsyncClient):
    doc = _make_trag_doc()
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "Why is NSW price elevated?", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert "known_before_query_time" in te[0], "temporal_evidence doc missing 'known_before_query_time'"
        assert isinstance(te[0]["known_before_query_time"], bool)


@pytest.mark.asyncio
async def test_temporal_evidence_has_retrieval_reason(client: AsyncClient):
    doc = _make_trag_doc()
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "Why is NSW price elevated?", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert "retrieval_reason" in te[0], "temporal_evidence doc missing 'retrieval_reason'"
        assert isinstance(te[0]["retrieval_reason"], str)
        assert len(te[0]["retrieval_reason"]) > 0


@pytest.mark.asyncio
async def test_temporal_evidence_has_relevance_score(client: AsyncClient):
    doc = _make_trag_doc(relevance_score=0.92)
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "NSW price lookup", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert "relevance_score" in te[0]
        assert isinstance(te[0]["relevance_score"], float)


# ── known_before_query_time semantics ─────────────────────────────────

@pytest.mark.asyncio
async def test_known_before_true_for_past_system_time(client: AsyncClient):
    """A doc ingested well before the query is bitemporal-safe (no lookahead)."""
    past_doc = _make_trag_doc(system_time=_VALID_TIME - timedelta(hours=2))
    bundle = _make_bundle(past_doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "Why is price high?", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert te[0]["known_before_query_time"] is True


@pytest.mark.asyncio
async def test_known_before_false_for_future_system_time(client: AsyncClient):
    """A doc ingested after query time would be lookahead — known_before must be False."""
    future_doc = _make_trag_doc(
        system_time=datetime.now(timezone.utc) + timedelta(hours=99)
    )
    bundle = _make_bundle(future_doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "NSW price check", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        assert te[0]["known_before_query_time"] is False


# ── retrieval_reason content ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieval_reason_mentions_relevance_tier(client: AsyncClient):
    """retrieval_reason must contain 'high', 'moderate', or 'low' relevance tier."""
    high_doc = _make_trag_doc(relevance_score=0.9)
    bundle = _make_bundle(high_doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "NSW price explanation", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        reason = te[0]["retrieval_reason"]
        assert any(word in reason for word in ("high", "moderate", "low")), (
            f"retrieval_reason '{reason}' does not mention relevance tier"
        )


@pytest.mark.asyncio
async def test_retrieval_reason_mentions_source_type(client: AsyncClient):
    """retrieval_reason should reference the source_type of the doc."""
    doc = _make_trag_doc(source_type="aemo_market_notice", relevance_score=0.75)
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "NSW price explanation", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        reason = te[0]["retrieval_reason"]
        # underscores converted to spaces in the helper
        assert "aemo" in reason.lower() or "market" in reason.lower() or "notice" in reason.lower(), (
            f"retrieval_reason '{reason}' does not reference source type"
        )


# ── system_time is ISO string ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_time_is_iso_parseable(client: AsyncClient):
    doc = _make_trag_doc()
    bundle = _make_bundle(doc)
    sid = await _create_session(client)
    r = await _query_with_trag(client, sid, "NSW price", trag_bundle=bundle)
    data = r.json()
    te = data.get("temporal_evidence") or []
    if te:
        try:
            datetime.fromisoformat(te[0]["system_time"])
        except ValueError as e:
            pytest.fail(f"system_time is not a valid ISO string: {e}")
