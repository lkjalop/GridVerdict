"""TemporalRAG — retriever.

Main entry point: `retrieve(query, session)`.

Source handlers (one per source_type) are async coroutines that return
list[TemporalDoc]. They are gated by try/except so a failing source never
blocks the pipeline — it contributes an empty list to the bundle.

No-leakage enforcement:
  Any document with system_time > query.system_time_at_query is removed
  before ranking. The count is reported in RetrievalBundle.leakage_filtered.

Ranking:
  relevance_score = 0.60 × temporal_proximity + 0.40 × source_credibility
  where temporal_proximity uses exponential decay from window midpoint.

Framework boundary: DB imports are deferred inside functions and wrapped
in try/except so the module is importable with no DB connection.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.engines.temporalrag.schema import (
    RetrievalBundle,
    SOURCE_CREDIBILITY,
    TemporalDoc,
    TemporalQuery,
    score_temporal_proximity,
)

logger = logging.getLogger(__name__)


async def retrieve(
    query: TemporalQuery,
    session: Any = None,
) -> RetrievalBundle:
    """Retrieve bitemporal evidence across all configured sources.

    Parameters
    ----------
    query : TemporalQuery
        Defines the valid-time window, system-time fence, region, and
        which source types to query.
    session : AsyncSession | None
        Optional SQLAlchemy async session for DB-backed sources
        (market_events, traces). If None, DB sources return empty lists.

    Returns
    -------
    RetrievalBundle
        Ranked, leakage-filtered documents with citation strings.
    """
    t0 = time.monotonic()

    source_coros = []
    source_names = []

    if "market_events" in query.source_types:
        source_coros.append(_fetch_market_events(query, session))
        source_names.append("market_events")
    if "notice" in query.source_types:
        source_coros.append(_fetch_notices(query))
        source_names.append("notice")
    if "trace" in query.source_types:
        source_coros.append(_fetch_traces(query, session))
        source_names.append("trace")
    if "analog" in query.source_types:
        source_coros.append(_fetch_analogs(query))
        source_names.append("analog")
    if "news" in query.source_types:
        source_coros.append(_fetch_news(query))
        source_names.append("news")

    raw_results = await asyncio.gather(*source_coros, return_exceptions=True)

    all_docs: list[TemporalDoc] = []
    for name, result in zip(source_names, raw_results):
        if isinstance(result, list):
            all_docs.extend(result)
        else:
            logger.debug("TemporalRAG source %s failed: %s", name, result)

    # No-leakage fence
    clean_docs: list[TemporalDoc] = []
    leakage_filtered = 0
    for doc in all_docs:
        if doc.passes_leakage_fence(query.system_time_at_query):
            clean_docs.append(doc)
        else:
            leakage_filtered += 1

    # Score and rank
    ranked = _rank(_dedupe_docs(clean_docs), query)
    ranked = ranked[: query.max_docs]

    source_counts: dict[str, int] = {}
    for doc in ranked:
        source_counts[doc.source_type] = source_counts.get(doc.source_type, 0) + 1

    elapsed_ms = (time.monotonic() - t0) * 1000
    return RetrievalBundle(
        query=query,
        docs=ranked,
        source_counts=source_counts,
        leakage_filtered=leakage_filtered,
        elapsed_ms=elapsed_ms,
    )


# ── Source handlers ───────────────────────────────────────────────────────────

def _dedupe_docs(docs: list[TemporalDoc]) -> list[TemporalDoc]:
    """Remove duplicate documents emitted by overlapping retrieval sources."""
    seen: set[tuple[str, str]] = set()
    unique: list[TemporalDoc] = []
    for doc in docs:
        key = (doc.source_type, doc.doc_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


async def _fetch_market_events(
    query: TemporalQuery,
    session: Any,
) -> list[TemporalDoc]:
    """Retrieve AEMO_DISPATCH_PRICE rows within the valid-time window."""
    if session is None:
        return []
    try:
        from sqlalchemy import select
        from app.db.models import MarketEvent

        stmt = (
            select(MarketEvent)
            .where(MarketEvent.valid_time >= query.valid_time_from)
            .where(MarketEvent.valid_time <= query.valid_time_to)
            .where(MarketEvent.source == "AEMO_DISPATCH_PRICE")
            .order_by(MarketEvent.valid_time.asc())
        )
        if query.region:
            stmt = stmt.where(MarketEvent.region == query.region.upper())

        result = await session.execute(stmt)
        rows = list(result.scalars().all())

        docs = []
        for row in rows:
            price = row.price_rrp or 0.0
            demand = row.demand_mw or 0.0
            region = row.region or ""
            docs.append(TemporalDoc(
                doc_id=str(row.id),
                source_type="market_events",
                valid_time=_ensure_tz(row.valid_time),
                system_time=_ensure_tz(row.system_time),
                content={
                    "region": region,
                    "price_rrp": price,
                    "demand_mw": demand,
                    "availability_mw": row.availability_mw,
                    "data": row.data or {},
                },
                citation=f"AEMO dispatch {region} {_fmt_dt(row.valid_time)}: "
                         f"${price:.2f}/MWh, {demand:.0f} MW",
            ))
        return docs
    except Exception as exc:
        logger.debug("_fetch_market_events error: %s", exc)
        return []


async def _fetch_traces(
    query: TemporalQuery,
    session: Any,
) -> list[TemporalDoc]:
    """Retrieve GridVerdict query traces (audit log) within the window."""
    if session is None:
        return []
    try:
        from sqlalchemy import select
        from app.db.models import Trace

        stmt = (
            select(Trace)
            .where(Trace.valid_time >= query.valid_time_from)
            .where(Trace.valid_time <= query.valid_time_to)
            .order_by(Trace.valid_time.desc())
            .limit(50)
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

        docs = []
        for row in rows:
            docs.append(TemporalDoc(
                doc_id=str(row.id),
                source_type="trace",
                valid_time=_ensure_tz(row.valid_time),
                system_time=_ensure_tz(row.system_time),
                content={
                    "query_text": getattr(row, "query_text", ""),
                    "verdict": getattr(row, "verdict", None),
                    "region": getattr(row, "region", ""),
                },
                citation=f"GridVerdict trace {_fmt_dt(row.valid_time)}",
            ))
        return docs
    except Exception as exc:
        logger.debug("_fetch_traces error: %s", exc)
        return []


async def _fetch_notices(query: TemporalQuery) -> list[TemporalDoc]:
    """Retrieve AEMO market notices from the in-process cache."""
    try:
        from app.data.cache import get_cache
        cache = get_cache()

        region = (query.region or "").upper()
        cached = await cache.get(f"notices_{region}") or await cache.get("notices")
        if not isinstance(cached, list):
            return []

        docs = []
        for item in cached:
            # Notices may lack system_time — default to their publication time
            vt_raw = item.get("publication_time") or item.get("valid_time")
            if vt_raw is None:
                continue
            vt = _parse_dt(vt_raw)
            if not (query.valid_time_from <= vt <= query.valid_time_to):
                continue
            st = _parse_dt(item.get("system_time") or vt_raw)
            notice_type = item.get("notice_type", "NOTICE")
            title = item.get("reason") or item.get("title") or ""
            docs.append(TemporalDoc(
                doc_id=str(item.get("notice_id", id(item))),
                source_type="notice",
                valid_time=vt,
                system_time=st,
                content=item,
                citation=f"AEMO {notice_type} {_fmt_dt(vt)}: {title[:80]}",
            ))
        return docs
    except Exception as exc:
        logger.debug("_fetch_notices error: %s", exc)
        return []


async def _fetch_analogs(query: TemporalQuery) -> list[TemporalDoc]:
    """Retrieve relevant HippoGraph nodes within the valid-time window."""
    try:
        from app.engines.hippograph.graph import get_graph

        graph = get_graph()
        region = query.region or ""
        region_nodes = graph.get_region_nodes(region, limit=2016) if region else []

        docs = []
        for node in region_nodes:
            vt = _ensure_tz(node.valid_time)
            if not (query.valid_time_from <= vt <= query.valid_time_to):
                continue
            # Analog nodes are derived data — system_time = valid_time for in-memory nodes
            price = node.feature_values.get("price_rrp", 0.0)
            demand = node.feature_values.get("demand_mw", 0.0)
            docs.append(TemporalDoc(
                doc_id=node.node_id,
                source_type="analog",
                valid_time=vt,
                system_time=vt,
                content={
                    "region": node.region,
                    "regime": node.regime,
                    "price_rrp": price,
                    "demand_mw": demand,
                    "feature_values": dict(node.feature_values),
                },
                citation=f"HippoGraph analog {node.region} {_fmt_dt(vt)}: "
                         f"${price:.0f}/MWh regime={node.regime}",
            ))
        return docs
    except Exception as exc:
        logger.debug("_fetch_analogs error: %s", exc)
        return []


async def _fetch_news(query: TemporalQuery) -> list[TemporalDoc]:
    """Retrieve RSS news items from the in-process cache within the window."""
    try:
        from app.data.cache import get_cache
        cache = get_cache()

        cached = await cache.get("nem_news")
        if not isinstance(cached, list):
            return []

        region_l = (query.region or "").lower()
        region_prefix = region_l[:3] if region_l else ""

        docs = []
        for item in cached:
            pub_raw = item.get("published") or item.get("pub_date")
            if pub_raw is None:
                continue
            vt = _parse_dt(pub_raw)
            if not (query.valid_time_from <= vt <= query.valid_time_to):
                continue
            # Filter to region-relevant items if region is specified
            if region_l:
                text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
                if region_l not in text and region_prefix not in text and "nem" not in text:
                    continue
            title = item.get("title") or ""
            source = item.get("source") or "RSS"
            docs.append(TemporalDoc(
                doc_id=str(item.get("id") or item.get("link") or id(item)),
                source_type="news",
                valid_time=vt,
                system_time=vt,  # RSS items have no separate system_time
                content=item,
                citation=f"{source} {_fmt_dt(vt)}: {title[:80]}",
            ))
        return docs
    except Exception as exc:
        logger.debug("_fetch_news error: %s", exc)
        return []


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank(docs: list[TemporalDoc], query: TemporalQuery) -> list[TemporalDoc]:
    """Score each document and return sorted descending by relevance_score."""
    for doc in docs:
        temporal = score_temporal_proximity(doc.valid_time, query)
        credibility = SOURCE_CREDIBILITY.get(doc.source_type, 0.5)
        doc.relevance_score = round(0.60 * temporal + 0.40 * credibility, 4)
    docs.sort(key=lambda d: d.relevance_score, reverse=True)
    return docs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_tz(dt: datetime) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_tz(value)
    text = str(value).replace("Z", "+00:00").replace(" ", "T")
    try:
        dt = datetime.fromisoformat(text)
        return _ensure_tz(dt)
    except ValueError:
        return datetime.now(timezone.utc)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%MZ") if dt else "unknown"
