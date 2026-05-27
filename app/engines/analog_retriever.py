"""Analog retriever — end-to-end: current FeatureVector → top-N historical analogs.

Called by scatter_gather T3. Takes the current dispatch price, converts it to
a FeatureVector, inserts it into HippoGraph, runs PPR, and returns the top-N
analog summaries as plain dicts (so routes and why_builder need not import
HippoGraph types directly).

Sprint B: driver-aware analog quality
  - Each analog dict now includes `driver_types` (inferred from regime + headroom)
  - `driver_similarity` (Jaccard overlap with query driver set)
  - `quality_score` (0.6 × PPR_norm + 0.4 × driver_similarity)
  - `rerank_analogs_with_drivers()` re-ranks using full driver_events + notices context,
    called from routes_query.py after market driver attribution completes.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from app.core.interfaces import FeatureVector
from app.engines.hippograph.graph import get_graph
from app.engines.hippograph.ppr import retrieve_analogs

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 10


# ── Driver-type helpers ────────────────────────────────────────────────────────

def _derive_query_driver_types(
    regime: str,
    headroom_mw: float,
    notices: list[dict[str, Any]] | None = None,
    driver_events: list[dict[str, Any]] | None = None,
) -> frozenset[str]:
    """Infer driver-type labels from current market state, notices, and driver events."""
    types: set[str] = set()
    if regime == "extreme":
        types.update(["high_price", "elevated_price"])
    elif regime == "spike":
        types.update(["high_price", "elevated_price"])
    elif regime == "elevated":
        types.add("elevated_price")
    if headroom_mw < 200:
        types.add("supply_squeeze")
    elif headroom_mw < 500:
        types.add("low_headroom")
    for notice in (notices or []):
        nt = str(notice.get("notice_type") or notice.get("type") or "").lower()
        if "constraint" in nt:
            types.add("constraint_binding")
        if "outage" in nt or "trip" in nt:
            types.add("generator_outage")
    for ev in (driver_events or []):
        ev_type = str(ev.get("driver_type") or ev.get("event_type") or "").lower()
        if "constraint" in ev_type:
            types.add("constraint_binding")
        if "outage" in ev_type:
            types.add("generator_outage")
        if "rebid" in ev_type:
            types.add("rebid")
    return frozenset(types)


def _infer_analog_driver_types(regime: str, headroom_mw: float) -> frozenset[str]:
    """Infer driver-type labels for an analog from its regime and headroom."""
    types: set[str] = set()
    if regime == "extreme":
        types.update(["high_price", "elevated_price"])
    elif regime == "spike":
        types.update(["high_price", "elevated_price"])
    elif regime == "elevated":
        types.add("elevated_price")
    if headroom_mw < 200:
        types.add("supply_squeeze")
    elif headroom_mw < 500:
        types.add("low_headroom")
    return frozenset(types)


def _driver_similarity_jaccard(q: frozenset[str], a: frozenset[str]) -> float:
    if not q and not a:
        return 1.0
    union = q | a
    return len(q & a) / len(union) if union else 1.0


def summarise_seasonal_events(
    region: str,
    season_buckets: list[dict[str, Any]],
    rows: list[Any],
    spike_threshold: float = 300.0,
) -> list[dict[str, Any]]:
    """Aggregate dispatch rows into per-season price statistics.

    Rows may be ORM objects or dicts with valid_time/price_rrp fields. This
    pure helper keeps the stats deterministic and easy to test without a DB.
    """
    summaries: list[dict[str, Any]] = []
    for bucket in season_buckets:
        start = _parse_bucket_dt(bucket["from_dt"])
        end = _parse_bucket_dt(bucket["to_dt"])
        prices = [
            float(_field(row, "price_rrp"))
            for row in rows
            if _field(row, "price_rrp") is not None
            and _field(row, "valid_time") is not None
            and start <= _parse_bucket_dt(_field(row, "valid_time")) < end
            and str(_field(row, "region") or region).upper() == region.upper()
        ]
        prices.sort()
        summaries.append({
            "label": bucket.get("label", f"{start.date()} to {end.date()}"),
            "region": region.upper(),
            "from_dt": start.isoformat(),
            "to_dt": end.isoformat(),
            "interval_count": len(prices),
            "mean_price": round(mean(prices), 2) if prices else None,
            "p90_price": round(_percentile(prices, 0.90), 2) if prices else None,
            "max_price": round(max(prices), 2) if prices else None,
            "spike_count": sum(1 for price in prices if price >= spike_threshold),
        })
    return summaries


async def retrieve_seasonal_summary(
    session,
    region: str,
    season_buckets: list[dict[str, Any]],
    spike_threshold: float = 300.0,
) -> list[dict[str, Any]]:
    """Retrieve and aggregate stored dispatch intervals for seasonal windows."""
    from sqlalchemy import select
    from app.db.models import MarketEvent

    if not season_buckets:
        return []
    starts = [_parse_bucket_dt(b["from_dt"]) for b in season_buckets]
    ends = [_parse_bucket_dt(b["to_dt"]) for b in season_buckets]
    stmt = (
        select(MarketEvent)
        .where(MarketEvent.region == region.upper())
        .where(MarketEvent.valid_time >= min(starts))
        .where(MarketEvent.valid_time < max(ends))
        .order_by(MarketEvent.valid_time.asc())
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    return summarise_seasonal_events(region, season_buckets, rows, spike_threshold)


def get_analogs(
    region: str,
    price_rrp: float,
    demand_mw: float,
    availability_mw: float,
    regime: str,
    valid_time: datetime | None = None,
    tenant_id: str = "system",
    top_k: int = _DEFAULT_TOP_K,
    notices: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve historical analogs for the given market state.

    Returns a list of dicts (serialisable, no HippoGraph types).
    Returns [] if the graph has insufficient history.
    """
    if valid_time is None:
        valid_time = datetime.now(timezone.utc)

    headroom = max(availability_mw - demand_mw, 0.0)
    price_max = 15_000.0
    query_driver_types = _derive_query_driver_types(regime, headroom, notices)

    fv = FeatureVector(
        event_id=f"qry-{valid_time.strftime('%Y%m%d%H%M%S')}-{region}",
        values={
            "price_rrp": price_rrp,
            "demand_mw": demand_mw,
            "availability_mw": availability_mw,
            "headroom_mw": headroom,
            "price_norm": min(1.0, price_rrp / price_max),
            "headroom_ratio": headroom / max(demand_mw, 1.0),
        },
        categorical={
            "region": region,
            "regime": regime,
            "source": "query",
        },
        valid_time=valid_time,
        tenant_id=tenant_id,
    )

    graph = get_graph()

    # Compute similarity edges for the new node before insertion
    # so PPR traversal can use them
    node = graph.insert(fv)
    graph.compute_similarity_edges(node.node_id)

    if graph.node_count() < 10:
        logger.debug("HippoGraph too sparse for analogs (%d nodes)", graph.node_count())
        return []

    try:
        analogs = retrieve_analogs(
            query_node_id=node.node_id,
            graph=graph,
            top_k=top_k,
            region=region,
        )
    except Exception as exc:
        logger.warning("PPR retrieval failed: %s", exc)
        return []

    max_ppr = max((a.ppr_score for a in analogs), default=1.0) or 1.0

    result_dicts: list[dict[str, Any]] = []
    for a in analogs:
        a_time = a.valid_time
        if hasattr(a_time, "tzinfo") and a_time.tzinfo is None:
            a_time = a_time.replace(tzinfo=valid_time.tzinfo)
        if a.region == region and abs((a_time - valid_time).total_seconds()) < 60:
            continue
        a_headroom = max(headroom - (a.headroom_delta or 0.0), 0.0)
        a_driver_types = _infer_analog_driver_types(a.regime, a_headroom)
        drv_sim = _driver_similarity_jaccard(query_driver_types, a_driver_types)
        ppr_norm = a.ppr_score / max_ppr
        quality = round(0.6 * ppr_norm + 0.4 * drv_sim, 4)
        result_dicts.append({
            "node_id": a.node_id,
            "valid_time": a.valid_time.isoformat() if hasattr(a.valid_time, "isoformat") else str(a.valid_time),
            "region": a.region,
            "regime": a.regime,
            "price_rrp": a.price_rrp,
            "demand_mw": a.demand_mw,
            "ppr_score": round(a.ppr_score, 4),
            "outcome": a.outcome,
            "cosine_similarity": a.cosine_similarity,
            "price_delta": a.price_delta,
            "demand_delta": a.demand_delta,
            "headroom_delta": a.headroom_delta,
            "regime_match": a.regime == regime,
            "match_reason": a.match_reason,
            "driver_types": sorted(a_driver_types),
            "driver_similarity": round(drv_sim, 4),
            "quality_score": quality,
        })

    result_dicts.sort(key=lambda x: x["quality_score"], reverse=True)
    return result_dicts


def rerank_analogs_with_drivers(
    analogs: list[dict[str, Any]],
    notices: list[dict[str, Any]] | None,
    driver_events: list[dict[str, Any]] | None,
    current_regime: str = "normal",
    current_headroom_mw: float = 1000.0,
) -> list[dict[str, Any]]:
    """Re-rank an already-computed analog list using full driver context.

    Called in routes_query.py after market driver attribution completes so that
    driver_events (constraint bindings, unit outages, rebid evidence) enrich
    the quality score beyond what was available during scatter_gather T3.

    Mutates and returns the same list with updated quality_score / driver_similarity.
    """
    if not analogs:
        return analogs

    q_types = _derive_query_driver_types(current_regime, current_headroom_mw, notices, driver_events)
    max_ppr = max((a.get("ppr_score", 0) or 0) for a in analogs) or 1.0

    for a in analogs:
        a_regime = a.get("regime", "normal")
        a_headroom = max(current_headroom_mw - (a.get("headroom_delta") or 0.0), 0.0)
        a_types = _infer_analog_driver_types(a_regime, a_headroom)
        drv_sim = _driver_similarity_jaccard(q_types, a_types)
        ppr_norm = (a.get("ppr_score", 0) or 0) / max_ppr
        a["driver_similarity"] = round(drv_sim, 4)
        a["quality_score"] = round(0.6 * ppr_norm + 0.4 * drv_sim, 4)
        if not a.get("driver_types"):
            a["driver_types"] = sorted(a_types)

    analogs.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    return analogs


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _parse_bucket_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = q * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac
