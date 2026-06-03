"""Query route — the core NLP endpoint.

POST /sessions/{id}/query   — submit a natural language query, get verdict + answer

Pipeline (all async, ~1s standard path):
  1. SecurityObserver pass 1 — input hygiene
  2. Decompose intent + entities
  3. ScatterGather — fetch live market + analogs + notices in parallel
  4. Derive verdict deterministically
  5. WhyEngine — build narrative
  6. SecurityObserver pass 4 — answer hygiene
  7. Persist Query row + return
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload
from app.api.deps import AEMOClient, Cache, get_current_user, get_db
from app.agents.scatter_gather import GatherResult, scatter_gather
from app.agents.claim_verifier import apply_verification, verify_answer
from app.agents.answer_planner import apply_plan_to_verdict, plan_answer
from app.agents.why_builder import build_seasonal_why, build_why
from app.agents.why_formatter import format_verdict
from app.agents.why_sources import SeasonalSources, assemble_why_sources
from app.core.trace import write_trace
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    FactualVerdict,
    IntentLabel,
    VerdictLabel,
)
# QueryDecomposition type hint (used in helper function signatures)
from app.core.schema import QueryDecomposition as _QueryDecompositionType  # noqa: F401
from app.data.aemo_live_client import AEMOLiveClient
from app.data.cache import MarketCache
from app.db.models import Query as QueryModel
from app.db.models import Session as SessionModel
from app.db.models import MarketEvent
from app.engines.decomposition import decompose as llm_decompose
from app.engines.query_restatement import restate_query
from app.engines.coverage_auditor import audit_coverage
from app.security.observer import get_observer, log_observer_event
from app.api.routes_security import append_observer_result
from app.api.metrics_registry import query_latency_ms, llm_decompose_latency_ms

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

# In-memory progress tracker — keyed by session_id (one active query per session)
# Each value: list of {"step": str, "message": str, "t_ms": int}
_query_progress: dict[str, list[dict]] = {}


def _progress(session_id: str, step: str, message: str, t0: float) -> None:
    """Write a progress step visible to the polling endpoint."""
    if session_id not in _query_progress:
        _query_progress[session_id] = []
    _query_progress[session_id].append({
        "step": step,
        "message": message,
        "t_ms": round((time.perf_counter() - t0) * 1000),
    })


class _EnrichBundle:
    """Holds all post-gather enrichment results from _enrich_context()."""
    __slots__ = (
        "temporal_evidence", "fuel_mix", "opennem_trend", "opennem_diurnal",
        "hist_dist", "period_stats", "intraday_prices", "intraday_fuel_timeline",
    )
    def __init__(self, temporal_evidence, fuel_mix, opennem_trend, opennem_diurnal,
                 hist_dist, period_stats, intraday_prices, intraday_fuel_timeline=None):
        self.temporal_evidence = temporal_evidence
        self.fuel_mix = fuel_mix
        self.opennem_trend = opennem_trend
        self.opennem_diurnal = opennem_diurnal
        self.hist_dist = hist_dist
        self.period_stats = period_stats
        self.intraday_prices = intraday_prices
        self.intraday_fuel_timeline = intraday_fuel_timeline


async def _enrich_context(
    gather: "GatherResult",
    decomp: "Any",
    region: str,
    db: "AsyncSession",
    session_id: str,
    t0: float,
    events: "list[dict]",    # mutated in-place
    sg_sources: "list[str]", # mutated in-place
    query_time: "datetime",
) -> "_EnrichBundle":
    """Run all post-gather enrichment in sequence: TemporalRAG, fuel mix, BOM 7-day,
    OpenNEM, historical price distribution, period stats, and intraday prices.
    Appends timeline events and source labels to events/sg_sources in-place."""
    from typing import Any as _Any

    # TemporalRAG — bitemporal evidence retrieval
    temporal_evidence: list[dict] = []
    try:
        from app.engines.temporalrag import TemporalQuery, retrieve as _trag_retrieve
        _anchor = gather.dispatch.valid_time if gather.dispatch else query_time
        _trag_query = TemporalQuery(
            valid_time_from=_anchor - timedelta(hours=4),
            valid_time_to=_anchor,
            system_time_at_query=query_time,
            region=region,
            max_docs=12,
        )
        async with db.begin_nested():
            _bundle = await _trag_retrieve(_trag_query, session=db)
        temporal_evidence = [
            {
                "doc_id": d.doc_id,
                "source_type": d.source_type,
                "valid_time": d.valid_time.isoformat(),
                "system_time": d.system_time.isoformat(),
                "known_before_query_time": d.system_time <= query_time,
                "relevance_score": round(d.relevance_score, 4),
                "retrieval_reason": _trag_retrieval_reason(d),
                "citation": d.citation,
                "content": {k: v for k, v in d.content.items()
                            if k in ("price_rrp", "demand_mw", "regime", "notice_type",
                                     "reason", "title", "region", "query_text", "verdict")},
            }
            for d in _bundle.docs
        ]
    except Exception as exc:
        logger.debug("TemporalRAG retrieval failed (non-fatal): %s", exc)

    # Fuel mix — per-fuel-type breakdown for source recommendation
    fuel_mix: dict | None = None
    try:
        from app.engines.fuel_mix import get_fuel_mix
        async with db.begin_nested():
            fuel_mix = await get_fuel_mix(
                region, db,
                weather=gather.weather,
                unit_events=gather.unit_events or None,
            )
    except Exception as exc:
        logger.debug("Fuel mix retrieval failed (non-fatal): %s", exc)

    # BOM 7-day forecast — injected when query requires forward weather scenario
    if decomp.requires_forecast and gather.weather is None:
        try:
            from app.mcp.bom_forecast_client import fetch_7day_forecast, forecast_to_scatter_context
            _bom = await fetch_7day_forecast(region)
            if _bom is not None:
                _bom_dict = forecast_to_scatter_context(_bom)
                from dataclasses import replace as _dc_replace
                gather = _dc_replace(gather, weather=_bom_dict)
                sg_sources.append("BOM_7DAY_FORECAST")
                events.append({
                    "step": "BOM_7DAY_FORECAST",
                    "t_ms": round((time.perf_counter() - t0) * 1000),
                    "region": region,
                    "days": len(_bom.days),
                    "heatwave_days": len(_bom.days_with_heatwave()),
                })
        except Exception as _bom_err:
            logger.debug("BOM 7-day forecast inject failed (non-fatal): %s", _bom_err)

    # OpenNEM trend/diurnal — real monthly + hourly data from OpenElectricity API
    opennem_trend: _Any | None = None
    opennem_diurnal: _Any | None = None
    _wants_opennem = decomp.requested_output in ("trend_analysis", "diurnal_analysis")
    if _wants_opennem:
        try:
            _progress(session_id, "opennem", "OpenNEM: fetching real market data...", t0)
            from app.mcp.opennem_client import get_trend_context, get_diurnal_context
            import asyncio as _asyncio
            if decomp.requested_output == "trend_analysis":
                opennem_trend = await _asyncio.wait_for(get_trend_context(region), timeout=8.0)
                sg_sources.append("OPENNEM_TREND")
            else:
                opennem_diurnal = await _asyncio.wait_for(get_diurnal_context(region), timeout=8.0)
                sg_sources.append("OPENNEM_DIURNAL")
        except Exception as _onem_err:
            logger.debug("OpenNEM fetch failed (non-fatal): %s", _onem_err)

    # Historical price distribution — for "is this cheap vs last year?" queries
    hist_dist: dict | None = None
    _wants_hist = (
        any(sq.get("type") == "historical_price_distribution" for sq in (decomp.sub_questions or []))
        or decomp.requires_history
        or decomp.intent.value in ("lookup", "explanation", "comparison", "action_recommendation")
    )
    if _wants_hist:
        try:
            async with db.begin_nested():
                from app.engines.historical_price import get_historical_price_distribution
                _anchor = gather.dispatch.valid_time if gather.dispatch else datetime.now(timezone.utc)
                _hist_period = next(
                    (sq.get("period", "last_year") for sq in (decomp.sub_questions or [])
                     if sq.get("type") == "historical_price_distribution"),
                    "last_year",
                )
                hist_dist = await get_historical_price_distribution(
                    db, region, _anchor, period=_hist_period,
                )
        except Exception as exc:
            logger.debug("Historical price distribution unavailable (non-fatal): %s", exc)

    # Specific period stats — direct aggregate for "what was the average in July 2023?"
    period_stats: dict | None = None
    _period_sq = next(
        (sq for sq in (decomp.sub_questions or []) if sq.get("type") == "specific_period_stats"),
        None,
    )
    if _period_sq:
        try:
            from app.engines.historical_price import get_period_stats
            period_stats = await get_period_stats(db, region, _period_sq["start"], _period_sq["end"])
        except Exception as exc:
            logger.debug("Period stats unavailable (non-fatal): %s", exc)

    # Intraday price history — for "earlier today wind was $15, why pay double now?"
    intraday_prices: list[dict] = []
    _has_intraday_sq = any(
        sq.get("type") == "intraday_price_cycle" for sq in (decomp.sub_questions or [])
    )
    if _has_intraday_sq:
        try:
            from app.db.session import db_session as _iday_session_factory
            from sqlalchemy import text as _text
            _now = datetime.now(timezone.utc)
            _today_start = _now.replace(hour=0, minute=0, second=0, microsecond=0)
            _cutoff = _now - timedelta(hours=1)
            async with _iday_session_factory() as _iday_session:
                _iday_result = await _iday_session.execute(_text("""
                    SELECT valid_time, price_rrp
                    FROM market_events
                    WHERE source = 'AEMO_DISPATCH_PRICE'
                      AND region = :region
                      AND price_rrp IS NOT NULL
                      AND valid_time >= :today_start
                      AND valid_time <= :cutoff
                    ORDER BY valid_time ASC
                    LIMIT 288
                """), {"region": region, "today_start": _today_start, "cutoff": _cutoff})
                _rows = _iday_result.fetchall()
            if _rows:
                for row in _rows:
                    _vt = row[0]
                    _vt_str = str(_vt)
                    try:
                        _hour = _vt.hour if hasattr(_vt, "hour") else int(_vt_str[11:13])
                    except Exception:
                        _hour = 0
                    intraday_prices.append({"time": _vt_str, "price": float(row[1]), "hour": _hour})
            events.append({
                "step": "INTRADAY_PRICES",
                "t_ms": round((time.perf_counter() - t0) * 1000),
                "region": region,
                "rows": len(intraday_prices),
            })
        except Exception as _iday_err:
            logger.debug("Intraday price fetch failed (non-fatal): %s", _iday_err)

    # Summarise fuel dispatch + deepen progress
    _fuel_by_type: dict[str, float] = {}
    if gather.unit_events:
        for ue in gather.unit_events:
            fuel = getattr(ue, "fuel_type", None) or "unknown"
            mw = float(getattr(ue, "total_cleared_mw", 0) or 0)
            _fuel_by_type[fuel] = round(_fuel_by_type.get(fuel, 0) + mw, 1)
    _binding_constraints = sum(
        1 for d in gather.driver_events if getattr(d, "constraint_id", None)
    ) if gather.driver_events else 0
    _deepen_parts = []
    if temporal_evidence: _deepen_parts.append(f"TemporalRAG {len(temporal_evidence)} docs")
    if fuel_mix: _deepen_parts.append("fuel mix")
    if hist_dist and hist_dist.get("available"): _deepen_parts.append("historical archive")
    if gather.weather: _deepen_parts.append("weather")
    _progress(session_id, "deepen",
              f"Deepening: {' · '.join(_deepen_parts) or 'evidence assembled'}...", t0)
    _progress(session_id, "plan", "Building evidence-grounded answer...", t0)

    _ev = {
        "step": "EVIDENCE_ASSEMBLED",
        "t_ms": round((time.perf_counter() - t0) * 1000),
        "temporal_docs": len(temporal_evidence),
        "fuel_sources": len((fuel_mix or {}).get("sources", [])),
        "fuel_dispatch_mw": _fuel_by_type or None,
        "binding_constraints": _binding_constraints or None,
        "analogs_matched": len(gather.analogs) if gather.analogs else 0,
        "driver_events": len(gather.driver_events) if gather.driver_events else 0,
    }
    if hist_dist and hist_dist.get("available"):
        _ev["hist_dist"] = {
            "period": hist_dist.get("period_label"),
            "median": round(hist_dist.get("median", 0), 2),
            "n_rows": hist_dist.get("count"),
        }
    events.append(_ev)

    # Intraday fuel timeline — for "why coal now vs wind earlier today?" queries
    intraday_fuel_timeline: dict | None = None
    _wants_fuel_tl = any(
        sq.get("type") == "intraday_fuel_timeline" for sq in (decomp.sub_questions or [])
    )
    if _wants_fuel_tl:
        try:
            from app.engines.fuel_mix import get_intraday_fuel_timeline, summarise_intraday_fuel_transition
            _raw_tl = await get_intraday_fuel_timeline(db, region, hours_back=12)
            intraday_fuel_timeline = summarise_intraday_fuel_transition(_raw_tl)
        except Exception as exc:
            logger.debug("Intraday fuel timeline unavailable (non-fatal): %s", exc)

    return _EnrichBundle(
        temporal_evidence=temporal_evidence,
        fuel_mix=fuel_mix,
        opennem_trend=opennem_trend,
        opennem_diurnal=opennem_diurnal,
        hist_dist=hist_dist,
        period_stats=period_stats,
        intraday_prices=intraday_prices,
        intraday_fuel_timeline=intraday_fuel_timeline,
    )


async def _check_tool_outputs(
    gather: "GatherResult",
    observer: "Any",
    user: "Any",
    db: "AsyncSession",
    query_id: str,
    trace_id: str,
) -> "list[dict]":
    """Build tool output manifest and run security gate 3.
    Returns the manifest (used later as tool_calls_log).
    Raises HTTPException(502) if the observer halts on anomalous data."""
    tool_outputs: list[dict] = []
    if gather.dispatch:
        tool_outputs.append({
            "source": "AEMO_DISPATCH_PRICE",
            "price_rrp": gather.dispatch.price_rrp,
            "demand_mw": gather.dispatch.demand_mw,
            "availability_mw": gather.dispatch.availability_mw,
        })
    for notice in gather.notices:
        tool_outputs.append({"source": "AEMO_NOTICE", **notice})
    for item in gather.news_items:
        tool_outputs.append({
            "source": "NEM_NEWS_RSS",
            "title": item.get("title", ""),
            "description": item.get("summary", ""),
            "link": item.get("link", ""),
        })
    if gather.weather:
        consensus = gather.weather.get("consensus", {})
        tool_outputs.append({
            "source": "WEATHER_CONSENSUS",
            "temperature_c": consensus.get("temperature_c"),
            "wind_speed_kmh": consensus.get("wind_speed_kmh"),
            "precipitation_mm": consensus.get("precipitation_mm"),
            "confidence": gather.weather.get("confidence"),
            "raw_ref": gather.weather.get("raw_ref", ""),
        })
    for driver in gather.driver_events:
        tool_outputs.append({"source": driver.get("source", "AEMO_MARKET_DRIVER"), **driver})
    for unit in gather.unit_events:
        tool_outputs.append({"source": unit.get("source", "AEMO_UNIT_DISPATCH"), **unit})

    tool_check = observer.pass_tool_output(tool_outputs)
    append_observer_result(tool_check, "tool_output")
    try:
        await log_observer_event(db, tool_check, user.tenant_id, query_id, trace_id)
    except Exception as _oe:
        logger.debug("Observer event log failed (non-fatal): %s", _oe)
    if tool_check.should_halt():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Tool output failed security validation: "
                f"{tool_check.signals[0].description if tool_check.signals else 'anomalous data'}"
            ),
        )
    return tool_outputs


class QueryRequest(BaseModel):
    text: str
    region: str = "NSW1"


class QueryResponse(BaseModel):
    query_id: str
    session_id: str
    intent: str
    verdict: FactualVerdict
    decomposition: dict
    viewport_type: str   # tells frontend which right-panel to render
    comparison_table: list[dict] | None = None   # populated for COMPARISON intent
    seasonal_summary: list[dict] | None = None
    analogs: list[dict] | None = None
    weather_consensus: dict | None = None
    temporal_evidence: list[dict] | None = None   # TemporalRAG top docs
    fuel_mix: dict | None = None                  # per-fuel-type breakdown
    historical_dist: dict | None = None           # historical price distribution (percentiles)
    evidence_quality: dict | None = None          # per-query trust strip summary
    provenance: list[dict] | None = None          # per-source SourceStatus records
    pipeline_events: list[dict] | None = None     # decision trace timeline steps
    suggested_questions: list[str] | None = None  # context-aware follow-up chips
    live_forecast: dict | None = None             # full 48-interval ensemble forecast for chart
    # Dispatch anchor — used by causal chain panel to build the correct time-indexed chain.
    # This is the interval the answer is ABOUT (historical for archive queries, live for current).
    dispatch_valid_time: str | None = None        # ISO — may differ from now() for historical queries
    dispatch_price_rrp: float | None = None       # $/MWh at dispatch_valid_time


@router.get("/sessions/{session_id}/progress")
async def get_query_progress(
    session_id: str,
    user: TokenPayload = Depends(get_current_user),
) -> dict:
    """Poll progress of the active query for this session.

    Returns list of step dicts while query is running; empty list when idle.
    Frontend polls every 500ms and shows messages after 5s elapsed.
    """
    steps = _query_progress.get(session_id, [])
    return {"session_id": session_id, "steps": steps, "active": bool(steps)}


@router.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def submit_query(
    session_id: str,
    body: QueryRequest,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    client: AEMOLiveClient = AEMOClient,
    cache: MarketCache = Cache,
):
    if not body.text.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query text is empty")
    if len(body.text) > 2000:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query too long (max 2000 chars)")

    query_id = f"qry-{uuid.uuid4().hex[:12]}"
    trace_id = f"trace-{uuid.uuid4().hex[:12]}"
    _t0 = time.perf_counter()
    _events: list[dict] = [{"step": "QUERY_RECEIVED", "t_ms": 0, "region": body.region, "text_len": len(body.text)}]

    # Initialise progress tracker for this session
    _query_progress[session_id] = []
    _progress(session_id, "security", "Security scan…", _t0)

    # --- 0. Security pass 1 — input hygiene ---
    observer = get_observer()
    input_check = observer.pass_input(body.text, user.tenant_id)
    append_observer_result(input_check, "input")
    try:
        await log_observer_event(db, input_check, user.tenant_id, query_id, trace_id)
    except Exception as _oe:
        logger.debug("Observer event log failed (non-fatal): %s", _oe)
    if input_check.should_halt():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input blocked by security observer: {input_check.signals[0].description if input_check.signals else 'risk threshold exceeded'}",
        )
    _events.append({"step": "SECURITY_INPUT", "t_ms": round((time.perf_counter() - _t0) * 1000), "result": "clean", "signals": len(input_check.signals)})

    # Verify session ownership
    from sqlalchemy import select
    result = await db.execute(
        select(SessionModel).where(
            SessionModel.id == session_id,
            SessionModel.tenant_id == user.tenant_id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    _VALID_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
    region = body.region.upper()
    if region not in _VALID_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported region '{region}'. Valid: {sorted(_VALID_REGIONS)}",
        )

    # --- 0.4. Session rolling context carry-forward (E2) ---
    # Load last 3 Q&A pairs for short/contextual follow-ups so multi-turn
    # conversations stay coherent over 5+ exchanges.
    _enriched_text = body.text
    _text_lower = body.text.lower().strip()
    _is_short = len(body.text.split()) <= 12
    _REGION_CODES = {"nsw1", "vic1", "qld1", "sa1", "tas1", "nsw", "vic", "qld", "sa", "tas"}
    _CONTEXT_STARTS = (
        "and ", "what about", "how about", "show me", "tell me",
        "also ", "instead", "rather", "compare",
    )
    _CONTEXT_WORDS = {
        "it", "that", "this", "there", "those", "same", "rather",
        "instead", "also", "other", "now", "currently",
    }
    _tokens = set(re.findall(r"\b[a-z0-9]+\b", _text_lower))
    _is_contextual = _is_short and (
        bool(_tokens & _CONTEXT_WORDS)
        or any(_text_lower.startswith(p) for p in _CONTEXT_STARTS)
        or bool(_tokens & _REGION_CODES)
        or (_text_lower.endswith("?") and len(body.text.split()) <= 6)
    )
    if _is_contextual:
        try:
            _prior_rows = (await db.execute(
                select(QueryModel)
                .where(QueryModel.session_id == session_id, QueryModel.tenant_id == user.tenant_id)
                .order_by(QueryModel.created_at.desc())
                .limit(3)
            )).scalars().all()
            if _prior_rows:
                _ctx_parts = [
                    f"Q{i+1}: {q.raw_query}"
                    for i, q in enumerate(reversed(_prior_rows))
                    if q.raw_query
                ]
                if _ctx_parts:
                    _enriched_text = (
                        f"[Session context: {'; '.join(_ctx_parts)}] "
                        f"[Current question]: {body.text}"
                    )
                    logger.debug(
                        "Session context applied: %d prior queries enriched", len(_ctx_parts)
                    )
        except Exception as _ctx_err:
            logger.debug("Session context lookup failed (non-fatal): %s", _ctx_err)

    # --- 0.5. Query restatement — extract sub-questions before decompose ---
    # Uses qwen3:14b /no_think (~1s). Falls back silently to empty result.
    restatement = await restate_query(_enriched_text)
    # Seed sub-questions into the decomposer text so it produces better
    # causal_targets and requested_output for multi-part queries.
    _decompose_text = restatement.seed_text(_enriched_text)

    # --- 1. Decompose (rules-first hybrid: rules → LLM enrichment → merge) ---
    _decomp_t0 = time.perf_counter()
    decomp = await llm_decompose(_decompose_text, region_hint=region, query_id=query_id)
    # Restore raw_query to original user text (not enriched/seeded version).
    if _decompose_text != body.text:
        decomp = decomp.model_copy(update={"raw_query": body.text})
    # Always populate sub_questions from the deterministic classifier if the
    # hybrid decomposer didn't already do it (e.g. Ollama down → rule-based only).
    if not decomp.sub_questions:
        from app.engines.decomposition import _classify_sub_questions
        _sq = _classify_sub_questions(body.text.lower(), decomp.requested_output or "")
        if _sq:
            decomp = decomp.model_copy(update={"sub_questions": _sq})
    llm_decompose_latency_ms.observe((time.perf_counter() - _decomp_t0) * 1000)

    # Intent-specific gather message — reveals the architecture in the demo
    _gather_msg_map = {
        "future_date_price_forecast": "Scatter-gather: AEMO dispatch · seasonal history · BOM 7-day forecast…",
        "trend_analysis":             "Scatter-gather: AEMO dispatch · 3yr price history · fuel mix trends…",
        "diurnal_analysis":           "Scatter-gather: AEMO dispatch · unit dispatch by hour · weather…",
        "fuel_source_recommendation": "Scatter-gather: AEMO dispatch · unit dispatch by fuel · LNN/LEAR/QRA…",
        "causal_explanation":         "Scatter-gather: AEMO dispatch · constraints · driver attribution · analogs…",
        "historical_analog_outcome":  "Scatter-gather: AEMO dispatch · 3yr analogs · historical context…",
        "regional_comparison":        "Scatter-gather: AEMO dispatch for all regions · interconnector flows…",
    }
    _gather_msg = _gather_msg_map.get(
        decomp.requested_output or "",
        "Scatter-gather agents: AEMO dispatch · LNN/LEAR/QRA forecast · analogs · constraints…",
    )
    _progress(session_id, "decompose", f"Intent: {decomp.intent.value} — {decomp.requested_output or 'routing'}…", _t0)
    _progress(session_id, "gather", _gather_msg, _t0)

    _events.append({
        "step": "DECOMPOSE",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "intent": decomp.intent.value,
        "regions": decomp.entities.get("regions", []),
        "requested_output": decomp.requested_output or "",
        "confidence": round(decomp.confidence, 2),
        "sub_questions": [sq.get("type") for sq in (decomp.sub_questions or [])],
        **({"clarifying": decomp.clarifying_question} if decomp.clarifying_question else {}),
    })

    # --- 1a. Adjacent query path (PARTIAL_SCOPE / EVIDENCE_BRIDGE / GEOGRAPHIC_REDIRECT) ---
    # These intents are handled entirely by adjacent_handlers — no scatter-gather needed.
    # The handler assembles a structured answer from reference data + any cached evidence.
    _ADJACENT_INTENTS = {
        IntentLabel.PARTIAL_SCOPE,
        IntentLabel.EVIDENCE_BRIDGE,
        IntentLabel.GEOGRAPHIC_REDIRECT,
    }
    if decomp.intent in _ADJACENT_INTENTS:
        from app.engines.adjacent_handlers import dispatch_adjacent_handler
        _adj_result = await dispatch_adjacent_handler(decomp, region, db)

        _adj_verdict = FactualVerdict(
            verdict=VerdictLabel.PARTIAL_SCOPE,
            action=ActionLabel.MONITOR,
            confidence=decomp.confidence,
            confidence_band=ConfidenceBand.MEDIUM if decomp.confidence >= 0.60 else ConfidenceBand.LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english=_adj_result.get("why_plain_english", decomp.clarifying_question or ""),
            answer_sections=_adj_result.get("sections", []),
            missing_data=_adj_result.get("missing_data", []),
            upgrade_path=_adj_result.get("upgrade_path", []),
            counterargument=(
                "GridVerdict only answered the NEM-relevant portion. "
                "External data sources are needed for the full answer — see 'Scope boundary' section."
            ),
            disclaimer=(
                "Partial scope response. NEM evidence from AEMO public data. "
                "External data sources cited are not verified in real-time. "
                "Not financial advice."
            ),
            trace_id=trace_id,
        )
        _events.append({
            "step": "ADJACENT_HANDLER",
            "t_ms": round((time.perf_counter() - _t0) * 1000),
            "intent": decomp.intent.value,
            "requested_output": decomp.requested_output or "",
            "geographic_market": decomp.geographic_market,
        })

        # Security pass on adjacent answer
        _adj_check = observer.pass_answer(_adj_verdict.model_dump(mode="json"))
        append_observer_result(_adj_check, "answer")
        if _adj_check.should_halt():
            _adj_verdict = FactualVerdict(
                verdict=VerdictLabel.INSUFFICIENT_DATA,
                action=ActionLabel.MONITOR,
                confidence=0.0,
                confidence_band=ConfidenceBand.VERY_LOW,
                as_of=datetime.now(timezone.utc),
                why_plain_english="Adjacent answer failed security validation.",
                counterargument="",
                trace_id=trace_id,
            )

        # Persist and return
        _adj_answer_dict = _adj_verdict.model_dump(mode="json")
        db.add(QueryModel(
            id=query_id, tenant_id=user.tenant_id, session_id=session_id,
            raw_query=body.text,
            decomposition=decomp.model_dump(mode="json"),
            answer=_adj_answer_dict,
            trace_id=trace_id,
            intent=decomp.intent.value,
            verdict=_adj_verdict.verdict.value,
            region=region,
        ))
        await write_trace(
            session=db, trace_id=trace_id, tenant_id=user.tenant_id,
            query_id=query_id, valid_time=datetime.now(timezone.utc),
            decomposition=decomp.model_dump(mode="json"),
            tool_calls=[{"source": "ADJACENT_HANDLER", "requested_output": decomp.requested_output}],
            answer=_adj_answer_dict,
            observer_result=_adj_check.to_dict(),
            prefill={"pipeline_events": _events},
        )
        await db.flush()
        session.updated_at = datetime.now(timezone.utc)

        _adj_suggested = _generate_followup_questions_adjacent(decomp, region)
        _events.append({"step": "COMPLETE", "t_ms": round((time.perf_counter() - _t0) * 1000)})
        _adj_decomp_dict = decomp.model_dump(mode="json")
        _adj_decomp_dict["trace_id"] = trace_id
        return QueryResponse(
            query_id=query_id,
            session_id=session_id,
            intent=decomp.intent.value,
            verdict=_adj_verdict,
            decomposition=_adj_decomp_dict,
            viewport_type=_intent_to_viewport(decomp.intent),
            pipeline_events=_events,
            suggested_questions=_adj_suggested,
        )

    # --- 1a. Ambiguity gate — return clarifying question instead of guessing ---
    # Fires when decomposer confidence is low AND a clarifying question is available.
    # Threshold 0.62: decomposer eval suite runs at 85%+ accuracy; below 0.62 is
    # genuine ambiguity where guessing produces worse answers than asking.
    # LOOKUP and OUT_OF_SCOPE are exempt — they can always be answered (or refused).
    _AMBIG_THRESHOLD = 0.62
    try:
        _decomp_conf = float(decomp.confidence)
    except (TypeError, ValueError):
        _decomp_conf = 1.0   # if confidence is not a float, don't fire the gate
    if (
        _decomp_conf < _AMBIG_THRESHOLD
        and decomp.clarifying_question
        and len(body.text.split()) > 8
        and decomp.intent not in (IntentLabel.LOOKUP, IntentLabel.OUT_OF_SCOPE)
    ):
        _clarif_verdict = FactualVerdict(
            verdict=VerdictLabel.NEEDS_CLARIFICATION,
            action=ActionLabel.ASK_CLARIFYING_QUESTION,
            confidence=decomp.confidence,
            confidence_band=ConfidenceBand.LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english=decomp.clarifying_question,
            counterargument="",
            missing_data=[],
            answer_sections=[
                {"title": "Clarification needed", "items": [decomp.clarifying_question]},
            ],
            answer_details={
                "headline": "Query is ambiguous — please clarify.",
                "ambiguities": decomp.ambiguities,
                "detected_intent": decomp.intent.value,
            },
        )
        _events.append({"step": "AMBIGUITY_GATE", "t_ms": round((time.perf_counter() - _t0) * 1000),
                         "confidence": round(decomp.confidence, 2), "fired": True})
        return QueryResponse(
            query_id=query_id,
            session_id=session_id,
            intent=decomp.intent.value,
            verdict=_clarif_verdict,
            decomposition=decomp.model_dump(mode="json"),
            viewport_type="answer",
            pipeline_events=_events,
        )

    # --- 1b. Security pass 2 — decomposition intent check ---
    decomp_check = observer.pass_decomposition(decomp.model_dump())
    append_observer_result(decomp_check, "decomposition")
    try:
        await log_observer_event(db, decomp_check, user.tenant_id, query_id, trace_id)
    except Exception as _oe:
        logger.debug("Observer event log failed (non-fatal): %s", _oe)
    if decomp_check.should_halt():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query intent blocked: {decomp_check.signals[0].description if decomp_check.signals else 'unsafe intent detected'}",
        )

    # --- 2. ScatterGather — parallel evidence collection ---
    # Comparison intent: gather multiple regions concurrently
    season_buckets = decomp.time_range.get("season_buckets") if decomp.time_range else None
    from app.mcp.weather_client import weather_query_relevant
    include_weather = weather_query_relevant(body.text, region=region)
    # Sprint Q: include pre-computed commentary events for queries needing historical context
    include_commentary = decomp.requires_history or decomp.requires_why or decomp.intent in (
        IntentLabel.RETROSPECTIVE, IntentLabel.EXPLANATION,
    )
    if season_buckets:
        return await _submit_seasonal_query(
            db=db,
            session=session,
            session_id=session_id,
            body=body,
            user=user,
            region=region,
            query_id=query_id,
            trace_id=trace_id,
            decomp=decomp,
            season_buckets=season_buckets,
            observer=observer,
        )

    # --- 2a. Historical routing (BKL-006 + BKL-013) ---
    # When the query is RETROSPECTIVE with a resolvable past anchor, route to the
    # archive DB rather than the live NEMWeb feed.  For multi-hop queries
    # (historical + forecast), run both and merge.
    #
    # IMPORTANT: "why is it NOT $X like YESTERDAY" is a LIVE query with historical
    # reference for context — it is NOT a retrospective. Detect negation comparison
    # patterns and keep them on the live path + historical price distribution (hist_dist).
    # Only RETROSPECTIVE intent ("what happened yesterday?") uses historical scatter.
    _negation_comparison = bool(
        decomp.requires_history
        and decomp.intent == IntentLabel.EXPLANATION
        and any(
            phrase in _text_lower
            for phrase in [
                "not $", "why isn't", "why is it not", "why hasn't", "lower than",
                "not as high", "not as elevated", "not like", "fallen from", "dropped from",
                "cheaper than", "below what", "below yesterday", "not $160", "not $100",
                "much lower", "why so low", "why so cheap", "used to be", "used to cost",
            ]
        )
    )
    _hist_anchor: "datetime | None" = None
    _try_historical = (
        decomp.intent == IntentLabel.RETROSPECTIVE
        or (
            decomp.requires_history
            and decomp.intent == IntentLabel.EXPLANATION
            and not _negation_comparison   # "not $X like yesterday" stays on live path
        )
    )
    if _try_historical:
        try:
            from app.engines.temporal_utils import resolve_time_anchor
            _time_range = decomp.time_range or {}
            _from_off = (
                _time_range.get("from_offset") or _time_range.get("from")
                if isinstance(_time_range, dict) else None
            )
            _hist_anchor = resolve_time_anchor(
                from_offset=_from_off,
                raw_query=body.text,
                now=datetime.now(timezone.utc),
            )
        except Exception as _ta_err:
            logger.debug("Historical anchor resolution failed (non-fatal): %s", _ta_err)

    comparison_gathers: dict[str, GatherResult] = {}
    _decomp_regions = [
        r.upper() for r in (decomp.entities.get("regions") or [])
        if r.upper() in {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
    ]
    # Run multi-region gather whenever the decomposer detected multiple regions
    # (covers COMPARISON intent AND queries like "how does NSW differ to other states?")
    _multi_region_query = decomp.intent == IntentLabel.COMPARISON or len(_decomp_regions) > 1

    if _hist_anchor is not None and not _multi_region_query:
        # Historical gather: pull evidence from DB archive at anchor_time
        from app.agents.scatter_gather import scatter_gather_historical, _merge_historical_and_live
        gather = await scatter_gather_historical(region, _hist_anchor, db)

        # Always merge live weather when the query explicitly mentions weather/wind/renewable —
        # scatter_historical has no weather task, so without this weather is always None.
        # Also merge live forecast context for forecast-needing queries.
        if include_weather or decomp.requires_forecast:
            try:
                import asyncio as _aio
                _live = await _aio.wait_for(
                    scatter_gather(
                        region, client, cache,
                        include_weather=include_weather,
                        include_commentary=False,
                    ),
                    timeout=10.0,
                )
                gather = _merge_historical_and_live(gather, _live)
            except Exception as _live_err:
                logger.debug("Historical weather/forecast merge failed (non-fatal): %s", _live_err)

        if gather.dispatch is None:
            # No archive data found — downgrade to insufficient data, don't run live gather
            # (the answer planner will explain the gap via _build_confidence_gap_explanation)
            logger.info(
                "Historical gather returned no dispatch data for %s @ %s — no archive rows",
                region, _hist_anchor.isoformat(),
            )

    elif _multi_region_query:
        comparison_regions = list(dict.fromkeys(_decomp_regions)) or [region]
        import asyncio as _asyncio
        results = await _asyncio.gather(
            *(scatter_gather(r, client, cache, include_weather=include_weather) for r in comparison_regions),
            return_exceptions=True,
        )
        for r, res in zip(comparison_regions, results):
            if isinstance(res, GatherResult):
                comparison_gathers[r] = res
        # Primary gather is the requested region (or first available)
        gather = comparison_gathers.get(region) or next(iter(comparison_gathers.values()), None)
        if gather is None:
            gather = await scatter_gather(
                region, client, cache,
                include_weather=include_weather,
                include_commentary=include_commentary,
            )
    else:
        gather: GatherResult = await scatter_gather(
            region, client, cache,
            include_weather=include_weather,
            include_commentary=include_commentary,
        )

    _sg_sources: list[str] = []
    if gather.dispatch: _sg_sources.append("AEMO_DISPATCH")
    if gather.analogs: _sg_sources.append(f"ANALOGS×{len(gather.analogs)}")
    if gather.notices: _sg_sources.append(f"NOTICES×{len(gather.notices)}")
    if gather.weather: _sg_sources.append("WEATHER")
    if comparison_gathers: _sg_sources.append(f"COMPARISON×{len(comparison_gathers)}")
    _dispatch_ts = gather.dispatch.valid_time.isoformat() if gather.dispatch else None
    _events.append({
        "step": "SCATTER_GATHER",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "sources": _sg_sources,
        "dispatch_price": round(gather.dispatch.price_rrp, 2) if gather.dispatch else None,
        "dispatch_interval": _dispatch_ts,
        "dispatch_source_url": "https://nemweb.com.au/Reports/Current/DispatchIS_Reports/",
        "analog_count": len(gather.analogs) if gather.analogs else 0,
        "dispatch_fresh": gather.dispatch_fresh if hasattr(gather, "dispatch_fresh") else None,
        "region_prices": {
            r: round(g.dispatch.price_rrp, 2)
            for r, g in comparison_gathers.items()
            if g.dispatch
        } if comparison_gathers else None,
        "region_intervals": {
            r: g.dispatch.valid_time.isoformat()
            for r, g in comparison_gathers.items()
            if g.dispatch
        } if comparison_gathers else None,
    })

    # --- 2b. Routing alignment check (adversarial critic for data anchor mismatch) ---
    # When the gathered dispatch is much older than now AND the intent is EXPLANATION,
    # the historical routing may have picked the wrong anchor. The user said "like yesterday"
    # as a comparison reference, not as the subject of the query.
    # Detect and self-correct by falling back to live scatter.
    if (
        gather.dispatch is not None
        and not gather.dispatch_fresh
        and decomp.intent == IntentLabel.EXPLANATION
        and _hist_anchor is not None
    ):
        _anchor_age_h = (datetime.now(timezone.utc) - gather.dispatch.valid_time).total_seconds() / 3600
        if _anchor_age_h > 2:
            # Data is more than 2 hours old for an EXPLANATION query — likely a routing mismatch.
            # The user probably wants TODAY's state explained, not 2h+ ago.
            # Self-correct: re-run live scatter and merge, preserving historical as context.
            logger.info(
                "Routing alignment check: historical anchor %s is %.1fh old for EXPLANATION intent — "
                "self-correcting to live scatter for %s",
                _hist_anchor.isoformat(), _anchor_age_h, region,
            )
            _events.append({
                "step": "ROUTING_CORRECTION",
                "t_ms": round((time.perf_counter() - _t0) * 1000),
                "reason": f"historical anchor {_anchor_age_h:.1f}h old for EXPLANATION intent — switching to live",
                "old_price": round(gather.dispatch.price_rrp, 2),
                "old_interval": gather.dispatch.valid_time.isoformat(),
            })
            try:
                from app.agents.scatter_gather import _merge_historical_and_live
                _live_corrected = await scatter_gather(
                    region, client, cache,
                    include_weather=include_weather,
                    include_commentary=include_commentary,
                )
                # Use live as primary; keep historical analogs, driver events, notices
                _corrected = _merge_historical_and_live(gather, _live_corrected)
                # Swap dispatch to live (main subject is current state)
                from dataclasses import replace as _dc_replace
                gather = _dc_replace(
                    _corrected,
                    dispatch=_live_corrected.dispatch,
                    dispatch_fresh=_live_corrected.dispatch_fresh,
                )
            except Exception as _rc_err:
                logger.debug("Routing correction live scatter failed (non-fatal): %s", _rc_err)

    if gather.dispatch:
        # Use savepoints (begin_nested) so a failing SELECT doesn't abort the main transaction.
        # A full rollback would undo previously committed observer events causing PK violations.
        try:
            async with db.begin_nested():
                gather.recent_dispatch = await _load_recent_dispatch_context(
                    db, region, gather.dispatch.valid_time
                )
        except Exception as exc:
            logger.debug("Recent dispatch context unavailable: %s", exc)
        try:
            async with db.begin_nested():
                from app.engines.driver_attribution import retrieve_market_drivers
                gather.driver_events = await retrieve_market_drivers(db, region, gather.dispatch.valid_time)
        except Exception as exc:
            logger.debug("Driver attribution unavailable: %s", exc)
        try:
            async with db.begin_nested():
                from app.engines.unit_attribution import retrieve_unit_dispatch
                gather.unit_events = await retrieve_unit_dispatch(db, region, gather.dispatch.valid_time)
        except Exception as exc:
            logger.debug("Unit attribution unavailable: %s", exc)

    # Sprint B: re-rank analogs with full driver context now that driver_events are loaded
    if gather.analogs and gather.dispatch:
        try:
            from app.engines.analog_retriever import rerank_analogs_with_drivers
            from domain.nem.adapter import classify_regime as _classify_regime
            _regime = _classify_regime(gather.dispatch.price_rrp, region)
            _headroom = max(gather.dispatch.availability_mw - gather.dispatch.demand_mw, 0.0)
            gather.analogs = rerank_analogs_with_drivers(
                gather.analogs, gather.notices, gather.driver_events, _regime, _headroom,
            )
        except Exception as exc:
            logger.debug("Analog driver re-rank unavailable (non-fatal): %s", exc)

    # --- Phases 2b–2e: Enrichment (TemporalRAG, fuel mix, BOM, OpenNEM, hist, intraday) ---
    _query_time = datetime.now(timezone.utc)
    _enrich = await _enrich_context(
        gather, decomp, region, db, session_id, _t0, _events, _sg_sources, _query_time,
    )
    temporal_evidence       = _enrich.temporal_evidence
    fuel_mix                = _enrich.fuel_mix
    opennem_trend           = _enrich.opennem_trend
    opennem_diurnal         = _enrich.opennem_diurnal
    hist_dist               = _enrich.hist_dist
    period_stats            = _enrich.period_stats
    intraday_prices         = _enrich.intraday_prices
    intraday_fuel_timeline  = _enrich.intraday_fuel_timeline

    # --- Security gate 3: tool output hygiene ---
    tool_outputs = await _check_tool_outputs(gather, observer, user, db, query_id, trace_id)

    # --- 3. Assemble why sources + build narrative ---
    why_sources = assemble_why_sources(decomp, gather, region)
    if intraday_prices:
        from dataclasses import replace as _dc_replace_ws
        why_sources = _dc_replace_ws(why_sources, intraday_prices=intraday_prices)
    evidence_quality = _build_evidence_quality(gather, temporal_evidence, fuel_mix, why_sources)
    provenance = (
        [s.to_dict() for s in gather.source_statuses.values()]
        if gather.source_statuses else None
    )
    why_output = build_why(why_sources)
    factual = format_verdict(why_output, why_sources, trace_id)

    # --- 3a. LNN evidence citation — inject EvidenceRefSchema when lnn_ltc is primary ---
    # When the LNN is the primary forecast model, its prediction is a first-class
    # evidence source and must appear in evidence_refs with a citable raw_ref.
    try:
        _live_fc = gather.live_forecast
        if (
            _live_fc
            and _live_fc.get("available")
            and _live_fc.get("primary_model") == "lnn_ltc"
        ):
            _lnn_fc_dict = next(
                (f for f in _live_fc.get("forecasts", []) if f.get("model") == "lnn_ltc"),
                None,
            )
            if _lnn_fc_dict and _lnn_fc_dict.get("p50"):
                from app.core.schema import EvidenceRefSchema
                _p50_val = _lnn_fc_dict["p50"][0] if isinstance(_lnn_fc_dict["p50"], list) else _lnn_fc_dict["p50"]
                _lnn_ev = EvidenceRefSchema(
                    source="LNN_LTC",
                    region=region,
                    interval=gather.dispatch.valid_time if gather.dispatch else datetime.now(timezone.utc),
                    field="p50_forecast",
                    value=float(_p50_val),
                    raw_ref=f"lnn_ltc:{region}:live-ltc-multistep",
                )
                _existing_refs = list(factual.evidence_refs or [])
                _existing_refs.append(_lnn_ev)
                factual = factual.model_copy(update={"evidence_refs": _existing_refs})
                logger.debug(
                    "LNN-LTC evidence injected for %s: p50=%.2f raw_ref=%s",
                    region, float(_p50_val), _lnn_ev.raw_ref,
                )
    except Exception as _lnn_cite_err:
        logger.debug("LNN evidence citation failed (non-fatal): %s", _lnn_cite_err)

    _events.append({
        "step": "VERDICT",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "verdict": factual.verdict.value,
        "action": factual.action.value if factual.action else None,
        "confidence": round(factual.confidence, 2),
        "band": factual.confidence_band.value if factual.confidence_band else None,
    })

    # --- 4. Claim Verifier — answer guard ---
    _cv_result = verify_answer(factual)
    factual = apply_verification(factual, _cv_result)
    try:
        factual = apply_plan_to_verdict(
            factual,
            plan_answer(
                why_sources,
                factual,
                analogs=gather.analogs,
                evidence_quality=evidence_quality,
                temporal_evidence=temporal_evidence,
                provenance=provenance,
                fuel_mix=fuel_mix,
                hist_dist=hist_dist,
                opennem_trend=opennem_trend,
                opennem_diurnal=opennem_diurnal,
                period_stats=period_stats,
                intraday_fuel_timeline=intraday_fuel_timeline,
            ),
        )
    except Exception as exc:
        # WARNING not debug — a planner failure produces wrong answers, not just missing enrichment
        logger.warning("Answer planner failed for %s (%s): %s", query_id, type(exc).__name__, exc)

    # Sprint U: per-sub-question confidence — weakest sub-question drives headline
    try:
        from app.agents.planner_helpers import apply_sub_question_scores
        factual = apply_sub_question_scores(factual, why_sources)
    except Exception as exc:
        logger.warning("Sub-question scoring failed for %s: %s", query_id, exc)
    _events.append({
        "step": "ANSWER_PLAN",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "planner": decomp.requested_output or "current_market_state",
        "claim_findings": len(_cv_result.findings),
    })
    if _cv_result.findings:
        logger.debug(
            "Claim verifier %s: %d finding(s) %s",
            query_id,
            len(_cv_result.findings),
            [f.rule for f in _cv_result.findings],
        )

    # --- 4.5. Adversarial critic — re-plan if answer misses sub-questions ---
    # Only fires when: low confidence + restatement detected gap risk + sub-questions available.
    # Uses qwen3:14b WITH thinking mode (~4-8s). Re-runs plan_answer only; no re-gather.
    if (
        factual.confidence < 0.6
        and not restatement.is_empty()
        and restatement.answer_gap_risk
    ):
        audit = await audit_coverage(
            restatement.sub_questions,
            factual.answer_sections or [],
            decomp.requested_output or "",
        )
        if audit.has_actionable_suggestion():
            logger.debug(
                "Coverage auditor re-routing %s → %s: %s",
                decomp.requested_output,
                audit.suggested_output,
                audit.reasoning,
            )
            _patched_decomp = decomp.model_copy(update={"requested_output": audit.suggested_output})
            _patched_sources = assemble_why_sources(_patched_decomp, gather, region)
            try:
                factual = apply_plan_to_verdict(
                    factual,
                    plan_answer(
                        _patched_sources,
                        factual,
                        analogs=gather.analogs,
                        evidence_quality=evidence_quality,
                        temporal_evidence=temporal_evidence,
                        provenance=provenance,
                        fuel_mix=fuel_mix,
                        hist_dist=hist_dist,
                        period_stats=period_stats,
                        intraday_fuel_timeline=intraday_fuel_timeline,
                    ),
                )
                decomp = _patched_decomp
            except Exception as exc:
                logger.warning("Critic re-plan failed (%s): %s", type(exc).__name__, exc)
        _events.append({
            "step": "COVERAGE_AUDIT",
            "t_ms": round((time.perf_counter() - _t0) * 1000),
            "re_routed": audit.has_actionable_suggestion(),
            "original_planner": decomp.requested_output,
            "suggested_planner": audit.suggested_output if audit.has_actionable_suggestion() else None,
        })

    # --- 5.5. Inject cross-region comparison narrative ---
    if comparison_gathers and len(comparison_gathers) > 1:
        _region_rows = sorted(
            [(r, g.dispatch.price_rrp) for r, g in comparison_gathers.items() if g.dispatch],
            key=lambda x: x[1],
        )
        if _region_rows:
            _cheapest_r, _cheapest_p = _region_rows[0]
            _priciest_r, _priciest_p = _region_rows[-1]
            _price_list = "  ·  ".join(
                f"{r} ${p:.0f}/MWh" for r, p in sorted(_region_rows, key=lambda x: x[0])
            )
            _comparison_para = (
                f"Cross-region snapshot: {_price_list}. "
                f"Cheapest: {_cheapest_r} at ${_cheapest_p:.0f}/MWh. "
                f"Priciest: {_priciest_r} at ${_priciest_p:.0f}/MWh. "
                f"Spread: ${(_priciest_p - _cheapest_p):.0f}/MWh."
            )
            factual = factual.model_copy(update={
                "why_plain_english": factual.why_plain_english + "\n\n" + _comparison_para
            })
            # Also inject as a dedicated answer section
            _existing = list(factual.answer_sections or [])
            _existing.insert(0, {
                "title": "Regional Comparison",
                "items": [
                    f"{r}: ${p:.2f}/MWh" for r, p in sorted(_region_rows, key=lambda x: x[0])
                ] + [
                    f"Spread {_cheapest_r}→{_priciest_r}: ${(_priciest_p - _cheapest_p):.0f}/MWh"
                ],
            })
            factual = factual.model_copy(update={"answer_sections": _existing})

    # --- 5.8. Confidence gap explainer — inject when LOW_CONFIDENCE or INSUFFICIENT_DATA ---
    if factual.verdict.value in ("LOW_CONFIDENCE", "INSUFFICIENT_DATA") and factual.confidence < 0.75:
        _gap_lines = _build_confidence_gap_explanation(factual, evidence_quality, gather)
        if _gap_lines:
            _existing_secs = list(factual.answer_sections or [])
            # Remove any existing "Missing" section and replace with gap explainer
            _existing_secs = [s for s in _existing_secs if s.get("title") != "Why low confidence"]
            _existing_secs.append({"title": "Why low confidence", "items": _gap_lines})
            factual = factual.model_copy(update={"answer_sections": _existing_secs})

    # --- 6. Security pass 4 — answer hygiene ---
    answer_check = observer.pass_answer(factual.model_dump(mode="json"))
    append_observer_result(answer_check, "answer")
    try:
        await log_observer_event(db, answer_check, user.tenant_id, query_id, trace_id)
    except Exception as _oe:
        logger.debug("Observer event log failed (non-fatal): %s", _oe)
    if answer_check.should_halt():
        # Downgrade to INSUFFICIENT_DATA rather than returning a bad answer
        factual = FactualVerdict(
            verdict=VerdictLabel.INSUFFICIENT_DATA,
            action=ActionLabel.MONITOR,
            confidence=0.0,
            confidence_band=ConfidenceBand.VERY_LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english="Answer failed security validation. Please rephrase your query.",
            counterargument="Security observer blocked this response.",
            trace_id=trace_id,
        )
    _events.append({
        "step": "SECURITY_OUTPUT",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "result": "blocked" if answer_check.should_halt() else "clean",
        "signals": len(answer_check.signals),
    })

    # --- 7. Persist query + trace ---
    answer_dict = factual.model_dump(mode="json")
    tool_calls_log = tool_outputs[:]  # tool outputs used as tool call log
    valid_time = (gather.dispatch.valid_time if gather.dispatch else datetime.now(timezone.utc))

    query_row = QueryModel(
        id=query_id,
        tenant_id=user.tenant_id,
        session_id=session_id,
        raw_query=body.text,
        decomposition=decomp.model_dump(mode="json"),
        answer=answer_dict,
        trace_id=trace_id,
        intent=decomp.intent.value,
        verdict=factual.verdict.value,
        region=region,
    )
    db.add(query_row)

    _events.append({"step": "COMPLETE", "t_ms": round((time.perf_counter() - _t0) * 1000)})
    _query_progress.pop(session_id, None)  # clear progress — response is on its way
    await write_trace(
        session=db,
        trace_id=trace_id,
        tenant_id=user.tenant_id,
        query_id=query_id,
        valid_time=valid_time,
        decomposition=decomp.model_dump(mode="json"),
        tool_calls=tool_calls_log,
        answer=answer_dict,
        observer_result=answer_check.to_dict(),
        prefill={"pipeline_events": _events},
    )

    await db.flush()

    # Update session updated_at
    session.updated_at = datetime.now(timezone.utc)

    viewport_type = _intent_to_viewport(decomp.intent)

    # Build comparison table when multiple regions were gathered
    comparison_table = None
    if comparison_gathers and len(comparison_gathers) > 1:
        from app.agents.why_sources import assemble_why_sources as _aws
        rows = []
        for r, g in sorted(comparison_gathers.items()):
            dp = g.dispatch
            if dp:
                rows.append({
                    "region": r,
                    "price_rrp": dp.price_rrp,
                    "demand_mw": dp.demand_mw,
                    "availability_mw": dp.availability_mw,
                    "headroom_mw": max(dp.availability_mw - dp.demand_mw, 0.0),
                    "regime": _aws(decomp, g, r).current.regime,
                    "is_fresh": g.dispatch_fresh,
                    "valid_time": dp.valid_time.isoformat(),
                })
        comparison_table = rows if rows else None

    query_latency_ms.observe((time.perf_counter() - _t0) * 1000)
    _decomp_dict = decomp.model_dump(mode="json")
    _decomp_dict["trace_id"] = trace_id
    if not restatement.is_empty():
        _decomp_dict["restatement"] = {
            "sub_questions": restatement.sub_questions,
            "primary_intent": restatement.primary_intent,
            "answer_gap_risk": restatement.answer_gap_risk,
        }

    # Generate context-aware follow-up question chips
    _suggested = _generate_followup_questions(decomp, factual, region, gather, hist_dist)

    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        intent=decomp.intent.value,
        verdict=factual,
        decomposition=_decomp_dict,
        viewport_type=viewport_type,
        comparison_table=comparison_table,
        seasonal_summary=None,
        analogs=gather.analogs,
        weather_consensus=gather.weather,
        temporal_evidence=temporal_evidence or None,
        fuel_mix=fuel_mix,
        historical_dist=hist_dist,
        evidence_quality=evidence_quality,
        provenance=provenance,
        pipeline_events=_events,
        suggested_questions=_suggested or None,
        live_forecast=gather.live_forecast if gather.live_forecast and gather.live_forecast.get("available") else None,
        dispatch_valid_time=gather.dispatch.valid_time.isoformat() if gather.dispatch else None,
        dispatch_price_rrp=float(gather.dispatch.price_rrp) if gather.dispatch else None,
    )


async def _submit_seasonal_query(
    db: AsyncSession,
    session: SessionModel,
    session_id: str,
    body: QueryRequest,
    user: TokenPayload,
    region: str,
    query_id: str,
    trace_id: str,
    decomp,
    season_buckets: list[dict],
    observer,
) -> QueryResponse:
    """Seasonal historical path: archive aggregate retrieval, no live scatter-gather."""
    from app.engines.analog_retriever import retrieve_seasonal_summary

    seasonal_summary = await retrieve_seasonal_summary(db, region, season_buckets)
    why_output = build_seasonal_why(SeasonalSources(region, season_buckets, seasonal_summary))
    factual = FactualVerdict(
        verdict=VerdictLabel.LOW_CONFIDENCE,
        action=ActionLabel.MONITOR,
        confidence=why_output.confidence,
        confidence_band=ConfidenceBand.LOW,
        as_of=datetime.now(timezone.utc),
        why_plain_english=why_output.why_plain_english,
        evidence_refs=why_output.evidence_refs,
        counterargument=why_output.counterargument,
        missing_data=why_output.missing_data,
        trace_id=trace_id,
    )
    _cv_result = verify_answer(factual)
    factual = apply_verification(factual, _cv_result)
    answer_check = observer.pass_answer(factual.model_dump(mode="json"))
    append_observer_result(answer_check, "answer")
    if answer_check.should_halt():
        factual = FactualVerdict(
            verdict=VerdictLabel.INSUFFICIENT_DATA,
            action=ActionLabel.MONITOR,
            confidence=0.0,
            confidence_band=ConfidenceBand.VERY_LOW,
            as_of=datetime.now(timezone.utc),
            why_plain_english="Seasonal answer failed security validation. Please rephrase your query.",
            counterargument="Security observer blocked this response.",
            trace_id=trace_id,
        )

    answer_dict = factual.model_dump(mode="json")
    db.add(QueryModel(
        id=query_id,
        tenant_id=user.tenant_id,
        session_id=session_id,
        raw_query=body.text,
        decomposition=decomp.model_dump(mode="json"),
        answer=answer_dict,
        trace_id=trace_id,
        intent=decomp.intent.value,
        verdict=factual.verdict.value,
        region=region,
    ))
    await write_trace(
        session=db,
        trace_id=trace_id,
        tenant_id=user.tenant_id,
        query_id=query_id,
        valid_time=datetime.now(timezone.utc),
        decomposition=decomp.model_dump(mode="json"),
        tool_calls=[{"source": "AEMO_ARCHIVE_DISPATCH_PRICE", "seasonal_summary": seasonal_summary}],
        answer=answer_dict,
        observer_result=answer_check.to_dict(),
    )
    await db.flush()
    session.updated_at = datetime.now(timezone.utc)

    return QueryResponse(
        query_id=query_id,
        session_id=session_id,
        intent=decomp.intent.value,
        verdict=factual,
        decomposition=decomp.model_dump(mode="json"),
        viewport_type="retrospective",
        comparison_table=None,
        seasonal_summary=seasonal_summary,
        analogs=None,
        weather_consensus=None,
    )



def _trag_retrieval_reason(doc) -> str:
    """Human-readable reason why this TemporalRAG document was retrieved."""
    score = doc.relevance_score
    src = doc.source_type.replace("_", " ")
    tier = "high" if score >= 0.8 else "moderate" if score >= 0.5 else "low"
    return f"{tier} relevance {src} ({score:.3f})"


def _intent_to_viewport(intent: IntentLabel) -> str:
    return {
        IntentLabel.ACTION_RECOMMENDATION: "verdict",
        IntentLabel.EXPLANATION: "why",
        IntentLabel.RETROSPECTIVE: "retrospective",
        IntentLabel.LOOKUP: "market_state",
        IntentLabel.COUNTERFACTUAL: "counterfactual",
        IntentLabel.TRACE_REPLAY: "trace_replay",
        IntentLabel.OUT_OF_SCOPE: "out_of_scope",
        IntentLabel.COMPARISON: "comparison",
        # Adjacent query intents — use dedicated partial-scope viewport
        IntentLabel.PARTIAL_SCOPE: "partial_scope",
        IntentLabel.EVIDENCE_BRIDGE: "partial_scope",
        IntentLabel.GEOGRAPHIC_REDIRECT: "partial_scope",
    }.get(intent, "verdict")


def _generate_followup_questions_adjacent(
    decomp: "QueryDecomposition",
    region: str,
) -> list[str]:
    """Generate context-aware follow-up questions for adjacent-intent responses."""
    requested = decomp.requested_output or ""
    intent = decomp.intent.value if decomp.intent else ""
    questions: list[str] = []

    if requested == "solar_household_context":
        questions = [
            f"What is the current {region} spot price right now?",
            f"How does the midday {region} price compare to the 6pm peak?",
            "Is SA spot cannibalisation worse than NSW for solar returns?",
        ]
    elif requested == "renewable_investment_price_context":
        questions = [
            f"What does AEMO ISP Step Change project for {region} prices in 2030?",
            f"How often does {region} see prices above $200/MWh — what's the spike risk?",
            "Compare NSW and SA for wind vs solar investment returns",
        ]
    elif requested == "geographic_redirect":
        geo = decomp.geographic_market or "WA"
        questions = [
            f"How does SA differ from NSW in renewable penetration?",
            f"What is the current spot price in SA (closest NEM analog to {geo})?",
            "Compare all NEM regions by price and renewable share right now",
        ]
    elif requested in ("policy_evidence_bridge",):
        questions = [
            f"What happened to {region} prices when Hazelwood closed?",
            "What is the AEMO ISP Step Change vs Slow Change price difference in 2030?",
            f"How much coal capacity is left in {region}?",
        ]
    elif requested in ("macro_mechanism_bridge",):
        # Real-user questions: connect the mechanism to something they can act on
        questions = [
            f"What is the current {region} spot price and which fuel type is setting it?",
            f"Is {region} electricity cheap or expensive compared to last year?",
            "Should I sign a fixed-rate energy contract now or stay on spot?",
        ]
    elif requested in ("gas_electricity_nexus",):
        questions = [
            f"Why is the {region} price elevated right now?",
            "What is the current east coast gas hub price?",
            "How often does gas set the NEM spot price vs coal?",
        ]
    elif requested in ("fiscal_budget_bridge",):
        questions = [
            "What is the Capacity Investment Scheme and how does it affect NEM prices?",
            "When will Rewiring the Nation transmission unlock the New England REZ?",
            f"What is the current {region} renewable penetration?",
        ]
    else:
        questions = [
            f"What is the current {region} spot price?",
            f"Why is {region} at this price right now?",
            "Compare all NEM regions by price right now",
        ]

    return questions[:3]


async def _load_recent_dispatch_context(
    db: AsyncSession,
    region: str,
    anchor_time: datetime,
    lookback_minutes: int = 70,
) -> list[dict]:
    """Load recent persisted dispatch rows for trend-aware answers."""
    if anchor_time.tzinfo is None:
        anchor_time = anchor_time.replace(tzinfo=timezone.utc)
    start = anchor_time - timedelta(minutes=lookback_minutes)
    stmt = (
        select(MarketEvent)
        .where(
            MarketEvent.source == "AEMO_DISPATCH_PRICE",
            MarketEvent.region == region.upper(),
            MarketEvent.valid_time >= start,
            MarketEvent.valid_time <= anchor_time,
        )
        .order_by(desc(MarketEvent.valid_time))
        .limit(32)
    )
    try:
        rows = (await db.execute(stmt)).scalars().all()
    except Exception as exc:
        logger.debug("Recent dispatch context unavailable for %s: %s", region, exc)
        return []

    return [
        {
            "valid_time": r.valid_time,
            "price_rrp": r.price_rrp,
            "demand_mw": r.demand_mw,
            "availability_mw": r.availability_mw,
            "headroom_mw": (
                max((r.availability_mw or 0.0) - (r.demand_mw or 0.0), 0.0)
                if r.availability_mw is not None and r.demand_mw is not None
                else None
            ),
            "raw_ref": r.raw_ref,
        }
        for r in rows
    ]


def _build_evidence_quality(gather, temporal_evidence: list, fuel_mix, why_sources) -> dict:
    """Compute a per-query evidence quality summary for the frontend trust strip."""
    # Dispatch freshness
    if gather.dispatch:
        dispatch_status = "fresh" if gather.dispatch_fresh else "stale"
        staleness = int(
            (datetime.now(timezone.utc) - gather.dispatch.valid_time).total_seconds()
        )
    else:
        dispatch_status = "offline"
        staleness = None

    # Notices
    notice_count = len(gather.notices)
    if notice_count > 0:
        notices_status = "stale" if gather.notices_stale else "found"
    elif gather.notices_stale:
        notices_status = "stale"
    else:
        notices_status = "none"

    # Analogs
    analog_count = len(gather.analogs)
    analogs_status = "ok" if analog_count >= 3 else ("thin" if analog_count > 0 else "none")

    # TemporalRAG
    temporal_count = len(temporal_evidence)

    # Forecast/model availability — read from per-model detail when present
    _model_map = {m.model: m for m in (why_sources.forecast.model_detail or [])}
    lear_ok = bool(_model_map.get("lear") and _model_map["lear"].available)
    qra_ok  = bool(_model_map.get("qra")  and _model_map["qra"].available)
    lnn_ok  = bool(_model_map.get("lnn")  and _model_map["lnn"].available)

    # Driver/constraint events
    constraint_count = len(why_sources.drivers.binding_constraints)

    # Unit dispatch
    unit_count = len(gather.unit_events)

    return {
        "dispatch": {"status": dispatch_status, "staleness_seconds": staleness},
        "notices": {"status": notices_status, "count": notice_count},
        "analogs": {"status": analogs_status, "count": analog_count},
        "temporal_rag": {
            "count": temporal_count,
            "status": "ok" if temporal_count > 0 else "none",
        },
        "constraints": {
            "count": constraint_count,
            "status": "ok" if constraint_count > 0 else "none",
        },
        "unit_dispatch": {
            "count": unit_count,
            "status": "ok" if unit_count > 0 else "none",
        },
        "models": {"lear": lear_ok, "lnn": lnn_ok, "qra": qra_ok},
        "rebids": {"status": "not_ingested"},
        "source_coverage": why_sources.source_coverage,
    }


def _generate_followup_questions(
    decomp: "QueryDecomposition",
    factual: "FactualVerdict",
    region: str,
    gather: "GatherResult",
    hist_dist: dict | None,
) -> list[str]:
    """Deterministically generate 3 context-aware follow-up question chips.

    Based on the current intent, verdict, and evidence gaps — no LLM call.
    """
    questions: list[str] = []
    intent = decomp.intent.value
    verdict = factual.verdict.value if factual.verdict else "UNKNOWN"
    price = gather.dispatch.price_rrp if gather.dispatch else None
    regime = gather.dispatch_regime if hasattr(gather, "dispatch_regime") else None

    requested_output = decomp.requested_output or ""
    raw_query = (decomp.raw_query or "").lower()
    _future_date = (decomp.time_range or {}).get("future_date_ref", False)

    # Q1: Primary follow-up — most contextually useful
    if _future_date:
        # User asked about a future date → connect to historical analogs and seasonal data
        questions.append(f"What were {region} prices on the same day last year?")
    elif "earlier today" in raw_query or "this morning" in raw_query or "pay double" in raw_query:
        # Diurnal comparison → explain the evening peak pattern
        questions.append(f"What is the typical {region} evening peak price from 5–8pm?")
    elif intent in ("lookup", "comparison"):
        questions.append(f"Why is {region} at this price right now?")
    elif intent == "explanation" and price and price > 150:
        questions.append(f"How long will {region} stay above $150/MWh?")
    elif intent == "explanation":
        questions.append(f"What is the {region} price forecast for the next hour?")
    elif intent in ("action_recommendation",) and requested_output == "fuel_source_recommendation":
        questions.append(f"What is the current {region} spot price and which fuel is marginal?")
    elif intent in ("action_recommendation",):
        questions.append(f"What is the current {region} spot price?")
    else:
        questions.append(f"Why is {region} at this price right now?")

    # Q2: Forward-looking context — tied to the specific question asked
    if _future_date:
        questions.append(f"What is the typical June price range in {region} — P10, P50, P90?")
    elif "earlier today" in raw_query or "this morning" in raw_query:
        questions.append(f"Is the current {region} price cheap or expensive vs last year?")
    elif price and price > 200:
        questions.append(f"Will {region} price drop below $200 in the next 30 minutes?")
    elif price and price < 60:
        questions.append(f"Is {region} unusually cheap right now — how does it compare to last year?")
    else:
        questions.append(f"What is the {region} price forecast for the next 30 minutes?")

    # Q3: Actionable / cross-region — real decision value
    if _future_date:
        questions.append(f"What weather is forecast for {region} next week that could affect prices?")
    elif "earlier today" in raw_query or "pay double" in raw_query:
        questions.append("Should I shift my load to morning hours when solar is generating?")
    elif intent == "comparison":
        questions.append("Which NEM state has the cheapest energy to buy right now?")
    elif hist_dist and hist_dist.get("classification") in ("elevated", "high", "spike"):
        questions.append(f"Is this {region} price spike going to persist or resolve soon?")
    elif requested_output == "fuel_source_recommendation":
        questions.append("When does wind generation typically drop and push prices up in NSW?")
    else:
        questions.append(f"Compare {region} to all other NEM states right now.")

    return questions[:3]


def _build_confidence_gap_explanation(
    factual: "FactualVerdict",
    evidence_quality: dict | None,
    gather: "GatherResult",
) -> list[str]:
    """Explain specifically WHY confidence is low, and what would resolve it.

    Returns a list of bullet points. Each bullet names the gap, why it matters,
    and how the system would behave differently if data were available.
    """
    lines: list[str] = []
    eq = evidence_quality or {}

    # Dispatch freshness
    dispatch_status = eq.get("dispatch", {}).get("status", "unknown")
    if dispatch_status in ("stale", "offline", "unknown"):
        lines.append(
            f"Live dispatch data is {dispatch_status} — price may not reflect current market. "
            "Fresh dispatch data (<5 min) would raise confidence by ~15%."
        )

    # Notices
    notices = eq.get("notices", {})
    if not notices.get("count"):
        lines.append(
            "No AEMO market notices found — cannot confirm generator outages, constraints, "
            "or market interventions. A notice would narrow the causal explanation."
        )

    # Constraints
    constraints = eq.get("constraints", {})
    if not constraints.get("count"):
        lines.append(
            "Binding constraint data unavailable for this interval (archive covers to 2024). "
            "Constraint violations would shift regime classification and affect confidence."
        )

    # Models
    models = eq.get("models", {})
    if not models.get("lnn"):
        lines.append(
            "LNN model is not yet trained — no machine-learning forecast for this region. "
            "This removes one ensemble member from the confidence calculation."
        )
    if not models.get("lear") and not models.get("qra"):
        lines.append(
            "LEAR and QRA models unavailable — quantile forecast uncertainty is unquantified. "
            "These models need dispatch history to calibrate."
        )

    # Missing data from answer sections
    missing_items = [
        m for s in (factual.answer_sections or [])
        if s.get("title") == "Missing"
        for m in s.get("items", [])
    ]
    for item in missing_items[:2]:
        lines.append(f"Missing: {item} — ingesting this would provide additional causal evidence.")

    if not lines:
        lines.append(
            f"Confidence is {round(factual.confidence * 100)}% due to limited corroborating evidence "
            "across the required data sources. Adding dispatch constraints and live market notices "
            "would improve confidence."
        )

    return lines[:5]


# ── BKL-027: Answer export endpoint ──────────────────────────────────────────

@router.get("/query/{query_id}/export")
async def export_query_analysis(
    query_id: str,
    format: str = "markdown",
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a query analysis as a formatted markdown report or JSON.

    Suitable for inclusion in regulatory submissions, board reports,
    and market analysis documentation. Includes full evidence table,
    claim map, and model provenance.
    """
    from fastapi.responses import Response
    row = (await db.execute(
        select(QueryModel)
        .where(QueryModel.id == query_id, QueryModel.tenant_id == user.tenant_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Query not found")

    if format == "json":
        return {
            "query_id": row.id,
            "raw_query": row.raw_query,
            "region": row.region,
            "intent": row.intent,
            "verdict": row.verdict,
            "answer": row.answer,
            "decomposition": row.decomposition,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    md = _build_markdown_export(row)
    return Response(content=md, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="gridverdict_{query_id}.md"'})


def _build_markdown_export(row: "QueryModel") -> str:
    """Build a self-contained markdown report from a stored query row."""
    answer = row.answer or {}
    sections = answer.get("answer_sections") or []
    evidence_refs = answer.get("evidence_refs") or answer.get("evidence_manifest") or []
    claim_map = answer.get("claim_map") or []
    confidence = answer.get("confidence", 0)
    verdict = answer.get("verdict", {}).get("value", row.verdict or "UNKNOWN") if isinstance(answer.get("verdict"), dict) else (row.verdict or "UNKNOWN")
    why_text = answer.get("why_plain_english", "")

    ts = row.created_at.strftime("%Y-%m-%d %H:%M AEST") if row.created_at else "Unknown"

    lines = [
        "# GridVerdict Market Analysis Report",
        "",
        f"**Query:** {row.raw_query}",
        f"**Date:** {ts}",
        f"**Region:** {row.region or 'N/A'}",
        f"**Intent:** {row.intent or 'N/A'}",
        f"**Verdict:** {verdict}  |  Confidence: {confidence:.0%}",
        "",
        "---",
        "",
    ]

    if why_text:
        lines += ["## Analysis", "", why_text, ""]

    for section in sections[:5]:
        title = section.get("title", "")
        items = section.get("items") or []
        if title and items:
            lines.append(f"## {title}")
            lines.append("")
            for item in items[:5]:
                lines.append(f"- {item}")
            lines.append("")

    if evidence_refs:
        lines += ["## Evidence References", "",
                  "| Source | Interval | Field | Value |",
                  "|---|---|---|---|"]
        for ref in evidence_refs[:10]:
            src = ref.get("source_table", ref.get("source", "—"))
            interval = ref.get("interval", "—")[:19] if ref.get("interval") else "—"
            field = ref.get("field", "—")
            value = str(ref.get("value", ref.get("raw_file_hash", "—")))[:40]
            lines.append(f"| {src} | {interval} | {field} | {value} |")
        lines.append("")

    if claim_map:
        lines += ["## Claim Map", "",
                  "| Type | Tier | Evidence ID |",
                  "|---|---|---|"]
        for claim in claim_map[:8]:
            ctype = claim.get("claim_type", claim.get("type", "—"))
            tier = claim.get("driver_tier", claim.get("tier", "—"))
            ev = claim.get("evidence_ref_id", claim.get("evidence_ref", "—"))
            lines.append(f"| {ctype} | {tier} | {ev} |")
        lines.append("")

    lines += [
        "---",
        "",
        "*Generated by GridVerdict. Decision-support only — not financial advice. "
        "Verify with AEMO evidence before acting. "
        f"Trace ID: {answer.get('trace_id', 'N/A')}*",
    ]

    return "\n".join(lines)


# ── BKL-028: Session research context helpers ─────────────────────────────────

async def _load_session_findings(session_id: str, cache: "MarketCache") -> list[str]:
    """Load accumulated research findings for this session (up to 5)."""
    try:
        findings = await cache.get(f"session_findings_{session_id}")
        return findings if isinstance(findings, list) else []
    except Exception:
        return []


async def _save_session_finding(
    session_id: str,
    cache: "MarketCache",
    factual: "FactualVerdict",
    region: str,
    query_text: str,
) -> None:
    """Extract and persist the single most important finding from this answer."""
    try:
        # Try confirmed/supported claims first
        finding: str | None = None
        for claim in (factual.claim_map or []):
            tier = getattr(claim, "driver_tier", None) or (
                claim.get("driver_tier") if isinstance(claim, dict) else None
            )
            if tier in ("confirmed", "supported"):
                desc_val = (
                    claim.get("description") if isinstance(claim, dict)
                    else getattr(claim, "description", None)
                )
                if desc_val:
                    finding = f"{region}: {str(desc_val)[:100]}"
                    break
        if not finding and factual.why_plain_english:
            # Fall back to first sentence of the answer
            first = factual.why_plain_english.split(".")[0]
            if len(first) > 20:
                finding = first[:120]
        if not finding:
            return

        existing = await _load_session_findings(session_id, cache)
        # Avoid exact duplicates; keep max 5 findings
        if finding not in existing:
            existing.append(finding)
        updated = existing[-5:]
        await cache.set(f"session_findings_{session_id}", updated)
    except Exception:
        pass
