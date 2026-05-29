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
    _is_contextual = _is_short and (
        any(w in _text_lower for w in _CONTEXT_WORDS)
        or any(_text_lower.startswith(p) for p in _CONTEXT_STARTS)
        or any(r in _text_lower for r in _REGION_CODES)
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
    _hist_anchor: "datetime | None" = None
    _try_historical = (
        decomp.intent == IntentLabel.RETROSPECTIVE
        or (decomp.requires_history and decomp.intent == IntentLabel.EXPLANATION)
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

        # Multi-hop: also run a live gather for the forecast half of the query
        if decomp.requires_forecast and gather.dispatch is not None:
            try:
                import asyncio as _aio
                _live = await _aio.wait_for(
                    scatter_gather(region, client, cache, include_weather=False),
                    timeout=8.0,
                )
                gather = _merge_historical_and_live(gather, _live)
            except Exception as _live_err:
                logger.debug("Multi-hop live gather failed (non-fatal): %s", _live_err)

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

    if gather.dispatch:
        gather.recent_dispatch = await _load_recent_dispatch_context(db, region, gather.dispatch.valid_time)
        try:
            from app.engines.driver_attribution import retrieve_market_drivers
            gather.driver_events = await retrieve_market_drivers(db, region, gather.dispatch.valid_time)
        except Exception as exc:
            logger.debug("Driver attribution unavailable: %s", exc)
        try:
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

    # --- 2b. TemporalRAG — bitemporal evidence retrieval ---
    temporal_evidence: list[dict] = []
    _query_time = datetime.now(timezone.utc)
    try:
        from datetime import timedelta
        from app.engines.temporalrag import TemporalQuery, retrieve as _trag_retrieve

        _anchor = gather.dispatch.valid_time if gather.dispatch else _query_time
        _trag_query = TemporalQuery(
            valid_time_from=_anchor - timedelta(hours=4),
            valid_time_to=_anchor,
            system_time_at_query=_query_time,
            region=region,
            max_docs=12,
        )
        _bundle = await _trag_retrieve(_trag_query, session=db)
        temporal_evidence = [
            {
                "doc_id": d.doc_id,
                "source_type": d.source_type,
                "valid_time": d.valid_time.isoformat(),
                "system_time": d.system_time.isoformat(),
                "known_before_query_time": d.system_time <= _query_time,
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

    # --- 2c. Fuel mix — per-fuel-type breakdown for source recommendation ---
    fuel_mix: dict | None = None
    try:
        from app.engines.fuel_mix import get_fuel_mix
        fuel_mix = await get_fuel_mix(region, db, weather=gather.weather)
    except Exception as exc:
        logger.debug("Fuel mix retrieval failed (non-fatal): %s", exc)

    # --- 2d. Historical price distribution — for "is this cheap vs last year?" queries ---
    hist_dist: dict | None = None
    _has_hist_sq = any(
        sq.get("type") == "historical_price_distribution"
        for sq in (decomp.sub_questions or [])
    )
    # Run historical distribution for any live-market query (not just explicit historical questions)
    # so the answer can always say "this is cheap/normal/elevated vs last year"
    _wants_hist = (
        _has_hist_sq
        or decomp.requires_history
        or decomp.intent.value in ("lookup", "explanation", "comparison", "action_recommendation")
    )
    if _wants_hist:
        try:
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

    # Summarise fuel dispatch by type for the evidence event
    _fuel_by_type: dict[str, float] = {}
    if gather.unit_events:
        for ue in gather.unit_events:
            fuel = getattr(ue, "fuel_type", None) or "unknown"
            mw = float(getattr(ue, "total_cleared_mw", 0) or 0)
            _fuel_by_type[fuel] = round(_fuel_by_type.get(fuel, 0) + mw, 1)
    _binding_constraints = sum(
        1 for d in gather.driver_events
        if getattr(d, "constraint_id", None)
    ) if gather.driver_events else 0
    _ev_assembled: dict = {
        "step": "EVIDENCE_ASSEMBLED",
        "t_ms": round((time.perf_counter() - _t0) * 1000),
        "temporal_docs": len(temporal_evidence),
        "fuel_sources": len((fuel_mix or {}).get("sources", [])),
        "fuel_dispatch_mw": _fuel_by_type or None,
        "binding_constraints": _binding_constraints or None,
        "analogs_matched": len(gather.analogs) if gather.analogs else 0,
        "driver_events": len(gather.driver_events) if gather.driver_events else 0,
    }
    if hist_dist and hist_dist.get("available"):
        _ev_assembled["hist_dist"] = {
            "period": hist_dist.get("period_label"),
            "median": round(hist_dist.get("median", 0), 2),
            "n_rows": hist_dist.get("count"),
        }
    _events.append(_ev_assembled)

    # --- 2e. Security pass 3 — tool output hygiene ---
    tool_outputs = []
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
            detail=f"Tool output failed security validation: {tool_check.signals[0].description if tool_check.signals else 'anomalous data'}",
        )

    # --- 3. Assemble why sources + build narrative ---
    why_sources = assemble_why_sources(decomp, gather, region)
    evidence_quality = _build_evidence_quality(gather, temporal_evidence, fuel_mix, why_sources)
    provenance = (
        [s.to_dict() for s in gather.source_statuses.values()]
        if gather.source_statuses else None
    )
    why_output = build_why(why_sources)
    factual = format_verdict(why_output, why_sources, trace_id)
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
            ),
        )
    except Exception as exc:
        logger.debug("Answer planner unavailable for %s: %s", query_id, exc)
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
                    ),
                )
                decomp = _patched_decomp
            except Exception as exc:
                logger.debug("Critic re-plan failed (non-fatal, keeping original): %s", exc)
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
    }.get(intent, "verdict")


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

    # Q1: Drill into the reason for current conditions
    if intent in ("lookup", "comparison"):
        questions.append(f"Why is {region} at this price right now?")
    elif intent == "explanation":
        questions.append(f"How long will {region} stay elevated?")
    elif intent in ("action_recommendation", "fuel_source_recommendation"):
        questions.append(f"What is the current {region} spot price?")

    # Q2: Forward-looking / forecast
    if price and price > 200:
        questions.append(f"Will {region} price drop below $200 in the next hour?")
    elif price and price < 100:
        questions.append(f"Is {region} cheap compared to last year?")
    else:
        questions.append(f"What is the {region} price forecast for the next 30 minutes?")

    # Q3: Evidence gap or cross-region
    missing = [m for s in (factual.answer_sections or []) if s.get("title") == "Missing"
               for m in s.get("items", [])]
    if missing:
        gap = missing[0].replace("_", " ")
        questions.append(f"Why is {gap} missing and does it matter?")
    elif intent == "comparison":
        questions.append("Which state has the cheapest energy to buy right now?")
    elif hist_dist and hist_dist.get("classification") in ("elevated", "high", "spike"):
        questions.append(f"Is {region} more expensive than usual this time of day?")
    else:
        questions.append(f"Compare {region} to all other NEM states now.")

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
