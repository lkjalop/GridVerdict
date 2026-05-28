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

    # --- 0.5. Query restatement — extract sub-questions before decompose ---
    # Uses qwen3:14b /no_think (~1s). Falls back silently to empty result.
    restatement = await restate_query(body.text)
    # Seed sub-questions into the decomposer text so it produces better
    # causal_targets and requested_output for multi-part queries.
    _decompose_text = restatement.seed_text(body.text)

    # --- 1. Decompose (rules-first hybrid: rules → LLM enrichment → merge) ---
    _decomp_t0 = time.perf_counter()
    decomp = await llm_decompose(_decompose_text, region_hint=region, query_id=query_id)
    # Restore raw_query to original user text (not the seeded version).
    if _decompose_text != body.text:
        decomp = decomp.model_copy(update={"raw_query": body.text})
    # Merge restatement sub_questions into decomp if the hybrid didn't populate them.
    if not decomp.sub_questions and not restatement.is_empty():
        from app.engines.decomposition import _classify_sub_questions
        _sq = _classify_sub_questions(body.text.lower(), decomp.requested_output or "")
        if _sq:
            decomp = decomp.model_copy(update={"sub_questions": _sq})
    llm_decompose_latency_ms.observe((time.perf_counter() - _decomp_t0) * 1000)

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
    include_weather = weather_query_relevant(body.text)
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

    comparison_gathers: dict[str, GatherResult] = {}
    if decomp.intent == IntentLabel.COMPARISON:
        comparison_regions = list({
            r.upper() for r in (decomp.entities.get("regions") or [region])
            if r.upper() in {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
        }) or [region]
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
    if _has_hist_sq or decomp.requires_history:
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

    # --- 7. Persist query + trace ---
    answer_dict = factual.model_dump(mode="json")
    tool_calls_log = tool_outputs[:]  # tool outputs used as tool call log
    valid_time = (gather.dispatch.valid_time if gather.dispatch else datetime.now(timezone.utc))

    query_row = QueryModel(
        id=query_id,
        tenant_id=user.tenant_id,
        session_id=session_id,
        raw_query=body.text,
        decomposition=decomp.model_dump(),
        answer=answer_dict,
        trace_id=trace_id,
        intent=decomp.intent.value,
        verdict=factual.verdict.value,
        region=region,
    )
    db.add(query_row)

    await write_trace(
        session=db,
        trace_id=trace_id,
        tenant_id=user.tenant_id,
        query_id=query_id,
        valid_time=valid_time,
        decomposition=decomp.model_dump(),
        tool_calls=tool_calls_log,
        answer=answer_dict,
        observer_result=answer_check.to_dict(),
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
    _decomp_dict = decomp.model_dump()
    if not restatement.is_empty():
        _decomp_dict["restatement"] = {
            "sub_questions": restatement.sub_questions,
            "primary_intent": restatement.primary_intent,
            "answer_gap_risk": restatement.answer_gap_risk,
        }
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
        decomposition=decomp.model_dump(),
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
        decomposition=decomp.model_dump(),
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
        decomposition=decomp.model_dump(),
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
