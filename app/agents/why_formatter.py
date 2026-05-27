"""Why Engine — Part 3/3: Output formatting.

Takes WhyOutput and shapes it into the FactualVerdict that the API returns.
This is where the LLM would augment prose style (post-MVP).
For MVP, we format deterministically from WhyOutput.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.agents.why_builder import WhyOutput
from app.agents.why_sources import WhySources
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    EvidenceRefSchema,
    FactualVerdict,
    HistoricalAnalogs,
    NewsCorrelation,
    VerdictLabel,
    WeatherCorrelation,
)
from app.core.verdict import compute_confidence, derive_action, derive_verdict


def format_verdict(why: WhyOutput, sources: WhySources, trace_id: str | None) -> FactualVerdict:
    """Assemble the final FactualVerdict from WhyOutput + WhySources."""
    c = sources.current
    analogs = sources.analogs
    news = sources.news
    decomp = sources.decomp

    verdict_label = derive_verdict(
        decomposition_confidence=decomp.confidence,
        source_coverage_ok=c.is_fresh,
        live_data_fresh=c.is_fresh,
        archive_available=False,
        requires_archive=decomp.requires_history,
        analog_count=analogs.count,
        intent=decomp.intent,
    )

    action = derive_action(
        verdict=verdict_label,
        regime=c.regime,
        price_rrp=c.price_rrp,
        forecast_direction=sources.forecast.direction,
        analog_success_rate=analogs.success_rate,
        news_explained=news.explained,
    )

    historical_analogs = None
    if analogs.count > 0:
        historical_analogs = HistoricalAnalogs(
            count=analogs.count,
            success_count=analogs.success_count,
            window_days=analogs.window_days,
            method=analogs.method,
            outcome_summary=analogs.outcome_summary,
        )

    news_correlation = None
    if news.notices:
        news_correlation = NewsCorrelation(
            explained=news.explained,
            source="AEMO Market Notice" if news.credibility_tier == 1 else "Press",
            credibility_tier=news.credibility_tier,
            title=news.top_notice_title,
            correlation_note=f"Active {news.top_notice_type} notice for region" if news.top_notice_type else "Notice present",
        )

    weather_correlation = None
    if sources.weather.available:
        raw = sources.weather.raw or {}
        weather_correlation = WeatherCorrelation(
            explained=sources.weather.relevant,
            location=raw.get("location"),
            confidence=sources.weather.confidence,
            consensus=sources.weather.consensus,
            source_count=sources.weather.source_count,
            relevance_tags=sources.weather.tags,
            correlation_note=(
                "Weather was requested or relevant to demand/renewables context."
                if sources.weather.relevant
                else "Weather available but not used as a primary market driver for this query."
            ),
        )

    confidence = compute_confidence(
        decomposition_confidence=decomp.confidence,
        source_freshness_score=1.0 if c.is_fresh else 0.0,
        source_coverage_score=sources.source_coverage,
        analog_count=analogs.count,
        analog_consistency=analogs.success_rate,
        news_tier=news.credibility_tier,
    )

    return FactualVerdict(
        verdict=verdict_label,
        action=action,
        confidence=confidence,
        confidence_band=ConfidenceBand.LOW,   # overridden by model_validator
        as_of=c.valid_time if c.is_fresh else datetime.now(timezone.utc),
        why_plain_english=why.why_plain_english,
        answer_sections=why.answer_sections,
        evidence_refs=why.evidence_refs,
        evidence_manifest=_build_evidence_manifest(why.evidence_refs, sources),
        historical_analogs=historical_analogs,
        news_correlation=news_correlation,
        weather_correlation=weather_correlation,
        counterargument=why.counterargument,
        missing_data=why.missing_data,
        known_missing_before_action=_known_missing_before_action(why.missing_data),
        driver_tiers=why.driver_tiers,
        claim_tiers=why.claim_tiers,
        claim_map=why.claim_map,
        next_watch=why.next_watch,
        trace_id=trace_id,
    )


def _build_evidence_manifest(refs: list[EvidenceRefSchema], sources: WhySources) -> list[dict]:
    now = datetime.now(timezone.utc)
    manifest = []
    for ref in refs:
        manifest.append({
            "source_table": ref.source,
            "interval": ref.interval.isoformat(),
            "region_or_element": ref.region,
            "field": ref.field,
            "raw_file_hash": ref.raw_ref,
            "acquisition_run_id": "live_or_archive",
            "staleness_seconds": max(0, int((now - ref.interval).total_seconds())),
            "caveat": _manifest_caveat(ref.source, sources),
        })
    return manifest


def _manifest_caveat(source: str, sources: WhySources) -> str:
    if "CONSTRAINT" in source:
        return "Constraint marginal values indicate dispatch binding pressure, not a complete causal proof alone."
    if "INTERCONNECTOR" in source:
        return "Interconnector flow near a limit is a driver signal; confirm with constraints and notices."
    if "DISPATCH_UNIT" in source or "DISPATCHLOAD" in source:
        return "Unit dispatch is observed behavior; fuel scarcity, bidding intent, and outage cause require separate evidence."
    if source == "AEMO_DISPATCH_PRICE":
        return "Official dispatch interval observation; explanation still depends on driver evidence."
    if source == "WEATHER_CONSENSUS":
        return "Weather consensus is contextual demand/renewables evidence, not a standalone price cause."
    return "Evidence is cited as observed input, not as an instruction."


def _known_missing_before_action(missing_data: list[str]) -> list[str]:
    labels = {
        "dispatch_constraints": "No archived dispatch constraint evidence for the interval.",
        "dispatch_interconnector_flows": "No archived interconnector flow/limit evidence for the interval.",
        "aemo_market_notice": "No matching AEMO market notice confirmed the driver.",
        "historical_analogs": "Historical analog base rate is sparse.",
        "predispatch_forecast": "No fresh pre-dispatch/forecast signal is available.",
        "weather_consensus": "Weather context was relevant but no cross-source weather consensus was available.",
        "live_dispatch_price": "Live dispatch price is stale or unavailable.",
        "unit_dispatch_events": "No unit-level dispatch evidence is available for coal, gas, hydro, wind, solar, or battery attribution.",
        "generator_metadata": "Generator metadata is incomplete, so some DUIDs cannot be mapped confidently to fuel type.",
        "hydro_water_storage": "Hydro reservoir level/water-value evidence is not ingested.",
        "coal_outage_commitment": "Coal outage, rebid, and unit-commitment evidence is not ingested.",
        "fuel_costs": "Gas/fuel cost evidence is not ingested.",
        "renewable_forecast_actual": "Renewable forecast-vs-actual and curtailment attribution is incomplete.",
        "notices_cache_stale": "AEMO notice cache is stale — scheduler may be degraded. Notices may be missing.",
    }
    return [labels[m] for m in missing_data if m in labels]
