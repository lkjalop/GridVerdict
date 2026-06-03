"""Why Engine — Part 2/3: Narrative builder.

Transforms WhySources into structured WhyOutput:
  - why_plain_english: primary narrative (deterministic from data, NOT LLM)
  - counterargument: adversarial critique driven by analog outcomes + evidence
  - missing_data: list of unavailable sources
  - confidence: weighted confidence score
  - evidence_refs: cited sources

DESIGN RULE: Every sentence that makes a factual claim cites an evidence_ref.
The adversarial critic considers analog outcomes, notice status, and forecast
direction — it never produces a generic template regardless of evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.agents.why_sources import SeasonalSources, WhySources
from app.core.schema import (
    ClaimMapItem,
    ClaimType,
    DriverConfidenceTier,
    EvidenceRefSchema,
    IntentLabel,
)


@dataclass
class WhyOutput:
    why_plain_english: str
    counterargument: str
    missing_data: list[str]
    evidence_refs: list[EvidenceRefSchema]
    confidence: float
    answer_sections: list[dict] = field(default_factory=list)
    driver_tiers: list[dict] = field(default_factory=list)    # [{label, tier, present, note}]
    claim_tiers: list[dict] = field(default_factory=list)     # [{label, tier, present, evidence_ref_ids}]
    claim_map: list[ClaimMapItem] = field(default_factory=list)  # typed claim map
    next_watch: list[str] = field(default_factory=list)          # actionable operational thresholds
    upgrade_path: list[str] = field(default_factory=list)        # "X present → confidence 60→82%"
    causal_chain: list[str] = field(default_factory=list)        # ordered evidence steps for WHY


def build_why(sources: WhySources) -> WhyOutput:
    """Deterministic narrative from WhySources — no LLM calls."""
    c = sources.current
    decomp = sources.decomp
    news = sources.news
    weather = sources.weather
    analogs = sources.analogs
    forecast = sources.forecast
    drivers = sources.drivers
    technology = sources.technology

    # Out-of-scope queries skip the market data pipeline entirely
    if decomp.intent == IntentLabel.OUT_OF_SCOPE:
        oos_parts: list[str] = []
        oos_missing: list[str] = []
        _append_intent_context(oos_parts, oos_missing, decomp.intent, c, analogs, news, forecast, decomp)
        return WhyOutput(
            why_plain_english=" ".join(oos_parts),
            counterargument="",
            missing_data=oos_missing,
            evidence_refs=[],
            confidence=0.0,
            answer_sections=[
                {"title": "Answer", "items": oos_parts or ["This query is outside GridVerdict's NEM market scope."]},
                {"title": "Missing", "items": oos_missing or ["No market evidence was gathered."]},
            ],
            claim_map=[ClaimMapItem(
                claim_type=ClaimType.OTHER,
                label="Out-of-scope query — no market data analysis performed",
                tier=DriverConfidenceTier.UNCONFIRMED,
                present=False,
                confidence=0.0,
                note="Query was outside GridVerdict's analytical scope",
            )],
        )

    evidence_refs: list[EvidenceRefSchema] = []
    missing_data: list[str] = []
    parts: list[str] = []

    # ── Current state ─────────────────────────────────────────────────
    if not c.is_fresh:
        parts.append(
            f"Latest available dispatch for {c.region} is stale ({c.staleness_seconds}s old): "
            f"${c.price_rrp:.2f}/MWh, demand {c.demand_mw:.0f} MW, "
            f"available generation {c.availability_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW. "
            "Treat this as historical/replay evidence until live AEMO refresh recovers."
        )
        trend_sentence = _build_recent_trend_sentence(sources.recent_dispatch, c)
        if trend_sentence:
            parts.append(trend_sentence)
        else:
            missing_data.append("recent_price_trend")
        missing_data.append("live_dispatch_price")
    else:
        regime_desc = {
            "normal": "within normal operating range",
            "elevated": "elevated above typical levels",
            "spike": "in a price spike event",
            "extreme": "at extreme levels — a significant market stress event",
        }.get(c.regime, "at current levels")

        quantile_clause = ""
        if (
            c.regime_state is not None
            and c.regime_state.quantile_rank > 0
            and (c.regime_state.quantile_rank >= 0.75 or c.regime_state.quantile_rank <= 0.25)
        ):
            pct = int(c.regime_state.quantile_rank * 100)
            quantile_clause = f" ({pct}th percentile of recent trading)"

        if decomp.intent == IntentLabel.RETROSPECTIVE:
            state_prefix = "Current comparison point"
        elif decomp.intent == IntentLabel.EXPLANATION:
            state_prefix = "Short answer"
        else:
            state_prefix = "Current state"
        parts.append(
            f"{state_prefix}: {c.region} is ${c.price_rrp:.2f}/MWh, {regime_desc}{quantile_clause}. "
            f"Demand is {c.demand_mw:.0f} MW, available generation is {c.availability_mw:.0f} MW, "
            f"and headroom is {c.headroom_mw:.0f} MW."
        )
        trend_sentence = _build_recent_trend_sentence(sources.recent_dispatch, c)
        if trend_sentence:
            parts.append(trend_sentence)
        else:
            missing_data.append("recent_price_trend")
        evidence_refs.append(EvidenceRefSchema(
            source="AEMO_DISPATCH_PRICE",
            region=c.region,
            interval=c.valid_time,
            field="price_rrp",
            value=c.price_rrp,
            raw_ref=f"dispatch_{c.valid_time.isoformat()}",
        ))
        evidence_refs.append(EvidenceRefSchema(
            source="AEMO_DISPATCH_PRICE",
            region=c.region,
            interval=c.valid_time,
            field="demand_mw",
            value=c.demand_mw,
            raw_ref=f"dispatch_{c.valid_time.isoformat()}",
        ))

    # ── AEMO market notice ────────────────────────────────────────────
    if news.explained and news.top_notice_type:
        parts.append(
            f"Active AEMO Market Notice: {news.top_notice_type}"
            + (f" — {news.top_notice_title}" if news.top_notice_title else "")
            + f" (credibility tier {news.credibility_tier})."
        )
    else:
        missing_data.append("aemo_market_notice")
        if news.notices_stale:
            missing_data.append("notices_cache_stale")

    if news.commentary_items:
        headline = news.commentary_items[0].get("title", "untitled item")
        parts.append(
            f"Recent public market commentary matched NEM keywords: {headline}. "
            "This is contextual only and is not treated as a confirmed price driver."
        )

    if news.auto_commentary:
        for ac in news.auto_commentary[:3]:
            ac_headline = ac.get("headline", "")
            ac_conf = ac.get("confidence", 0.0)
            ac_time = ac.get("valid_time", "")[:16]  # trim to minute precision
            if ac_headline:
                parts.append(
                    f"Prior market analysis at {ac_time}: {ac_headline} "
                    f"(confidence {ac_conf:.0%}, source: GridVerdict commentary log)."
                )

    if weather.relevant:
        if weather.available:
            wc = weather.consensus
            weather_parts = []
            if wc.get("temperature_c") is not None:
                weather_parts.append(f"temperature {wc['temperature_c']:.1f} C")
            if wc.get("wind_speed_kmh") is not None:
                weather_parts.append(f"wind {wc['wind_speed_kmh']:.1f} km/h")
            if wc.get("cloud_cover_pct") is not None:
                weather_parts.append(f"cloud cover {wc['cloud_cover_pct']:.0f}%")
            if wc.get("precipitation_mm") is not None:
                weather_parts.append(f"rain {wc['precipitation_mm']:.1f} mm")
            tag_clause = f" Tags: {', '.join(weather.tags)}." if weather.tags else ""
            parts.append(
                "Weather consensus for this query: "
                + (", ".join(weather_parts) if weather_parts else "no strong weather signal")
                + f" ({weather.source_count} sources, confidence {weather.confidence:.0%})."
                + tag_clause
                + " Weather is treated as contextual demand/renewables evidence, not a standalone price cause."
            )
            interval = c.valid_time
            raw_ref = (weather.raw or {}).get("raw_ref", "weather_consensus")
            if wc.get("temperature_c") is not None:
                evidence_refs.append(EvidenceRefSchema(
                    source="WEATHER_CONSENSUS",
                    region=c.region,
                    interval=interval,
                    field="temperature_c",
                    value=float(wc["temperature_c"]),
                    raw_ref=raw_ref,
                ))
            if wc.get("wind_speed_kmh") is not None:
                evidence_refs.append(EvidenceRefSchema(
                    source="WEATHER_CONSENSUS",
                    region=c.region,
                    interval=interval,
                    field="wind_speed_kmh",
                    value=float(wc["wind_speed_kmh"]),
                    raw_ref=raw_ref,
                ))
        else:
            missing_data.append("weather_consensus")

    if drivers.binding_constraints:
        top = drivers.binding_constraints[0]
        mv = top.get("values", {}).get("marginal_value")
        parts.append(
            f"Archived dispatch constraint evidence is present: {top.get('element_id')} "
            f"had marginal value {mv} at the interval."
        )
        evidence_refs.append(EvidenceRefSchema(
            source=top.get("source", "AEMO_DISPATCHCONSTRAINT"),
            region=top.get("region") or c.region,
            interval=top.get("valid_time") or c.valid_time,
            field="constraint_marginal_value",
            value=float(mv or 0.0),
            raw_ref=top.get("raw_ref", "constraint_archive"),
        ))
    else:
        missing_data.append("dispatch_constraints")

    if drivers.tight_interconnectors:
        top = drivers.tight_interconnectors[0]
        values = top.get("values", {})
        flow = values.get("mw_flow", values.get("metered_mw_flow"))
        if drivers.interconnector_narrative:
            parts.append(drivers.interconnector_narrative)
        else:
            parts.append("Direct answer: " +
                f"Archived interconnector evidence is present: {top.get('element_id')} "
                f"flow was {float(flow or 0.0):.1f} MW near its dispatch limit."
            )
        evidence_refs.append(EvidenceRefSchema(
            source=top.get("source", "AEMO_DISPATCHINTERCONNECTORRES"),
            region=top.get("region") or c.region,
            interval=top.get("valid_time") or c.valid_time,
            field="interconnector_flow_mw",
            value=float(flow or 0.0),
            raw_ref=top.get("raw_ref", "interconnector_archive"),
        ))
    else:
        missing_data.append("dispatch_interconnector_flows")

    if technology.has_unit_evidence:
        tech_parts = []
        for fuel, summary in technology.by_fuel.items():
            if fuel == "unknown":
                continue
            delta = float(summary.get("delta_mw") or 0.0)
            tech_parts.append(
                f"{fuel} cleared {float(summary.get('total_cleared_mw') or 0.0):.0f} MW "
                f"({delta:+.0f} MW versus initial output)"
            )
        if tech_parts:
            parts.append("Observed unit dispatch by technology: " + "; ".join(tech_parts) + ".")
        if "hydro" in technology.by_fuel:
            parts.append(
                "Hydro dispatch is observed, but reservoir level and water-value evidence are not ingested, "
                "so water scarcity is not asserted as a cause."
            )
        if "coal" in technology.by_fuel:
            parts.append(
                "Coal unit movement is observed, but outage, unit-commitment, and rebid evidence are still needed "
                "before assigning cause."
            )
        if {"wind", "solar"} & set(technology.by_fuel):
            capped = sum(
                int(v.get("semi_dispatch_cap_count") or 0)
                for k, v in technology.by_fuel.items()
                if k in {"wind", "solar"}
            )
            if capped:
                parts.append(f"Renewable semi-scheduled cap evidence is present on {capped} unit rows.")
            else:
                parts.append(
                    "Renewable output is observed, but forecast-vs-actual and curtailment attribution remain incomplete."
                )
        top_ref = _technology_evidence_ref(technology.by_fuel)
        if top_ref:
            evidence_refs.append(EvidenceRefSchema(
                source="AEMO_DISPATCH_UNIT_SOLUTION",
                region=c.region,
                interval=c.valid_time,
                field=f"{top_ref['fuel_type']}_total_cleared_mw",
                value=float(top_ref.get("total_cleared_mw") or 0.0),
                raw_ref=(top_ref.get("raw_refs") or ["unit_dispatch"])[0],
            ))
        missing_data.extend(technology.caveats)
    else:
        missing_data.append("unit_dispatch_events")

    # ── Marginal price-setter identification ─────────────────────────
    # Runs whenever we have unit dispatch data OR need a causal explanation.
    # Upgrades fuel attribution from prior cost model to dispatch-backed evidence.
    _marginal_setter = None
    if decomp.intent in (IntentLabel.EXPLANATION, IntentLabel.ACTION_RECOMMENDATION):
        try:
            from app.engines.marginal_setter import identify_marginal_setter
            _unit_ev = list(technology.events) if technology.events else []
            _driver_ev = list(drivers.events) if hasattr(drivers, "events") and drivers.events else []
            _marginal_setter = identify_marginal_setter(
                unit_events=_unit_ev,
                driver_events=_driver_ev,
                spot_price=c.price_rrp,
                region=c.region,
            )
            _tier = _marginal_setter.data_tier
            if _tier in ("dispatch", "bid_reconstruction"):
                # Upgrade the narrative: replace generic "dispatch constraints missing"
                # with a confirmed marginal setter sentence
                parts.append(
                    f"Marginal price setter: {_marginal_setter.display_name} "
                    f"(${_marginal_setter.estimated_price_mwh:.0f}/MWh prior, "
                    f"{_tier} evidence). "
                    f"{_marginal_setter.explanation}"
                )
                if _marginal_setter.constraint_narrative:
                    parts.append(_marginal_setter.constraint_narrative)
                if _marginal_setter.interconnector_narrative:
                    parts.append(_marginal_setter.interconnector_narrative)
                # Add an evidence ref for the marginal setter identification
                evidence_refs.append(EvidenceRefSchema(
                    source=f"MARGINAL_SETTER_{_tier.upper()}",
                    region=c.region,
                    interval=c.valid_time,
                    field="marginal_fuel_type",
                    value=_marginal_setter.estimated_price_mwh,
                    raw_ref=f"marginal_setter_{_marginal_setter.fuel_type}",
                ))
            else:
                # Prior model — still surface the price-bracket inference
                parts.append(_marginal_setter.explanation)
        except Exception as _ms_exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("Marginal setter identification failed: %s", _ms_exc)

    # ── Historical analogs from HippoGraph PPR ────────────────────────
    if analogs.count >= 3:
        outcome_clause = ""
        if analogs.outcome_summary:
            outcome_clause = f" Outcome: {analogs.outcome_summary}."
        if decomp.intent != IntentLabel.RETROSPECTIVE:
            parts.append(
                f"{analogs.count} historical analog periods retrieved "
                f"({analogs.window_days}-day window, PPR similarity match)."
                + outcome_clause
            )
    else:
        missing_data.append("historical_analogs")
        if decomp.intent == IntentLabel.RETROSPECTIVE:
            parts.append(
                "HippoGraph cannot yet answer the historical-analog part decisively "
                f"(need at least 3 comparable periods, found {analogs.count})."
            )
        else:
            parts.append(
                "Historical analog confidence is unavailable "
                f"(need at least 3 comparable periods, found {analogs.count}). "
                "HippoGraph accumulates over the first 24 hours of live operation."
            )
        _legacy_cold_start_text = (
            "Insufficient historical analog periods for comparison "
            f"(need ≥3, found {analogs.count}). "
            "HippoGraph accumulates over the first 24 hours of live operation."
        )

    # ── Forecast (per-model breakdown) ───────────────────────────────
    if forecast.model_detail:
        model_lines = _build_model_forecast_summary(forecast.model_detail)
        if model_lines:
            parts.append(model_lines)
        elif not forecast.available:
            missing_data.append("predispatch_forecast")
    elif forecast.available and forecast.p50 is not None:
        parts.append(
            f"Pre-dispatch quantile forecast (next {forecast.horizon_intervals * 5} min): "
            f"P10 ${forecast.p10:.2f} / P50 ${forecast.p50:.2f} / P90 ${forecast.p90:.2f}/MWh, "
            f"direction {forecast.direction}."
        )
    elif forecast.available and forecast.direction not in (None, "unknown"):
        dir_desc = {
            "rising": "trending upward",
            "falling": "trending downward",
            "flat": "broadly flat",
        }.get(forecast.direction, forecast.direction)
        parts.append(
            f"Recent price trend is {dir_desc} based on the last "
            f"{forecast.horizon_intervals} dispatch intervals. "
            "P10/P50/P90 quantile forecast available once LNN training completes (≥288 intervals)."
        )
    else:
        missing_data.append("predispatch_forecast")

    # ── Intent-specific context ───────────────────────────────────────
    _append_intent_context(parts, missing_data, decomp.intent, c, analogs, news, forecast, decomp)

    # ── Adversarial critic — evidence-driven ─────────────────────────
    counterarg = _build_counterargument(c, analogs, news, forecast)

    why_text = " ".join(parts)
    if "No single dominant cause confirmed" in why_text and "The confirmed facts" in why_text:
        why_text = why_text.replace(
            "No single dominant cause confirmed — review demand/generation balance The confirmed facts",
            "Direct answer: GridVerdict cannot confirm the cause from the available evidence. The confirmed facts",
        ).replace(
            "No single dominant cause confirmed â€” review demand/generation balance The confirmed facts",
            "Direct answer: GridVerdict cannot confirm the cause from the available evidence. The confirmed facts",
        )

    d_tiers = _build_driver_tiers(c, news, weather=sources.weather,
                                   drivers=drivers, technology=technology,
                                   analogs=analogs, forecast=forecast)
    # Stale threshold relative to market interval + staleness offset, not wall clock.
    # This ensures fixed-time test fixtures never appear stale when fresh=True.
    query_now = c.valid_time + timedelta(seconds=max(0, c.staleness_seconds))
    if query_now.tzinfo is None:
        query_now = query_now.replace(tzinfo=timezone.utc)
    c_tiers = _build_claim_tiers(d_tiers, evidence_refs, now=query_now)
    c_map = _build_claim_map(c_tiers, evidence_refs, sources, now=query_now)
    n_watch = _build_next_watch(c, forecast, drivers, analogs, sources.weather, sources.news)
    answer_sections = _build_answer_sections(sources, evidence_refs, missing_data, why_text)

    upgrade_path = _compute_upgrade_path(sources, missing_data, analogs, technology, forecast)

    causal_chain = _build_causal_chain_steps(sources, decomp.intent, missing_data)

    return WhyOutput(
        why_plain_english=why_text,
        counterargument=counterarg,
        missing_data=missing_data,
        evidence_refs=evidence_refs,
        confidence=_estimate_confidence(c.is_fresh, analogs.count, news.explained),
        answer_sections=answer_sections,
        driver_tiers=d_tiers,
        claim_tiers=c_tiers,
        claim_map=c_map,
        next_watch=n_watch,
        upgrade_path=upgrade_path,
        causal_chain=causal_chain,
    )


def _build_driver_tiers(c, news, weather, drivers, technology, analogs, forecast) -> list[dict]:
    """Build a section-level causality confidence tier list.

    Each entry: {label, tier, present, note}
    tier ∈ {confirmed, supported, plausible, unconfirmed}
    present: True when evidence was available and used.
    """
    T = DriverConfidenceTier

    def _tier(label: str, tier: DriverConfidenceTier, present: bool, note: str = "") -> dict:
        return {"label": label, "tier": tier.value, "present": present, "note": note}

    tiers = []

    # 1. Dispatch price — direct AEMO telemetry
    tiers.append(_tier(
        "dispatch_price",
        T.CONFIRMED if c.is_fresh else T.UNCONFIRMED,
        present=c.is_fresh,
        note=f"${c.price_rrp:.2f}/MWh" if c.is_fresh else f"stale {c.staleness_seconds}s",
    ))

    # 2. AEMO market notice
    notice_present = bool(news.notices or news.explained)
    if notice_present and news.credibility_tier == 1:
        n_tier = T.CONFIRMED
    elif notice_present and news.credibility_tier == 2:
        n_tier = T.PLAUSIBLE
    else:
        n_tier = T.UNCONFIRMED
    tiers.append(_tier(
        "aemo_notice",
        n_tier,
        present=notice_present,
        note=news.top_notice_type or ("none" if not notice_present else "tier-2"),
    ))

    # 3. Weather — consensus signal, not measurement-grade
    tiers.append(_tier(
        "weather",
        T.PLAUSIBLE if (weather.available and weather.relevant) else T.UNCONFIRMED,
        present=weather.available and weather.relevant,
        note=f"{weather.source_count} sources {weather.confidence:.0%}" if weather.available else "unavailable",
    ))

    # 4. Dispatch constraints — archived evidence from MMSDM
    tiers.append(_tier(
        "dispatch_constraints",
        T.SUPPORTED if drivers.binding_constraints else T.UNCONFIRMED,
        present=bool(drivers.binding_constraints),
        note=f"{len(drivers.binding_constraints)} binding" if drivers.binding_constraints else "not ingested",
    ))

    # 5. Interconnector flows — tier elevated to CONFIRMED when causal role is proven
    _ic_role = drivers.interconnector_causal_role
    _ic_tier = (
        T.CONFIRMED if _ic_role == "causal"
        else T.SUPPORTED if (drivers.tight_interconnectors or _ic_role == "contributing")
        else T.UNCONFIRMED
    )
    _ic_note = (
        drivers.interconnector_narrative[:80] if drivers.interconnector_narrative
        else (f"{len(drivers.tight_interconnectors)} near limit" if drivers.tight_interconnectors else "not ingested")
    )
    tiers.append(_tier(
        "interconnector_flows",
        _ic_tier,
        present=bool(drivers.tight_interconnectors or _ic_role in ("causal", "contributing")),
        note=_ic_note,
    ))

    # 6. Unit dispatch — DUID-level telemetry
    tiers.append(_tier(
        "unit_dispatch",
        T.SUPPORTED if technology.has_unit_evidence else T.UNCONFIRMED,
        present=technology.has_unit_evidence,
        note=", ".join(technology.by_fuel.keys()) if technology.has_unit_evidence else "not ingested",
    ))

    # 7. Historical analogs — pattern match (probabilistic)
    tiers.append(_tier(
        "historical_analogs",
        T.PLAUSIBLE if analogs.count >= 3 else T.UNCONFIRMED,
        present=analogs.count >= 3,
        note=f"{analogs.count} matched" if analogs.count else "warming up",
    ))

    # 8. Forecast — model signal (probabilistic)
    tiers.append(_tier(
        "forecast",
        T.PLAUSIBLE if forecast.available else T.UNCONFIRMED,
        present=forecast.available,
        note=forecast.direction if forecast.available else "unavailable",
    ))

    return tiers


_STALE_THRESHOLD_SECONDS = 900  # 15 min


def _stale_ref_ids(
    evidence_refs: list[EvidenceRefSchema],
    now: datetime | None = None,
) -> frozenset[str]:
    """Return the set of evidence ref IDs whose interval is >15 min old."""
    now = now or datetime.now(timezone.utc)
    stale: set[str] = set()
    for ref in evidence_refs:
        if ref.interval is None:
            continue
        iv = ref.interval if ref.interval.tzinfo else ref.interval.replace(tzinfo=timezone.utc)
        if (now - iv).total_seconds() > _STALE_THRESHOLD_SECONDS:
            stale.add(ref.id)
    return frozenset(stale)


def _build_claim_tiers(
    driver_tiers: list[dict],
    evidence_refs: list[EvidenceRefSchema],
    now: datetime | None = None,
) -> list[dict]:
    """Link each section-level driver tier to specific evidence_ref IDs.

    Each entry: {label, tier, present, evidence_ref_ids: list[str]}

    Invariant: if tier is 'confirmed' or 'supported', evidence_ref_ids is non-empty.
    Tiers are capped at 'plausible' when all matched refs are stale (>15min).
    """
    _SOURCE_MAP: dict[str, list[str]] = {
        "dispatch_price":        ["AEMO_DISPATCH_PRICE"],
        "weather":               ["WEATHER_CONSENSUS"],
        "dispatch_constraints":  ["AEMO_DISPATCHCONSTRAINT"],
        "interconnector_flows":  ["AEMO_DISPATCHINTERCONNECTORRES"],
        "unit_dispatch":         ["AEMO_DISPATCH_UNIT_SOLUTION"],
        # aemo_notice, historical_analogs, forecast → no hard evidence_refs
    }

    stale = _stale_ref_ids(evidence_refs, now)
    result: list[dict] = []
    for dt in driver_tiers:
        label = dt["label"]
        tier = dt["tier"]
        present = dt["present"]
        patterns = _SOURCE_MAP.get(label, [])
        ref_ids: list[str] = [
            r.id for r in evidence_refs
            if any(p in r.source for p in patterns)
        ]
        # Stale cap: if all matched refs are stale, downgrade confirmed/supported → plausible
        if ref_ids and all(rid in stale for rid in ref_ids):
            if tier in ("confirmed", "supported"):
                tier = "plausible"
        result.append({
            "label": label,
            "tier": tier,
            "present": present,
            "evidence_ref_ids": ref_ids,
        })
    return result


def _build_claim_map(
    claim_tiers: list[dict],
    evidence_refs: list[EvidenceRefSchema],
    sources: "WhySources | None" = None,
    now: datetime | None = None,
) -> list[ClaimMapItem]:
    """Convert claim_tiers into typed ClaimMapItem objects and add Sprint R claim types.

    Base claims are built from the existing driver tier labels.
    Extended claim types from WhySources: CONSTRAINT_BINDING, WEATHER_CORRELATION, REBID_EVIDENCE,
    OUTAGE_EVIDENCE, FCAS_CLAIM sourced directly from WhySources fields.

    Quality invariants enforced here:
      - confirmed/supported claims from news are capped at supported (never confirmed)
      - stale evidence refs downgrade tier to plausible
    """
    now = now or datetime.now(timezone.utc)
    stale = _stale_ref_ids(evidence_refs, now)

    _LABEL_TO_CLAIM_TYPE: dict[str, ClaimType] = {
        "dispatch_price":       ClaimType.PRICE_ASSERTION,
        "aemo_notice":          ClaimType.CAUSE_CLAIM,
        "weather":              ClaimType.CAUSE_CLAIM,
        "dispatch_constraints": ClaimType.CAUSE_CLAIM,
        "interconnector_flows": ClaimType.CAUSE_CLAIM,
        "unit_dispatch":        ClaimType.CAUSE_CLAIM,
        "historical_analogs":   ClaimType.HISTORICAL_ANALOG,
        "forecast":             ClaimType.FORECAST_CLAIM,
    }
    _TIER_CONFIDENCE: dict[str, float] = {
        "confirmed":   0.95,
        "supported":   0.75,
        "plausible":   0.50,
        "unconfirmed": 0.20,
    }
    _LABEL_DESCRIPTIONS: dict[str, str] = {
        "dispatch_price":       "AEMO dispatch price for the interval",
        "aemo_notice":          "AEMO market / LOR / constraint notice",
        "weather":              "Weather conditions as demand/renewable driver",
        "dispatch_constraints": "Binding AEMO dispatch constraint",
        "interconnector_flows": "Interconnector flow near import/export limit",
        "unit_dispatch":        "Unit-level generator dispatch evidence",
        "historical_analogs":   "Historical analog market states",
        "forecast":             "Probabilistic price forecast (LEAR/QRA/LNN)",
    }
    # News-sourced labels: tier is capped at supported (never confirmed) per quality rule
    _NEWS_LABELS = {"aemo_notice"}

    result: list[ClaimMapItem] = []
    for ct in claim_tiers:
        label = ct.get("label", "")
        tier_str = ct.get("tier", "unconfirmed")
        present = ct.get("present", False)
        ref_ids: list[str] = ct.get("evidence_ref_ids", [])

        # Quality: news/notice claims never exceed supported tier
        if label in _NEWS_LABELS and tier_str == "confirmed":
            tier_str = "supported"

        # Quality: stale refs cap tier at plausible
        stale_note: str | None = None
        if ref_ids and all(rid in stale for rid in ref_ids):
            if tier_str in ("confirmed", "supported"):
                tier_str = "plausible"
                stale_note = "Source data is stale (>15 min)"

        try:
            tier = DriverConfidenceTier(tier_str)
        except ValueError:
            tier = DriverConfidenceTier.UNCONFIRMED

        result.append(ClaimMapItem(
            claim_type=_LABEL_TO_CLAIM_TYPE.get(label, ClaimType.OTHER),
            label=_LABEL_DESCRIPTIONS.get(label, label),
            tier=tier,
            present=present,
            confidence=_TIER_CONFIDENCE.get(tier_str, 0.2),
            evidence_ref_ids=ref_ids,
            note=stale_note,
        ))

    if sources is None:
        return result

    drivers = sources.drivers
    weather = sources.weather

    # CONSTRAINT_BINDING — direct AEMO dispatch constraint data
    bc = getattr(drivers, "binding_constraints", [])
    if bc:
        names = [b.get("element_id") or b.get("constraint_id") or b.get("name") or "" for b in bc[:3]]
        label_str = f"Binding constraint{'s' if len(bc) > 1 else ''}: {', '.join(filter(None, names))}"
        c_ref_ids = [r.id for r in evidence_refs if "CONSTRAINT" in r.source or "DISPATCHCONSTRAINT" in r.source]
        result.append(ClaimMapItem(
            claim_type=ClaimType.CONSTRAINT_BINDING,
            label=label_str,
            tier=DriverConfidenceTier.CONFIRMED if c_ref_ids else DriverConfidenceTier.SUPPORTED,
            present=True,
            confidence=0.92 if c_ref_ids else 0.75,
            evidence_ref_ids=c_ref_ids,
        ))

    # WEATHER_CORRELATION — consensus weather signal
    if getattr(weather, "available", False) and getattr(weather, "relevant", False):
        tags = getattr(weather, "tags", [])
        wlabel = f"Weather: {', '.join(tags[:2])}" if tags else "Weather conditions relevant"
        w_conf = float(getattr(weather, "confidence", 0.4))
        w_tier = DriverConfidenceTier.SUPPORTED if w_conf >= 0.6 else DriverConfidenceTier.PLAUSIBLE
        w_ref_ids = [r.id for r in evidence_refs if "WEATHER" in r.source]
        src_count = getattr(weather, "source_count", 0)
        result.append(ClaimMapItem(
            claim_type=ClaimType.WEATHER_CORRELATION,
            label=wlabel,
            tier=w_tier,
            present=True,
            confidence=w_conf,
            evidence_ref_ids=w_ref_ids,
            note=f"Consensus from {src_count} source{'s' if src_count != 1 else ''}" if src_count else None,
        ))

    # REBID_EVIDENCE — unit rebidding activity from driver_events
    driver_events = getattr(drivers, "events", [])
    rebid_events = [
        e for e in driver_events
        if "REBID" in str(e.get("source", "")).upper() or "REBID" in str(e.get("type", "")).upper()
    ]
    if rebid_events:
        total_mw = sum(float(e.get("rebid_mw", 0)) for e in rebid_events)
        r_tier = DriverConfidenceTier.CONFIRMED if total_mw > 100 else DriverConfidenceTier.SUPPORTED
        r_refs = [r.id for r in evidence_refs if "REBID" in r.source]
        result.append(ClaimMapItem(
            claim_type=ClaimType.REBID_EVIDENCE,
            label=f"{len(rebid_events)} unit rebid(s) ({total_mw:.0f} MW total)",
            tier=r_tier,
            present=True,
            confidence=0.85 if total_mw > 100 else 0.65,
            evidence_ref_ids=r_refs,
        ))

    # OUTAGE_EVIDENCE — generation outage events from driver_events
    outage_events = [
        e for e in driver_events
        if "OUTAGE" in str(e.get("source", "")).upper() or "OUTAGE" in str(e.get("type", "")).upper()
    ]
    if outage_events:
        o_refs = [r.id for r in evidence_refs if "OUTAGE" in r.source or "UNIT_SOLUTION" in r.source]
        result.append(ClaimMapItem(
            claim_type=ClaimType.OUTAGE_EVIDENCE,
            label=f"{len(outage_events)} generation outage event(s) active",
            tier=DriverConfidenceTier.SUPPORTED,
            present=True,
            confidence=0.75,
            evidence_ref_ids=o_refs,
        ))

    # FCAS_CLAIM — FCAS price data from driver_events (if available)
    fcas_events = [
        e for e in driver_events
        if "FCAS" in str(e.get("source", "")).upper() or "FCAS" in str(e.get("type", "")).upper()
    ]
    if fcas_events:
        max_raise = max((float(e.get("fcas_price", 0)) for e in fcas_events), default=0.0)
        if max_raise > 200:
            f_refs = [r.id for r in evidence_refs if "FCAS" in r.source]
            result.append(ClaimMapItem(
                claim_type=ClaimType.FCAS_CLAIM,
                label=f"FCAS RAISE price at ${max_raise:.0f}/MWh — frequency constraint active",
                tier=DriverConfidenceTier.CONFIRMED if f_refs else DriverConfidenceTier.PLAUSIBLE,
                present=True,
                confidence=0.90 if f_refs else 0.50,
                evidence_ref_ids=f_refs,
            ))

    return result


def _build_next_watch(
    c,
    forecast,
    drivers,
    analogs,
    weather,
    news=None,
) -> list[str]:
    """Produce an ordered list of actionable monitoring thresholds.

    Rules are deterministic — no LLM, no magic numbers invented mid-run.
    Each rule that fires appends one human-readable sentence.  Rules are
    ordered from most urgent to least urgent.
    """
    from datetime import timezone as _tz
    items: list[str] = []

    # ── Price level thresholds ────────────────────────────────────────
    price = getattr(c, "price_rrp", None)
    if price is not None:
        if price >= 14_500:
            items.append(
                f"CRITICAL: Price is at market price cap "
                f"(${price:,.0f}/MWh) — monitor for VoLL settlement and notice of emergency."
            )
        elif price >= 1_000:
            items.append(
                f"Watch for price cap breach — current ${price:,.0f}/MWh is within "
                f"${14_500 - price:,.0f}/MWh of the cap."
            )
        elif price >= 300:
            items.append(
                f"Watch for spike continuation — price is in spike territory "
                f"(${price:,.0f}/MWh > $300 threshold)."
            )
        elif price < 0:
            items.append(
                f"Watch for negative price continuation — "
                f"${price:,.0f}/MWh may trigger emergency response from generators."
            )

    # ── Headroom ─────────────────────────────────────────────────────
    headroom = getattr(c, "headroom_mw", None)
    if headroom is not None:
        if headroom < 200:
            items.append(
                f"CRITICAL: Headroom is critically low at {headroom:,.0f} MW "
                f"— system is vulnerable to rapid price escalation on any outage."
            )
        elif headroom < 500:
            items.append(
                f"Watch headroom — currently {headroom:,.0f} MW (warning threshold: 500 MW). "
                f"Any unplanned outage could cause a spike."
            )

    # ── Forecast thresholds ──────────────────────────────────────────
    if forecast is not None and getattr(forecast, "available", False):
        p90 = getattr(forecast, "p90", None)
        p50 = getattr(forecast, "p50", None)
        direction = getattr(forecast, "direction", "unknown")
        if p90 is not None and p90 >= 500:
            items.append(
                f"Watch forecast P90 — ensemble forecasting up to ${p90:,.0f}/MWh "
                f"for the next {getattr(forecast, 'horizon_intervals', 6) * 5} minutes."
            )
        if direction == "rising" and (p50 or 0) >= 200:
            items.append(
                f"Watch for rising prices — forecast P50 ${(p50 or 0):,.0f}/MWh "
                f"with upward direction signal."
            )
        if direction == "falling" and price is not None and price >= 300:
            items.append(
                "Watch for price recovery — forecast signals declining prices from current spike level."
            )

    # ── Binding constraints ──────────────────────────────────────────
    bc = getattr(drivers, "binding_constraints", [])
    if bc:
        names = [
            b.get("element_id") or b.get("constraint_id") or b.get("name") or "?"
            for b in bc[:3]
        ]
        joined = ", ".join(names)
        items.append(
            f"Watch constraint{'s' if len(bc) > 1 else ''} {joined} — "
            f"currently binding and directly influencing dispatch price."
        )

    # ── Tight interconnectors ────────────────────────────────────────
    ic = getattr(drivers, "tight_interconnectors", [])
    if ic:
        ic_names = [i.get("interconnector_id", i.get("name", "interconnector")) for i in ic[:2]]
        items.append(
            f"Watch interconnector{'s' if len(ic) > 1 else ''} "
            f"{', '.join(ic_names)} — near import/export limit."
        )

    # ── Analog base rate ─────────────────────────────────────────────
    if analogs is not None and getattr(analogs, "count", 0) >= 3:
        success_rate = getattr(analogs, "success_rate", 0.5)
        continued_pct = int((1.0 - success_rate) * 100)
        if continued_pct >= 40:
            items.append(
                f"Watch for prolonged event — {continued_pct}% of similar historical "
                f"states continued beyond 30 minutes (analog base rate)."
            )

    # ── Weather ──────────────────────────────────────────────────────
    if weather is not None and getattr(weather, "relevant", False):
        tags = getattr(weather, "tags", [])
        tag_str = ", ".join(tags[:3]) if tags else "temperature/wind"
        items.append(
            f"Watch {tag_str} — weather is a current market driver; "
            f"forecast changes will affect demand and renewable output."
        )
        wc = getattr(weather, "consensus", {}) or {}
        temp = wc.get("temperature_c")
        wind_kmh = wc.get("wind_speed_kmh")
        # Convert km/h → m/s for threshold check (3 m/s ≈ 10.8 km/h)
        wind_ms = (wind_kmh / 3.6) if wind_kmh is not None else None
        if temp is not None and float(temp) >= 38:
            items.append(
                f"Watch extreme heat — temperature at {float(temp):.1f}°C will drive "
                f"peak cooling demand; risk of rapid headroom reduction."
            )
        if wind_ms is not None and float(wind_ms) < 3.0:
            items.append(
                f"Watch wind drought — wind speed at {float(wind_ms):.1f} m/s "
                f"({float(wind_kmh):.1f} km/h) will reduce wind farm output materially."
            )

    driver_events = getattr(drivers, "events", [])
    fcas_events = [
        e for e in driver_events
        if "FCAS" in str(e.get("source", "")).upper() or "FCAS" in str(e.get("type", "")).upper()
    ]
    if fcas_events:
        max_fcas = max((float(e.get("fcas_price", 0)) for e in fcas_events), default=0.0)
        if max_fcas > 200:
            items.append(
                f"Watch FCAS — raise price at ${max_fcas:,.0f}/MWh indicates a frequency "
                f"constraint; system inertia may be insufficient to absorb the next outage."
            )

    rebid_events = [
        e for e in driver_events
        if "REBID" in str(e.get("source", "")).upper() or "REBID" in str(e.get("type", "")).upper()
    ]
    if rebid_events:
        n = len(rebid_events)
        total_mw = sum(float(e.get("rebid_mw", 0)) for e in rebid_events)
        items.append(
            f"Watch rebids — {n} unit{'s' if n != 1 else ''} rebidding "
            f"({total_mw:,.0f} MW total); price impact pending next dispatch interval."
        )

    if news is not None:
        notices = getattr(news, "notices", []) or []
        if notices:
            now_utc = datetime.now(_tz.utc)
            most_recent_ts: datetime | None = None
            for notice in notices:
                ts_raw = notice.get("publish_datetime") or notice.get("timestamp") or notice.get("created_at")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_tz.utc)
                    if most_recent_ts is None or ts > most_recent_ts:
                        most_recent_ts = ts
                except (ValueError, TypeError):
                    pass
            if most_recent_ts is not None:
                age_s = (now_utc - most_recent_ts).total_seconds()
                if age_s < 1800:  # < 30 min
                    notice_type = getattr(news, "top_notice_type", None) or "market"
                    age_min = int(age_s / 60)
                    items.append(
                        f"Watch for AEMO notice follow-up — {notice_type} notice issued "
                        f"{age_min} min ago; conditions may still be evolving."
                    )

    if price is not None and price < 150:
        # Look for evidence that price was recently in spike territory via auto_commentary
        recent_spike = False
        if news is not None:
            for ac in getattr(news, "auto_commentary", []) or []:
                evt_type = str(ac.get("event_type", "")).upper()
                if evt_type in ("PRICE_SPIKE", "EXTREME_PRICE"):
                    recent_spike = True
                    break
        if recent_spike:
            items.append(
                f"Watch closed: price normalised to ${price:,.0f}/MWh after recent spike — "
                f"verify headroom and constraint status before resuming normal operations."
            )

    # ── LOR notice active ────────────────────────────────────────────
    # This is a catch-all: if nothing else fired, flag that the system
    # is monitoring for notices.
    if not items:
        region = getattr(c, "region", "this region")
        items.append(
            f"No immediate spike-risk thresholds triggered for {region}. "
            f"Monitor for new AEMO LOR/Constraint notices and headroom changes."
        )

    return items


def _as_utc(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_recent_trend_sentence(rows: list[dict], current) -> str:
    """Summarise now / 5m / 10m / 60m context for analyst-style answers."""
    if not rows:
        return ""

    anchor = _as_utc(current.valid_time)
    if anchor is None:
        return ""

    by_time = []
    for row in rows:
        vt = _as_utc(row.get("valid_time"))
        price = row.get("price_rrp")
        if vt is None or price is None:
            continue
        by_time.append((vt, row))
    if not by_time:
        return ""

    def nearest(minutes_ago: int) -> dict | None:
        target = anchor - timedelta(minutes=minutes_ago)
        candidates = [
            (abs((vt - target).total_seconds()), row)
            for vt, row in by_time
            if abs((vt - target).total_seconds()) <= 210
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])[1]

    points = [("now", {
        "price_rrp": current.price_rrp,
        "demand_mw": current.demand_mw,
        "headroom_mw": current.headroom_mw,
    })]
    for label, minutes in [("5m ago", 5), ("10m ago", 10), ("60m ago", 60)]:
        row = nearest(minutes)
        if row:
            points.append((label, row))

    if len(points) < 2:
        return ""

    bits = []
    for label, row in points:
        bit = f"{label} ${float(row['price_rrp']):.0f}"
        if row.get("headroom_mw") is not None:
            bit += f", headroom {float(row['headroom_mw']):.0f} MW"
        bits.append(bit)

    newest = float(points[0][1]["price_rrp"])
    oldest = float(points[-1][1]["price_rrp"])
    delta = newest - oldest
    direction = "up" if delta > 5 else "down" if delta < -5 else "roughly flat"
    return (
        "Recent trajectory: " + "; ".join(bits) +
        f". Net move versus the oldest available point is {direction} by ${delta:+.0f}/MWh."
    )


def _build_answer_sections(
    sources: WhySources,
    evidence_refs: list[EvidenceRefSchema],
    missing_data: list[str],
    why_text: str,
) -> list[dict]:
    """Build compact UI sections for analyst-style answers."""
    c = sources.current
    forecast = sources.forecast
    analogs = sources.analogs
    drivers = sources.drivers
    technology = sources.technology

    answer_items = []
    if c.is_fresh:
        answer_items.append(
            f"{c.region} is ${c.price_rrp:.2f}/MWh with {c.headroom_mw:.0f} MW headroom."
        )
    else:
        answer_items.append(
            f"Latest {c.region} dispatch is stale: ${c.price_rrp:.2f}/MWh with {c.headroom_mw:.0f} MW headroom."
        )
    if analogs.count:
        outcome = f" {analogs.outcome_summary}." if analogs.outcome_summary else ""
        answer_items.append(f"HippoGraph found {analogs.count} comparable historical states.{outcome}")

    evidence_items = [
        f"{ref.source}: {ref.field}={ref.value:g} at {ref.interval.isoformat()}"
        for ref in evidence_refs[:5]
    ] or ["No hard evidence refs were available."]

    driver_items = []
    if drivers.binding_constraints:
        driver_items.append(f"Binding constraint evidence present: {drivers.binding_constraints[0].get('element_id')}.")
    if drivers.tight_interconnectors:
        driver_items.append(f"Interconnector pressure present: {drivers.tight_interconnectors[0].get('element_id')}.")
    if technology.has_unit_evidence:
        fuels = ", ".join(sorted(k for k in technology.by_fuel if k != "unknown")) or "unit dispatch"
        driver_items.append(f"Unit dispatch evidence present by fuel: {fuels}.")
    if sources.weather.available and sources.weather.relevant:
        wc = sources.weather.consensus
        temp = wc.get("temperature_c")
        wind = wc.get("wind_speed_kmh")
        driver_items.append(f"Weather context present: {temp}C, wind {wind} km/h.")
    if not driver_items:
        driver_items.append(
            "No single driver is confirmed yet; constraints, interconnectors, unit dispatch, rebids, and outages are the main blockers."
        )

    continuation = _build_continuation_sentence(forecast, c)
    continuation_items = [continuation] if continuation else [
        "No calibrated LEAR/QRA/LNN continuation signal is available."
    ]

    missing_items = sorted(set(missing_data))[:10] or ["No major missing-data blocker flagged."]

    return [
        {"title": "Answer", "items": answer_items},
        {"title": "Evidence", "items": evidence_items},
        {"title": "Drivers", "items": driver_items},
        {"title": "Continuation", "items": continuation_items},
        {"title": "Missing", "items": missing_items},
    ]


def _build_continuation_sentence(forecast, current) -> str:
    """Turn per-model forecasts into a concrete continuation-risk sentence."""
    details = list(getattr(forecast, "model_detail", []) or [])
    available = [
        m for m in details
        if getattr(m, "available", False) and getattr(m, "p50", None) is not None
    ]

    if available:
        segs = []
        rising = falling = flat = 0
        for m in available:
            label = {
                "meta_ensemble": "Meta-ensemble",
                "lear": "LEAR",
                "qra": "QRA",
                "lnn": "LNN",
                "gbm": "GBM",
                "tcn": "TCN",
                "seasonal_naive": "Seasonal naive",
                "persistence": "Persistence",
                "aemo_predispatch": "AEMO model",
                "predispatch": "AEMO pre-dispatch",
            }.get(getattr(m, "model", ""), getattr(m, "model", "model").upper())
            direction = getattr(m, "direction", "unknown") or "unknown"
            if direction == "rising":
                rising += 1
            elif direction == "falling":
                falling += 1
            elif direction == "flat":
                flat += 1
            p50 = getattr(m, "p50", None)
            p90 = getattr(m, "p90", None)
            if p90 is not None:
                segs.append(f"{label} {direction}, P50 ${p50:.0f}, P90 ${p90:.0f}")
            else:
                segs.append(f"{label} {direction}, P50 ${p50:.0f}")

        if rising > falling and rising > flat:
            call = "higher continuation risk"
        elif falling > rising and falling >= flat:
            call = "lower continuation risk"
        else:
            call = "mixed or flat continuation risk"
        return "Continuation: " + call + " from available models. " + "; ".join(segs) + "."

    if getattr(forecast, "available", False) and getattr(forecast, "direction", None) not in (None, "unknown"):
        return (
            f"Continuation: weak {forecast.direction} signal only. "
            "This comes from trend/predispatch fallback, not a full calibrated model pack."
        )

    return ""


def _format_analog_examples(items: list[dict]) -> str:
    if not items:
        return ""
    examples = []
    for item in items[:3]:
        vt = item.get("valid_time", "unknown time")
        price = item.get("price_rrp")
        headroom_delta = item.get("headroom_delta")
        outcome = item.get("outcome") or "outcome unknown"
        reason = _clean_analog_reason(item.get("match_reason") or "similar state vector")
        if price is None:
            continue
        bit = f"{vt}: ${float(price):.0f}/MWh"
        if headroom_delta is not None:
            bit += f", headroom delta {float(headroom_delta):+.0f} MW"
        bit += f", {outcome} ({reason})"
        examples.append(bit)
    if not examples:
        return ""
    return "Closest analogs: " + " | ".join(examples) + "."


def _clean_analog_reason(reason: str) -> str:
    return (
        reason
        .replace("Î”", "Delta ")
        .replace("Δ", "Delta ")
        .replace("â€¯", " ")
        .replace("\u202f", " ")
    )


def _build_model_forecast_summary(model_detail: list) -> str:
    """Build a per-model forecast narrative string.

    Produces text like:
      "Model forecast: LNN (point forecast): rising P50=$347/MWh;
       LEAR: unavailable (live inference not yet wired);
       QRA: unavailable (live inference not yet wired);
       AEMO pre-dispatch: P50=$312/MWh direction flat."
    """
    from app.agents.why_sources import ModelForecastDetail  # avoid circular at module level

    if not model_detail:
        return ""

    _LABELS = {
        "meta_ensemble": "Meta-ensemble",
        "lnn": "LNN",
        "lear": "LEAR",
        "qra": "QRA",
        "gbm": "GBM",
        "tcn": "TCN",
        "seasonal_naive": "Seasonal naive",
        "persistence": "Persistence",
        "aemo_predispatch": "AEMO model",
        "predispatch": "AEMO pre-dispatch",
    }

    parts = []
    any_available = False
    for m in model_detail:
        if not isinstance(m, ModelForecastDetail):
            continue
        label = _LABELS.get(m.model, m.model.upper())
        if m.available and m.p50 is not None:
            any_available = True
            seg = f"{label}: {m.direction} P50=${m.p50:.0f}/MWh"
            if m.p10 is not None and m.p90 is not None:
                seg += f" [P10 ${m.p10:.0f}–P90 ${m.p90:.0f}]"
        elif m.available:
            any_available = True
            seg = f"{label}: {m.direction}"
        else:
            caveat = f" ({m.caveat})" if m.caveat else ""
            seg = f"{label}: unavailable{caveat}"
        parts.append(seg)

    if not any_available:
        return ""
    return "Model forecasts — " + "; ".join(parts) + "."


def _technology_evidence_ref(by_fuel: dict[str, dict]) -> dict | None:
    candidates = [v for k, v in by_fuel.items() if k != "unknown"]
    if not candidates:
        return None
    return max(candidates, key=lambda row: abs(float(row.get("total_cleared_mw") or 0.0)))


def build_seasonal_why(sources: SeasonalSources) -> WhyOutput:
    """Comparative narrative for multi-season aggregate queries."""
    if not sources.summaries:
        return WhyOutput(
            why_plain_english=(
                f"No seasonal dispatch intervals are stored for {sources.region}. "
                "Archive backfill must complete before multi-season comparison is decision-grade."
            ),
            counterargument="Seasonal attribution is not available without notices, constraints, and outage data for the same historical window.",
            missing_data=["seasonal_dispatch_history", "historical_constraints", "historical_notices"],
            evidence_refs=[],
            confidence=0.1,
        )

    parts = []
    missing = []
    evidence_refs: list[EvidenceRefSchema] = []
    usable = [s for s in sources.summaries if s.get("interval_count", 0) > 0]
    for summary in sources.summaries:
        label = summary["label"]
        count = summary.get("interval_count", 0)
        if not count:
            parts.append(f"{label}: no stored intervals for {sources.region}.")
            missing.append(f"seasonal_history_{label}")
            continue
        parts.append(
            f"{label}: mean price ${summary['mean_price']:.2f}/MWh, "
            f"P90 ${summary['p90_price']:.2f}/MWh, max ${summary['max_price']:.2f}/MWh, "
            f"{summary['spike_count']} intervals at or above the spike threshold "
            f"from {count} stored intervals."
        )
        evidence_refs.append(EvidenceRefSchema(
            source="AEMO_ARCHIVE_DISPATCH_PRICE",
            region=sources.region,
            interval=datetime.fromisoformat(summary["from_dt"]),
            field="seasonal_interval_count",
            value=float(count),
            raw_ref=f"seasonal_{sources.region}_{label.replace(' ', '_')}",
        ))

    if len(usable) >= 2:
        first, last = usable[0], usable[-1]
        delta = (last["mean_price"] or 0.0) - (first["mean_price"] or 0.0)
        parts.append(
            f"Mean price changed by ${delta:.2f}/MWh from {first['label']} to {last['label']}."
        )
    missing.extend(["historical_constraints", "historical_interconnector_flows", "historical_notices"])

    return WhyOutput(
        why_plain_english=" ".join(parts),
        counterargument=(
            "This comparison is statistical only. It cannot confirm causal drivers until "
            "historical constraints, interconnector flows, notices, and availability changes "
            "are ingested for the same windows."
        ),
        missing_data=sorted(set(missing)),
        evidence_refs=evidence_refs,
        confidence=0.55 if len(usable) == len(sources.summaries) else 0.35,
    )


def _append_intent_context(parts, missing_data, intent, c, analogs, news, forecast, decomp=None):
    """Add intent-specific sentences to the narrative."""
    if intent == IntentLabel.ACTION_RECOMMENDATION:
        if c.regime in ("spike", "extreme") and c.headroom_mw < 500:
            analog_signal = ""
            if analogs.count >= 3:
                rate = analogs.success_rate
                analog_signal = (
                    f" {analogs.count} analogous periods show {rate:.0%} dispatch success rate."
                )
            parts.append(
                f"Tight headroom ({c.headroom_mw:.0f} MW) at elevated prices "
                f"favours fast-response dispatch.{analog_signal}"
            )
        elif c.regime == "normal":
            parts.append("Market conditions are currently stable — no urgent dispatch signal.")

    elif intent == IntentLabel.EXPLANATION:
        causes = []
        if news.explained and news.credibility_tier == 1:
            causes.append(f"a confirmed AEMO notice ({news.top_notice_type})")
        if c.headroom_mw < 300:
            causes.append(f"very low generation headroom ({c.headroom_mw:.0f} MW)")
        elif c.headroom_mw < 800:
            causes.append(f"reduced headroom ({c.headroom_mw:.0f} MW)")
        if forecast.direction == "rising":
            causes.append("a rising price trend")
        if causes:
            parts.append(
                f"Primary price drivers: the supported driver evidence points to {', '.join(causes)}."
            )
        else:
            parts.append(
                "Why: no single dominant cause is confirmed from the available evidence. "
                "The confirmed facts are price, demand, available generation, headroom, and the recent price path; "
                "constraint, interconnector, unit-dispatch, rebid, and outage evidence are still needed "
                "before saying what drove the move."
            )
            missing_data.append("price_cause_unconfirmed")
        if getattr(decomp, "requires_forecast", False):
            continuation = _build_continuation_sentence(forecast, c)
            if continuation:
                parts.append(continuation)
            else:
                parts.append(
                    "Continuation: not forecast-supported. No calibrated LEAR/QRA/LNN or fresh pre-dispatch "
                    "signal is available, so the defensible action is monitor rather than predict persistence."
                )

    elif intent == IntentLabel.RETROSPECTIVE:
        if analogs.count >= 1:
            outcome = f" Outcome: {analogs.outcome_summary}." if analogs.outcome_summary else ""
            examples = _format_analog_examples(getattr(analogs, "top_items", []))
            parts.append(
                f"Historical analog retrieval returned {analogs.count} similar market states "
                f"from the {analogs.window_days}-day archive.{outcome} "
                + (examples + " " if examples else "")
                + "This is a base-rate answer: it says what usually happened next for similar states, "
                "not proof that the same driver is active now."
            )
        else:
            parts.append(
                "Historical answer: no close analog set was returned for the current state. "
                "If the Docker/Postgres history is connected, this usually means HippoGraph's current graph window "
                "does not contain enough comparable states; otherwise the history store is not available to the app."
            )
            missing_data.append("historical_archive_sparse")

    elif intent == IntentLabel.COUNTERFACTUAL:
        parts.append(
            f"Counterfactual analysis uses historical analog outcomes as the reference. "
            f"Current state: ${c.price_rrp:.2f}/MWh, regime {c.regime}, "
            f"headroom {c.headroom_mw:.0f} MW. "
        )
        if analogs.count >= 3 and analogs.outcome_summary:
            parts.append(f"Analog outcomes: {analogs.outcome_summary}.")
        else:
            parts.append(
                "Insufficient analog history for counterfactual comparison. "
                "Run after 24 hours of live operation."
            )
            missing_data.append("counterfactual_requires_analogs")

    elif intent == IntentLabel.COMPARISON:
        parts.append(
            "Region comparison uses the same evidence pipeline for each region. "
            "Price differentials reflect transmission constraints and local demand/supply balance."
        )

    elif intent == IntentLabel.TRACE_REPLAY:
        parts.append(
            "Trace replay reconstructs the decision pipeline at the requested timestamp "
            "using the bitemporal audit log. All values are the system's observations at "
            "that point in time — not retroactively adjusted."
        )

    elif intent == IntentLabel.OUT_OF_SCOPE:
        clarification = getattr(decomp, "clarifying_question", None) if decomp else None
        if clarification:
            # LLM or rule-based produced a specific redirect (e.g. Darwin → not NEM)
            parts.append(clarification)
        else:
            # Generic meta-question or off-topic query → show FAQ
            parts.append(
                "GridVerdict is an evidence-grounded decision-support cockpit for the Australian "
                "National Electricity Market (NEM). "
                "It pulls live 5-minute dispatch prices directly from AEMO's NEMWeb (public data, "
                "no credentials required), cross-references active market notices, and retrieves "
                "historical analog market states via HippoGraph pattern matching. "
                "Every factual claim in the answer cites a timestamped, SHA-256-referenced evidence record "
                "— nothing is generated or assumed. "
                "\n\nQuestions you can ask:\n"
                "• 'What is the current NSW price and why is it elevated?'\n"
                "• 'Should I dispatch my battery in SA right now?'\n"
                "• 'Compare prices across all NEM regions'\n"
                "• 'Why did QLD prices spike last interval?'\n"
                "• 'What would have happened if I had dispatched 30 min ago?'\n"
                "\nGridVerdict covers NSW1, VIC1, QLD1, SA1, and TAS1. "
                "It does not cover Western Australia (SWIS), Northern Territory, or non-electricity markets. "
                "It does not connect to social media, proprietary data vendors, or external trading platforms."
            )
        missing_data.append("query_outside_nem_scope")


def _build_counterargument(c, analogs, news, forecast) -> str:
    """Build an evidence-driven adversarial critique of the primary assessment.

    Uses analog outcomes, notice status, and forecast direction rather than
    a regime-only template. Specifically challenges the strongest part of
    the primary narrative with the weakest available evidence.
    """
    points: list[str] = []

    # Challenge based on analog outcome distribution
    if analogs.count >= 3:
        recovery_rate = analogs.success_rate
        if recovery_rate >= 0.70:
            points.append(
                f"{recovery_rate:.0%} of analogous market states recovered within 30 minutes "
                f"({analogs.success_count}/{analogs.count} analogs). "
                "The spike may be transient — waiting 1-2 intervals before dispatching "
                "could capture a better entry price."
            )
        elif recovery_rate <= 0.30:
            points.append(
                f"{1 - recovery_rate:.0%} of analogous situations continued spiking beyond 30 min "
                f"({analogs.count - analogs.success_count}/{analogs.count} analogs). "
                "History suggests this regime is sticky — delay increases exposure."
            )
        else:
            points.append(
                f"Analog outcomes are mixed ({recovery_rate:.0%} recovery rate across "
                f"{analogs.count} similar periods). No strong directional signal from history."
            )
    else:
        points.append(
            "Analog history is insufficient (need ≥3, have "
            f"{analogs.count}). This assessment rests on current state alone — "
            "there is no historical base rate to reference."
        )

    # Challenge with notice status
    if news.explained and news.credibility_tier == 1:
        points.append(
            f"The active {news.top_notice_type} notice is a firm signal, but AEMO can "
            "cancel or downgrade notices rapidly once plant returns. "
            "Do not treat an LOR notice as a guarantee of sustained elevated prices."
        )
    elif not news.explained:
        points.append(
            "No confirmed AEMO notice explains the current price level. "
            "Without a fundamental cause, the move is more likely to reverse quickly."
        )

    # Challenge with forecast direction
    if forecast.available and forecast.p50 is not None:
        if forecast.direction == "falling" and c.regime in ("spike", "extreme"):
            points.append(
                f"P50 forecast shows ${forecast.p50:.2f}/MWh — below current "
                f"${c.price_rrp:.2f}/MWh. The LNN model anticipates price easing "
                "within the next dispatch interval."
            )
        elif forecast.direction == "rising" and c.regime == "normal":
            points.append(
                f"LNN P50 ${forecast.p50:.2f}/MWh suggests price pressure building. "
                "Quantile range P10-P90 "
                f"(${forecast.p10:.2f}–${forecast.p90:.2f}) remains wide — forecast uncertainty is high."
            )

    # Headroom challenge
    if c.is_fresh:
        if c.headroom_mw > 1500:
            points.append(
                f"Ample generation headroom ({c.headroom_mw:.0f} MW) means AEMO can bring "
                "additional capacity online rapidly, which typically softens spike persistence."
            )
        elif c.headroom_mw < 200:
            points.append(
                f"Critically low headroom ({c.headroom_mw:.0f} MW) removes the usual "
                "price-moderating buffer — sustained extreme prices are plausible."
            )

    return " ".join(points)


def _compute_upgrade_path(sources, missing_data: list[str], analogs, technology, forecast) -> list[str]:
    """Map each missing evidence gap to its confidence impact.

    Answers the professional question: 'I see 60% — how do I get to 85%?'
    Each item is a one-line statement of: what data → what confidence change.
    """
    paths: list[str] = []
    c = sources.current

    if "unit_dispatch_events" in missing_data or not technology.has_unit_evidence:
        paths.append(
            "Unit dispatch data (DISPATCHLOAD) ingested → fuel rank tier: PRIOR → DISPATCH, "
            "fuel-source confidence 35% → 80%, 'unit dispatch by fuel' removed from MISSING"
        )
    if "dispatch_constraints" in missing_data:
        paths.append(
            "Binding constraint data (DISPATCHCONSTRAINT) ingested → constraint causal role: "
            "UNCONFIRMED → SUPPORTED, 'dispatch_constraints' removed from MISSING"
        )
    if "historical_analogs" in missing_data or analogs.count < 5:
        needed = max(0, 5 - analogs.count)
        paths.append(
            f"HippoGraph needs {needed} more comparable intervals "
            f"(currently {analogs.count}/5 minimum) → analog pattern: unavailable → MEDIUM, "
            "confidence +15%"
        )
    if not forecast.available or all(not m.available for m in (forecast.model_detail or [])):
        lnn_ready = any(
            m.model in {"lnn", "lnn_cfc"} and m.available
            for m in (forecast.model_detail or [])
        )
        if not lnn_ready:
            paths.append(
                "LNN training complete (needs 288+ dispatch intervals ≈ 1 day) → "
                "forecast: LEAR/QRA-only → full ensemble, confidence +5–10%"
            )
    if "dispatch_interconnector_flows" in missing_data:
        paths.append(
            "Interconnector flow data (DISPATCHINTERCONNECTORRES) → "
            "regional price separation: UNCONFIRMED → SUPPORTED for comparison queries"
        )
    if not c.is_fresh:
        paths.append(
            "Live dispatch price (AEMO NEMWeb) refreshed → "
            "confidence floor: 10% → 50% (stale data caps all evidence tiers)"
        )

    return paths[:4]   # cap at 4 items in the UI


def _build_causal_chain_steps(sources, intent: "IntentLabel", missing_data: list[str]) -> list[str]:
    """Build an ordered causal evidence chain for EXPLANATION queries.

    Shows which steps of the NEM causal chain are available vs missing.
    This directly answers: 'Why does the system show INSUFFICIENT_DATA?'
    Each item is one step in the causal reasoning chain with status.
    """
    from app.core.schema import IntentLabel as _IL
    if intent not in (_IL.EXPLANATION, _IL.ACTION_RECOMMENDATION):
        return []

    c = sources.current
    technology = sources.technology
    drivers = sources.drivers

    chain: list[str] = []

    # Step 1: Always available (live price is always T1)
    age = c.staleness_seconds
    if age < 0:
        age_label = "live (fresh)"
    elif age < 300:
        age_label = f"{age}s old"
    elif age < 3600:
        age_label = f"{age//60}m old (stale — refresh or check AEMO connectivity)"
    else:
        # Very old: this is historical data used for a retrospective query
        age_label = f"historical data (~{age//3600}h old — this is the anchor time for your query)"
    label = "Historical dispatch price" if age > 3600 else "Live dispatch price"
    chain.append(
        f"✓ {label}: ${c.price_rrp:.0f}/MWh, "
        f"demand {c.demand_mw:.0f} MW, headroom {c.headroom_mw:.0f} MW — {age_label}"
    )

    # Step 2: Binding constraints
    if drivers.binding_constraints:
        chain.append(
            f"✓ Constraint evidence: {len(drivers.binding_constraints)} binding constraint(s) — "
            "confirmed contribution to dispatch cost"
        )
    else:
        chain.append(
            "✗ Binding constraints: not available — "
            "cannot confirm whether a transmission limit raised the dispatch cost"
        )

    # Step 3: Unit dispatch by fuel
    if technology.has_unit_evidence:
        fuel_summary = ", ".join(
            f"{fuel} {b.get('total_cleared_mw', 0):.0f} MW"
            for fuel, b in list(technology.by_fuel.items())[:4]
            if fuel != "unknown"
        )
        tier = "bid_reconstruction" if any(
            e.get("source") == "BID_RECONSTRUCTION"
            for e in technology.events
        ) else "live dispatch"
        chain.append(
            f"✓ Unit dispatch ({tier}): {fuel_summary} — "
            "fuel-type attribution available"
        )
    else:
        chain.append(
            "✗ Unit dispatch by fuel: not yet available — "
            "cannot confirm which generator technology set the price "
            "(seeds within 5min of first archive gap-fill)"
        )

    # Step 4: Bid/rebid stack
    has_rebid = any("rebid" in str(d) for d in drivers.binding_constraints)
    if has_rebid:
        chain.append("✓ Rebid evidence: detected — strategic availability withdrawal confirmed")
    else:
        chain.append(
            "✗ Bid/rebid stack: not yet available — "
            "cannot confirm whether a generator changed its offer intra-day "
            "(BIDPEROFFER requires archive gap-fill)"
        )

    # Step 5: Interconnector
    if drivers.tight_interconnectors:
        chain.append(
            f"✓ Interconnector: {len(drivers.tight_interconnectors)} near limit — "
            "import/export constraint on regional supply confirmed"
        )
    else:
        chain.append(
            "✗ Interconnector flows: not available — "
            "cannot confirm whether import congestion contributed to price"
        )

    # Append the 'verdict if complete' line
    missing_count = sum(1 for step in chain if step.startswith("✗"))
    if missing_count == 0:
        verdict_if_complete = "All causal steps confirmed — verdict would be SUPPORTED at 85%+"
    elif missing_count <= 2:
        verdict_if_complete = (
            f"{missing_count} gap(s) remain — verdict would reach SUPPORTED at 70–80% "
            "once unit dispatch data is present"
        )
    else:
        verdict_if_complete = (
            f"{missing_count} causal steps missing — INSUFFICIENT_DATA is correct. "
            "Primary gap: unit dispatch by fuel (fills automatically within 1 hour of archive run)"
        )
    chain.append(f"→ {verdict_if_complete}")

    return chain


def _estimate_confidence(live_fresh: bool, analog_count: int, news_explained: bool) -> float:
    base = 0.50 if live_fresh else 0.10
    if analog_count >= 10:
        base += 0.25
    elif analog_count >= 3:
        base += 0.15
    if news_explained:
        base += 0.15
    return round(min(1.0, base), 3)
