"""Query decomposer — structured intent extraction from natural language.

Backend priority:
  1. Ollama (local, default) — mistral or any model running on localhost:11434
  2. Claude (Anthropic API) — fallback when Ollama is unreachable
  3. Rule-based — final fallback, no LLM dependency

The decomposer calls an LLM with a structured JSON prompt and validates the
output against the QueryDecomposition schema. If the LLM output is malformed,
it falls back to the rule-based classifier rather than raising.

SECURITY: Raw user text is passed to the LLM inside a role=user message ONLY.
The system prompt is static, loaded from this file, and never includes user content.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.schema import IntentLabel, QueryDecomposition
from app.engines.geo_aliases import (
    build_geo_prompt_section,
    correct_region_hallucinations,
    detect_non_nem,
    detect_regions,
    known_regions,
)
from app.engines.temporal_utils import extract_season_buckets
from config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_SYSTEM_PROMPT = """You are a query decomposer for GridVerdict, an Australian National Electricity Market (NEM) decision-support system.

Your task: extract structured intent from a natural language query about the NEM.

Return ONLY valid JSON matching this schema (no markdown, no explanation, no <think> tags):
{
  "intent": one of ["action_recommendation", "explanation", "retrospective", "counterfactual", "comparison", "lookup", "trace_replay", "out_of_scope"],
  "entities": {
    "regions": list of NEM region codes (NSW1, VIC1, QLD1, SA1, TAS1),
    "generators": list of generator names if mentioned,
    "technologies": list of technology types (battery, hydro, gas, solar, wind, coal)
  },
  "time_range": {
    "type": "current" | "historical" | "forecast" | "range",
    "from_offset": null or offset description (e.g., "1h ago", "3 months ago", "yesterday"),
    "to_offset": null or offset description
  },
  "requires_why": true if question asks for cause/reason/explanation,
  "requires_history": true if question refers to past events,
  "requires_forecast": true if question asks about future,
  "requires_backtest": true if question asks to simulate or backtest,
  "requires_portfolio": false,
  "confidence": float 0.0-1.0 reflecting how clearly the query maps to NEM topics,
  "ambiguities": list of strings describing anything unclear,
  "clarifying_question": null or a single helpful string question if confidence < 0.6
}

## NEM region codes and Australian geography

The NEM has exactly 5 regions. Map any Australian place name to its NEM region:
- NSW1: New South Wales and ACT — includes Sydney, Canberra, Newcastle, Wollongong, Penrith, Parramatta, Orange, Griffith (Griffith is in NSW, not Victoria), Dubbo, Wagga Wagga, Albury, Broken Hill
- VIC1: Victoria — includes Melbourne, Geelong, Ballarat, Bendigo, Shepparton, Mildura
- QLD1: Queensland — includes Brisbane, Gold Coast, Cairns, Townsville, Rockhampton, Toowoomba
- SA1: South Australia — includes Adelaide, Port Augusta, Whyalla, Port Pirie
- TAS1: Tasmania — includes Hobart, Launceston, Devonport

## Regions NOT in the NEM (cannot answer data questions about these)

These are connected to separate grids and have NO data in GridVerdict:
- Northern Territory (Darwin, Alice Springs, Katherine) — uses the Darwin-Katherine Interconnected System, NOT the NEM
- Western Australia (Perth, Broome, Kalgoorlie) — uses SWIS/NWIS, NOT the NEM
- If the user asks about these, set intent="out_of_scope" and use clarifying_question to explain and suggest a NEM region

## Intent rules

- "should I dispatch / bid / act" → action_recommendation
- "why is / what caused / explain / reason for" → explanation
- "last time / historical / what happened / when did" → retrospective
- "what if / simulate / counterfactual / what would" → counterfactual
- "compare / versus / vs / difference between" → comparison
- "how does this work / what is GridVerdict / what can you do" → out_of_scope, clarifying_question = "GridVerdict answers live and historical NEM electricity market questions. Try: 'Why is the NSW price elevated?' or 'Should I dispatch my battery in SA right now?'"
- Unrelated to electricity → out_of_scope, confidence < 0.2

## Ambiguity rules

- Set confidence < 0.6 when: region is unclear, time period is vague, or the query mixes non-NEM geography
- Always add an ambiguity entry when a place name maps to a NEM region (e.g. "Penrith mapped to NSW1")
- Always add an ambiguity entry when a non-NEM region is mentioned (e.g. "Darwin is not in the NEM")
- Never invent information not in the query

## Historical data availability

GridVerdict is a live system. Historical analog data accumulates from the first day of operation. Inform users: "Historical analog retrieval covers the current operation window. For extended history (months), note that the archive backfill expands incrementally." Set requires_history=true for any past-looking query.
"""

_SYSTEM_PROMPT = _SYSTEM_PROMPT + "\n\n## Authoritative geography alias table\n\n" + build_geo_prompt_section()


async def decompose(
    text: str,
    region_hint: str = "NSW1",
    query_id: str | None = None,
) -> QueryDecomposition:
    """Decompose a query using the configured LLM backend.

    Falls back through: ollama → claude → rule_based on failure.
    Never raises — always returns a valid QueryDecomposition.
    """
    backend = _settings.decomposer_backend

    if backend == "ollama" or backend == "auto":
        try:
            return await _decompose_ollama(text, region_hint, query_id)
        except Exception as exc:
            logger.warning("Ollama decomposer failed (%s), trying Claude", exc)

    if backend in ("claude", "auto") or _settings.anthropic_api_key:
        try:
            return await _decompose_claude(text, region_hint, query_id)
        except Exception as exc:
            logger.warning("Claude decomposer failed (%s), falling back to rule-based", exc)

    return _decompose_rules(text, region_hint, query_id)


_REASONING_MODELS = {"qwen3", "qwq", "deepseek-r1", "marco-o1"}


def _is_reasoning_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(r in lower for r in _REASONING_MODELS)


async def _decompose_ollama(
    text: str, region_hint: str, query_id: str | None
) -> QueryDecomposition:
    """Call Ollama's /api/chat endpoint.

    Prepends /no_think to the user message on qwen3/deepseek-r1 family models
    to disable chain-of-thought and avoid spending all tokens on reasoning traces.
    """
    model = _settings.ollama_model
    user_content = f"/no_think {text}" if _is_reasoning_model(model) else text

    # Note: format:"json" is omitted for reasoning models (qwen3/deepseek-r1) —
    # the grammar constraint silences their output when combined with /no_think.
    # We rely on the explicit system-prompt instruction instead and strip any
    # stray markdown in _parse_llm_output.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 768,
            "num_ctx": 4096,    # ensure long system prompt fits
        },
    }
    if not _is_reasoning_model(model):
        payload["format"] = "json"
    async with httpx.AsyncClient(
        base_url=_settings.ollama_base_url,
        timeout=httpx.Timeout(_settings.ollama_timeout_s),
    ) as client:
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = data["message"]["content"]

    return _parse_llm_output(raw, text, region_hint, query_id)


async def _decompose_claude(
    text: str, region_hint: str, query_id: str | None
) -> QueryDecomposition:
    """Call Claude via Anthropic API."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=_settings.anthropic_api_key)
    message = await client.messages.create(
        model=_settings.anthropic_model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        temperature=0.0,
    )
    raw = message.content[0].text
    return _parse_llm_output(raw, text, region_hint, query_id)


def _parse_llm_output(
    raw: str, original_text: str, region_hint: str, query_id: str | None
) -> QueryDecomposition:
    """Parse and validate LLM JSON output into QueryDecomposition.
    Falls back to rule-based on any parse or validation error.
    """
    try:
        # Strip think/reasoning tags (qwen3, deepseek-r1) and code fences
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        clean = re.sub(r"```(?:json)?", "", clean).strip()
        data: dict[str, Any] = json.loads(clean)

        # Normalise intent
        intent_str = data.get("intent", "lookup")
        try:
            intent = IntentLabel(intent_str)
        except ValueError:
            intent = IntentLabel.LOOKUP

        # Ensure region_hint is in entities if LLM found no regions
        entities = data.get("entities", {})
        if not entities.get("regions"):
            entities["regions"] = [region_hint]
        entities, region_corrections = correct_region_hallucinations(entities, original_text)

        confidence = float(data.get("confidence", 0.75))
        confidence = max(0.0, min(1.0, confidence))
        ambiguities = list(data.get("ambiguities", []))
        ambiguities.extend(region_corrections)
        time_range = data.get("time_range", {"type": "current"})
        season_buckets = extract_season_buckets(original_text)
        if season_buckets:
            time_range = dict(time_range or {})
            time_range["type"] = "seasonal"
            time_range["season_buckets"] = season_buckets
        requires_why = bool(data.get("requires_why", False))
        requires_history = bool(data.get("requires_history", False)) or bool(season_buckets)
        requires_forecast = bool(data.get("requires_forecast", False))
        causal_targets = list(data.get("causal_targets") or _extract_causal_targets(original_text.lower()))
        if intent == IntentLabel.EXPLANATION:
            causal_targets = _expand_explanation_targets(causal_targets, requires_forecast)
        _llm_requested = data.get("requested_output") or ""
        _rule_requested = _requested_output_for(
            original_text.lower(), intent, requires_why, requires_history, requires_forecast
        )
        # Rule-based wins for strong signal types; LLM value used only when rule falls back to
        # a generic bucket that the LLM may have classified more specifically.
        _generic_outputs = {
            "current_market_state", "causal_explanation",
            "causal_explanation_with_forecast", "forecast",
        }
        requested_output = (
            _rule_requested
            if _rule_requested not in _generic_outputs
            else (_llm_requested or _rule_requested)
        )

        return QueryDecomposition(
            query_id=query_id or f"qry-{_uuid8()}",
            raw_query=original_text,
            intent=intent,
            entities=entities,
            time_range=time_range,
            requires_why=requires_why,
            requires_history=requires_history,
            requires_forecast=requires_forecast,
            requires_backtest=bool(data.get("requires_backtest", False)),
            requires_portfolio=bool(data.get("requires_portfolio", False)),
            requires_live_market=bool(data.get("requires_live_market", intent != IntentLabel.OUT_OF_SCOPE)),
            requires_incident_timeline=bool(data.get(
                "requires_incident_timeline",
                intent in (IntentLabel.EXPLANATION, IntentLabel.ACTION_RECOMMENDATION),
            )),
            requires_bess_context=bool(data.get("requires_bess_context", False)),
            causal_targets=causal_targets,
            spike_thresholds=list(data.get("spike_thresholds") or []),
            action_context=data.get("action_context"),
            missing_inputs=list(data.get("missing_inputs") or []),
            requested_output=requested_output,
            confidence=confidence,
            ambiguities=ambiguities,
            clarifying_question=data.get("clarifying_question"),
            region_corrections=region_corrections,
        )
    except Exception as exc:
        logger.warning("LLM output parse failed (%s) — using rule-based fallback", exc)
        return _decompose_rules(original_text, region_hint, query_id)


def _decompose_rules(
    text: str, region_hint: str, query_id: str | None
) -> QueryDecomposition:
    """Rule-based intent classifier — deterministic, no LLM dependency."""
    lower = text.lower()

    intent = IntentLabel.LOOKUP
    confidence = 0.65
    ambiguities: list[str] = []
    clarifying_question: str | None = None

    # Intent signals (order matters — more specific signals first)
    unsafe_action = bool(re.search(
        r"\b(submit|place|send|execute|lodge)\b.*\b(real|live|actual)?\s*(dispatch\s+)?bid\b|\b(dispatch\s+bid|bid\s+\d+)",
        lower,
    ))
    if unsafe_action:
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.95
        clarifying_question = (
            "GridVerdict cannot submit, place, or execute market bids. "
            "It can only provide evidence-grounded NEM decision support."
        )
    elif any(w in lower for w in [
        "replay", "trace", "what did you", "your decision at",
        "system know at the time", "know at the time", "known at the time",
    ]):
        intent = IntentLabel.TRACE_REPLAY
        confidence = 0.82
    elif any(w in lower for w in [
        "should i", "dispatch now", "what action", "recommend",
        "before i act", "missing before i act", "should reserve",
        "reserve capacity",
    ]):
        intent = IntentLabel.ACTION_RECOMMENDATION
        confidence = 0.82
    elif any(w in lower for w in [
        "why is", "why are", "what is driving", "reason for", "explain",
        "because", "what changed", "evidence supports", "what evidence",
        "price move", "backed by", "market balance", "why did", "causing",
        "cause", "fluctuate", "fluctuation", "moved from", "back down",
    ]):
        intent = IntentLabel.EXPLANATION
        confidence = 0.85
    elif any(w in lower for w in ["what would", "if i had", "counterfactual", "simulate", "what if"]):
        intent = IntentLabel.COUNTERFACTUAL
        confidence = 0.80
    elif any(w in lower for w in [
        "last time", "historical", "when did", "what happened",
        "has this happened", "happened before", "similar", "before",
        "analogs", "yesterday",
    ]):
        intent = IntentLabel.RETROSPECTIVE
        confidence = 0.80
    elif any(w in lower for w in ["compare", "versus", "vs", "difference between"]):
        intent = IntentLabel.COMPARISON
        confidence = 0.78
    elif any(w in lower for w in ["forecast", "tomorrow", "will price", "going to", "expected"]):
        intent = IntentLabel.LOOKUP
        confidence = 0.75

    # Meta-questions about the platform itself
    _meta_phrases = [
        "how does this work", "how does gridverdict", "what is gridverdict",
        "what can you do", "how does it work", "tell me about yourself",
        "what are your capabilities", "how do you work",
    ]
    if any(p in lower for p in _meta_phrases):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.95

    # OOS detection — non-electricity topics
    _oos_words = ["stock", "forex", "crypto", "bitcoin", "election", "football"]
    _external_platforms = [
        "bloomberg", "refinitiv", "ice nexus", "nem-review", "nem review",
        "opennem", "wattclarity", "iress", "factset",
    ]
    if any(w in lower for w in _oos_words):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
    if "weather" in lower and not _has_market_weather_context(lower):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
    if any(p in lower for p in _external_platforms):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
        clarifying_question = (
            "For a comparison of GridVerdict with other NEM data platforms, see the About tab."
        )

    non_nem_notes = detect_non_nem(text)
    if non_nem_notes:
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
        ambiguities.extend(non_nem_notes)
        clarifying_question = (
            f"{non_nem_notes[0]} GridVerdict currently covers NSW1, VIC1, QLD1, SA1, and TAS1."
        )

    # Region detection
    _ALL_REGIONS = known_regions()
    regions = []
    # "all regions" / "all NEM regions" / "each region" → expand to all 5
    if any(phrase in lower for phrase in ["all region", "all nem region", "each region", "every region", "all states", "all nem"]):
        regions = _ALL_REGIONS[:]
    else:
        alias_regions, alias_notes = detect_regions(text)
        regions.extend(alias_regions)
        ambiguities.extend(alias_notes)
    if not regions:
        regions = [region_hint]
    technologies = _extract_technologies(lower)

    # Flag non-NEM geography as ambiguities even when a NEM region is detected
    for place, note in [
        ("darwin", "Darwin is in the Northern Territory — not connected to the NEM."),
        ("alice springs", "Alice Springs is in the NT — not in the NEM."),
        ("perth", "Perth uses the SWIS grid — not part of the NEM."),
        ("western australia", "Western Australia has its own grid (SWIS/NWIS), not in the NEM."),
    ]:
        if place in lower:
            ambiguities.append(note)
            if intent != IntentLabel.OUT_OF_SCOPE:
                clarifying_question = (
                    f"{note} The NEM covers NSW1, VIC1, QLD1, SA1, TAS1. "
                    "Which NEM region did you mean?"
                )

    if regions:
        for place, nem in [("penrith", "NSW1"), ("griffith", "NSW1"), ("parramatta", "NSW1"),
                           ("wollongong", "NSW1"), ("newcastle", "NSW1"), ("canberra", "NSW1"),
                           ("geelong", "VIC1"), ("ballarat", "VIC1"), ("bendigo", "VIC1"),
                           ("gold coast", "QLD1"), ("cairns", "QLD1"), ("townsville", "QLD1"),
                           ("adelaide", "SA1"), ("hobart", "TAS1")]:
            if place in lower and nem not in regions:
                ambiguities.append(f"{place.title()} mapped to {nem}")

    season_buckets = extract_season_buckets(text)
    if season_buckets and intent == IntentLabel.LOOKUP:
        intent = IntentLabel.RETROSPECTIVE
        confidence = 0.82

    requires_history = any(w in lower for w in [
        "historical", "last", "yesterday", "when", "previous", "analogs",
        "happened before", "similar", "what happened afterwards",
        "fluctuate", "fluctuation", "back down", "moved from", "price path",
    ])
    requires_forecast = any(w in lower for w in [
        "forecast", "tomorrow", "will", "going to", "expected", "predict",
        "likely to continue", "continue", "persist", "persistence",
    ])
    if intent == IntentLabel.EXPLANATION and requires_forecast:
        requires_history = True
    requires_backtest = any(w in lower for w in ["backtest", "simulate", "what would", "counterfactual"])
    requires_why = intent in (IntentLabel.ACTION_RECOMMENDATION, IntentLabel.EXPLANATION)
    if season_buckets:
        requires_history = True
    causal_targets = _extract_causal_targets(lower)
    if intent == IntentLabel.EXPLANATION:
        causal_targets = _expand_explanation_targets(causal_targets, requires_forecast)
    requested_output = _requested_output_for(lower, intent, requires_why, requires_history, requires_forecast)

    return QueryDecomposition(
        query_id=query_id or f"qry-{_uuid8()}",
        raw_query=text,
        intent=intent,
        entities={
            "regions": regions,
            **({"technologies": technologies} if technologies else {}),
        },
        time_range={
            "type": "seasonal" if season_buckets else (
                "historical" if requires_history else ("forecast" if requires_forecast else "current")
            ),
            **({"season_buckets": season_buckets} if season_buckets else {}),
        },
        requires_why=requires_why,
        requires_history=requires_history,
        requires_forecast=requires_forecast,
        requires_backtest=requires_backtest,
        requires_portfolio=False,
        requires_live_market=intent != IntentLabel.OUT_OF_SCOPE,
        requires_incident_timeline=intent in (IntentLabel.EXPLANATION, IntentLabel.ACTION_RECOMMENDATION),
        causal_targets=causal_targets,
        requested_output=requested_output,
        confidence=confidence,
        ambiguities=ambiguities,
        clarifying_question=clarifying_question,
        region_corrections=[],
    )


def _has_market_weather_context(lower: str) -> bool:
    """True when a weather mention is tied to NEM price/demand/renewables."""
    market_terms = [
        "nem", "aemo", "electric", "energy", "power", "price", "dispatch",
        "demand", "load", "wind", "solar", "renewable", "generator", "headroom",
        "nsw", "nsw1", "vic", "vic1", "qld", "qld1", "sa1", "tas", "tas1",
    ]
    return any(term in lower for term in market_terms)


def _extract_causal_targets(lower: str) -> list[str]:
    targets: list[str] = []
    for word, target in [
        ("price", "price"),
        ("fluctuate", "price_path"),
        ("fluctuation", "price_path"),
        ("back down", "price_path"),
        ("fcas", "fcas"),
        ("weather", "weather"),
        ("wind", "weather"),
        ("solar", "weather"),
        ("notice", "aemo_notice"),
        ("rss", "news"),
        ("news", "news"),
        ("constraint", "constraint"),
        ("interconnector", "interconnector"),
        ("rebid", "rebid"),
        ("outage", "outage"),
        ("unit", "unit_dispatch"),
        ("demand", "demand"),
        ("headroom", "headroom"),
        ("forecast", "forecast"),
        ("continue", "forecast"),
        ("persist", "forecast"),
        ("analog", "historical_analog"),
        ("similar", "historical_analog"),
    ]:
        if word in lower and target not in targets:
            targets.append(target)
    return targets


def _extract_technologies(lower: str) -> list[str]:
    technologies = []
    for token in ["coal", "solar", "hydro", "wind", "gas", "battery"]:
        if token in lower and token not in technologies:
            technologies.append(token)
    return technologies


def _expand_explanation_targets(targets: list[str], requires_forecast: bool) -> list[str]:
    expanded = list(targets)
    for target in ["price", "demand", "headroom", "constraints", "interconnectors", "rebids", "outages", "unit_dispatch"]:
        if target not in expanded:
            expanded.append(target)
    if requires_forecast:
        for target in ["forecast", "historical_analog"]:
            if target not in expanded:
                expanded.append(target)
    return expanded


def _requested_output_for(
    lower: str,
    intent: IntentLabel,
    requires_why: bool,
    requires_history: bool,
    requires_forecast: bool,
) -> str:
    if intent == IntentLabel.TRACE_REPLAY:
        return "trace_replay"
    if any(w in lower for w in ["fluctuate", "fluctuation", "moved from", "back down", "price path"]):
        return "price_fluctuation_attribution"
    # Fuel/source comparison must come before ACTION_RECOMMENDATION and COMPARISON —
    # "why coal instead of hydro, should I be cautious?" is a fuel question even if
    # "should I" is present. Multi-part queries must not lose their primary intent.
    if any(t in lower for t in ["coal", "solar", "hydro", "wind", "gas", "battery"]) and any(
        w in lower for w in [
            "buy", "best", "source", "instead", "prefer", "choose",
            "compare", "versus", "vs", "contributes", "contribute",
            "cautious", "changed", "change", "switch", "now instead",
        ]
    ):
        return "fuel_source_recommendation"
    if intent == IntentLabel.ACTION_RECOMMENDATION:
        return "portfolio_action"
    if intent == IntentLabel.COMPARISON:
        return "regional_comparison"
    if "stale" in lower or "fresh" in lower or "data status" in lower or "source status" in lower:
        return "data_freshness_status"
    if any(w in lower for w in ["weather", "notice", "notices", "rss", "news"]):
        return "weather_notice_news_correlation"
    if intent == IntentLabel.RETROSPECTIVE or "similar" in lower or "happened before" in lower:
        return "historical_analog_outcome"
    if requires_why and requires_forecast:
        return "causal_explanation_with_forecast"
    if requires_why:
        return "causal_explanation"
    if requires_forecast:
        return "forecast"
    return "current_market_state"


def _uuid8() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
