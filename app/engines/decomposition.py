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

from app.core.schema import IntentLabel, QueryDecomposition, VerdictLabel
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
  "intent": one of ["action_recommendation", "explanation", "retrospective", "counterfactual", "comparison", "lookup", "trace_replay", "out_of_scope", "partial_scope", "evidence_bridge", "geographic_redirect"],
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
- "rooftop solar / solar panels on my house / feed-in tariff" → partial_scope (NEM price signal answerable, FiT rate not)
- "buying a wind farm / commercial solar investment" + price question → partial_scope
- "if interest rates rise what happens to renewables" → evidence_bridge (LCOE mechanism answerable)
- "western australia electricity / WA energy" → geographic_redirect
- "government policy / coal investment / carbon price" + price impact → evidence_bridge

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
    """Hybrid decomposer: rules FIRST for safety, LLM SECOND for entity/sub-question refinement.

    Pipeline:
      1. Rule-based runs always — establishes intent, routing, OOS/unsafe detection.
      2. If OOS or unsafe: return rule result immediately (LLM never sees it).
      3. If LLM available: run in parallel to enrich entities and sub_questions.
      4. Merge: rule-based routing wins; LLM contributes better entity extraction.
    Never raises — always returns a valid QueryDecomposition.
    """
    # Step 1: Rules always run first — safety gate, deterministic routing
    rule_decomp = _decompose_rules(text, region_hint, query_id)

    # Step 2: Hard stop for OOS/unsafe and adjacent intents — LLM never processes these
    if rule_decomp.intent in (
        IntentLabel.OUT_OF_SCOPE,
        IntentLabel.PARTIAL_SCOPE,
        IntentLabel.EVIDENCE_BRIDGE,
        IntentLabel.GEOGRAPHIC_REDIRECT,
    ):
        return rule_decomp

    backend = _settings.decomposer_backend
    if backend == "rule_based":
        return rule_decomp

    # Step 3: LLM for entity/sub-question enrichment only
    llm_decomp: QueryDecomposition | None = None
    if backend == "ollama" or backend == "auto":
        try:
            llm_decomp = await _decompose_ollama(text, region_hint, query_id)
        except Exception as exc:
            logger.debug("Ollama decomposer failed (%s), using rule-based", exc)

    if llm_decomp is None and (backend in ("claude", "auto") or _settings.anthropic_api_key):
        try:
            llm_decomp = await _decompose_claude(text, region_hint, query_id)
        except Exception as exc:
            logger.debug("Claude decomposer failed (%s), using rule-based", exc)

    if llm_decomp is None:
        return rule_decomp

    # Step 4: Merge — rules win on routing/safety, LLM wins on entity extraction
    return _merge_decompositions(rule_decomp, llm_decomp, text)


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


def _merge_decompositions(
    rule: QueryDecomposition,
    llm: QueryDecomposition,
    original_text: str,
) -> QueryDecomposition:
    """Merge rule-based and LLM decompositions.

    Rules ALWAYS win on: intent, routing (requested_output), requires_* flags.
    LLM wins on: entity extraction (technologies, generators), confidence refinement.
    Both contribute: ambiguities, causal_targets, sub_questions.
    """
    # Log when LLM disagrees with rules — the rules always win for structural intents
    _STRUCTURAL_INTENTS = {
        IntentLabel.COMPARISON, IntentLabel.RETROSPECTIVE,
        IntentLabel.TRACE_REPLAY, IntentLabel.COUNTERFACTUAL,
        IntentLabel.OUT_OF_SCOPE, IntentLabel.PARTIAL_SCOPE,
        IntentLabel.GEOGRAPHIC_REDIRECT,
    }
    if rule.intent != llm.intent:
        if rule.intent in _STRUCTURAL_INTENTS:
            logger.debug(
                "LLM intent '%s' overridden by rules '%s' for structural intent (query: %.60s)",
                llm.intent.value, rule.intent.value, original_text,
            )
        # For non-structural intents (explanation/lookup/action), rules still win but
        # we do not log — the difference is less important.

    # LLM entity extraction is generally better (catches fuel types, generators)
    merged_entities = dict(rule.entities)
    for key, vals in (llm.entities or {}).items():
        if vals:
            # Regions: rule wins when it detected more regions than the LLM
            # (e.g., rule expanded "other states" → all NEM; LLM may only see one)
            if key == "regions" and len(merged_entities.get("regions", [])) > len(vals):
                continue
            merged_entities[key] = vals

    # Combine ambiguities from both, deduplicated
    merged_ambiguities = list(dict.fromkeys(rule.ambiguities + llm.ambiguities))

    # Causal targets: union
    merged_targets = list(dict.fromkeys(rule.causal_targets + llm.causal_targets))

    # Classify sub_questions from the original text (deterministic, no extra LLM call)
    sub_questions = _classify_sub_questions(original_text.lower(), rule.requested_output or "")

    # Explicit intent enforcement — model_copy preserves all rule fields not in update,
    # but we include intent explicitly so there is no ambiguity about who wins.
    return rule.model_copy(update={
        "intent": rule.intent,
        "requested_output": rule.requested_output,
        "requires_forecast": rule.requires_forecast,
        "requires_history": rule.requires_history,
        "time_range": rule.time_range,
        "entities": merged_entities,
        "ambiguities": merged_ambiguities,
        "causal_targets": merged_targets,
        "sub_questions": sub_questions,
        # Take slightly higher confidence from LLM if it's more certain
        "confidence": max(rule.confidence, llm.confidence * 0.9),
    })


def _classify_sub_questions(lower: str, requested_output: str) -> list[dict]:
    """Deterministically classify sub-questions from query text.

    Returns a list of typed sub-question dicts. Each dict has a 'type' key
    and optional 'period', 'entities', 'fuels' keys.
    """
    questions: list[dict] = []

    if _asks_for_regional_comparison(lower):
        questions.append({"type": "regional_comparison"})

    # Historical price distribution
    hist_periods = {
        "last year": "last_year", "past year": "last_year", "12 months": "last_year",
        "last month": "last_month", "past month": "last_month",
        "last week": "last_week", "this week": "last_week",
        "last quarter": "last_quarter",
        "historically": "historical", "normally": "historical", "usual": "historical",
        "typical": "historical", "average": "historical",
    }
    for phrase, period in hist_periods.items():
        if phrase in lower:
            questions.append({"type": "historical_price_distribution", "period": period})
            break

    # Fuel source comparison
    fuels_mentioned = [f for f in ["coal", "solar", "hydro", "wind", "gas", "battery"] if f in lower]
    if requested_output == "fuel_source_recommendation" or (
        fuels_mentioned and any(w in lower for w in ["best", "buy", "instead", "compare", "vs", "versus", "cautious", "changed", "switch"])
    ):
        questions.append({"type": "fuel_source_comparison", "fuels": fuels_mentioned})

    # Price fluctuation / sequence
    if any(w in lower for w in ["fluctuate", "fluctuation", "moved from", "back down", "spike", "dropped"]):
        questions.append({"type": "price_fluctuation"})

    # Current price reason
    if any(w in lower for w in ["why is", "what caused", "reason", "explain", "driving", "elevated", "high"]):
        questions.append({"type": "current_price_reason"})

    # Forecast / outlook
    if any(w in lower for w in ["will", "forecast", "going to", "continue", "persist", "expect"]):
        questions.append({"type": "forecast_outlook"})

    # Regime change ("what changed", "before", "now instead")
    if any(w in lower for w in ["what changed", "changed", "before", "now instead", "switch", "regime"]):
        questions.append({"type": "regime_change"})

    # Intraday comparison: "earlier today wind was $X, why pay double now?"
    # Detects the daily price cycle question that needs diurnal context
    _intraday_comparison = any(phrase in lower for phrase in [
        "earlier today", "this morning", "today earlier",
        "pay double", "double the price", "more expensive than earlier",
        "was at", "was only", "was cheaper", "was better",
    ]) and any(w in lower for w in ["now", "currently", "today", "why", "buy"])
    if _intraday_comparison:
        questions.append({"type": "intraday_price_cycle"})
        if {"type": "price_fluctuation"} not in questions:
            questions.append({"type": "price_fluctuation"})

    # Future date price forecast ("prices on monday june 8th")
    if any(phrase in lower for phrase in ["next week", "june", "july", "august", "monday", "next month"]):
        if any(w in lower for w in ["prices", "price", "what", "expect", "forecast", "buy"]):
            if {"type": "forecast_outlook"} not in questions:
                questions.append({"type": "forecast_outlook"})

    # Diurnal / time-of-day price pattern
    _is_diurnal = any(w in lower for w in [
        "diurnal", "daily pattern", "daily cycle", "time of day",
        "peak vs off-peak", "morning peak", "evening peak", "solar window",
        "price throughout the day", "by hour", "hourly pattern", "intraday pattern",
        "typical day", "across the day",
    ])
    if _is_diurnal:
        questions.append({"type": "diurnal_pattern"})

    # Trend / multi-period historical series
    _is_trend = any(w in lower for w in [
        "monthly trend", "annual trend", "quarterly trend",
        "year over year", "yoy", "month by month", "quarter by quarter",
        "by month", "by quarter", "time series", "over time",
        "how much has", "how have", "how has",
    ])
    if _is_trend and {"type": "historical_price_distribution", "period": "last_year"} not in questions:
        questions.append({"type": "trend_analysis"})

    return questions


def _decompose_rules(
    text: str, region_hint: str, query_id: str | None
) -> QueryDecomposition:
    """Rule-based intent classifier — deterministic, no LLM dependency."""
    # Strip all known prefixes that inject context keywords into routing.
    # Session carry-forward format: "[Session context: Q1: ...] [Current question]: raw"
    # Restatement seed format:      "[The user is asking: ...]"
    _seed_prefix_re = re.compile(
        r"^\[(?:the user is asking|session context)[^\]]*\]\s*"
        r"(?:\[current question\]:\s*)?",
        re.IGNORECASE | re.DOTALL,
    )
    _clean_text = _seed_prefix_re.sub("", text)
    lower = _clean_text.lower()

    intent = IntentLabel.LOOKUP
    confidence = 0.65
    ambiguities: list[str] = []
    clarifying_question: str | None = None
    adjacent_context: dict | None = None
    geographic_market: str | None = None

    # ── Pre-classify causal-how pattern before the elif chain ─────────────────
    # "How does/do X affect/impact Y?" is always an EXPLANATION (causal mechanism)
    # even when it doesn't contain "why/reason/explain". Must be computed here so
    # the elif chain can reference it without nested re-evaluation.
    _causal_how = bool(
        re.search(r"\bhow\s+do(?:es)?\b|\bhow\s+w(?:ill|ould)\b", lower)
        and re.search(r"\b(affect|impact|influence|drive|alter|shape|change)\b", lower)
        and not re.search(r"\bcompare\b|\bversus\b|\b vs \b|\bdifference between\b", lower)
    )

    # Future-date reference detection (e.g. "monday june 8th", "next week", "in two days")
    _future_date_re = re.compile(
        r"\bnext\s+(?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|month|fortnight)\b"
        r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2})\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?\b"
        r"|\b(?:june|july|august|september|october|november|december|january|february|march|april|may)\s+\d{1,2}(?:st|nd|rd|th)?\b"
        r"|\bin\s+(?:a\s+)?(?:couple\s+of\s+)?(?:few\s+)?(?:days?|weeks?|months?)\b"
        r"|\bday\s+after\s+tomorrow\b|\bweek\s+after\s+(?:next|tomorrow)\b"
    )
    _has_future_date = bool(_future_date_re.search(lower))

    # ── Safety gates (always TRULY_OOS — LLM never sees these) ──────────────
    unsafe_action = bool(re.search(
        r"\b(submit|place|send|execute|lodge)\b.*\b(real|live|actual)?\s*(dispatch\s+)?bid\b|\b(dispatch\s+bid|bid\s+\d+)",
        lower,
    ))
    # Financial products — truly OOS (ASX Energy requires paid license)
    _financial_oos = any(w in lower for w in [
        "futures contract", "swap contract", "forward contract",
        "electricity futures", "energy futures", "cap contract",
        "hedge my load", "hedge my", "hedging contract",
    ])
    if unsafe_action:
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.95
        clarifying_question = (
            "GridVerdict cannot submit, place, or execute market bids. "
            "It can only provide evidence-grounded NEM decision support."
        )
    elif _financial_oos:
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
        clarifying_question = (
            "GridVerdict covers live NEM spot market analytics. Financial derivatives "
            "(futures, swaps, caps, hedging contracts) are traded on ASX Energy and "
            "require separate financial market data. Try asking about current spot "
            "price conditions or dispatch decisions instead."
        )
    elif any(w in lower for w in [
        "replay", "trace", "what did you", "your decision at",
        "system know at the time", "know at the time", "known at the time",
    ]):
        intent = IntentLabel.TRACE_REPLAY
        confidence = 0.82
    elif _causal_how:
        # "How does X affect/impact Y?" — causal explanation, intercept before ACTION check
        intent = IntentLabel.EXPLANATION
        confidence = 0.83
    elif (
        not any(w in lower for w in [
            "why is", "why are", "why did", "reason for", "what caused",
            "what is driving", "explain", "what changed",
        ])
        and any(w in lower for w in [
        "should i", "dispatch now", "what action", "recommend",
        "before i act", "missing before i act", "should reserve",
        "reserve capacity",
        "cheapest to procure", "cheapest to buy", "which fuel",
        "what to buy", "best to buy",
        "optimise my", "optimize my", "charge/discharge schedule",
        ])
    ):
        intent = IntentLabel.ACTION_RECOMMENDATION
        confidence = 0.82
    elif any(w in lower for w in [
        "why is", "why are", "what is driving", "reason for", "explain",
        "because", "what changed", "evidence supports", "what evidence",
        "price move", "backed by", "market balance", "why did", "causing",
        "cause", "fluctuate", "fluctuation", "moved from", "back down",
        "which source", "what source",
        # Causal attribution phrases: "is X the main reason", "is Y causing"
        "main reason", "main cause", "the reason", "the cause",
        "is that the reason", "is it because",
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
        # Q16 fix: "this morning at X", "earlier today at X"
        "this morning", "earlier today", "this afternoon", "this evening",
        "at 7am", "at 8am", "at noon", "at midnight",
        # Monthly / annual / trend historical queries
        "last year", "past year", "last month", "past month",
        "last quarter", "last june", "last july", "last august",
        "last september", "last october", "last november", "last december",
        "last january", "last february", "last march", "last april", "last may",
        "last summer", "last winter", "last spring", "last autumn",
        "in 2024", "in 2023", "in 2022", "in 2021",
        "monthly trend", "annual trend", "quarterly trend",
        "year over year", "yoy", "month by month", "quarter by quarter",
        "by month", "by quarter", "time series", "price history",
        "over the past", "how much has", "how have", "how has",
        "average price", "average prices", "average spot",
    ]):
        # Regional comparison overrides retrospective when multi-region is clear
        if _asks_for_regional_comparison(lower):
            intent = IntentLabel.COMPARISON
            confidence = 0.78
        else:
            intent = IntentLabel.RETROSPECTIVE
            confidence = 0.80
    elif any(w in lower for w in [
        "compare", "versus", "vs", "difference between",
        "differ to", "differ from", "different to", "different from",
        "how does", "how do",
    ]) and (any(w in lower for w in ["state", "region", "other", "compare"]) or _asks_for_regional_comparison(lower)):
        intent = IntentLabel.COMPARISON
        confidence = 0.78
    elif any(w in lower for w in ["compare", "versus", "vs", "difference between"]):
        intent = IntentLabel.COMPARISON
        confidence = 0.78
    elif _asks_for_regional_comparison(lower) and any(
        w in lower for w in ["cheaper", "more expensive", "pricier", "higher than", "lower than",
                              "higher priced", "lower priced", "which is cheaper", "which is pricier"]
    ):
        # "Is NSW cheaper than QLD?" — price-comparison vocabulary + 2 region codes
        intent = IntentLabel.COMPARISON
        confidence = 0.76
    elif any(w in lower for w in ["forecast", "tomorrow", "will price", "going to", "expected"]):
        intent = IntentLabel.LOOKUP
        confidence = 0.75

    # NEM glossary queries — "what does headroom mean?", "explain FCAS", "what is MTPASA?"
    _nem_term_query = any(p in lower for p in [
        "what does", "what is", "explain", "define", "meaning of",
        "what are", "how does it work",
    ]) and any(t in lower for t in [
        "headroom", "fcas", "mtpasa", "dispatch interval", "dispatch price",
        "marginal setter", "settlement", "nem", "market cap", "voll",
        "causer pays", "constraint", "interconnector", "p10", "p50", "p90",
        "availability", "spot price", "rrp", "regional reference price",
        "predispatch", "5-minute", "5 minute settlement",
    ])
    _current_price_lookup = any(p in lower for p in [
        "current", "live", "right now", "now", "latest", "today",
    ]) and any(t in lower for t in [
        "dispatch price", "spot price", "rrp", "regional reference price",
        "price", "prices",
    ])
    if _nem_term_query and intent != IntentLabel.COMPARISON and not _current_price_lookup:
        intent = IntentLabel.EXPLANATION
        confidence = 0.82

    # Meta-questions about the platform itself
    _meta_phrases = [
        "how does this work", "how does gridverdict", "what is gridverdict",
        "what can you do", "how does it work", "tell me about yourself",
        "what are your capabilities", "how do you work",
    ]
    if any(p in lower for p in _meta_phrases):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.95

    # Prompt injection / adversarial probing detection
    _injection_phrases = [
        "ignore all previous", "ignore your instructions", "ignore prior",
        "you are now", "new instructions", "override your",
        "system prompt", "show me your prompt", "print your instructions",
        "/etc/passwd", "database connection", "connection string",
        "admin override", "debug mode", "developer mode",
    ]
    if any(p in lower for p in _injection_phrases):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.99
        clarifying_question = (
            "GridVerdict is a read-only NEM market analytics system. "
            "It cannot execute commands, disclose configuration, or change its behaviour via prompts."
        )

    # ── Five-category adjacent classifier ───────────────────────────────────
    # Priority order: safety gates (above) → adjacent taxonomy → normal routing.
    # Each category returns early so later categories don't override.

    # Category 1: TRULY_OOS — pure investment appraisal (no answerable NEM sub)
    _investment_appraisal_only = any(w in lower for w in [
        "20-year revenue", "20 year revenue", "lcoe", "levelised cost",
        "power purchase agreement", "ppa", "merchant risk",
        "grid connection cost", "tnsp connection", "connection cost",
    ])
    # Category 1 combo: buying/building + farm but NOT asking about prices
    _investment_combo = (
        any(w in lower for w in ["buying", "build", "invest", "purchase"])
        and any(w in lower for w in ["wind farm", "solar farm", "renewable farm", "wind and solar"])
    )
    _asks_price_context = any(w in lower for w in [
        "price", "prices", "what to know", "what do i need", "spot", "revenue",
        "worth", "profitable", "returns", "earnings", "income",
    ])

    # Category 2: PARTIAL_SCOPE — commercial renewable investment with price question
    _commercial_renewable_investment = (
        (_investment_combo or any(w in lower for w in [
            "buying a wind", "buying a solar", "build a wind farm", "build a solar farm",
            "invest in renewable",
        ]))
        and _asks_price_context
        and not _investment_appraisal_only
    )

    # Category 3: PARTIAL_SCOPE — household/residential solar
    _household_solar = (
        any(w in lower for w in [
            "solar panel", "solar panels", "rooftop solar", "home solar",
            "residential solar", "house solar", "put solar on",
            "install solar", "solar on my house", "solar on my home",
            "solar on my roof", "solar power", "solar system",
            "feed-in tariff", "fit rate", "fit payment", "solar rebate",
            "solar revenue", "sell back", "solar export",
        ])
        and any(w in lower for w in [
            "house", "home", "roof", "property", "household",
            "my", "i", "revenue", "earn", "make money", "worth", "pay",
            "feed", "export", "sell", "return", "investment", "payback",
        ])
        and not _investment_combo  # not commercial scale
    )

    # Category 4: EVIDENCE_BRIDGE — macro forces that affect NEM (answerable mechanism)
    _interest_rate_bridge = any(w in lower for w in [
        "interest rate", "interest rates", "rba cash rate", "cost of capital",
        "discount rate", "borrowing cost", "monetary policy", "rate rise",
        "rate cut", "rate hike", "cost of debt", "financing cost",
    ]) and any(w in lower for w in [
        "renewable", "solar", "wind", "coal", "gas", "energy", "electricity",
        "power", "price", "invest", "lcoe", "generator",
    ])
    _policy_bridge = any(w in lower for w in [
        "government policy", "policy change", "new policy",
        "government invest", "government subsidy", "government fund",
        "government announce", "government plan", "legislation",
        "carbon price", "carbon tax", "emissions trading", "safeguard mechanism",
        "capacity investment scheme", "cis", "renewable energy target", "ret",
        "ltesa", "ress", "eraring", "coal subsidy", "coal investment",
        "coal policy", "coal policies",
    ]) and any(w in lower for w in [
        "price", "electricity", "energy", "nem", "renewable", "coal", "gas",
        "market", "dispatch", "power", "affect", "change", "impact",
    ])
    _lng_gas_bridge = any(w in lower for w in [
        "lng", "liquefied natural gas", "gas export", "wallumbilla",
        "gas price", "domestic gas", "gas market", "jkm", "asian gas",
        "gas crisis", "energy crisis 2022",
    ]) and any(w in lower for w in [
        "electricity", "power", "price", "nem", "affect", "link", "nexus",
    ])
    _budget_fiscal_bridge = any(w in lower for w in [
        "government budget", "federal budget", "budget 2024", "budget 2025",
        "fiscal", "treasury", "midyear economic", "myefo",
        "energy spending", "clean energy finance", "arena funding",
        "cefc", "rewiring the nation",
    ])

    # Category 5: GEOGRAPHIC_REDIRECT — non-NEM electricity markets
    _wa_query = any(w in lower for w in [
        "western australia", "west australia", "perth", "wa energy",
        "swis", "wem", "western power", "synergy",
        "broome", "kalgoorlie", "albany", "geraldton",
    ])
    _nt_query = any(w in lower for w in [
        "northern territory", "darwin energy", "alice springs energy",
        "territory generation", "power water", "nt grid",
        "darwin electricity", "darwin grid", "darwin market",
        "alice springs electricity", "katherine electricity",
    ])
    _nz_query = any(w in lower for w in [
        "new zealand energy", "new zealand electricity", "nz electricity", "nz grid",
        "meridian energy", "contact energy", "genesis energy nz",
    ])

    # ── Apply categories (first match wins, lower priority = more fallback) ──

    if _wa_query or _nt_query or _nz_query:
        intent = IntentLabel.GEOGRAPHIC_REDIRECT
        confidence = 0.88
        if _wa_query:
            geographic_market = "WA_WEM"
            clarifying_question = (
                "Western Australia operates the Wholesale Electricity Market (WEM) — "
                "a capacity market design separate from the NEM. GridVerdict covers the "
                "NEM (NSW1, VIC1, QLD1, SA1, TAS1). I can compare WA's market structure "
                "to NEM regions, or answer NEM questions. What would be most useful?"
            )
        elif _nt_query:
            geographic_market = "NT_GRID"
            clarifying_question = (
                "The Northern Territory operates an isolated grid (Darwin-Katherine system) "
                "not connected to the NEM. GridVerdict covers NEM regions. I can explain "
                "how isolated grids differ from the NEM, or answer NEM questions."
            )
        else:
            geographic_market = "NZ_GRID"
            clarifying_question = (
                "New Zealand operates a separate wholesale market under the Electricity Authority. "
                "GridVerdict covers the Australian NEM. I can explain structural differences "
                "or answer NEM questions instead."
            )
        adjacent_context = {
            "answerable_sub": "NEM market structure, regional price comparison, dispatch mechanics",
            "unanswerable_sub": f"Live data for {geographic_market} (not in AEMO/NEM)",
            "bridge_mechanism": "Market design comparison between NEM and non-NEM grids",
            "redirect_resource": (
                "AEMO WEM dashboard (aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem)"
                if _wa_query else "NT Power and Water Corporation (powerwater.com.au)"
            ),
        }

    elif _household_solar and intent not in (
        IntentLabel.OUT_OF_SCOPE,
        IntentLabel.TRACE_REPLAY,
        IntentLabel.GEOGRAPHIC_REDIRECT,
    ):
        intent = IntentLabel.PARTIAL_SCOPE
        confidence = 0.72
        adjacent_context = {
            "answerable_sub": (
                "Wholesale spot price distribution during solar export hours (9am–3pm), "
                "solar cannibalisation trend (midday price suppression from PV penetration), "
                "seasonal P10/P50/P90, evening peak pricing relevant to battery value"
            ),
            "unanswerable_sub": (
                "Actual feed-in tariff rates (retailer-set), STC rebate value (federal scheme), "
                "equipment + installation costs, payback period calculation"
            ),
            "bridge_mechanism": (
                "Wholesale spot prices set the reference signal retailers use when "
                "pricing feed-in tariffs — GridVerdict provides this signal directly."
            ),
            "redirect_resource": (
                "AER retailer comparison tool (energy.gov.au/households/solar-panels), "
                "Clean Energy Regulator for STC scheme (cleanenergyregulator.gov.au)"
            ),
        }
        clarifying_question = (
            "GridVerdict can show you the wholesale price signal underlying feed-in tariffs: "
            "solar-window spot prices, cannibalisation trends, and seasonal patterns. "
            "For actual FiT rates and STC rebates, check your retailer or the AER tool."
        )

    elif _commercial_renewable_investment and intent not in (
        IntentLabel.OUT_OF_SCOPE,
        IntentLabel.TRACE_REPLAY,
        IntentLabel.GEOGRAPHIC_REDIRECT,
    ):
        intent = IntentLabel.PARTIAL_SCOPE
        confidence = 0.70
        adjacent_context = {
            "answerable_sub": (
                "Spot price distributions by season and time-of-day for the region, "
                "solar/wind capture rate analysis (cannibalisation effect), "
                "price volatility and spike frequency, AEMO ISP scenario price trajectories"
            ),
            "unanswerable_sub": (
                "LCOE calculation, IRR/NPV, grid connection costs (TNSP study), "
                "PPA pricing, RESS/LTESA eligibility, 20-year revenue forecast"
            ),
            "bridge_mechanism": (
                "Spot market price distributions are the primary input to revenue modelling. "
                "AEMO ISP Step Change scenario provides long-run price trajectory bands."
            ),
            "redirect_resource": (
                "AEMO ISP 2024 (aemo.com.au/isp), CSIRO GenCost 2023-24, "
                "Clean Energy Finance Corporation (cefc.com.au)"
            ),
        }
        clarifying_question = (
            "GridVerdict can provide the spot market price context for your investment analysis "
            "(capture rates, seasonal distributions, ISP scenario trajectories). "
            "For LCOE, grid connection, and PPA modelling, see AEMO ISP and CSIRO GenCost."
        )

    elif (_interest_rate_bridge or _policy_bridge or _lng_gas_bridge or _budget_fiscal_bridge) and (
        intent in (IntentLabel.LOOKUP, IntentLabel.COMPARISON, IntentLabel.RETROSPECTIVE)
        or (intent == IntentLabel.EXPLANATION and not _causal_how)
    ):
        intent = IntentLabel.EVIDENCE_BRIDGE
        confidence = 0.68
        if _interest_rate_bridge:
            adjacent_context = {
                "answerable_sub": (
                    "Technology SRMC structure (fuel cost vs capex split per technology), "
                    "LCOE sensitivity by discount rate using CSIRO GenCost inputs, "
                    "historical dispatch cost evolution, current fuel-technology dispatch ranking"
                ),
                "unanswerable_sub": (
                    "Actual equilibrium price impact of a rate change "
                    "(requires capacity expansion model — AEMO Plexos/ISP territory)"
                ),
                "bridge_mechanism": (
                    "Interest rates affect LCOE (capex-heavy technologies like solar/wind "
                    "are most sensitive), not SRMC (spot dispatch cost). "
                    "Spot prices respond to SRMC; investment pipeline responds to LCOE."
                ),
                "redirect_resource": (
                    "AEMO ISP 2024 scenario outputs, CSIRO GenCost 2023-24, "
                    "RBA Statement on Monetary Policy for rate expectations"
                ),
            }
            clarifying_question = (
                "GridVerdict can show how interest rates affect the levelised cost of each "
                "technology (solar/wind are most rate-sensitive) and current dispatch economics. "
                "For equilibrium price impact, AEMO ISP scenario outputs are the reference."
            )
        elif _lng_gas_bridge:
            adjacent_context = {
                "answerable_sub": (
                    "Current east coast gas hub prices (Wallumbilla/STTM), "
                    "gas generator SRMC from hub price + heat rate, "
                    "gas price → NEM spot price correlation, 2022 energy crisis mechanism"
                ),
                "unanswerable_sub": (
                    "Real-time JKM (Asian LNG spot price) — requires Platts/ICIS subscription"
                ),
                "bridge_mechanism": (
                    "JKM sets the LNG export netback price → east coast domestic gas price floor → "
                    "gas generator SRMC → NEM spot price during gas-marginal intervals (~40% of peak hours)"
                ),
                "redirect_resource": (
                    "ACCC Gas Inquiry quarterly reports (accc.gov.au/gas-inquiry), "
                    "AEMO Gas Bulletin Board (gbb.aemo.com.au)"
                ),
            }
        elif _policy_bridge:
            adjacent_context = {
                "answerable_sub": (
                    "Current technology dispatch mix and coal/gas marginal-setting frequency, "
                    "historical price impact of capacity retirements (Hazelwood 2017, Liddell 2023), "
                    "AEMO ISP scenario comparison (Step Change vs Slow Change)"
                ),
                "unanswerable_sub": (
                    "Long-run equilibrium price under new policy (requires ISP capacity expansion model)"
                ),
                "bridge_mechanism": (
                    "New generation capacity takes 7–10 years to materialise. "
                    "Short-run: no price effect. Long-run: dispatch mix shift → baseload price change. "
                    "Evidence: Hazelwood closure raised Victorian baseload by ~$30–50/MWh."
                ),
                "redirect_resource": (
                    "AEMO ISP 2024 Progressive Change and Slow Change scenarios, "
                    "DOGE/DCCEEW energy policy register"
                ),
            }
            clarifying_question = (
                "GridVerdict can show how coal currently shapes NEM dispatch and "
                "what history shows about capacity changes affecting prices. "
                "For new policy scenario modelling, AEMO ISP is the reference."
            )
        elif _budget_fiscal_bridge:
            adjacent_context = {
                "answerable_sub": (
                    "Current NEM market state and any AEMO notices related to funded programs, "
                    "budget energy measure summary from Australian Government Budget documents"
                ),
                "unanswerable_sub": (
                    "Detailed fiscal modelling, program effectiveness analysis"
                ),
                "bridge_mechanism": (
                    "Government energy spending (ARENA, CEFC, Rewiring the Nation) affects "
                    "transmission investment and renewable deployment — which feeds into "
                    "NEM supply mix and long-run prices."
                ),
                "redirect_resource": (
                    "Australian Government Budget (budget.gov.au), "
                    "ARENA (arena.gov.au), CEFC (cefc.com.au)"
                ),
            }

    # OOS detection — non-electricity topics (unchanged)
    _oos_words = [
        "stock", "forex", "crypto", "bitcoin", "election", "football",
        "futures", "swap contract", "forward contract", "options contract",
    ]
    _external_platforms = [
        "bloomberg", "refinitiv", "ice nexus", "nem-review", "nem review",
        "opennem", "wattclarity", "iress", "factset",
    ]
    if (
        ("source" in lower or "data" in lower)
        and any(term in lower for term in ["stale", "fresh", "missing", "status"])
    ):
        intent = IntentLabel.LOOKUP
        confidence = 0.82

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
    # Pure investment appraisal (no price context) — still truly OOS
    if _investment_appraisal_only and intent not in (
        IntentLabel.PARTIAL_SCOPE, IntentLabel.EVIDENCE_BRIDGE,
        IntentLabel.GEOGRAPHIC_REDIRECT, IntentLabel.OUT_OF_SCOPE,
    ):
        intent = IntentLabel.OUT_OF_SCOPE
        confidence = 0.90
        clarifying_question = (
            "GridVerdict covers live NEM spot market analytics. Investment appraisal "
            "(LCOE, PPAs, grid connection, 20-year revenue) requires AEMO ISP projections, "
            "CSIRO GenCost data, and TNSP connection studies. "
            "Try asking about current spot conditions or dispatch decisions instead."
        )

    # Region detection (runs before OOS check so mixed NEM+non-NEM queries are handled correctly)
    _ALL_REGIONS = known_regions()
    regions = []
    _explicit_regions: list[str] = []
    if any(phrase in lower for phrase in ["all region", "all nem region", "each region", "every region", "all states", "all nem"]):
        regions = _ALL_REGIONS[:]
        _explicit_regions = _ALL_REGIONS[:]
    else:
        alias_regions, alias_notes = detect_regions(text)
        regions.extend(alias_regions)
        ambiguities.extend(alias_notes)
        _explicit_regions = alias_regions
    if not regions:
        regions = [region_hint]

    # Mixed queries can ask for both a recommendation and a regional comparison.
    # "how does NSW compare to other states?" → gather all five.
    if regions != _ALL_REGIONS and _asks_for_regional_comparison(lower):
        regions = _ALL_REGIONS[:]
        _explicit_regions = _ALL_REGIONS[:]

    technologies = _extract_technologies(lower)

    non_nem_notes = detect_non_nem(text)
    if non_nem_notes:
        ambiguities.extend(non_nem_notes)
        if _explicit_regions:
            # Mixed query: NEM regions detected alongside non-NEM place — answer NEM portion.
            # If intent was COMPARISON or GEOGRAPHIC_REDIRECT it can't be fully fulfilled
            # (one entity is non-NEM), so degrade to LOOKUP so the NEM portion is answered.
            if intent in (IntentLabel.COMPARISON, IntentLabel.GEOGRAPHIC_REDIRECT):
                intent = IntentLabel.LOOKUP
                confidence = max(confidence - 0.1, 0.60)
            clarifying_question = (
                f"{non_nem_notes[0]} "
                f"Answering for the NEM region(s) detected: {', '.join(_explicit_regions)}."
            )
        elif intent != IntentLabel.GEOGRAPHIC_REDIRECT:
            # Only non-NEM geography mentioned, no NEM region — full refusal
            # (pure GEOGRAPHIC_REDIRECT queries already have their clarifying question set)
            intent = IntentLabel.OUT_OF_SCOPE
            confidence = 0.90
            clarifying_question = (
                f"{non_nem_notes[0]} GridVerdict currently covers NSW1, VIC1, QLD1, SA1, and TAS1."
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

    # Past calendar year reference — "june 2024", "prices in 2023", "Q3 2022"
    # Catches cases where year and month are separated ("in june 2024" ≠ "in 2024")
    if intent == IntentLabel.LOOKUP and re.search(r'\b20(1[5-9]|2[0-4])\b', lower):
        intent = IntentLabel.RETROSPECTIVE
        requires_history = True
        confidence = max(confidence, 0.75)

    # Price threshold extraction — "$150", "below $200", "above $300", "P90", "under $100"
    import re as _re
    _threshold_raw = _re.findall(
        r'(?:below|above|under|over|less than|more than|exceed|drop to|reach|hit|target)?\s*'
        r'\$\s*(\d[\d,]*(?:\.\d+)?)\s*/?\s*mwh?|'
        r'(?:below|above|under|over|less than|more than|exceed|drop to|reach|hit|target)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)',
        lower,
    )
    spike_thresholds = []
    for match in _threshold_raw:
        raw = (match[0] or match[1]).replace(",", "")
        try:
            val = float(raw)
            if 0 < val < 20000:
                spike_thresholds.append(val)
        except ValueError:
            pass
    # Also detect percentile references ("P90", "P10", "90th percentile")
    _pct_refs = _re.findall(r'\b[Pp](\d+)\b|\b(\d+)(?:th|rd|nd|st)\s+percentile', text)
    price_percentiles = [int(p[0] or p[1]) for p in _pct_refs if int(p[0] or p[1]) <= 99]

    requires_history = any(w in lower for w in [
        "historical", "last", "yesterday", "when", "previous", "analogs",
        "happened before", "similar", "what happened afterwards",
        "fluctuate", "fluctuation", "back down", "moved from", "price path",
        # Intraday retrospective: user comparing current price to earlier today
        "earlier today", "this morning", "earlier this", "today earlier",
        "was at", "was only", "was better", "was cheaper",
        "pay double", "more expensive than", "double the price",
    ])
    _fy_future = bool(re.search(r'\bfy\s*20(2[5-9]|3\d)\b', lower, re.IGNORECASE)) or any(w in lower for w in [
        "next financial year", "next fiscal year", "next fy",
        "q1 fy", "q2 fy", "q3 fy", "q4 fy",
        "fy 2026", "fy 2027", "fy 2028", "2026-27", "2027-28",
    ])
    requires_forecast = _has_future_date or _fy_future or any(w in lower for w in [
        "forecast", "tomorrow", "will", "going to", "expected", "predict",
        "likely to continue", "continue", "persist", "persistence",
        "next week", "week after", "day after",
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
            **({"future_date_ref": True} if _has_future_date else {}),
        },
        requires_why=requires_why,
        requires_history=requires_history,
        requires_forecast=requires_forecast,
        requires_backtest=requires_backtest,
        requires_portfolio=False,
        requires_live_market=intent not in (
            IntentLabel.OUT_OF_SCOPE,
            IntentLabel.GEOGRAPHIC_REDIRECT,
        ),
        requires_incident_timeline=intent in (IntentLabel.EXPLANATION, IntentLabel.ACTION_RECOMMENDATION),
        causal_targets=causal_targets,
        spike_thresholds=spike_thresholds,
        requested_output=requested_output,
        confidence=confidence,
        ambiguities=ambiguities + (
            [f"Threshold detected: ${t:.0f}/MWh" for t in spike_thresholds[:2]]
            if spike_thresholds else []
        ),
        clarifying_question=clarifying_question,
        region_corrections=[],
        adjacent_context=adjacent_context,
        geographic_market=geographic_market,
    )


def _has_market_weather_context(lower: str) -> bool:
    """True when a weather mention is tied to NEM price/demand/renewables."""
    market_terms = [
        "nem", "aemo", "electric", "energy", "power", "price", "dispatch",
        "demand", "load", "wind", "solar", "renewable", "generator", "headroom",
        "nsw", "nsw1", "vic", "vic1", "qld", "qld1", "sa1", "tas", "tas1",
    ]
    return any(term in lower for term in market_terms)


def _asks_for_regional_comparison(lower: str) -> bool:
    """True when a query asks to compare NEM regions or states."""
    phrases = [
        "other state", "other states", "other region", "other regions",
        "rest of the nem", "rest of nem", "rest of the market",
        "across state", "across states", "across region", "across regions",
        "compared to other", "compare to other", "compare with other",
        "all other", "every state", "each state", "each region",
        "all nem region", "all nem regions", "all states",
        "differ to other", "differ from other",
        "different to other", "different from other",
    ]
    if any(phrase in lower for phrase in phrases):
        return True
    _has_compare_verb = any(
        term in lower for term in [
            "compare", "versus", "vs", "difference between",
            "cheaper", "cheaper than", "more expensive", "pricier",
            "higher than", "lower than", "higher priced", "lower priced",
            "differ", "different", "how does", "how do",
        ]
    )
    if not _has_compare_verb:
        return False
    # Catches "compare NSW and QLD" — two or more region codes (with or without "1") counts as regional
    _REGION_CODES = ["nsw1", "vic1", "qld1", "sa1", "tas1", "nsw", "vic", "qld", " sa ", "tas"]
    _region_hits = sum(1 for code in _REGION_CODES if code in lower)
    if _region_hits >= 2:
        return True
    return any(scope in lower for scope in ["state", "states", "region", "regions", "nem"])


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
    # Adjacent intents have their own output types
    if intent == IntentLabel.GEOGRAPHIC_REDIRECT:
        return "geographic_redirect"
    if intent == IntentLabel.EVIDENCE_BRIDGE:
        if any(w in lower for w in ["interest rate", "rba", "cost of capital", "discount rate", "borrowing"]):
            return "macro_mechanism_bridge"
        if any(w in lower for w in ["lng", "gas price", "wallumbilla", "gas market", "domestic gas"]):
            return "gas_electricity_nexus"
        # Fiscal/budget check before generic policy — "government" appears in both but budget is more specific
        if any(w in lower for w in ["budget", "fiscal", "myefo", "arena", "cefc", "rewiring", "capacity investment scheme", "cis"]):
            return "fiscal_budget_bridge"
        if any(w in lower for w in ["policy", "government", "coal subsidy", "carbon price", "coal invest"]):
            return "policy_evidence_bridge"
        return "evidence_bridge"
    if intent == IntentLabel.PARTIAL_SCOPE:
        if any(w in lower for w in [
            "solar panel", "rooftop solar", "feed-in", "household solar", "home solar",
        ]):
            return "solar_household_context"
        return "renewable_investment_price_context"
    if intent == IntentLabel.TRACE_REPLAY:
        return "trace_replay"
    if any(w in lower for w in ["fluctuate", "fluctuation", "moved from", "back down", "price path"]):
        return "price_fluctuation_attribution"
    # Diurnal / time-of-day pattern — before fuel check (solar/wind keywords present)
    _is_diurnal_out = any(w in lower for w in [
        "diurnal", "daily pattern", "daily cycle", "time of day",
        "peak vs off-peak", "morning peak", "evening peak", "solar window",
        "price throughout the day", "by hour", "hourly pattern",
        "intraday pattern", "typical day", "across the day",
        "cheapest time", "cheapest hour", "cheapest part of",
        "best time to buy", "best time to purchase", "when is electricity cheapest",
        "when to buy power", "cheapest time of day",
    ])
    if _is_diurnal_out:
        return "diurnal_analysis"
    # Monthly / annual trend — requires historical context
    _trend_compound = any(w in lower for w in [
        "monthly trend", "annual trend", "quarterly trend", "price trend", "price trends",
        "year over year", "yoy", "month by month", "quarter by quarter",
        "by month", "by quarter", "time series", "price history",
        "how much has", "how have", "how has",
        "average price", "average prices", "average spot",
    ])
    _trend_period_plus = any(w in lower for w in ["monthly", "annual", "quarterly", "yearly"]) and \
                         any(w in lower for w in ["trend", "average", "pattern", "breakdown", "split", "distribution"])
    _is_trend_out = _trend_compound or _trend_period_plus
    if _is_trend_out and not requires_forecast:
        return "trend_analysis"
    # Fuel/source comparison must come before ACTION_RECOMMENDATION and COMPARISON —
    # "why coal instead of hydro, should I be cautious?" is a fuel question even if
    # "should I" is present. Multi-part queries must not lose their primary intent.
    fuel_specific = any(t in lower for t in ["coal", "solar", "hydro", "wind", "gas", "battery"])
    fuel_question = any(
        w in lower for w in [
            "buy", "best", "source", "instead", "prefer", "choose",
            "compare", "versus", "vs", "contributes", "contribute",
            "cautious", "changed", "change", "switch", "now instead",
            "cheaper", "cheapest", "fuel",
        ]
    )
    generic_source_question = (
        not any(term in lower for term in ["stale", "fresh", "missing", "status"])
    ) and any(
        phrase in lower
        for phrase in ["which source", "what source", "source is normally", "normally cheaper"]
    )
    # Generic "which energy/power source" without naming a specific fuel
    _generic_fuel_question = any(w in lower for w in [
        "energy source", "power source", "electricity source", "generation source",
        "cheapest source", "cheapest energy", "best energy", "which source",
        "which energy", "which power",
    ])
    if (fuel_specific and fuel_question) or generic_source_question or _generic_fuel_question:
        return "fuel_source_recommendation"
    if intent == IntentLabel.ACTION_RECOMMENDATION:
        return "portfolio_action"
    if intent == IntentLabel.COMPARISON:
        return "regional_comparison"
    if "stale" in lower or "fresh" in lower or "data status" in lower or "source status" in lower:
        return "data_freshness_status"
    # Word-boundary matching avoids false hits from substrings (e.g. "rain" in "constraints")
    _has_weather = bool(re.search(r"\b(weather|temperature|wind|rain)\b", lower))
    _has_notice = bool(re.search(r"\b(notice|notices|aemo\s+notice)\b", lower))
    _has_news = bool(re.search(r"\b(rss|news|energy\s+news)\b", lower))
    # "How does X affect/impact Y?" — mechanism question, not data-source investigation
    _is_causal_how = bool(
        re.search(r"\bhow\s+do(?:es)?\b|\bhow\s+w(?:ill|ould)\b", lower)
        and re.search(r"\b(affect|impact|influence|drive|alter|shape|change)\b", lower)
    )
    # Gas/LNG mechanism — wins over generic causal_how because the nexus is a named adjacent handler
    _is_gas_mechanism = any(w in lower for w in [
        "lng", "gas price", "wallumbilla", "gas market", "domestic gas", "gas cost",
        "gas affects", "gas impact", "lng price",
    ])
    if _is_gas_mechanism:
        return "gas_electricity_nexus"
    # Causal mechanism queries win over weather/notice/news (user asks about mechanism, not sources)
    if _is_causal_how and requires_why and requires_forecast:
        return "causal_explanation_with_forecast"
    if _is_causal_how and requires_why:
        return "causal_explanation"
    # Data-source investigation: weather/notice/news as subjects of the question
    if _has_weather or _has_notice or _has_news:
        return "weather_notice_news_correlation"
    if intent == IntentLabel.RETROSPECTIVE or "similar" in lower or "happened before" in lower:
        return "historical_analog_outcome"
    if requires_why and requires_forecast:
        return "causal_explanation_with_forecast"
    if requires_why:
        return "causal_explanation"
    # Government policy — route to policy bridge even for LOOKUP intent
    _is_policy = any(w in lower for w in [
        "government policy", "energy policy", "renewable policy", "climate policy",
        "legislation", "policy", "regulation", "scheme", "capacity investment",
    ]) and any(w in lower for w in ["government", "federal", "state", "policy", "minister"])
    if _is_policy:
        return "policy_evidence_bridge"
    # Sustainability / persistence — what keeps price at this level?
    if any(w in lower for w in ["sustainable", "can this last", "will this hold", "how long will"]):
        return "causal_explanation"
    # FY / fiscal year future queries
    _fy_out = bool(re.search(r'\bfy\s*20(2[5-9]|3\d)\b', lower, re.IGNORECASE)) or any(w in lower for w in [
        "next financial year", "next fiscal year", "q1 fy", "q2 fy", "q3 fy", "q4 fy",
        "fy 2026", "fy 2027", "2026-27", "2027-28",
    ])
    if _fy_out:
        if any(w in lower for w in ["cost", "bill", "spend", "budget", "afford", "business", "manage"]):
            return "fiscal_budget_bridge"
        return "forecast"
    if requires_forecast:
        return "forecast"
    return "current_market_state"


def _uuid8() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
