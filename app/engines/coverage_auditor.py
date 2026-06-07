"""Coverage auditor — checks whether the planned answer covers what was asked.

Distinct from ClaimVerifier (which asks "is this claim true?").
This asks "did we answer the sub-questions?" — a routing/coverage check.

Priority order:
  1. Deterministic rule-based check (always runs, <1ms, no network)
  2. Ollama LLM with thinking mode (optional enhancement, 4–8s)

The deterministic check maps sub-question types → expected answer section titles.
If a sub-question has no matching section, it suggests a routing fix.

Called after answer planning, only when confidence < 0.6 AND answer_gap_risk=True.
Always falls back to AuditResult(passes=True) on any failure — never blocks.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

_SYSTEM_PROMPT = """\
You are an adversarial reviewer for an Australian electricity market analytics system.

You receive:
1. The sub-questions the user was actually asking (extracted before routing)
2. The answer sections the system produced
3. The routing label the system used

Return JSON:
{
  "passes": true or false,
  "gaps": ["describe each unaddressed sub-question"],
  "suggested_output": null or one of [
    "price_fluctuation_attribution",
    "fuel_source_recommendation",
    "causal_explanation",
    "causal_explanation_with_forecast",
    "historical_analog_outcome",
    "regional_comparison",
    "current_market_state"
  ],
  "reasoning": "one sentence"
}

Be strict:
- If the user asked about fuel sources and the answer only shows price/demand → gap
- If the user described a price sequence and the answer doesn't acknowledge it → gap
- If the user asked for comparison and got a bare snapshot → gap
- suggested_output must be a specific routing fix, not null, when passes=false"""

_VALID_OUTPUTS = {
    "price_fluctuation_attribution",
    "fuel_source_recommendation",
    "causal_explanation",
    "causal_explanation_with_forecast",
    "historical_analog_outcome",
    "regional_comparison",
    "current_market_state",
}

# Sub-question type → required answer section keyword(s) (any match = covered)
_SQ_COVERAGE_MAP: dict[str, list[str]] = {
    "current_price_reason":           ["answer", "evidence", "drivers", "why"],
    "fuel_source_comparison":         ["fuel", "source", "coal", "gas", "solar", "wind", "hydro"],
    "historical_price_distribution":  ["historical", "median", "p50", "p90", "last year", "percentile", "archive"],
    "forecast_outlook":               ["forecast", "continuation", "p10", "p50", "p90", "will"],
    "regime_change":                  ["analog", "historical", "changed", "regime"],
    "price_fluctuation":              ["answer", "evidence", "fluctuat", "movement", "path"],
    "regional_comparison":            ["comparison", "region", "state", "spread", "all nem"],
    "intraday_price_cycle":           ["answer", "morning", "evening", "solar", "diurnal", "earlier"],
    "intraday_fuel_timeline":         ["fuel", "solar cliff", "coal", "wind", "transition", "today"],
    "specific_period_stats":          ["average", "mean", "median", "period", "archive", "interval"],
    "diurnal_pattern":                ["diurnal", "time of day", "pattern", "peak", "morning", "evening"],
    "trend_analysis":                 ["trend", "monthly", "annual", "year over year", "yoy"],
    "fcas_opportunity":               ["fcas", "ancillary", "raise", "lower", "contingency"],
    "interconnector_causality":       ["interconnector", "flow", "qni", "heywood", "basslink"],
}

# Sub-question type → suggested routing fix when coverage gap detected
_SQ_ROUTING_FIX: dict[str, str] = {
    "fuel_source_comparison":        "fuel_source_recommendation",
    "price_fluctuation":             "price_fluctuation_attribution",
    "regional_comparison":           "regional_comparison",
    "historical_price_distribution": "causal_explanation",
    "forecast_outlook":              "causal_explanation_with_forecast",
    "regime_change":                 "historical_analog_outcome",
    "intraday_price_cycle":          "fuel_source_recommendation",
    "intraday_fuel_timeline":        "fuel_source_recommendation",
}


@dataclass
class AuditResult:
    passes: bool = True
    gaps: list[str] = field(default_factory=list)
    suggested_output: str | None = None
    reasoning: str = ""

    def has_actionable_suggestion(self) -> bool:
        return (
            not self.passes
            and self.suggested_output is not None
            and self.suggested_output in _VALID_OUTPUTS
        )


_PASSING = AuditResult(passes=True)


def _deterministic_audit(
    sub_questions: list[str],
    answer_sections: list[dict[str, Any]],
    current_requested_output: str,
) -> AuditResult:
    """Fast deterministic coverage check.

    Checks each sub-question type against answer section titles and items.
    Returns AuditResult immediately — no network call.
    """
    if not sub_questions:
        return _PASSING

    # Flatten all answer text for keyword scanning
    all_text = " ".join(
        (s.get("title", "") + " " + " ".join(s.get("items") or []))
        for s in answer_sections
    ).lower()

    gaps: list[str] = []
    suggested: str | None = None

    for sq in sub_questions:
        sq_type = sq if isinstance(sq, str) else str(sq)
        # Normalise: handle both plain strings and dict types
        if sq_type.startswith("{") and "type" in sq_type:
            try:
                sq_type = json.loads(sq_type.replace("'", '"')).get("type", sq_type)
            except Exception:
                pass

        keywords = _SQ_COVERAGE_MAP.get(sq_type)
        if keywords is None:
            continue  # unknown sub-question type — skip

        covered = any(kw in all_text for kw in keywords)
        if not covered:
            gaps.append(f"Sub-question '{sq_type}' has no matching evidence in answer sections")
            # Take the routing fix for the first uncovered sub-question
            if suggested is None:
                candidate = _SQ_ROUTING_FIX.get(sq_type)
                if candidate and candidate != current_requested_output and candidate in _VALID_OUTPUTS:
                    suggested = candidate

    if not gaps:
        return _PASSING

    return AuditResult(
        passes=False,
        gaps=gaps[:3],
        suggested_output=suggested,
        reasoning=f"Deterministic check: {len(gaps)} sub-question(s) unaddressed in answer sections.",
    )


async def audit_coverage(
    sub_questions: list[str],
    answer_sections: list[dict[str, Any]],
    current_requested_output: str,
) -> AuditResult:
    """Coverage audit — deterministic first, Ollama enhancement optional.

    Always runs the fast deterministic check first.
    If deterministic finds gaps, returns immediately (no Ollama call needed).
    If deterministic passes AND Ollama is available, runs LLM for deeper analysis.
    Falls back to _PASSING on any failure — never blocks the pipeline.
    """
    if not sub_questions:
        return _PASSING

    # ── Step 1: deterministic check (always, <1ms) ────────────────────────────
    det_result = _deterministic_audit(sub_questions, answer_sections, current_requested_output)
    if not det_result.passes and det_result.has_actionable_suggestion():
        logger.debug(
            "Coverage auditor (deterministic): gaps=%s, re-routing to %s",
            det_result.gaps, det_result.suggested_output,
        )
        return det_result

    # ── Step 2: Ollama enhancement (optional, only when deterministic passes) ──
    # Skip if: rule_based mode, no gaps found by deterministic (save latency),
    # or this is a simple single sub-question query.
    if _settings.decomposer_backend == "rule_based":
        return det_result
    if det_result.passes and len(sub_questions) <= 1:
        return _PASSING  # single sub-question, deterministic says OK — skip LLM

    model = _settings.ollama_model
    all_items = [item for sec in answer_sections for item in (sec.get("items") or [])]
    answer_text = "\n".join(f"- {item}" for item in all_items[:12]) or "(no answer items)"
    section_titles = [s.get("title", "") for s in answer_sections]

    user_content = (
        f"Sub-questions the user was asking:\n"
        + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
        + f"\n\nAnswer produced (sections: {section_titles}):\n{answer_text}"
        + f"\n\nCurrent routing label: {current_requested_output}"
        + "\n\nDoes this answer address each sub-question? /no_think"
    )

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 2048},
    }

    try:
        async with httpx.AsyncClient(
            base_url=_settings.ollama_base_url,
            timeout=httpx.Timeout(5.0),   # 5s max — fast fail, not 12s
        ) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]

        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"```(?:json)?", "", clean).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return det_result  # fall back to deterministic
        data: dict = json.loads(match.group())

        suggested = data.get("suggested_output") or None
        if suggested and suggested not in _VALID_OUTPUTS:
            suggested = None

        result = AuditResult(
            passes=bool(data.get("passes", True)),
            gaps=[str(g) for g in (data.get("gaps") or [])],
            suggested_output=suggested,
            reasoning=str(data.get("reasoning") or ""),
        )
        if not result.passes:
            logger.debug(
                "Coverage auditor (LLM): gaps=%s, suggesting=%s",
                result.gaps, result.suggested_output,
            )
        return result

    except Exception as exc:
        logger.debug("Coverage auditor LLM failed (using deterministic result): %s", exc)
        return det_result  # fall back to deterministic result, not blind _PASSING
