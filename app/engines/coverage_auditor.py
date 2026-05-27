"""Coverage auditor — checks whether the planned answer covers what was asked.

Distinct from ClaimVerifier (which asks "is this claim true?").
This asks "did we answer the sub-questions?" — a routing/coverage check, not a truth check.

Called after answer planning, only when confidence < 0.6 AND answer_gap_risk=True.
Uses qwen3:14b WITH thinking mode enabled (~4-8s).
Always falls back to AuditResult(passes=True) on any failure — never blocks.

If a routing gap is found, routes_query.py re-plans ONCE with the suggested_output.
No loop: one re-plan attempt, then return whatever we have.
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

Your job: assess whether the answer addresses each sub-question.

Return JSON:
{
  "passes": true or false,
  "gaps": ["describe each unaddressed sub-question and why"],
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



async def audit_coverage(
    sub_questions: list[str],
    answer_sections: list[dict[str, Any]],
    current_requested_output: str,
) -> AuditResult:
    """Run adversarial critique with thinking mode ON.

    Returns AuditResult. Falls back to _PASSING on any failure.
    """
    if not sub_questions or _settings.decomposer_backend == "rule_based":
        return _PASSING

    model = _settings.ollama_model

    all_items = [item for sec in answer_sections for item in (sec.get("items") or [])]
    answer_text = "\n".join(f"- {item}" for item in all_items[:12]) or "(no answer items)"
    section_titles = [s.get("title", "") for s in answer_sections]

    user_content = (
        f"Sub-questions the user was asking:\n"
        + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
        + f"\n\nAnswer produced (sections: {section_titles}):\n{answer_text}"
        + f"\n\nCurrent routing label: {current_requested_output}"
        + "\n\nDoes this answer address each sub-question?"
    )

    # Thinking mode ON — no /no_think prefix. qwen3 will reason before answering.
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600, "num_ctx": 3072},
    }

    try:
        async with httpx.AsyncClient(
            base_url=_settings.ollama_base_url,
            timeout=httpx.Timeout(12.0),
        ) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]

        # Strip think/reasoning block — keep only the JSON answer
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"```(?:json)?", "", clean).strip()

        # Find first JSON object in the cleaned output
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            logger.debug("Adversarial critic: no JSON object in output")
            return _PASSING
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
                "Adversarial critic FAIL — gaps: %s, suggesting: %s",
                result.gaps,
                result.suggested_output,
            )
        return result

    except Exception as exc:
        logger.debug("Adversarial critic failed (non-fatal, passing): %s", exc)
        return _PASSING
