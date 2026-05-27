"""Query restatement — thin LLM pass that restates what the user is actually asking.

Runs before decomposition. Uses qwen3:14b with /no_think (~1s).
Always falls back to an empty result on any failure — never blocks the hot path.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

_SYSTEM_PROMPT = """\
You are a query-understanding assistant for an Australian electricity market analytics platform (NEM).

Given a user query, output JSON with:
- sub_questions: list of 2-3 concrete sub-questions the user is actually asking
- primary_intent: one of [price_explanation, fuel_comparison, price_fluctuation, forecast, \
historical_lookup, market_status, out_of_scope]
- explicit_entities: {"regions": [], "fuels": [], "price_values": [], "time_refs": []}
- answer_gap_risk: true if a simple current-state lookup would miss what the user wants

Rules:
- sub_questions must be specific, not generic ("What caused the $143→$167 price rise?" not "Why did price change?")
- price_values: extract numeric prices from the query text (e.g. 143, 167, 165, 140)
- fuels: extract any mentioned fuel types (coal, solar, hydro, wind, gas, battery)
- answer_gap_risk=true when: query is multi-part, mentions a price sequence, asks for attribution/comparison

Return ONLY valid JSON, no explanation."""

_EXAMPLE_USER = (
    "why did the price fluctuate from 143 to 167 to 165 and then back down to 140? "
    "which fuel source or other reasons?"
)
_EXAMPLE_ASSISTANT = json.dumps({
    "sub_questions": [
        "What caused the price to spike from $143 to $167/MWh?",
        "Why did it partially recover to $165 then fall back to $140?",
        "Which fuel type was the marginal generator during these moves?",
    ],
    "primary_intent": "price_fluctuation",
    "explicit_entities": {
        "regions": [],
        "fuels": [],
        "price_values": [143, 167, 165, 140],
        "time_refs": [],
    },
    "answer_gap_risk": True,
})

_REASONING_PREFIXES = {"qwen3", "qwq", "deepseek-r1", "marco-o1"}


def _is_reasoning_model(model: str) -> bool:
    lower = model.lower()
    return any(p in lower for p in _REASONING_PREFIXES)


@dataclass
class RestatementResult:
    sub_questions: list[str] = field(default_factory=list)
    primary_intent: str = ""
    explicit_entities: dict = field(default_factory=dict)
    answer_gap_risk: bool = False

    def is_empty(self) -> bool:
        return not self.sub_questions

    def seed_text(self, original: str) -> str:
        """Prefix to prepend to the original query for the decomposer.

        Gives the decomposer LLM concrete sub-questions as context so it
        produces better causal_targets and requested_output.
        """
        if not self.sub_questions:
            return original
        joined = "; ".join(self.sub_questions)
        return f"[The user is asking: {joined}]\n{original}"


_EMPTY = RestatementResult()


async def restate_query(text: str) -> RestatementResult:
    """Restate the query into explicit sub-questions using qwen3:14b /no_think.

    Returns RestatementResult. Falls back to _EMPTY on any error — safe to ignore.
    """
    if _settings.decomposer_backend == "rule_based":
        return _EMPTY

    model = _settings.ollama_model
    # For reasoning models, /no_think suppresses chain-of-thought and speeds up output.
    user_content = f"/no_think {text}" if _is_reasoning_model(model) else text

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _EXAMPLE_USER},
            {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400, "num_ctx": 2048},
    }
    # Do NOT set format:"json" for reasoning models — grammar constraint silences output
    # when combined with /no_think (same pattern as decomposition.py).
    if not _is_reasoning_model(model):
        payload["format"] = "json"

    try:
        async with httpx.AsyncClient(
            base_url=_settings.ollama_base_url,
            timeout=httpx.Timeout(4.0),  # tight budget — must not slow the hot path
        ) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]

        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"```(?:json)?", "", clean).strip()
        data: dict = json.loads(clean)

        return RestatementResult(
            sub_questions=[str(q) for q in (data.get("sub_questions") or [])[:3]],
            primary_intent=str(data.get("primary_intent") or ""),
            explicit_entities=dict(data.get("explicit_entities") or {}),
            answer_gap_risk=bool(data.get("answer_gap_risk", False)),
        )
    except Exception as exc:
        logger.debug("Query restatement failed (non-fatal, using empty): %s", exc)
        return _EMPTY
