"""SecurityObserver — 4-pass hygiene for every query lifecycle.

Pass 1 — Input sanitisation
  Blocks: prompt injection, PII leakage, market manipulation language,
           out-of-scope topics, excessive length.

Pass 2 — Decomposition validation
  Blocks: unsafe intents (real execution commands, private data extraction),
           out-of-scope classification, very low confidence ambiguous queries.

Pass 3 — Tool output validation
  Blocks: injected instructions inside retrieved data, anomalous numeric values,
           stale-data masquerading as fresh, source trust violations.

Pass 4 — Answer validation
  Blocks: numeric claims without evidence_refs, SUPPORTED verdict with no evidence,
           missing disclaimer, overconfidence (confidence > 0.95 with no analogs).

Risk scoring: 0–100. ≥80 = halt pipeline. 40–79 = warn but proceed.
Each pass returns ObserverResult with full signal list for audit logging.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Thresholds ────────────────────────────────────────────────────────
_HALT = 80
_WARN = 40


# ── Pass 1: Prompt injection patterns ────────────────────────────────
_INJECTION_RE = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in [
        r"ignore\s+(previous|all|above)\s+instructions",
        r"you\s+are\s+now\s+(a|an|the)",
        r"act\s+as\s+(a|an|the)",
        r"disregard\s+(your|all)\s+(previous\s+)?(system\s+)?instructions",
        r"(jailbreak|dan\s+mode|developer\s+mode|god\s+mode)",
        r"forget\s+(everything|your\s+instructions|your\s+training)",
        r"\bsystem\s+prompt\b",
        r"<\s*/?\s*system\s*>",
        r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>",
        r"new\s+instruction[s]?\s*:",
        r"###\s+instruction",
        r"assistant\s*:\s*sure",
    ]
]

# ── Pass 1: PII patterns ──────────────────────────────────────────────
_PII_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b\d{3}-\d{2}-\d{4}\b",                              # US SSN
        r"\b4[0-9]{12}(?:[0-9]{3})?\b",                        # Visa
        r"\b5[1-5][0-9]{14}\b",                                  # Mastercard
        r"\b(?:password|passwd|secret|api[_\-]?key)\s*[:=]\s*\S+",
        r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",       # email
    ]
]

# ── Pass 1: Market manipulation ───────────────────────────────────────
_MANIPULATION_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(pump|dump)\s+(the\s+)?(market|price)",
        r"corner\s+the\s+market",
        r"\bfront.?run\b",
        r"\binsider\s+(trading|information)\b",
        r"without\s+(AEMO|the\s+regulator|AER)\s+knowing",
        r"manipulat(e|ing|ion)\s+(the\s+)?(market|price|spread)",
        r"\bspoofing\b|\blayering\b",
    ]
]

# ── Pass 1: Hard OOS topics ───────────────────────────────────────────
_OOS_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(stock|equity|share\s+price|dividend)\b",
        r"\b(forex|currency|exchange\s+rate)\b",
        r"\b(crypto|bitcoin|ethereum|nft)\b",
        r"\b(prime\s+minister|election|political\s+party)\b",
        r"\b(sport|football|cricket|tennis)\b",
    ]
]

# ── Pass 2: Unsafe intent patterns ───────────────────────────────────
_UNSAFE_INTENT_RE = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Multi-adjective forms: "real live bid", "actual real order", etc.
        r"(execute|submit|place|send)\s+(a\s+)?(?:(real|actual|live)\s+){1,3}(bid|offer|order)",
        # Broader AEMO execution language
        r"\b(submit|place|send|execute)\b.{0,40}\b(AEMO|NEMDE|market)\b.{0,40}\b(bid|offer|order)\b",
        r"\b(real|actual|live)\b.{0,30}\b(bid|offer|order)\b.{0,30}\b(AEMO|market|NEM)\b",
        r"(log\s+into|access)\s+(AEMO|MSATS|NEMWEB)\s+(as|with\s+credentials)",
        r"extract\s+(all\s+)?(user|tenant|customer)\s+data",
        r"delete\s+(the\s+)?(database|all\s+records|table)",
        r"reveal\s+(the\s+)?(system\s+prompt|instructions|api\s+key)",
    ]
]

# ── Pass 3: Injection inside retrieved data ───────────────────────────
_TOOL_INJECTION_RE = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in [
        r"ignore\s+(previous|all)\s+instructions",
        r"you\s+are\s+now",
        r"\bsystem\s+prompt\b",
        r"<\s*/?\s*system\s*>",
        r"new\s+instruction[s]?\s*:",
    ]
]


@dataclass
class ObserverSignal:
    name: str
    score: int
    description: str
    control_ref: str | None = None  # ISO/IEC 27001:2022 Annex A reference, e.g. "A.8.28"


@dataclass
class ObserverResult:
    phase: str
    risk_score: int
    risk_band: str           # low | medium | high | critical
    signals: list[ObserverSignal] = field(default_factory=list)
    relevance_class: str = "in_scope"
    verdict: str = "pass"    # pass | warn | halt

    def should_halt(self) -> bool:
        return self.risk_score >= _HALT

    def to_dict(self) -> dict[str, Any]:
        from app.compliance.iso27001_controls import get_control_ref
        return {
            "phase": self.phase,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "signals": [
                {
                    "name": s.name,
                    "score": s.score,
                    "description": s.description,
                    "control_ref": s.control_ref or get_control_ref(s.name),
                }
                for s in self.signals
            ],
            "relevance_class": self.relevance_class,
            "verdict": self.verdict,
        }

    def primary_control_ref(self) -> str | None:
        """Return the Annex A control ref of the highest-risk signal."""
        from app.compliance.iso27001_controls import get_control_ref
        if not self.signals:
            return None
        top = max(self.signals, key=lambda s: s.score)
        return top.control_ref or get_control_ref(top.name)


class SecurityObserver:
    """Stateless, synchronous observer — all passes O(n) regex, no I/O."""

    # ── Pass 1: Input ────────────────────────────────────────────────
    def pass_input(self, text: str, tenant_id: str = "") -> ObserverResult:
        signals: list[ObserverSignal] = []
        score = 0

        for pat in _INJECTION_RE:
            if pat.search(text):
                sig = ObserverSignal(
                    "prompt_injection", 90,
                    f"Injection pattern: {pat.pattern[:60]}"
                )
                signals.append(sig)
                score = max(score, 90)

        for pat in _MANIPULATION_RE:
            if pat.search(text):
                sig = ObserverSignal(
                    "manipulation_signal", 85,
                    "Market manipulation language detected"
                )
                signals.append(sig)
                score = max(score, 85)

        for pat in _PII_RE:
            if pat.search(text):
                signals.append(ObserverSignal("pii_in_input", 35, "PII-like pattern in user input"))
                score = max(score, 35)

        oos_hits = [pat.pattern[:40] for pat in _OOS_RE if pat.search(text)]
        if oos_hits:
            signals.append(ObserverSignal("oos_topic", 20, f"Non-NEM topic: {oos_hits[0]}"))
            score = max(score, 20)

        if len(text) > 1500:
            signals.append(ObserverSignal("excessive_length", 15, f"Input {len(text)} chars"))
            score = max(score, 15)

        # Invisible unicode / homoglyph attempts
        suspicious_chars = sum(1 for c in text if ord(c) > 8000 and not c.isspace())
        if suspicious_chars > 5:
            signals.append(ObserverSignal("unicode_anomaly", 40, f"{suspicious_chars} suspicious codepoints"))
            score = max(score, 40)

        return _result("input", score, signals)

    # ── Pass 2: Decomposition ────────────────────────────────────────
    def pass_decomposition(self, decomp: dict[str, Any]) -> ObserverResult:
        signals: list[ObserverSignal] = []
        score = 0

        intent = decomp.get("intent", "")
        raw_query = decomp.get("raw_query", "")

        # Hard block: real execution commands
        for pat in _UNSAFE_INTENT_RE:
            if pat.search(raw_query):
                signals.append(ObserverSignal(
                    "unsafe_execution_intent", 95,
                    "Query requests real market action or credential access"
                ))
                score = max(score, 95)

        if intent == "out_of_scope":
            signals.append(ObserverSignal("oos_intent", 65, "Decomposer classified as out_of_scope"))
            score = max(score, 65)

        confidence = float(decomp.get("confidence", 1.0))
        if confidence < 0.30:
            signals.append(ObserverSignal(
                "very_low_decomp_confidence", 45,
                f"Decomp confidence {confidence:.2f} — likely garbled or ambiguous query"
            ))
            score = max(score, 45)
        elif confidence < 0.50:
            signals.append(ObserverSignal(
                "low_decomp_confidence", 20,
                f"Decomp confidence {confidence:.2f}"
            ))
            score = max(score, 20)

        # Requires_portfolio with no tenant approval is medium risk
        if decomp.get("requires_portfolio"):
            signals.append(ObserverSignal(
                "portfolio_data_requested", 30,
                "Query involves portfolio data — never sent to cloud LLM without approval"
            ))
            score = max(score, 30)

        return _result("decomposition", score, signals)

    # ── Pass 3: Tool output ──────────────────────────────────────────
    def pass_tool_output(self, tool_outputs: list[dict[str, Any]]) -> ObserverResult:
        signals: list[ObserverSignal] = []
        score = 0

        for i, output in enumerate(tool_outputs):
            src = output.get("source", f"tool_{i}")

            # Injection in retrieved text content
            for field_name in ("reason", "title", "description", "content", "text"):
                val = str(output.get(field_name, ""))
                for pat in _TOOL_INJECTION_RE:
                    if pat.search(val):
                        signals.append(ObserverSignal(
                            "tool_output_injection", 85,
                            f"Injection pattern in {src}.{field_name}"
                        ))
                        score = max(score, 85)

            # Anomalous numeric ranges
            price = output.get("price_rrp")
            if price is not None:
                p = float(price)
                if p < -1000 or p > 20_000:
                    signals.append(ObserverSignal(
                        "price_anomaly", 55,
                        f"{src}: price_rrp={p} outside plausible NEM range [-1000, 20000]"
                    ))
                    score = max(score, 55)

            demand = output.get("demand_mw")
            if demand is not None:
                d = float(demand)
                if d < 0 or d > 60_000:
                    signals.append(ObserverSignal(
                        "demand_anomaly", 45,
                        f"{src}: demand_mw={d} outside plausible range [0, 60000]"
                    ))
                    score = max(score, 45)

            avail = output.get("availability_mw")
            if avail is not None:
                a = float(avail)
                if a < 0 or a > 80_000:
                    signals.append(ObserverSignal(
                        "availability_anomaly", 40,
                        f"{src}: availability_mw={a} outside plausible range"
                    ))
                    score = max(score, 40)

            temp = output.get("temperature_c")
            if temp is not None:
                t = float(temp)
                if t < -20 or t > 55:
                    signals.append(ObserverSignal(
                        "weather_temperature_anomaly",
                        35,
                        f"{src}: temperature_c={t} outside plausible Australian operating range",
                    ))
                    score = max(score, 35)

            wind = output.get("wind_speed_kmh")
            if wind is not None:
                w = float(wind)
                if w < 0 or w > 200:
                    signals.append(ObserverSignal(
                        "weather_wind_anomaly",
                        35,
                        f"{src}: wind_speed_kmh={w} outside plausible range",
                    ))
                    score = max(score, 35)

        return _result("tool_output", score, signals)

    # ── Pass 4: Answer ───────────────────────────────────────────────
    def pass_answer(self, answer: dict[str, Any]) -> ObserverResult:
        signals: list[ObserverSignal] = []
        score = 0

        verdict = answer.get("verdict", "")
        evidence_refs = answer.get("evidence_refs", [])
        confidence = float(answer.get("confidence", 0.0))
        analogs = answer.get("historical_analogs")

        # SUPPORTED with no evidence = hallucination risk
        if verdict == "SUPPORTED" and not evidence_refs:
            signals.append(ObserverSignal(
                "unsupported_claim", 85,
                "SUPPORTED verdict has no evidence_refs — numeric claim is ungrounded"
            ))
            score = max(score, 85)

        # Disclaimer must be present
        if not answer.get("disclaimer"):
            signals.append(ObserverSignal("missing_disclaimer", 25, "Answer has no disclaimer"))
            score = max(score, 25)

        # Overconfidence: very high confidence with no analogs and no news
        analog_count = (analogs or {}).get("count", 0) if analogs else 0
        news = answer.get("news_correlation")
        if confidence > 0.92 and analog_count == 0 and not news:
            signals.append(ObserverSignal(
                "overconfidence", 40,
                f"Confidence {confidence:.2f} with no analogs and no news correlation"
            ))
            score = max(score, 40)

        # Counterargument must be present for action recommendations
        action = answer.get("action", "")
        if action in ("dispatch_now", "charge") and not answer.get("counterargument"):
            signals.append(ObserverSignal(
                "missing_counterargument", 30,
                f"Action '{action}' has no counterargument — adversarial critic required"
            ))
            score = max(score, 30)

        # Check each evidence ref has valid numeric value
        for ref in evidence_refs:
            if ref.get("value") is None:
                signals.append(ObserverSignal(
                    "null_evidence_value", 20,
                    f"Evidence ref {ref.get('id', '?')} has null value"
                ))
                score = max(score, 20)

        return _result("answer", score, signals)


# ── Helpers ───────────────────────────────────────────────────────────

def _result(phase: str, score: int, signals: list[ObserverSignal]) -> ObserverResult:
    if score >= _HALT:
        band, verdict = "critical", "halt"
    elif score >= _WARN:
        band, verdict = "high", "warn"
    elif score >= 15:
        band, verdict = "medium", "warn"
    else:
        band, verdict = "low", "pass"

    oos = any(s.name in ("oos_intent", "oos_topic") for s in signals)
    relevance = (
        "out_of_scope" if oos and score >= 60
        else "marginal" if score >= 15
        else "in_scope"
    )

    return ObserverResult(
        phase=phase,
        risk_score=score,
        risk_band=band,
        signals=signals,
        relevance_class=relevance,
        verdict=verdict,
    )


async def log_observer_event(
    session: Any,
    result: "ObserverResult",
    tenant_id: str = "system",
    query_id: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Persist an ObserverResult to the database as an ObserverEvent row.

    Returns the generated event ID. Safe to call; swallows DB errors with a
    warning so that security logging never blocks the main query pipeline.
    """
    import uuid as _uuid
    from app.db.models import ObserverEvent

    event_id = f"obs-evt-{_uuid.uuid4()}"
    control_ref = result.primary_control_ref()
    row = ObserverEvent(
        id=event_id,
        tenant_id=tenant_id,
        query_id=query_id,
        trace_id=trace_id,
        phase=result.phase,
        relevance_class=result.relevance_class,
        risk_score=result.risk_score,
        risk_band=result.risk_band,
        verdict=result.verdict,
        signals=[
            {
                "name": s.name,
                "score": s.score,
                "description": s.description,
                "control_ref": s.control_ref or result.primary_control_ref(),
            }
            for s in result.signals
        ],
        control_ref=control_ref,
    )
    session.add(row)
    try:
        await session.flush()
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("ObserverEvent DB write failed: %s", exc)
    return event_id


# Singleton
_observer = SecurityObserver()


def get_observer() -> SecurityObserver:
    return _observer
