"""Deterministic verdict derivation — no LLM, no randomness.

The verdict label and confidence band are computed from measurable inputs.
The LLM narrates the result; it never produces the verdict itself.
"""
from __future__ import annotations

from .schema import ActionLabel, ConfidenceBand, IntentLabel, VerdictLabel


def derive_confidence_band(score: float) -> ConfidenceBand:
    if score >= 0.80:
        return ConfidenceBand.HIGH
    if score >= 0.60:
        return ConfidenceBand.MEDIUM
    if score >= 0.40:
        return ConfidenceBand.LOW
    return ConfidenceBand.VERY_LOW


def derive_verdict(
    decomposition_confidence: float,
    source_coverage_ok: bool,
    live_data_fresh: bool,
    archive_available: bool,
    requires_archive: bool,
    analog_count: int,
    min_analog_count: int = 3,
    intent: IntentLabel | None = None,
) -> VerdictLabel:
    """Deterministic verdict from measurable source/coverage signals."""
    # Adjacent intents short-circuit before data checks — they have their own
    # verdict logic driven by the adjacent_handlers module, not live data.
    if intent == IntentLabel.OUT_OF_SCOPE:
        return VerdictLabel.OUT_OF_SCOPE
    if intent == IntentLabel.GEOGRAPHIC_REDIRECT:
        return VerdictLabel.PARTIAL_SCOPE   # structural answer, no live data needed
    if intent in (IntentLabel.PARTIAL_SCOPE, IntentLabel.EVIDENCE_BRIDGE):
        return VerdictLabel.PARTIAL_SCOPE   # capped — never reaches SUPPORTED

    if not live_data_fresh:
        return VerdictLabel.INSUFFICIENT_DATA
    if requires_archive and not archive_available:
        return VerdictLabel.INSUFFICIENT_DATA
    if requires_archive and analog_count < min_analog_count:
        return VerdictLabel.LOW_CONFIDENCE
    if not source_coverage_ok:
        return VerdictLabel.LOW_CONFIDENCE
    if decomposition_confidence < 0.50:
        return VerdictLabel.NEEDS_CLARIFICATION
    if decomposition_confidence < 0.72:
        return VerdictLabel.LOW_CONFIDENCE
    return VerdictLabel.SUPPORTED


def derive_action(
    verdict: VerdictLabel,
    regime: str,
    price_rrp: float,
    forecast_direction: str,        # "rising" | "falling" | "flat" | "unknown"
    analog_success_rate: float,
    news_explained: bool,
) -> ActionLabel:
    """Deterministic action from market signals — LLM refines the narrative only."""
    if verdict in (VerdictLabel.INSUFFICIENT_DATA, VerdictLabel.NEEDS_CLARIFICATION):
        return ActionLabel.MONITOR
    if verdict == VerdictLabel.OUT_OF_SCOPE:
        return ActionLabel.REFUSE
    if verdict == VerdictLabel.PARTIAL_SCOPE:
        return ActionLabel.MONITOR   # partial answer — user should read the scope note

    if regime in ("spike", "extreme") and forecast_direction == "rising":
        if analog_success_rate >= 0.70:
            return ActionLabel.DISPATCH_NOW
        return ActionLabel.MONITOR

    if regime == "elevated" and forecast_direction == "rising" and news_explained:
        return ActionLabel.DISPATCH_NOW

    if regime in ("normal",) and forecast_direction == "falling":
        return ActionLabel.CHARGE

    return ActionLabel.HOLD


def compute_confidence(
    decomposition_confidence: float,
    source_freshness_score: float,   # 0-1, 1=fully fresh
    source_coverage_score: float,    # 0-1, 1=all sources available
    analog_count: int,
    analog_consistency: float,       # fraction of analogs agreeing with recommendation
    news_tier: int | None,           # 1=AEMO notice, 2=press, None=unexplained
) -> float:
    """Weighted confidence combining all measurable uncertainty signals."""
    weights = {
        "decomp": 0.20,
        "freshness": 0.20,
        "coverage": 0.15,
        "analogs": 0.25,
        "news": 0.20,
    }
    analog_score = min(1.0, analog_count / 10) * analog_consistency
    news_score = {1: 1.0, 2: 0.75, None: 0.40}.get(news_tier, 0.40)

    raw = (
        weights["decomp"] * decomposition_confidence
        + weights["freshness"] * source_freshness_score
        + weights["coverage"] * source_coverage_score
        + weights["analogs"] * analog_score
        + weights["news"] * news_score
    )
    return round(min(1.0, max(0.0, raw)), 3)
