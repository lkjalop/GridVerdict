"""Claim Verifier — answer guard that runs before the response leaves the API.

Checks the FactualVerdict against 6 hard rules:
  1. Numeric claim  → evidence_ref required
  2. Causal claim   → claim_tier required
  3. Confirmed/supported claim_tier → evidence_ref_ids required (invariant)
  4. Forecast claim → forecast model source in evidence_refs required
  5. "caused by" wording → confirmed or supported driver_tier must exist
  6. Rebid/outage language → supported+ claim_tier for rebid/outage/unit_dispatch required

When a rule fails:
  - warn      → adds to missing_data + counterargument note (verdict unchanged)
  - downgrade → also downgrades SUPPORTED → LOW_CONFIDENCE + caps confidence at 0.55

apply_verification() rebuilds the FactualVerdict via model_validate so that
Pydantic validators (confidence_band, numeric_claims_need_evidence) re-run
on the corrected values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.schema import FactualVerdict, VerdictLabel

# ── Regex patterns ─────────────────────────────────────────────────────────────

_NUMERIC_RE = [
    re.compile(r'\$[\d,]+\.?\d*\s*/?\s*MWh', re.IGNORECASE),   # $500.25/MWh
    re.compile(r'\b[\d,]+\s*MW(?:h)?\b', re.IGNORECASE),         # 5,000 MW / 300 MWh
    re.compile(r'\b\d+\.?\d*\s*%'),                               # 45%  / 12.5%
]

_CAUSAL_RE = re.compile(
    r'\b(caused by|due to|driven by|results? in|led to|because of|attributable to|'
    r'owing to|driving the|responsible for|contributing to)\b',
    re.IGNORECASE,
)

_CAUSED_BY_EXACT = re.compile(r'\bcaused by\b', re.IGNORECASE)

_OUTAGE_RE = re.compile(
    r'\b(rebid|rebidding|rebids|forced outage|unit tripped|unit trip|unit offline|'
    r'availability withdrawal|generator trip|generator outage|plant outage|plant trip)\b',
    re.IGNORECASE,
)

_OUTAGE_TIERS = frozenset({"rebid", "outage", "unit_dispatch"})

_FORECAST_RE = re.compile(
    r'\b(forecast|predicted|projected|expected to|will reach|anticipates?|expects?|'
    r'pre-dispatch|predispatch)\b',
    re.IGNORECASE,
)

_FORECAST_SOURCES = frozenset({
    "AEMO_PREDISPATCH_30MIN",
    "AEMO_PREDISPATCH_5MIN",
    "LNN_FORECAST",
    "LEAR_FORECAST",
    "QRA_FORECAST",
})

_HIGH_TIERS = frozenset({"confirmed", "supported"})

# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class ClaimFinding:
    rule: str        # identifier e.g. "numeric_without_evidence"
    severity: str    # "warn" | "downgrade"
    detail: str      # human-readable explanation for logs / audit


@dataclass
class VerificationResult:
    passed: bool
    findings: list[ClaimFinding] = field(default_factory=list)
    suggested_verdict: VerdictLabel | None = None
    appended_counterargument: str = ""
    appended_missing_data: list[str] = field(default_factory=list)
    corrected_claim_tiers: list[dict] | None = None  # None = no correction


# ── Public API ─────────────────────────────────────────────────────────────────


def verify_answer(factual: FactualVerdict) -> VerificationResult:
    """Run all 5 claim-verification rules against a FactualVerdict.

    Returns a VerificationResult. Does NOT mutate factual.
    """
    findings: list[ClaimFinding] = []
    suggested_verdict: VerdictLabel | None = None
    extra_counterargument: list[str] = []
    extra_missing: list[str] = []
    corrected_tiers: list[dict] | None = None

    text = factual.why_plain_english or ""
    refs = factual.evidence_refs or []
    claim_tiers = list(factual.claim_tiers or [])
    driver_tiers = list(factual.driver_tiers or [])
    ref_sources = {r.source for r in refs}

    # ── Rule 1: numeric claim → evidence_ref required ──────────────────
    if _has_numeric_claim(text) and not refs:
        findings.append(ClaimFinding(
            rule="numeric_without_evidence",
            severity="warn",
            detail=(
                "Narrative contains numeric claims (price/MW/%) but evidence_refs is empty. "
                "Numbers must trace to a cited source."
            ),
        ))
        extra_missing.append("evidence_for_numeric_claims")
        extra_counterargument.append(
            "Numeric values in this answer lack direct evidence references."
        )

    # ── Rule 2: causal claim → claim_tier required ─────────────────────
    if _has_causal_language(text) and not claim_tiers:
        findings.append(ClaimFinding(
            rule="causal_without_claim_tiers",
            severity="warn",
            detail=(
                "Causal language detected in narrative but claim_tiers list is empty. "
                "Causal claims must be graded against evidence tiers."
            ),
        ))
        extra_missing.append("causal_evidence_tiers")
        extra_counterargument.append(
            "Causal claims in this answer have not been graded against evidence tiers."
        )

    # ── Rule 3: confirmed/supported claim_tier → evidence_ref_ids required
    corrected = list(claim_tiers)
    tier_corrected = False
    for i, ct in enumerate(corrected):
        if ct.get("tier") in _HIGH_TIERS and not ct.get("evidence_ref_ids"):
            label = ct.get("label", "unknown")
            tier = ct["tier"]
            findings.append(ClaimFinding(
                rule="high_tier_without_evidence_refs",
                severity="downgrade",
                detail=(
                    f"claim_tier '{label}' is '{tier}' but evidence_ref_ids is empty. "
                    "Confirmed/supported tiers require direct evidence references."
                ),
            ))
            corrected[i] = {**ct, "tier": "plausible"}
            tier_corrected = True
            extra_counterargument.append(
                f"'{label}' driver was marked {tier} but lacks direct evidence "
                "references — downgraded to plausible."
            )

    if tier_corrected:
        corrected_tiers = corrected

    # ── Rule 4: forecast claim → forecast model source required ───────────
    if _has_forecast_language(text) and not (ref_sources & _FORECAST_SOURCES):
        findings.append(ClaimFinding(
            rule="forecast_without_model_evidence",
            severity="warn",
            detail=(
                "Forecast language detected but no forecast model source "
                "(PREDISPATCH / LNN / LEAR / QRA) is in evidence_refs."
            ),
        ))
        extra_missing.append("forecast_model_source")
        extra_counterargument.append(
            "Forecast references in this answer are not backed by a cited model result."
        )

    # ── Rule 5: "caused by" → confirmed or supported driver required ──────
    if _CAUSED_BY_EXACT.search(text):
        has_strong_driver = any(
            dt.get("tier") in _HIGH_TIERS and dt.get("present", False)
            for dt in driver_tiers
        )
        if not has_strong_driver:
            findings.append(ClaimFinding(
                rule="caused_by_without_confirmed_driver",
                severity="downgrade",
                detail=(
                    '"caused by" wording used but no confirmed or supported driver tier exists. '
                    "This is an overclaim — the causal attribution is not substantiated."
                ),
            ))
            extra_counterargument.append(
                '"Caused by" wording requires a confirmed or supported market driver. '
                "This causal claim is not yet substantiated to that standard."
            )
            if factual.verdict == VerdictLabel.SUPPORTED:
                suggested_verdict = VerdictLabel.LOW_CONFIDENCE

    # ── Rule 6: rebid/outage language → supported+ claim_tier required ───
    if _OUTAGE_RE.search(text):
        has_outage_evidence = any(
            ct.get("tier") in _HIGH_TIERS and ct.get("category") in _OUTAGE_TIERS
            for ct in claim_tiers
        )
        if not has_outage_evidence:
            findings.append(ClaimFinding(
                rule="outage_rebid_without_evidence_tier",
                severity="downgrade",
                detail=(
                    "Rebid or outage language detected but no supported/confirmed claim_tier "
                    "for rebid, outage, or unit_dispatch category exists. "
                    "Attribution to a specific plant action requires direct bid or dispatch evidence."
                ),
            ))
            extra_counterargument.append(
                "Rebid or outage attribution requires a supported or confirmed evidence tier "
                "for rebid/outage/unit_dispatch. This causal claim is not yet substantiated."
            )
            if factual.verdict == VerdictLabel.SUPPORTED:
                suggested_verdict = VerdictLabel.LOW_CONFIDENCE

    # ── Aggregate downgrade decision ──────────────────────────────────────
    has_downgrade = any(f.severity == "downgrade" for f in findings)
    if (
        has_downgrade
        and factual.verdict == VerdictLabel.SUPPORTED
        and suggested_verdict is None
    ):
        suggested_verdict = VerdictLabel.LOW_CONFIDENCE

    if has_downgrade:
        try:
            from app.api.metrics_registry import claim_verifier_downgrades_total
            claim_verifier_downgrades_total.inc()
        except Exception:
            pass

    return VerificationResult(
        passed=not findings,
        findings=findings,
        suggested_verdict=suggested_verdict,
        appended_counterargument=" ".join(extra_counterargument),
        appended_missing_data=extra_missing,
        corrected_claim_tiers=corrected_tiers,
    )


def apply_verification(factual: FactualVerdict, result: VerificationResult) -> FactualVerdict:
    """Apply a VerificationResult to produce a corrected FactualVerdict.

    Rebuilds the model via model_validate so Pydantic validators (confidence_band,
    numeric_claims_need_evidence) re-execute on the updated values.

    Fields modified: verdict, confidence, counterargument, missing_data, claim_tiers.
    Fields never modified: why_plain_english, evidence_refs, action, trace_id.
    """
    if result.passed:
        return factual

    updates: dict = {}

    # Verdict downgrade (only downgrade, never upgrade)
    if (
        result.suggested_verdict is not None
        and _verdict_rank(factual.verdict) > _verdict_rank(result.suggested_verdict)
    ):
        updates["verdict"] = result.suggested_verdict
        updates["confidence"] = min(factual.confidence, 0.55)

    # Counterargument extension (append verifier note, but only once)
    if result.appended_counterargument:
        existing = (factual.counterargument or "").rstrip()
        verifier_note = f"[Verifier: {result.appended_counterargument}]"
        if verifier_note not in existing:
            sep = " " if existing else ""
            updates["counterargument"] = f"{existing}{sep}{verifier_note}"

    # Missing data extension (deduplicated)
    if result.appended_missing_data:
        existing_missing = list(factual.missing_data or [])
        for item in result.appended_missing_data:
            if item not in existing_missing:
                existing_missing.append(item)
        if existing_missing != list(factual.missing_data or []):
            updates["missing_data"] = existing_missing

    # Corrected claim_tiers (tiers downgraded from confirmed/supported to plausible)
    if result.corrected_claim_tiers is not None:
        updates["claim_tiers"] = result.corrected_claim_tiers

    if not updates:
        return factual

    # Rebuild via model_validate to re-run validators (especially confidence_band)
    data = factual.model_dump()
    data.update(updates)
    return FactualVerdict.model_validate(data)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _verdict_rank(v: VerdictLabel) -> int:
    """Higher = better verdict; used to ensure apply_verification only downgrades."""
    return {
        VerdictLabel.SUPPORTED: 4,
        VerdictLabel.LOW_CONFIDENCE: 3,
        VerdictLabel.NEEDS_CLARIFICATION: 2,
        VerdictLabel.INSUFFICIENT_DATA: 1,
        VerdictLabel.OUT_OF_SCOPE: 0,
    }.get(v, 2)


def _has_numeric_claim(text: str) -> bool:
    return any(p.search(text) for p in _NUMERIC_RE)


def _has_causal_language(text: str) -> bool:
    return bool(_CAUSAL_RE.search(text))


def _has_forecast_language(text: str) -> bool:
    return bool(_FORECAST_RE.search(text))
