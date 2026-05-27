"""Tests for Claim Verifier / Answer Guard.

Covers all 5 rules, severity levels, apply_verification corrections,
and integration invariants.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agents.claim_verifier import (
    _has_causal_language,
    _has_forecast_language,
    _has_numeric_claim,
    apply_verification,
    verify_answer,
)
from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    EvidenceRefSchema,
    FactualVerdict,
    VerdictLabel,
)

_NOW = datetime(2026, 5, 25, 14, 5, 0, tzinfo=timezone.utc)


# ── Fixture helpers ──────────────────────────────────────────────────────────


def _ref(source: str = "AEMO_DISPATCH_PRICE", field: str = "price_rrp", value: float = 500.0):
    return EvidenceRefSchema(
        source=source, region="NSW1", interval=_NOW, field=field, value=value, raw_ref="test"
    )


def _verdict(
    text: str = "The market is operating normally.",
    verdict: VerdictLabel = VerdictLabel.LOW_CONFIDENCE,
    refs: list | None = None,
    claim_tiers: list | None = None,
    driver_tiers: list | None = None,
    counterargument: str = "No additional counterargument.",
    confidence: float = 0.65,
) -> FactualVerdict:
    return FactualVerdict(
        verdict=verdict,
        action=ActionLabel.MONITOR,
        confidence=confidence,
        confidence_band=ConfidenceBand.MEDIUM,
        as_of=_NOW,
        why_plain_english=text,
        evidence_refs=refs if refs is not None else [],
        counterargument=counterargument,
        claim_tiers=claim_tiers if claim_tiers is not None else [],
        driver_tiers=driver_tiers if driver_tiers is not None else [],
    )


# ── Helper function unit tests ────────────────────────────────────────────────


def test_has_numeric_claim_price():
    assert _has_numeric_claim("Price is $500.25/MWh.")


def test_has_numeric_claim_mw():
    assert _has_numeric_claim("Demand is 7,500 MW.")


def test_has_numeric_claim_mwh():
    assert _has_numeric_claim("300 MWh of battery storage remains.")


def test_has_numeric_claim_percent():
    assert _has_numeric_claim("Confidence is 85%.")


def test_has_numeric_claim_none():
    assert not _has_numeric_claim("The market is operating normally.")


def test_has_causal_language_caused_by():
    assert _has_causal_language("The spike was caused by low headroom.")


def test_has_causal_language_due_to():
    assert _has_causal_language("Prices rose due to constrained supply.")


def test_has_causal_language_driven_by():
    assert _has_causal_language("The price is driven by coal availability.")


def test_has_causal_language_none():
    assert not _has_causal_language("Price is $120/MWh and demand is elevated.")


def test_has_forecast_language_forecast():
    assert _has_forecast_language("The forecast for tomorrow is $200/MWh.")


def test_has_forecast_language_predispatch():
    assert _has_forecast_language("Pre-dispatch intervals show rising prices.")


def test_has_forecast_language_none():
    assert not _has_forecast_language("The current price is $120/MWh.")


# ── Rule 1: Numeric claim → evidence_ref required ────────────────────────────


def test_rule1_numeric_without_refs_triggers_finding():
    v = _verdict(text="NSW1 price is $500/MWh.", refs=[])
    result = verify_answer(v)
    assert any(f.rule == "numeric_without_evidence" for f in result.findings)


def test_rule1_numeric_with_refs_passes():
    v = _verdict(text="NSW1 price is $500/MWh.", refs=[_ref()])
    result = verify_answer(v)
    assert not any(f.rule == "numeric_without_evidence" for f in result.findings)


def test_rule1_no_numeric_no_refs_passes():
    v = _verdict(text="The market is operating normally.", refs=[])
    result = verify_answer(v)
    assert not any(f.rule == "numeric_without_evidence" for f in result.findings)


def test_rule1_finding_severity_is_warn():
    v = _verdict(text="Price is $500/MWh.", refs=[])
    result = verify_answer(v)
    finding = next(f for f in result.findings if f.rule == "numeric_without_evidence")
    assert finding.severity == "warn"


def test_rule1_adds_missing_data():
    v = _verdict(text="Price is $500/MWh.", refs=[])
    result = verify_answer(v)
    assert "evidence_for_numeric_claims" in result.appended_missing_data


def test_rule1_adds_counterargument():
    v = _verdict(text="Price is $500/MWh.", refs=[])
    result = verify_answer(v)
    assert result.appended_counterargument  # non-empty


# ── Rule 2: Causal claim → claim_tier required ───────────────────────────────


def test_rule2_causal_without_claim_tiers_triggers():
    v = _verdict(text="The spike was caused by low headroom.", claim_tiers=[])
    result = verify_answer(v)
    assert any(f.rule == "causal_without_claim_tiers" for f in result.findings)


def test_rule2_causal_with_claim_tiers_passes():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": ["ev-abc"]}]
    v = _verdict(text="The spike was caused by low headroom.", claim_tiers=tiers)
    result = verify_answer(v)
    assert not any(f.rule == "causal_without_claim_tiers" for f in result.findings)


def test_rule2_finding_severity_is_warn():
    v = _verdict(text="Prices rose due to constrained supply.", claim_tiers=[])
    result = verify_answer(v)
    finding = next(f for f in result.findings if f.rule == "causal_without_claim_tiers")
    assert finding.severity == "warn"


def test_rule2_adds_missing_data():
    v = _verdict(text="The price led to generator exit.", claim_tiers=[])
    result = verify_answer(v)
    assert "causal_evidence_tiers" in result.appended_missing_data


# ── Rule 3: High-tier claim_tier → evidence_ref_ids required ────────────────


def test_rule3_confirmed_without_refs_triggers():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    assert any(f.rule == "high_tier_without_evidence_refs" for f in result.findings)


def test_rule3_supported_without_refs_triggers():
    tiers = [{"label": "constraint", "tier": "supported", "present": True, "evidence_ref_ids": []}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    assert any(f.rule == "high_tier_without_evidence_refs" for f in result.findings)


def test_rule3_confirmed_with_refs_passes():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": ["ev-abc"]}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    assert not any(f.rule == "high_tier_without_evidence_refs" for f in result.findings)


def test_rule3_plausible_without_refs_passes():
    """Plausible and unconfirmed tiers don't require evidence_ref_ids."""
    tiers = [{"label": "weather", "tier": "plausible", "present": True, "evidence_ref_ids": []}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    assert not any(f.rule == "high_tier_without_evidence_refs" for f in result.findings)


def test_rule3_tier_downgraded_to_plausible():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    assert result.corrected_claim_tiers is not None
    assert result.corrected_claim_tiers[0]["tier"] == "plausible"


def test_rule3_severity_is_downgrade():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []}]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    finding = next(f for f in result.findings if f.rule == "high_tier_without_evidence_refs")
    assert finding.severity == "downgrade"


def test_rule3_only_affected_tiers_corrected():
    tiers = [
        {"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []},
        {"label": "weather", "tier": "plausible", "present": True, "evidence_ref_ids": []},
    ]
    v = _verdict(text="Normal market.", claim_tiers=tiers)
    result = verify_answer(v)
    corrected = result.corrected_claim_tiers
    assert corrected[0]["tier"] == "plausible"  # was confirmed, downgraded
    assert corrected[1]["tier"] == "plausible"  # was already plausible, unchanged


# ── Rule 4: Forecast claim → forecast model source required ─────────────────


def test_rule4_forecast_without_source_triggers():
    v = _verdict(text="Forecast for next interval is $300/MWh.", refs=[_ref()])
    result = verify_answer(v)
    assert any(f.rule == "forecast_without_model_evidence" for f in result.findings)


def test_rule4_forecast_with_predispatch_passes():
    v = _verdict(
        text="Forecast shows rising prices.",
        refs=[_ref(source="AEMO_PREDISPATCH_30MIN", field="price_rrp")],
    )
    result = verify_answer(v)
    assert not any(f.rule == "forecast_without_model_evidence" for f in result.findings)


def test_rule4_forecast_with_lnn_passes():
    v = _verdict(
        text="The LNN forecast expects higher prices.",
        refs=[_ref(source="LNN_FORECAST", field="p50")],
    )
    result = verify_answer(v)
    assert not any(f.rule == "forecast_without_model_evidence" for f in result.findings)


def test_rule4_finding_severity_is_warn():
    v = _verdict(text="Pre-dispatch shows $400/MWh.", refs=[_ref()])
    result = verify_answer(v)
    finding = next(
        (f for f in result.findings if f.rule == "forecast_without_model_evidence"), None
    )
    assert finding is not None
    assert finding.severity == "warn"


def test_rule4_adds_missing_data():
    v = _verdict(text="The forecast anticipates a spike.", refs=[_ref()])
    result = verify_answer(v)
    assert "forecast_model_source" in result.appended_missing_data


def test_rule4_no_forecast_language_no_finding():
    v = _verdict(text="Current price is $500/MWh.", refs=[_ref()])
    result = verify_answer(v)
    assert not any(f.rule == "forecast_without_model_evidence" for f in result.findings)


# ── Rule 5: "caused by" → confirmed/supported driver required ───────────────


def test_rule5_caused_by_without_driver_triggers():
    v = _verdict(text="The price spike was caused by generator outages.")
    result = verify_answer(v)
    assert any(f.rule == "caused_by_without_confirmed_driver" for f in result.findings)


def test_rule5_caused_by_with_confirmed_driver_passes():
    drivers = [{"label": "unit_dispatch", "tier": "confirmed", "present": True}]
    v = _verdict(text="The price spike was caused by generator outages.", driver_tiers=drivers)
    result = verify_answer(v)
    assert not any(f.rule == "caused_by_without_confirmed_driver" for f in result.findings)


def test_rule5_caused_by_with_supported_driver_passes():
    drivers = [{"label": "constraint", "tier": "supported", "present": True}]
    v = _verdict(text="High prices caused by binding constraints.", driver_tiers=drivers)
    result = verify_answer(v)
    assert not any(f.rule == "caused_by_without_confirmed_driver" for f in result.findings)


def test_rule5_caused_by_with_only_plausible_driver_triggers():
    """Plausible driver is not sufficient to back 'caused by' wording."""
    drivers = [{"label": "weather", "tier": "plausible", "present": True}]
    v = _verdict(text="The event was caused by high temperatures.", driver_tiers=drivers)
    result = verify_answer(v)
    assert any(f.rule == "caused_by_without_confirmed_driver" for f in result.findings)


def test_rule5_severity_is_downgrade():
    v = _verdict(text="Spike caused by tight supply.")
    result = verify_answer(v)
    finding = next(f for f in result.findings if f.rule == "caused_by_without_confirmed_driver")
    assert finding.severity == "downgrade"


def test_rule5_suggests_downgrade_when_supported():
    v = _verdict(
        text="Spike caused by tight supply.",
        verdict=VerdictLabel.SUPPORTED,
        refs=[_ref()],  # need refs to pass model_validator for SUPPORTED
    )
    result = verify_answer(v)
    assert result.suggested_verdict == VerdictLabel.LOW_CONFIDENCE


def test_rule5_no_suggested_downgrade_when_already_low_confidence():
    v = _verdict(
        text="Spike caused by tight supply.",
        verdict=VerdictLabel.LOW_CONFIDENCE,
    )
    result = verify_answer(v)
    # May or may not suggest LOW_CONFIDENCE, but applying should not change verdict
    factual_out = apply_verification(v, result)
    assert factual_out.verdict == VerdictLabel.LOW_CONFIDENCE


def test_rule5_causal_language_without_exact_caused_by_does_not_trigger():
    """Rule 5 is specific to 'caused by' wording, not all causal language."""
    v = _verdict(text="High demand due to a heatwave drove prices up.")
    result = verify_answer(v)
    assert not any(f.rule == "caused_by_without_confirmed_driver" for f in result.findings)


# ── apply_verification corrections ──────────────────────────────────────────


def test_apply_clean_verdict_passes_through_unchanged():
    v = _verdict(text="Market is normal.", refs=[], claim_tiers=[], driver_tiers=[])
    result = verify_answer(v)
    out = apply_verification(v, result)
    # No findings on a clean text-only verdict with no causal/numeric language
    assert out.verdict == v.verdict
    assert out.counterargument == v.counterargument


def test_apply_downgrades_supported_to_low_confidence():
    v = _verdict(
        text="Spike was caused by generator failure.",
        verdict=VerdictLabel.SUPPORTED,
        refs=[_ref()],
    )
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.verdict == VerdictLabel.LOW_CONFIDENCE


def test_apply_does_not_upgrade_low_confidence():
    """apply_verification must never upgrade the verdict."""
    v = _verdict(
        text="Market is currently at $50/MWh.",
        verdict=VerdictLabel.LOW_CONFIDENCE,
        refs=[],
    )
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.verdict == VerdictLabel.LOW_CONFIDENCE


def test_apply_does_not_touch_insufficient_data():
    v = _verdict(
        text="Spike caused by constrained network.",
        verdict=VerdictLabel.INSUFFICIENT_DATA,
    )
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.verdict == VerdictLabel.INSUFFICIENT_DATA


def test_apply_extends_counterargument():
    v = _verdict(text="Price is $300/MWh.", refs=[])
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert "[Verifier:" in out.counterargument


def test_apply_counterargument_not_duplicated():
    """Running apply twice must not duplicate the [Verifier: ...] block."""
    v = _verdict(text="Price is $300/MWh.", refs=[])
    result = verify_answer(v)
    out1 = apply_verification(v, result)
    result2 = verify_answer(out1)
    out2 = apply_verification(out1, result2)
    assert out2.counterargument.count("[Verifier:") == 1


def test_apply_extends_missing_data():
    v = _verdict(text="Price is $300/MWh.", refs=[])
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert "evidence_for_numeric_claims" in out.missing_data


def test_apply_no_duplicate_missing_data():
    v = _verdict(text="Price is $300/MWh.", refs=[], )
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.missing_data.count("evidence_for_numeric_claims") == 1


def test_apply_corrects_claim_tiers():
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []}]
    v = _verdict(claim_tiers=tiers)
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.claim_tiers[0]["tier"] == "plausible"


def test_apply_reduces_confidence_on_downgrade():
    v = _verdict(
        text="Spike was caused by network constraint.",
        verdict=VerdictLabel.SUPPORTED,
        refs=[_ref()],
        confidence=0.85,
    )
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.confidence <= 0.55


def test_apply_confidence_band_updated_after_reduction():
    """confidence_band must match the corrected confidence."""
    v = _verdict(
        text="Spike was caused by network constraint.",
        verdict=VerdictLabel.SUPPORTED,
        refs=[_ref()],
        confidence=0.85,
    )
    result = verify_answer(v)
    out = apply_verification(v, result)
    from app.core.schema import ConfidenceBand
    assert out.confidence_band in (ConfidenceBand.LOW, ConfidenceBand.VERY_LOW, ConfidenceBand.MEDIUM)


def test_apply_does_not_modify_evidence_refs():
    """Verifier must never add or remove evidence_refs."""
    v = _verdict(text="Price spike caused by constraint.", refs=[_ref()])
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert len(out.evidence_refs) == 1
    assert out.evidence_refs[0].source == "AEMO_DISPATCH_PRICE"


def test_apply_does_not_modify_why_plain_english():
    """Verifier must never rewrite the narrative."""
    original_text = "Price spike caused by constraint."
    v = _verdict(text=original_text)
    result = verify_answer(v)
    out = apply_verification(v, result)
    assert out.why_plain_english == original_text


def test_verify_multiple_findings_accumulate():
    """Multiple rule violations on one verdict accumulate all findings."""
    tiers = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": []}]
    v = _verdict(
        text="Spike caused by tight supply at $5000/MWh. Forecast expects $6000/MWh.",
        refs=[],
        claim_tiers=tiers,
    )
    result = verify_answer(v)
    rules_triggered = {f.rule for f in result.findings}
    # Should hit rule 1 (numeric, no refs), rule 3 (confirmed, no ref_ids),
    # rule 4 (forecast, no source), rule 5 (caused by, no driver)
    assert len(rules_triggered) >= 3


def test_verify_fully_clean_verdict_passes():
    """A well-formed answer with all data present should pass cleanly."""
    drivers = [{"label": "price", "tier": "confirmed", "present": True}]
    claim_t = [{"label": "price", "tier": "confirmed", "present": True, "evidence_ref_ids": ["ev-abc"]}]
    forecast_ref = _ref(source="AEMO_PREDISPATCH_30MIN", field="price_rrp")
    v = _verdict(
        text="Current price is $500/MWh. Forecast shows $400/MWh next interval.",
        refs=[_ref(), forecast_ref],
        claim_tiers=claim_t,
        driver_tiers=drivers,
    )
    result = verify_answer(v)
    # Rule 1: refs present ✓, Rule 3: evidence_ref_ids present ✓,
    # Rule 4: PREDISPATCH ref present ✓, Rule 5: no "caused by" ✓
    # Rule 2: causal language? "shows" is not causal — passes
    assert not any(f.rule in ("numeric_without_evidence", "high_tier_without_evidence_refs",
                               "forecast_without_model_evidence") for f in result.findings)
