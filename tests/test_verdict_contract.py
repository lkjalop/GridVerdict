"""Tests for the FactualVerdict Pydantic contract.

Verifies:
- SUPPORTED verdict requires ≥1 evidence_ref (model_validator)
- confidence_band is auto-derived from confidence (model_validator)
- All non-SUPPORTED verdicts accept empty evidence_refs
- EvidenceRefSchema auto-generates IDs
- Field-level constraints (confidence 0..1)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.schema import (
    ActionLabel,
    ConfidenceBand,
    EvidenceRefSchema,
    FactualVerdict,
    VerdictLabel,
)

_NOW = datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc)


def _base(**overrides) -> dict:
    """Return a minimal valid FactualVerdict payload."""
    d = {
        "verdict": VerdictLabel.LOW_CONFIDENCE,
        "action": ActionLabel.HOLD,
        "confidence": 0.65,
        "confidence_band": ConfidenceBand.MEDIUM,
        "as_of": _NOW,
        "why_plain_english": "Data is stale — confidence is low.",
        "evidence_refs": [],
        "counterargument": "Conditions could change rapidly.",
        "missing_data": [],
        "disclaimer": (
            "Simulation and decision-support only. Not financial advice. "
            "Not a market participant. Uses public AEMO/NEMWEB data."
        ),
    }
    d.update(overrides)
    return d


def _evidence_ref(value: float = 347.5, region: str = "NSW1") -> EvidenceRefSchema:
    return EvidenceRefSchema(
        source="AEMO_DISPATCH_PRICE",
        region=region,
        interval=_NOW,
        field="price_rrp",
        value=value,
        raw_ref="dispatch_test_ref",
    )


# ── SUPPORTED verdict ─────────────────────────────────────────────────

class TestSupportedVerdict:
    def test_supported_with_evidence_is_valid(self):
        v = FactualVerdict(**_base(
            verdict=VerdictLabel.SUPPORTED,
            action=ActionLabel.DISPATCH_NOW,
            confidence=0.84,
            evidence_refs=[_evidence_ref()],
        ))
        assert v.verdict == VerdictLabel.SUPPORTED
        assert len(v.evidence_refs) == 1

    def test_supported_without_evidence_raises(self):
        with pytest.raises(ValidationError) as exc:
            FactualVerdict(**_base(
                verdict=VerdictLabel.SUPPORTED,
                evidence_refs=[],
            ))
        assert "evidence_ref" in str(exc.value).lower()

    def test_supported_multiple_evidence_refs(self):
        v = FactualVerdict(**_base(
            verdict=VerdictLabel.SUPPORTED,
            action=ActionLabel.DISPATCH_NOW,
            confidence=0.84,
            evidence_refs=[_evidence_ref(347.5), _evidence_ref(8420.0)],
        ))
        assert len(v.evidence_refs) == 2


# ── Non-SUPPORTED verdicts accept empty evidence ───────────────────────

class TestNonSupportedVerdicts:
    @pytest.mark.parametrize("verdict,action", [
        (VerdictLabel.LOW_CONFIDENCE, ActionLabel.HOLD),
        (VerdictLabel.INSUFFICIENT_DATA, ActionLabel.MONITOR),
        (VerdictLabel.OUT_OF_SCOPE, ActionLabel.REFUSE),
        (VerdictLabel.NEEDS_CLARIFICATION, ActionLabel.ASK_CLARIFYING_QUESTION),
    ])
    def test_non_supported_empty_evidence_valid(self, verdict, action):
        v = FactualVerdict(**_base(verdict=verdict, action=action, evidence_refs=[]))
        assert v.verdict == verdict

    def test_insufficient_data_can_have_evidence(self):
        v = FactualVerdict(**_base(
            verdict=VerdictLabel.INSUFFICIENT_DATA,
            action=ActionLabel.MONITOR,
            evidence_refs=[_evidence_ref()],
        ))
        assert v.verdict == VerdictLabel.INSUFFICIENT_DATA


# ── Confidence band auto-derivation ───────────────────────────────────

class TestConfidenceBand:
    @pytest.mark.parametrize("confidence,expected_band", [
        (1.00, ConfidenceBand.HIGH),
        (0.80, ConfidenceBand.HIGH),
        (0.85, ConfidenceBand.HIGH),
        (0.79, ConfidenceBand.MEDIUM),
        (0.60, ConfidenceBand.MEDIUM),
        (0.70, ConfidenceBand.MEDIUM),
        (0.59, ConfidenceBand.LOW),
        (0.40, ConfidenceBand.LOW),
        (0.50, ConfidenceBand.LOW),
        (0.39, ConfidenceBand.VERY_LOW),
        (0.00, ConfidenceBand.VERY_LOW),
        (0.20, ConfidenceBand.VERY_LOW),
    ])
    def test_band_derived_from_confidence(self, confidence, expected_band):
        v = FactualVerdict(**_base(confidence=confidence))
        assert v.confidence_band == expected_band, (
            f"confidence={confidence}: expected {expected_band}, got {v.confidence_band}"
        )

    def test_provided_band_overridden_by_validator(self):
        # Even if caller sets the wrong band, the validator fixes it
        v = FactualVerdict(**_base(confidence=0.90, confidence_band=ConfidenceBand.VERY_LOW))
        assert v.confidence_band == ConfidenceBand.HIGH


# ── EvidenceRefSchema ──────────────────────────────────────────────────

class TestEvidenceRefSchema:
    def test_id_auto_generated(self):
        ev = _evidence_ref()
        assert ev.id.startswith("ev-")
        assert len(ev.id) == 11  # "ev-" + 8 hex chars

    def test_two_refs_have_unique_ids(self):
        ids = {_evidence_ref().id for _ in range(20)}
        assert len(ids) == 20

    def test_explicit_id_respected(self):
        ev = EvidenceRefSchema(
            id="ev-custom01",
            source="AEMO_DISPATCH_PRICE",
            region="VIC1",
            interval=_NOW,
            field="demand_mw",
            value=7500.0,
            raw_ref="ref-x",
        )
        assert ev.id == "ev-custom01"

    def test_region_optional(self):
        ev = EvidenceRefSchema(
            source="AEMO_NOTICE",
            interval=_NOW,
            field="notice_type",
            value=1.0,
            raw_ref="ref-y",
        )
        assert ev.region is None


# ── Disclaimer and counterargument ────────────────────────────────────

class TestDisclaimerAndCounterargument:
    def test_default_disclaimer_contains_not_financial_advice(self):
        v = FactualVerdict(**_base())
        assert "Not financial advice" in v.disclaimer

    def test_evidence_manifest_and_known_missing_defaults(self):
        v = FactualVerdict(**_base())
        assert v.evidence_manifest == []
        assert v.known_missing_before_action == []

    def test_custom_disclaimer_accepted(self):
        v = FactualVerdict(**_base(disclaimer="Custom disclaimer text."))
        assert v.disclaimer == "Custom disclaimer text."

    def test_counterargument_required_field(self):
        with pytest.raises(ValidationError):
            FactualVerdict(**{k: v for k, v in _base().items() if k != "counterargument"})


# ── Field constraints ──────────────────────────────────────────────────

class TestFieldConstraints:
    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            FactualVerdict(**_base(confidence=1.01))

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FactualVerdict(**_base(confidence=-0.01))

    def test_confidence_zero_valid(self):
        v = FactualVerdict(**_base(confidence=0.0))
        assert v.confidence == 0.0
        assert v.confidence_band == ConfidenceBand.VERY_LOW

    def test_confidence_one_valid(self):
        v = FactualVerdict(**_base(confidence=1.0))
        assert v.confidence == 1.0
        assert v.confidence_band == ConfidenceBand.HIGH


# ── Fixture-driven contract check ─────────────────────────────────────

class TestFixtureDrivenContract:
    def test_supported_valid_fixture_deserialises(self, load_fixture):
        data = load_fixture("sample_answers/supported_valid.json")
        v = FactualVerdict.model_validate(data)
        assert v.verdict == VerdictLabel.SUPPORTED
        assert len(v.evidence_refs) >= 1
        assert v.trace_id is not None

    def test_supported_no_evidence_fixture_raises(self, load_fixture):
        data = load_fixture("sample_answers/supported_no_evidence.json")
        with pytest.raises(ValidationError):
            FactualVerdict.model_validate(data)
