"""Sprint P tests: claim_map, next_watch, eval scoring infrastructure."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent


def _rule_based(query: str, region_hint: str = "NSW1"):
    """Call _decompose_rules directly — no LLM, no settings required."""
    import os
    os.environ.setdefault("GRIDVERDICT_DEV_NO_AUTH", "true")
    os.environ.setdefault("JWT_SECRET", "test-placeholder-32-chars-minimum!!")
    from app.engines.decomposition import _decompose_rules
    return _decompose_rules(query, region_hint, "test-id")


# ---------------------------------------------------------------------------
# claim_map tests
# ---------------------------------------------------------------------------

class TestClaimMap:
    def test_claim_map_items_are_typed(self):
        from app.core.schema import ClaimMapItem, ClaimType
        item = ClaimMapItem(
            claim_id="test-001",
            claim_type=ClaimType.PRICE_ASSERTION,
            label="dispatch_price",
            tier="confirmed",
            present=True,
            confidence=0.95,
            evidence_ref_ids=["ref-1"],
        )
        assert item.claim_type == ClaimType.PRICE_ASSERTION
        assert item.confidence == pytest.approx(0.95)
        assert item.evidence_ref_ids == ["ref-1"]

    def test_claim_map_item_defaults(self):
        from app.core.schema import ClaimMapItem, ClaimType
        item = ClaimMapItem(
            claim_id="test-002",
            claim_type=ClaimType.OTHER,
            label="unknown",
            tier="unconfirmed",
            present=False,
        )
        assert item.confidence == pytest.approx(0.5)
        assert item.evidence_ref_ids == []
        assert item.note is None

    def test_claim_type_all_eight_values(self):
        from app.core.schema import ClaimType
        # Sprint P original 8 types — Sprint R adds more; use subset check
        expected = {
            "PRICE_ASSERTION", "DEMAND_ASSERTION", "CAUSE_CLAIM",
            "FORECAST_CLAIM", "ACTION_RECOMMENDATION", "PROBABILITY_CLAIM",
            "HISTORICAL_ANALOG", "OTHER",
        }
        assert expected <= {m.name for m in ClaimType}, (
            f"Missing ClaimType names: {expected - {m.name for m in ClaimType}}"
        )

    def test_build_claim_map_from_claim_tiers(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        claim_tiers = [
            {"label": "dispatch_price", "tier": "confirmed", "present": True},
            {"label": "aemo_notice", "tier": "supported", "present": True},
            {"label": "historical_analogs", "tier": "plausible", "present": True},
            {"label": "forecast", "tier": "unconfirmed", "present": False},
        ]
        items = _build_claim_map(claim_tiers, [])
        assert len(items) == 4
        types = {i.claim_type for i in items}
        assert ClaimType.PRICE_ASSERTION in types
        assert ClaimType.CAUSE_CLAIM in types
        assert ClaimType.HISTORICAL_ANALOG in types
        assert ClaimType.FORECAST_CLAIM in types

    def test_build_claim_map_confidence_mapping(self):
        from app.agents.why_builder import _build_claim_map
        from app.core.schema import ClaimType
        claim_tiers = [
            {"label": "dispatch_price", "tier": "confirmed", "present": True},
            {"label": "aemo_notice", "tier": "supported", "present": True},
            {"label": "forecast", "tier": "plausible", "present": True},
            {"label": "weather", "tier": "unconfirmed", "present": False},
        ]
        items = _build_claim_map(claim_tiers, [])
        by_type = {i.claim_type: i for i in items}
        assert by_type[ClaimType.PRICE_ASSERTION].confidence == pytest.approx(0.95)
        # aemo_notice and weather both map to CAUSE_CLAIM; check by tier directly
        aemo = next(i for i in items if "notice" in i.label.lower())
        assert aemo.confidence == pytest.approx(0.75)
        fc = next(i for i in items if i.claim_type == ClaimType.FORECAST_CLAIM)
        assert fc.confidence == pytest.approx(0.50)
        weather = next(i for i in items if "weather" in i.label.lower())
        assert weather.confidence == pytest.approx(0.20)

    def test_build_claim_map_empty_input(self):
        from app.agents.why_builder import _build_claim_map
        items = _build_claim_map([], [])
        assert items == []

    def test_claim_map_in_factual_verdict_schema(self):
        from app.core.schema import FactualVerdict
        fields = FactualVerdict.model_fields
        assert "claim_map" in fields

    def test_factual_verdict_claim_map_defaults_empty(self):
        from app.core.schema import FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand
        from datetime import datetime, timezone
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.HOLD,
            confidence=0.7,
            confidence_band=ConfidenceBand.MEDIUM,
            as_of=datetime.now(timezone.utc),
            why_plain_english="test",
            evidence_refs=[],
            evidence_manifest=[],
            counterargument="none",
            missing_data=[],
        )
        assert v.claim_map == []


# ---------------------------------------------------------------------------
# next_watch tests
# ---------------------------------------------------------------------------

class TestNextWatch:
    def test_next_watch_in_factual_verdict_schema(self):
        from app.core.schema import FactualVerdict
        assert "next_watch" in FactualVerdict.model_fields

    def test_next_watch_defaults_empty(self):
        from app.core.schema import FactualVerdict, VerdictLabel, ActionLabel, ConfidenceBand
        from datetime import datetime, timezone
        v = FactualVerdict(
            verdict=VerdictLabel.LOW_CONFIDENCE,
            action=ActionLabel.HOLD,
            confidence=0.7,
            confidence_band=ConfidenceBand.MEDIUM,
            as_of=datetime.now(timezone.utc),
            why_plain_english="test",
            evidence_refs=[],
            evidence_manifest=[],
            counterargument="none",
            missing_data=[],
        )
        assert v.next_watch == []

    def test_build_next_watch_high_price(self):
        from app.agents.why_builder import _build_next_watch
        from unittest.mock import MagicMock
        c = MagicMock()
        c.price_rrp = 1500.0
        c.headroom_mw = 600.0
        c.regime = "normal"
        c.is_fresh = True
        forecast = MagicMock()
        forecast.direction = "flat"
        forecast.available = False
        forecast.p90 = 200.0
        forecast.p50 = 100.0
        drivers = []
        analogs = MagicMock()
        analogs.count = 0
        analogs.success_rate = 0.0
        weather = MagicMock()
        weather.relevant = False
        weather.tags = []
        items = _build_next_watch(c, forecast, drivers, analogs, weather)
        # $1500 triggers the ≥$1000 branch: "Watch for price cap breach"
        assert any("cap" in w.lower() or "1,500" in w for w in items)

    def test_build_next_watch_negative_price(self):
        from app.agents.why_builder import _build_next_watch
        from unittest.mock import MagicMock
        c = MagicMock()
        c.price_rrp = -50.0
        c.headroom_mw = 2000.0
        c.regime = "normal"
        c.is_fresh = True
        forecast = MagicMock()
        forecast.direction = "flat"
        forecast.available = False
        forecast.p90 = 100.0
        forecast.p50 = 50.0
        drivers = []
        analogs = MagicMock()
        analogs.count = 0
        analogs.success_rate = 0.0
        weather = MagicMock()
        weather.relevant = False
        weather.tags = []
        items = _build_next_watch(c, forecast, drivers, analogs, weather)
        assert any("negative" in w.lower() for w in items)

    def test_build_next_watch_low_headroom(self):
        from app.agents.why_builder import _build_next_watch
        from unittest.mock import MagicMock
        c = MagicMock()
        c.price_rrp = 100.0
        c.headroom_mw = 100.0
        c.regime = "normal"
        c.is_fresh = True
        forecast = MagicMock()
        forecast.direction = "flat"
        forecast.available = False
        forecast.p90 = 150.0
        forecast.p50 = 80.0
        drivers = []
        analogs = MagicMock()
        analogs.count = 0
        analogs.success_rate = 0.0
        weather = MagicMock()
        weather.relevant = False
        weather.tags = []
        items = _build_next_watch(c, forecast, drivers, analogs, weather)
        assert any("headroom" in w.lower() for w in items)

    def test_build_next_watch_normal_returns_fallback(self):
        from app.agents.why_builder import _build_next_watch
        from unittest.mock import MagicMock
        c = MagicMock()
        c.price_rrp = 80.0
        c.headroom_mw = 1500.0
        c.regime = "normal"
        c.is_fresh = True
        forecast = MagicMock()
        forecast.direction = "flat"
        forecast.available = False
        forecast.p90 = 100.0
        forecast.p50 = 60.0
        drivers = []
        analogs = MagicMock()
        analogs.count = 0
        analogs.success_rate = 0.0
        weather = MagicMock()
        weather.relevant = False
        weather.tags = []
        items = _build_next_watch(c, forecast, drivers, analogs, weather)
        assert len(items) >= 1

    def test_why_output_has_next_watch_field(self):
        from app.agents.why_builder import WhyOutput
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(WhyOutput)}
        assert "next_watch" in field_names

    def test_why_output_has_claim_map_field(self):
        from app.agents.why_builder import WhyOutput
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(WhyOutput)}
        assert "claim_map" in field_names


# ---------------------------------------------------------------------------
# Eval scoring infrastructure tests
# ---------------------------------------------------------------------------

class TestEvalScoring:
    """Tests that the eval script loads, runs, and scores correctly."""

    def test_fixture_loads_and_has_200_plus_cases(self):
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        assert len(data["cases"]) >= 200, f"Expected >=200 cases, got {len(data['cases'])}"

    def test_fixture_all_cases_have_required_fields(self):
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        required = {"id", "query", "expected_intent", "min_confidence"}
        for case in data["cases"]:
            missing = required - set(case.keys())
            assert not missing, f"[{case.get('id')}] missing fields: {missing}"

    def test_fixture_intent_labels_valid(self):
        from app.core.schema import IntentLabel
        valid = {i.value for i in IntentLabel}
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        for case in data["cases"]:
            intent = case["expected_intent"]
            assert intent in valid, f"[{case['id']}] unknown intent: {intent}"

    def test_fixture_all_intent_types_represented(self):
        from app.core.schema import IntentLabel
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        found = {c["expected_intent"] for c in data["cases"]}
        for intent in IntentLabel:
            assert intent.value in found, f"Intent {intent.value} not represented in fixture"

    def test_fixture_region_codes_valid(self):
        valid_regions = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        for case in data["cases"]:
            if "expected_regions" in case:
                for r in case["expected_regions"]:
                    assert r in valid_regions, f"[{case['id']}] invalid region: {r}"

    def test_fixture_no_duplicate_ids(self):
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        ids = [c["id"] for c in data["cases"]]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_rule_based_decomposer_intent_accuracy_above_70pct(self):
        """Rule-based decomposer must reach ≥70% intent accuracy on the eval set."""
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        cases = data["cases"]
        correct = 0
        for case in cases:
            result = _rule_based(case["query"], case.get("region_hint", "NSW1"))
            if result.intent.value == case["expected_intent"]:
                correct += 1
        accuracy = correct / len(cases)
        assert accuracy >= 0.70, f"Rule-based intent accuracy {accuracy:.1%} < 70%"

    def test_rule_based_region_overlap_perfect(self):
        """Rule-based must correctly detect at least one expected region in every case."""
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        failures = []
        for case in data["cases"]:
            if "expected_regions" not in case:
                continue
            result = _rule_based(case["query"], case.get("region_hint", "NSW1"))
            actual = set(result.entities.get("regions", []))
            expected = set(case["expected_regions"])
            if not (expected & actual):
                failures.append(case["id"])
        assert not failures, f"Region detection failures: {failures}"

    def test_eval_script_exists_and_is_importable(self):
        script = _REPO / "scripts" / "eval_decomposer.py"
        assert script.exists(), "scripts/eval_decomposer.py not found"

    def test_eval_script_runs_and_exits_clean(self):
        """Smoke-test: eval script runs to completion with exit 0."""
        result = subprocess.run(
            [sys.executable, "scripts/eval_decomposer.py", "--backend", "rule_based"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"eval_decomposer.py exited {result.returncode}\n"
            f"stdout: {result.stdout[-1000:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )
        assert "OVERALL INTENT ACCURACY" in result.stdout
        assert "CONFUSION MATRIX" in result.stdout
        assert "Macro F1" in result.stdout

    def test_eval_report_json_saved(self):
        report_path = _REPO / "reports" / "decomposer_eval_report.json"
        assert report_path.exists(), "Report JSON not found — run eval_decomposer.py first"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert "intent_accuracy" in data
        assert "confusion_matrix" in data
        assert "per_intent" in data
        assert data["intent_accuracy"] >= 0.70

    def test_counterfactual_precision_perfect(self):
        """Counterfactual and comparison both achieve precision 1.0 in rule-based mode."""
        fixture = _REPO / "tests" / "fixtures" / "decomposer_eval_set.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        fp_count = {"counterfactual": 0, "comparison": 0}
        for case in data["cases"]:
            result = _rule_based(case["query"], case.get("region_hint", "NSW1"))
            if result.intent.value in fp_count and result.intent.value != case["expected_intent"]:
                fp_count[result.intent.value] += 1
        assert fp_count["counterfactual"] == 0, "counterfactual has false positives"
        assert fp_count["comparison"] == 0, "comparison has false positives"


# ---------------------------------------------------------------------------
# QueryDecomposition new-field tests
# ---------------------------------------------------------------------------

class TestQueryDecompositionNewFields:
    def test_requires_live_market_default_false(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.LOOKUP,
            entities={},
            time_range={"type": "current"},
            requires_why=False,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
        )
        assert q.requires_live_market is False

    def test_requires_incident_timeline_default_false(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.LOOKUP,
            entities={},
            time_range={"type": "current"},
            requires_why=False,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
        )
        assert q.requires_incident_timeline is False

    def test_requires_bess_context_default_false(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.LOOKUP,
            entities={},
            time_range={"type": "current"},
            requires_why=False,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
        )
        assert q.requires_bess_context is False

    def test_causal_targets_default_empty(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.LOOKUP,
            entities={},
            time_range={"type": "current"},
            requires_why=False,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
        )
        assert q.causal_targets == []

    def test_spike_thresholds_default_empty(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.LOOKUP,
            entities={},
            time_range={"type": "current"},
            requires_why=False,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
        )
        assert q.spike_thresholds == []

    def test_new_fields_accept_values(self):
        from app.core.schema import QueryDecomposition, IntentLabel
        q = QueryDecomposition(
            query_id="test",
            raw_query="test",
            intent=IntentLabel.ACTION_RECOMMENDATION,
            entities={},
            time_range={"type": "current"},
            requires_why=True,
            requires_history=False,
            requires_forecast=False,
            requires_backtest=False,
            requires_portfolio=False,
            confidence=0.8,
            ambiguities=[],
            requires_live_market=True,
            requires_bess_context=True,
            requires_incident_timeline=False,
            causal_targets=["demand", "constraint"],
            spike_thresholds=[300.0, 1000.0],
            action_context="battery dispatch",
            requested_output="recommendation",
        )
        assert q.requires_live_market is True
        assert q.requires_bess_context is True
        assert q.causal_targets == ["demand", "constraint"]
        assert q.spike_thresholds == [300.0, 1000.0]
        assert q.action_context == "battery dispatch"
