from __future__ import annotations

from app.engines.llm_eval import evaluate_rows


def test_llm_eval_report_passes_when_thresholds_met():
    rows = [
        {
            "id": "a",
            "expected_intent": "lookup",
            "actual_intent": "lookup",
            "schema_valid": True,
            "latency_ms": 10.0,
        },
        {
            "id": "b",
            "expected_intent": "out_of_scope",
            "actual_intent": "out_of_scope",
            "schema_valid": True,
            "latency_ms": 20.0,
        },
    ]
    report = evaluate_rows(rows, "qwen3:14b", "ollama", 30.0, min_intent_accuracy=0.8)
    assert report["passed"]
    assert report["intent_accuracy"] == 1.0
    assert report["out_of_scope_accuracy"] == 1.0


def test_llm_eval_report_fails_on_oos_miss_even_if_accuracy_is_high():
    rows = [
        {
            "id": "a",
            "expected_intent": "lookup",
            "actual_intent": "lookup",
            "schema_valid": True,
            "latency_ms": 10.0,
        },
        {
            "id": "b",
            "expected_intent": "out_of_scope",
            "actual_intent": "lookup",
            "schema_valid": True,
            "latency_ms": 20.0,
        },
    ]
    report = evaluate_rows(rows, "qwen3:14b", "ollama", 30.0, min_intent_accuracy=0.5)
    assert not report["passed"]
    assert report["failures"][0]["id"] == "b"
