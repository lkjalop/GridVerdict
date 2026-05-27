"""LLM decomposer quality reporting.

This module turns the existing fixed decomposer fixture set into a CI-friendly
JSON report. It can run against rule-based or live LLM backends, but the report
format stays stable so qwen/llama/mistral comparisons are reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.core.schema import QueryDecomposition
from app.engines.decomposition import decompose


DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "decomposer_eval_set.json"


async def run_decomposer_eval(
    fixture_path: str | Path = DEFAULT_FIXTURE,
    decompose_fn: Callable[[str, str, str | None], Awaitable[QueryDecomposition]] = decompose,
    model_name: str = "configured",
    backend: str = "configured",
    min_intent_accuracy: float = 0.80,
    require_all_oos: bool = True,
) -> dict[str, Any]:
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    cases = fixture["cases"]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        t0 = time.perf_counter()
        result = await decompose_fn(case["query"], case.get("region_hint", "NSW1"), case["id"])
        rows.append({
            "id": case["id"],
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "actual_intent": result.intent.value,
            "expected_regions": case.get("expected_regions", []),
            "actual_regions": result.entities.get("regions", []),
            "confidence": result.confidence,
            "schema_valid": isinstance(result, QueryDecomposition),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    return evaluate_rows(
        rows,
        model_name=model_name,
        backend=backend,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        min_intent_accuracy=min_intent_accuracy,
        require_all_oos=require_all_oos,
    )


def evaluate_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    backend: str,
    elapsed_ms: float,
    min_intent_accuracy: float = 0.80,
    require_all_oos: bool = True,
) -> dict[str, Any]:
    total = len(rows)
    intent_correct = [r for r in rows if r["actual_intent"] == r["expected_intent"]]
    oos_rows = [r for r in rows if r["expected_intent"] == "out_of_scope"]
    oos_correct = [r for r in oos_rows if r["actual_intent"] == "out_of_scope"]
    schema_valid = [r for r in rows if r.get("schema_valid")]
    latencies = [r["latency_ms"] for r in rows]
    intent_accuracy = len(intent_correct) / max(total, 1)
    oos_accuracy = len(oos_correct) / max(len(oos_rows), 1)
    passed = intent_accuracy >= min_intent_accuracy and (
        not require_all_oos or len(oos_correct) == len(oos_rows)
    )
    return {
        "model": model_name,
        "backend": backend,
        "total_cases": total,
        "intent_accuracy": round(intent_accuracy, 4),
        "schema_validation_rate": round(len(schema_valid) / max(total, 1), 4),
        "out_of_scope_accuracy": round(oos_accuracy, 4),
        "mean_latency_ms": round(sum(latencies) / max(total, 1), 1),
        "elapsed_ms": elapsed_ms,
        "thresholds": {
            "min_intent_accuracy": min_intent_accuracy,
            "require_all_oos": require_all_oos,
        },
        "passed": passed,
        "failures": [
            r for r in rows
            if r["actual_intent"] != r["expected_intent"]
            or (r["expected_intent"] == "out_of_scope" and r["actual_intent"] != "out_of_scope")
        ],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GridVerdict decomposer quality eval")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--model", default="configured")
    parser.add_argument("--backend", default="configured")
    parser.add_argument("--out", default="")
    parser.add_argument("--min-intent-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    report = asyncio.run(run_decomposer_eval(
        fixture_path=args.fixture,
        model_name=args.model,
        backend=args.backend,
        min_intent_accuracy=args.min_intent_accuracy,
    ))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
