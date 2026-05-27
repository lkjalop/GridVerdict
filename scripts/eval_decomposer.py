#!/usr/bin/env python3
"""Evaluate the query decomposer against the fixture eval set.

Usage
-----
    python scripts/eval_decomposer.py [options]

Options
-------
    --fixture PATH      Fixture JSON file (default: tests/fixtures/decomposer_eval_set.json)
    --output PATH       Write JSON report here (default: reports/decomposer_eval_report.json)
    --backend NAME      rule_based (default), ollama, claude
    --intent-only       Skip field-level scoring, print only the confusion matrix
    --verbose           Print per-case failures

Outputs
-------
    - Per-intent precision / recall / F1
    - 8×8 confusion matrix
    - Per-field accuracy table (intent, region, requires_forecast, requires_history, requires_why, confidence)
    - Overall accuracy summary
    - JSON report saved to --output

Field scoring notes
-------------------
    Fields marked requires_live_market / requires_bess_context / requires_incident_timeline
    are LLM-only fields: not set by the rule-based decomposer. They are skipped in
    rule_based mode and only scored when --backend ollama or --backend claude is used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow running without a fully configured .env (rule-based decomposer needs no LLM).
# Must be set before any app import that triggers config.settings.get_settings().
os.environ.setdefault("GRIDVERDICT_DEV_NO_AUTH", "true")
os.environ.setdefault("JWT_SECRET", "eval-script-placeholder-not-used-in-rule-based-mode")

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.schema import IntentLabel

ALL_INTENTS: list[str] = [i.value for i in IntentLabel]

# Fields scored in rule-based mode
RULE_FIELDS = ["requires_forecast", "requires_history", "requires_why"]
# Fields only scored when an LLM backend is used
LLM_ONLY_FIELDS = ["requires_live_market", "requires_bess_context", "requires_incident_timeline"]


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------

def _run_rule_based(query: str, region_hint: str, case_id: str):
    from app.engines.decomposition import _decompose_rules
    return _decompose_rules(query, region_hint, case_id)


async def _run_llm(query: str, region_hint: str, case_id: str, backend: str):
    from app.engines.decomposition import _decompose_ollama, _decompose_claude
    if backend == "ollama":
        return await _decompose_ollama(query, region_hint, case_id)
    return await _decompose_claude(query, region_hint, case_id)


def run_decomposer(case: dict, backend: str) -> Any:
    query = case["query"]
    region_hint = case.get("region_hint", "NSW1")
    case_id = case["id"]
    if backend == "rule_based":
        return _run_rule_based(query, region_hint, case_id)
    return asyncio.run(_run_llm(query, region_hint, case_id, backend))


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(case: dict, actual, backend: str) -> dict:
    result: dict[str, Any] = {
        "id": case["id"],
        "query": case["query"],
        "expected_intent": case["expected_intent"],
        "actual_intent": actual.intent.value,
        "intent_correct": actual.intent.value == case["expected_intent"],
        "actual_confidence": actual.confidence,
    }

    # Region scoring (overlap: at least one expected region must be in actual)
    if "expected_regions" in case:
        expected = set(case["expected_regions"])
        actual_regions = set(actual.entities.get("regions", []))
        result["regions_overlap"] = bool(expected & actual_regions)
        result["regions_exact"] = expected == actual_regions
        result["expected_regions"] = sorted(expected)
        result["actual_regions"] = sorted(actual_regions)

    # Confidence band
    min_conf = case.get("min_confidence", 0.0)
    max_conf = case.get("max_confidence", 1.0)
    result["confidence_in_range"] = min_conf <= actual.confidence <= max_conf

    # Rule-based boolean flags
    for field in RULE_FIELDS:
        if field in case:
            expected_val = bool(case[field])
            actual_val = bool(getattr(actual, field, False))
            result[f"{field}_correct"] = expected_val == actual_val
            result[f"{field}_expected"] = expected_val
            result[f"{field}_actual"] = actual_val

    # LLM-only flags (skip in rule-based mode)
    if backend != "rule_based":
        for field in LLM_ONLY_FIELDS:
            if field in case:
                expected_val = bool(case[field])
                actual_val = bool(getattr(actual, field, False))
                result[f"{field}_correct"] = expected_val == actual_val
                result[f"{field}_expected"] = expected_val
                result[f"{field}_actual"] = actual_val

    return result


# ---------------------------------------------------------------------------
# Confusion matrix + per-intent stats
# ---------------------------------------------------------------------------

def build_confusion_matrix(results: list[dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {i: {j: 0 for j in ALL_INTENTS} for i in ALL_INTENTS}
    for r in results:
        exp = r["expected_intent"]
        act = r["actual_intent"]
        if exp in matrix and act in matrix:
            matrix[exp][act] += 1
    return matrix


def per_intent_stats(results: list[dict]) -> dict[str, dict]:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for r in results:
        exp = r["expected_intent"]
        act = r["actual_intent"]
        support[exp] += 1
        if exp == act:
            tp[exp] += 1
        else:
            fn[exp] += 1
            fp[act] += 1

    stats = {}
    for intent in ALL_INTENTS:
        p_denom = tp[intent] + fp[intent]
        r_denom = tp[intent] + fn[intent]
        precision = tp[intent] / p_denom if p_denom else 0.0
        recall = tp[intent] / r_denom if r_denom else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats[intent] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp[intent],
            "fp": fp[intent],
            "fn": fn[intent],
            "support": support[intent],
        }
    return stats


def field_accuracy(results: list[dict], field: str) -> tuple[int, int] | None:
    """Returns (correct, total) for cases where `field` was evaluated."""
    key = f"{field}_correct"
    relevant = [r for r in results if key in r]
    if not relevant:
        return None
    correct = sum(1 for r in relevant if r[key])
    return correct, len(relevant)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

_ABBREV = {
    "action_recommendation": "action",
    "explanation": "explan",
    "retrospective": "retro ",
    "counterfactual": "cfact ",
    "comparison": "compar",
    "lookup": "lookup",
    "trace_replay": "trace ",
    "out_of_scope": "OOS   ",
}


def print_confusion_matrix(matrix: dict[str, dict[str, int]]) -> None:
    col_w = 7
    col_headers = "".join(f"{_ABBREV[i]:>{col_w}}" for i in ALL_INTENTS)
    header = f"{'exp / act':<12}" + col_headers
    print(header)
    print("-" * len(header))
    for exp in ALL_INTENTS:
        row = f"{_ABBREV[exp]:<12}"
        for act in ALL_INTENTS:
            val = matrix[exp][act]
            cell = str(val) if val > 0 else "."
            row += f"{cell:>{col_w}}"
        print(row)


def print_per_intent(stats: dict[str, dict]) -> None:
    w = 22
    print(f"\n{'Intent':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'N':>5}")
    print("-" * 62)
    for intent in ALL_INTENTS:
        s = stats[intent]
        print(
            f"{intent:<22} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}"
            f" {s['tp']:>5} {s['fp']:>5} {s['fn']:>5} {s['support']:>5}"
        )


def print_field_table(results: list[dict], backend: str) -> None:
    fields = RULE_FIELDS + (LLM_ONLY_FIELDS if backend != "rule_based" else [])
    print(f"\n{'Field':<30} {'Correct':>8} {'Total':>7} {'Acc':>8}  Note")
    print("-" * 70)
    for field in fields:
        fa = field_accuracy(results, field)
        if fa is None:
            note = "no fixture cases"
            print(f"{field:<30} {'—':>8} {'—':>7} {'—':>8}  {note}")
            continue
        correct, total = fa
        acc = correct / total if total else 0.0
        note = "LLM-only (skipped in rule_based)" if field in LLM_ONLY_FIELDS and backend == "rule_based" else ""
        print(f"{field:<30} {correct:>8} {total:>7} {acc:>8.1%}  {note}")

    # Regions
    region_cases = [r for r in results if "regions_overlap" in r]
    if region_cases:
        overlap_correct = sum(1 for r in region_cases if r["regions_overlap"])
        exact_correct = sum(1 for r in region_cases if r["regions_exact"])
        n = len(region_cases)
        print(f"{'region_overlap':<30} {overlap_correct:>8} {n:>7} {overlap_correct/n:>8.1%}")
        print(f"{'region_exact':<30} {exact_correct:>8} {n:>7} {exact_correct/n:>8.1%}")

    # Confidence band
    conf_cases = [r for r in results if "confidence_in_range" in r]
    if conf_cases:
        in_band = sum(1 for r in conf_cases if r["confidence_in_range"])
        n = len(conf_cases)
        print(f"{'confidence_in_band':<30} {in_band:>8} {n:>7} {in_band/n:>8.1%}")


def print_failures(results: list[dict]) -> None:
    failures = [r for r in results if not r["intent_correct"]]
    if not failures:
        print("\nAll intents correct.")
        return
    print(f"\nIntent failures ({len(failures)}):")
    for r in failures:
        q = r["query"][:60] + ("..." if len(r["query"]) > 60 else "")
        print(f"  [{r['id']}] exp={r['expected_intent']:<22} act={r['actual_intent']:<22} | {q}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", default="tests/fixtures/decomposer_eval_set.json")
    p.add_argument("--output", default="reports/decomposer_eval_report.json")
    p.add_argument("--backend", choices=["rule_based", "ollama", "claude"], default="rule_based")
    p.add_argument("--intent-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = data["cases"]
    print(f"GridVerdict Decomposer Eval — {len(cases)} cases, backend={args.backend}")
    print(f"Fixture: {fixture_path}  |  Version: {data.get('version', '?')}\n")

    results: list[dict] = []
    errors: list[str] = []

    for i, case in enumerate(cases):
        try:
            actual = run_decomposer(case, args.backend)
            results.append(evaluate_case(case, actual, args.backend))
        except Exception as exc:
            errors.append(f"[{case['id']}] {exc}")
            print(f"  ERROR [{case['id']}]: {exc}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            done = i + 1
            correct = sum(1 for r in results if r["intent_correct"])
            print(f"  ... {done}/{len(cases)} cases  intent_acc={correct/done:.1%}")

    if not results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    total = len(results)
    intent_correct = sum(1 for r in results if r["intent_correct"])
    intent_acc = intent_correct / total

    print(f"\n{'='*62}")
    print(f"OVERALL INTENT ACCURACY: {intent_correct}/{total} = {intent_acc:.1%}")
    print(f"{'='*62}\n")

    # Confusion matrix
    matrix = build_confusion_matrix(results)
    print("CONFUSION MATRIX (rows=expected, cols=actual):\n")
    print_confusion_matrix(matrix)

    # Per-intent stats
    stats = per_intent_stats(results)
    print_per_intent(stats)

    if not args.intent_only:
        print(f"\nFIELD-LEVEL ACCURACY:")
        print_field_table(results, args.backend)

    if args.verbose:
        print_failures(results)

    # Macro-averaged F1
    scored_intents = [i for i in ALL_INTENTS if stats[i]["support"] > 0]
    macro_f1 = sum(stats[i]["f1"] for i in scored_intents) / len(scored_intents) if scored_intents else 0.0
    print(f"\nMacro F1 (over {len(scored_intents)} intents with support): {macro_f1:.3f}")

    if errors:
        print(f"\nErrors during evaluation: {len(errors)}")
        for e in errors:
            print(f"  {e}")

    # Save JSON report
    report = {
        "fixture": str(fixture_path),
        "version": data.get("version"),
        "backend": args.backend,
        "total_cases": total,
        "intent_accuracy": round(intent_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_intent": stats,
        "confusion_matrix": matrix,
        "errors": errors,
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
