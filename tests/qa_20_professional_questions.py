"""
GridVerdict — 20-Question Professional QA Suite
================================================
Simulates real questions from energy market professionals:
  - NEM operators/retailers buying spot
  - BESS/generator asset owners
  - Market analysts studying price formation
  - Risk managers assessing hedging needs
  - Compliance/regulatory staff

Each question has:
  - expected_intent: what the decomposer should classify
  - expected_verdict: what the verdict should be (or list of acceptable)
  - required_content: strings that must appear somewhere in the full response JSON
  - forbidden_content: strings that must NOT appear (hollow answers)
  - expected_scatter: sources that should be present in the gather
  - description: why this question is in the QA set

Run: python tests/qa_20_professional_questions.py
"""
from __future__ import annotations
import json, time, sys, io, urllib.request, urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"
AUTH = {"Authorization": "Bearer dev-no-auth-bypass", "Content-Type": "application/json"}
TIMEOUT = 150   # seconds per query — qwen3:14b takes 30-50s


def req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, headers=AUTH, method=method)
    with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


@dataclass
class QACase:
    id: str
    question: str
    region: str
    category: str
    description: str
    expected_intent: list[str]                    # acceptable intent values
    expected_verdict: list[str]                   # acceptable verdict values
    required_content: list[str] = field(default_factory=list)   # must be in JSON response
    forbidden_content: list[str] = field(default_factory=list)  # must NOT be in response
    expected_weather: bool = False                # weather data should be present
    expected_oos: bool = False                    # should be OUT_OF_SCOPE
    routing_check: str | None = None             # specific routing check


QA_CASES: list[QACase] = [

    # ── CATEGORY 1: Simple lookups ────────────────────────────────────────────
    QACase(
        id="Q01",
        question="What is the current NSW price and demand right now?",
        region="NSW1",
        category="lookup",
        description="Simplest professional query — should give live price, demand, headroom. Tests T1 scatter.",
        expected_intent=["lookup"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE"],
        required_content=["price_rrp", "demand_mw"],
        forbidden_content=["Weather evidence is unavailable", "No relevant AEMO notice"],
    ),

    QACase(
        id="Q02",
        question="What's the current market regime in SA right now — is it spike, elevated or normal?",
        region="SA1",
        category="lookup",
        description="Regime classification query. SA1 is most volatile region. Tests regime detection.",
        expected_intent=["lookup", "explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["SA1"],
        forbidden_content=[],
    ),

    QACase(
        id="Q03",
        question="Compare prices across all NEM regions right now — which is cheapest and most expensive?",
        region="NSW1",
        category="comparison",
        description="Multi-region comparison. Should trigger all-5-region scatter. Critical for operators buying spot.",
        expected_intent=["comparison"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["VIC1", "QLD1", "SA1", "TAS1"],
        forbidden_content=[],
    ),

    # ── CATEGORY 2: Causal explanation ───────────────────────────────────────
    QACase(
        id="Q04",
        question="Why is NSW price so low right now? What is causing this?",
        region="NSW1",
        category="explanation",
        description="Core causal query. Should use live scatter, show constraints, binding data, analog pattern.",
        expected_intent=["explanation"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["demand", "headroom"],
        forbidden_content=["Weather evidence is unavailable for this query.\nNo relevant AEMO notice"],
    ),

    QACase(
        id="Q05",
        question="Is the NSW price low today because of good weather and high renewable output?",
        region="NSW1",
        category="explanation_weather",
        description="CRITICAL: Previously routed to historical scatter losing weather. Fix verified here.",
        expected_intent=["explanation"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["weather", "wind", "temperature"],
        forbidden_content=["Weather evidence is unavailable for this query.\nNo relevant AEMO notice"],
        expected_weather=True,
    ),

    QACase(
        id="Q06",
        question="Why is SA price higher than NSW right now? Is it the interconnector?",
        region="SA1",
        category="explanation_comparison",
        description="Interconnector causality + comparison. Tests driver attribution and multi-region gather.",
        expected_intent=["explanation", "comparison"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["NSW1"],
        forbidden_content=[],
    ),

    # ── CATEGORY 3: Historical comparison (tests routing fix) ─────────────────
    QACase(
        id="Q07",
        question="Why is NSW price not $160+ like it was yesterday? What changed?",
        region="NSW1",
        category="negation_comparison",
        description="CRITICAL: The exact screenshot bug. 'not X like yesterday' = live query. Tests negation routing fix.",
        expected_intent=["explanation"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["demand", "headroom"],  # should have live data, not 27h old
        forbidden_content=["1649m old", "98990s ago"],  # must NOT show stale archive data
        routing_check="live_scatter",  # dispatch should be recent, not 27h ago
    ),

    QACase(
        id="Q08",
        question="What happened to the NSW price at midnight last night? Why was it so high?",
        region="NSW1",
        category="retrospective",
        description="True retrospective — SHOULD use historical scatter. Opposite of Q07.",
        expected_intent=["retrospective", "explanation"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["analogs"],
        forbidden_content=[],
        routing_check="historical_scatter",  # this one SHOULD use archive data
    ),

    QACase(
        id="Q09",
        question="Have we seen NSW headroom this large before? What happens next when headroom is above 9000 MW?",
        region="NSW1",
        category="retrospective_analogs",
        description="Analog retrieval query. Tests HippoGraph PPR with specific headroom condition.",
        expected_intent=["retrospective", "explanation"],
        expected_verdict=["INSUFFICIENT_DATA", "LOW_CONFIDENCE", "SUPPORTED"],
        required_content=["analog"],
        forbidden_content=[],
    ),

    # ── CATEGORY 4: Forecast/action ───────────────────────────────────────────
    QACase(
        id="Q10",
        question="Will NSW price stay low for the next hour or is a spike likely?",
        region="NSW1",
        category="forecast",
        description="Forecast query. Tests LNN/LEAR/QRA ensemble and predispatch data.",
        expected_intent=["explanation", "lookup"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["forecast", "P50"],
        forbidden_content=[],
    ),

    QACase(
        id="Q11",
        question="Should I dispatch my BESS into the NSW market right now? What is the expected revenue?",
        region="NSW1",
        category="action_bess",
        description="BESS dispatch decision. Tests action recommendation + portfolio engine + FCAS context.",
        expected_intent=["action_recommendation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["confidence"],
        forbidden_content=[],
    ),

    # ── CATEGORY 5: Fuel/source ───────────────────────────────────────────────
    QACase(
        id="Q12",
        question="Which fuel type is cheapest to procure from right now in NSW — solar, wind, coal or gas?",
        region="NSW1",
        category="fuel_source",
        description="Fuel source recommendation. Tests fuel_mix engine. Should show [prior model] label.",
        expected_intent=["action_recommendation", "explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["fuel", "prior"],
        forbidden_content=[],
    ),

    QACase(
        id="Q13",
        question="Is solar output the main reason NSW price is low right now?",
        region="NSW1",
        category="fuel_causality",
        description="Fuel causality — solar specifically. Tests whether weather+fuel data confirms solar driver.",
        expected_intent=["explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["solar", "weather"],
        forbidden_content=[],
        expected_weather=True,
    ),

    # ── CATEGORY 6: Complex multi-part ─────────────────────────────────────────
    QACase(
        id="Q14",
        question="Is NSW cheap compared to last year and which fuel source should I buy from at this price?",
        region="NSW1",
        category="multi_part",
        description="Multi-part: historical distribution + fuel recommendation. Tests sub_questions and multi-planner.",
        expected_intent=["action_recommendation", "explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["median", "fuel"],
        forbidden_content=[],
    ),

    QACase(
        id="Q15",
        question="How does NSW compare to SA and VIC right now, and what is causing the price spread?",
        region="NSW1",
        category="comparison_causal",
        description="Comparison + causal attribution. Tests multi-region scatter + interconnector context.",
        expected_intent=["comparison", "explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["SA1", "VIC1"],
        forbidden_content=[],
    ),

    # ── CATEGORY 7: Specific time queries ─────────────────────────────────────
    QACase(
        id="Q16",
        question="What was the NSW price this morning at 7am? Why was it different from now?",
        region="NSW1",
        category="time_specific",
        description="Specific time reference. Tests temporal anchor resolution and historical routing.",
        expected_intent=["retrospective", "explanation"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=[],
        forbidden_content=[],
    ),

    QACase(
        id="Q17",
        question="Show me the historical price distribution for NSW at this time of day — is $22 cheap or normal?",
        region="NSW1",
        category="historical_distribution",
        description="Historical price distribution. Tests hist_dist module, P25/median/P75/P90 output.",
        expected_intent=["explanation", "retrospective", "lookup"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA"],
        required_content=["median", "P90"],
        forbidden_content=[],
    ),

    # ── CATEGORY 8: Ambiguous and edge cases ───────────────────────────────────
    QACase(
        id="Q18",
        question="Is this a good time for a large electricity consumer to switch to spot contracts in NSW?",
        region="NSW1",
        category="ambiguous_action",
        description="Ambiguous — 'good time' could mean many things. Tests whether decomposer handles correctly.",
        expected_intent=["action_recommendation", "explanation", "lookup"],
        expected_verdict=["SUPPORTED", "LOW_CONFIDENCE", "INSUFFICIENT_DATA", "NEEDS_CLARIFICATION"],
        required_content=[],
        forbidden_content=[],
    ),

    QACase(
        id="Q19",
        question="What is the electricity price in Perth right now and should I buy from wind farms there?",
        region="NSW1",
        category="out_of_scope",
        description="OOS — Perth/WA is NOT in the NEM. Must detect and refuse, not guess.",
        expected_intent=["out_of_scope"],
        expected_verdict=["OUT_OF_SCOPE", "INSUFFICIENT_DATA"],
        required_content=["Western Australia", "SWIS"],
        forbidden_content=[],
        expected_oos=True,
    ),

    QACase(
        id="Q20",
        question="Should I buy electricity futures or swap contracts to hedge my NSW load for next quarter?",
        region="NSW1",
        category="out_of_scope_financial",
        description="Financial products OOS — ASX futures are out of scope. Must detect and explain.",
        expected_intent=["out_of_scope"],
        expected_verdict=["OUT_OF_SCOPE", "INSUFFICIENT_DATA"],
        required_content=[],
        forbidden_content=[],
        expected_oos=True,
    ),
]


@dataclass
class QAResult:
    case: QACase
    passed: bool
    elapsed_s: float
    verdict: str
    intent: str
    confidence: float
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scatter_sources: list[str] = field(default_factory=list)
    answer_preview: str = ""
    error: str = ""
    dispatch_age_minutes: float = 0.0
    routing_correction: bool = False
    weather_present: bool = False


def evaluate(case: QACase, answer: dict, elapsed: float) -> QAResult:
    v = answer.get("verdict", {})
    verdict = v.get("verdict", "")
    intent = answer.get("intent", "")
    confidence = v.get("confidence", 0)
    events = answer.get("pipeline_events", [])
    sections = v.get("answer_sections", [])
    full_json = json.dumps(answer).lower()

    failures = []
    warnings = []

    # Check intent
    if intent not in case.expected_intent:
        failures.append(f"Intent={intent!r}, expected one of {case.expected_intent}")

    # Check verdict
    if verdict not in case.expected_verdict:
        failures.append(f"Verdict={verdict!r}, expected one of {case.expected_verdict}")

    # OOS check
    if case.expected_oos and verdict not in ("OUT_OF_SCOPE", "INSUFFICIENT_DATA"):
        failures.append(f"OOS query got verdict={verdict!r} — should refuse")

    # Required content
    for token in case.required_content:
        if token.lower() not in full_json:
            failures.append(f"Required content missing: {token!r}")

    # Forbidden content
    for token in case.forbidden_content:
        if token.lower() in full_json:
            failures.append(f"Forbidden content found: {token!r}")

    # Weather check
    sg_events = [e for e in events if e.get("step") == "SCATTER_GATHER"]
    has_weather = any("WEATHER" in str(e.get("sources", [])) for e in sg_events)
    if case.expected_weather and not has_weather:
        failures.append("Weather expected in scatter but not present")

    # Dispatch age check (for routing fix)
    dispatch_age_min = 0.0
    if sg_events and sg_events[0].get("dispatch_interval"):
        try:
            dt_str = sg_events[0]["dispatch_interval"][:19]
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            dispatch_age_min = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        except Exception:
            pass

    if case.routing_check == "live_scatter" and dispatch_age_min > 120:
        failures.append(
            f"Routing fix failed: dispatch is {dispatch_age_min:.0f}m old — should be live (< 120m)"
        )
    if case.routing_check == "historical_scatter" and dispatch_age_min < 60:
        warnings.append(
            f"Historical query but dispatch only {dispatch_age_min:.0f}m old — may have used live scatter"
        )

    # Routing correction
    has_rc = any(e.get("step") == "ROUTING_CORRECTION" for e in events)

    # Answer preview
    preview_parts = []
    for sec in sections[:2]:
        for item in sec.get("items", [])[:1]:
            preview_parts.append(f"[{sec['title']}] {str(item)[:80]}")
    preview = " | ".join(preview_parts) or v.get("why_plain_english", "")[:120]

    scatter_sources = sg_events[0].get("sources", []) if sg_events else []

    return QAResult(
        case=case,
        passed=len(failures) == 0,
        elapsed_s=elapsed,
        verdict=verdict,
        intent=intent,
        confidence=confidence,
        failures=failures,
        warnings=warnings,
        scatter_sources=scatter_sources,
        answer_preview=preview,
        dispatch_age_minutes=dispatch_age_min,
        routing_correction=has_rc,
        weather_present=has_weather,
    )


def run_qa(cases: list[QACase], session_id: str) -> list[QAResult]:
    results = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i:02d}/20] {case.id} — {case.category.upper()}", flush=True)
        print(f"  Q: {case.question[:90]}", flush=True)
        t0 = time.time()
        try:
            answer = req("POST", f"/api/sessions/{session_id}/query", {
                "text": case.question,
                "region": case.region,
            })
            elapsed = time.time() - t0
            result = evaluate(case, answer, elapsed)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} | {result.intent} → {result.verdict} | {result.confidence:.0%} | {elapsed:.1f}s", flush=True)
            print(f"  Scatter: {result.scatter_sources}", flush=True)
            if result.routing_correction:
                print(f"  ⚡ Routing correction fired", flush=True)
            if result.weather_present:
                print(f"  🌤  Weather: present", flush=True)
            if result.dispatch_age_minutes > 0:
                print(f"  📅 Dispatch age: {result.dispatch_age_minutes:.0f}m", flush=True)
            if result.failures:
                for f in result.failures:
                    print(f"  ❌ {f}", flush=True)
            if result.warnings:
                for w in result.warnings:
                    print(f"  ⚠  {w}", flush=True)
            if result.answer_preview:
                print(f"  → {result.answer_preview[:100]}", flush=True)
        except urllib.error.HTTPError as exc:
            elapsed = time.time() - t0
            body = exc.read().decode("utf-8", errors="replace")[:200]
            result = QAResult(
                case=case, passed=False, elapsed_s=elapsed,
                verdict="ERROR", intent="ERROR", confidence=0,
                error=f"HTTP {exc.code}: {body}",
            )
            result.failures.append(f"HTTP error: {exc.code}")
            print(f"  ✗ HTTP {exc.code}: {body}", flush=True)
        except Exception as exc:
            elapsed = time.time() - t0
            result = QAResult(
                case=case, passed=False, elapsed_s=elapsed,
                verdict="ERROR", intent="ERROR", confidence=0,
                error=str(exc),
            )
            result.failures.append(f"Exception: {exc}")
            print(f"  ✗ Exception: {exc}", flush=True)
        results.append(result)
    return results


def print_report(results: list[QAResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\n" + "=" * 80)
    print(f"QA REPORT — {passed}/{total} PASSED  ({passed/total:.0%})")
    print("=" * 80)

    print("\n── FAILURES ──────────────────────────────────────────────────────────────────")
    for r in results:
        if not r.passed:
            print(f"\n  {r.case.id} [{r.case.category}]: {r.case.question[:70]}")
            for f in r.failures:
                print(f"    ❌ {f}")

    print("\n── PERFORMANCE ───────────────────────────────────────────────────────────────")
    for r in results:
        flag = "✓" if r.passed else "✗"
        weather_flag = "🌤" if r.weather_present else "  "
        rc_flag = "⚡" if r.routing_correction else "  "
        print(
            f"  {flag} {r.case.id} {weather_flag}{rc_flag} | "
            f"{r.elapsed_s:5.1f}s | {r.intent:<22} | {r.verdict:<20} | "
            f"{r.confidence:.0%} | age={r.dispatch_age_minutes:.0f}m"
        )

    print("\n── SCATTER COVERAGE ──────────────────────────────────────────────────────────")
    all_sources: dict[str, int] = {}
    for r in results:
        for s in r.scatter_sources:
            all_sources[s] = all_sources.get(s, 0) + 1
    for src, count in sorted(all_sources.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {src:<30} {bar} ({count})")

    print("\n── CATEGORY SUMMARY ──────────────────────────────────────────────────────────")
    by_cat: dict[str, list[QAResult]] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)
    for cat, cat_results in sorted(by_cat.items()):
        p = sum(1 for r in cat_results if r.passed)
        print(f"  {cat:<30} {p}/{len(cat_results)}")

    total_time = sum(r.elapsed_s for r in results)
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    print("GridVerdict Professional QA — 20 Questions")
    print(f"Server: {BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("-" * 80)

    # Create one session for all questions (stateful context carry-forward)
    try:
        sess = req("POST", "/api/sessions", {"region": "NSW1"})
        session_id = sess["id"]
        print(f"Session: {session_id[:12]}...")
    except Exception as exc:
        print(f"Failed to create session: {exc}")
        sys.exit(1)

    results = run_qa(QA_CASES, session_id)
    print_report(results)

    # Exit code for CI
    failed = sum(1 for r in results if not r.passed)
    sys.exit(0 if failed == 0 else 1)
