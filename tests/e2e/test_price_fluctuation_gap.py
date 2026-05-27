"""Price fluctuation query gap — Playwright e2e tests.

GAP DOCUMENTED HERE:
  User query: "why did the price fluctuate from 143 to 167 to 165 and then back
  down to 140 again? what is causing the fluctuations? which fuel source?"

  BEFORE FIX: System returns a bare snapshot lookup (NSW1: $140.25/MWh, demand
  8106 MW, headroom 3954 MW. Regime: elevated.) with no acknowledgment of the
  price sequence the user described. No fuel-source guidance.

  AFTER FIX:
  - Routing: decomposition.requested_output == "price_fluctuation_attribution"
  - Content: answer acknowledges the user-described price path ($143→$167→$165→$140)
  - Fuel: explains why fuel-source attribution requires unit dispatch evidence

Run:
    E2E_TESTS=1 pytest tests/e2e/test_price_fluctuation_gap.py -v
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2E_TESTS") != "1",
    reason="Set E2E_TESTS=1 to run (requires: playwright install chromium)",
)

# ── Constants ──────────────────────────────────────────────────────────────────

_FLUCTUATION_QUERY = (
    "why did the price fluctuate from 143 to 167 to 165 and then back down "
    "to 140 again? what is causing the fluctuations? which fuel source or other reasons?"
)

# Mock verdict showing the BEFORE-fix bare-lookup behavior.
# This is what the screenshot showed: _plan_lookup output with no price path analysis.
_MOCK_VERDICT_BEFORE_FIX = {
    "verdict": "INSUFFICIENT_DATA",
    "action": "monitor",
    "confidence": 0.35,
    "confidence_band": "low",
    "why_plain_english": "Minimal answer — current snapshot only.",
    "answer_sections": [
        {
            "title": "Answer",
            "items": [
                "NSW1: $140.25/MWh, demand 8106 MW, headroom 3954 MW.",
                "Regime: elevated.",
            ],
        },
        {
            "title": "Evidence",
            "items": ["Dispatch interval: 2026-05-27T12:20:00+00:00."],
        },
    ],
    "missing_data": ["recent_price_trend", "unit_dispatch", "rebid_stack"],
    "counterargument": None,
    "evidence_refs": [],
    "claim_map": [],
    "disclaimer": "Simulation only. Not financial advice.",
}

# Mock verdict showing the DESIRED behavior after fix.
# System acknowledges user-described price path and explains what's needed for attribution.
_MOCK_VERDICT_AFTER_FIX = {
    "verdict": "INSUFFICIENT_DATA",
    "action": "monitor",
    "confidence": 0.45,
    "confidence_band": "low",
    "why_plain_english": "Price path acknowledged; driver attribution not yet confirmed.",
    "answer_sections": [
        {
            "title": "Answer",
            "items": [
                "You described a price path of $143 → $167 → $165 → $140/MWh; live reading is $140.25/MWh.",
                "The observed swing of $27/MWh is consistent with a minor price variation event.",
                "Confirmed driver and fuel-source attribution requires unit dispatch, bids/rebids, and constraint evidence.",
            ],
        },
        {
            "title": "Evidence",
            "items": [
                "Current dispatch: demand 8106 MW, available generation 12060 MW, headroom 3954 MW.",
            ],
        },
        {
            "title": "Drivers",
            "items": [
                "Supported/plausible: live dispatch price/demand/headroom.",
                "Unconfirmed blockers: constraints, interconnectors, unit dispatch, AEMO notice.",
            ],
        },
        {
            "title": "Missing",
            "items": [
                "unit dispatch by fuel type",
                "bid and rebid stack",
                "recent price trend from DB",
            ],
        },
    ],
    "missing_data": ["unit_dispatch", "rebid_stack", "recent_price_trend"],
    "counterargument": "Price fluctuations at this magnitude can have multiple overlapping causes.",
    "evidence_refs": [],
    "claim_map": [],
    "disclaimer": "Simulation only. Not financial advice.",
}


def _load(page, app_url: str):
    page.goto(app_url, wait_until="load")
    page.wait_for_function("window.Alpine !== undefined", timeout=10000)


def _get_alpine_data(page):
    return page.evaluate("Alpine.$data(document.querySelector('#root'))")


# ── API routing tests (httpx, no browser) ─────────────────────────────────────

class TestFluctuationQueryRouting:
    """Verify the rule-based decomposer routes fluctuation queries correctly."""

    def test_fluctuation_query_routes_to_price_attribution(self, app_url: str):
        """Rule-based decomposer must set requested_output='price_fluctuation_attribution'
        for a query containing 'fluctuate/fluctuation/back down' keywords.
        """
        with httpx.Client(base_url=app_url, timeout=30.0) as client:
            # Create a session
            sess_resp = client.post("/api/sessions", json={"region": "NSW1"})
            assert sess_resp.status_code == 201, f"Session create failed: {sess_resp.text}"
            session_id = sess_resp.json()["id"]

            # Submit the fluctuation query
            qry_resp = client.post(
                f"/api/sessions/{session_id}/query",
                json={"text": _FLUCTUATION_QUERY, "region": "NSW1"},
            )
            assert qry_resp.status_code == 200, (
                f"Query failed ({qry_resp.status_code}): {qry_resp.text[:300]}"
            )
            body = qry_resp.json()

        decomp = body.get("decomposition", {})
        requested = decomp.get("requested_output", "")
        intent = decomp.get("intent", "")

        assert intent == "explanation", (
            f"Expected intent='explanation' for fluctuation query, got {intent!r}.\n"
            "The rule-based decomposer should detect 'fluctuate'/'fluctuation' and set explanation intent."
        )
        assert requested == "price_fluctuation_attribution", (
            f"Expected requested_output='price_fluctuation_attribution', got {requested!r}.\n"
            "Fix: ensure _requested_output_for() in decomposition.py matches fluctuation keywords."
        )

    def test_fluctuation_query_requires_history(self, app_url: str):
        """Fluctuation queries must set requires_history=True so dispatch context is sought.

        GAP: Before fix, requires_history was False for fluctuation queries,
        meaning the system didn't know to load recent_dispatch context.
        """
        with httpx.Client(base_url=app_url, timeout=30.0) as client:
            sess_resp = client.post("/api/sessions", json={"region": "NSW1"})
            assert sess_resp.status_code == 201
            session_id = sess_resp.json()["id"]

            qry_resp = client.post(
                f"/api/sessions/{session_id}/query",
                json={"text": _FLUCTUATION_QUERY, "region": "NSW1"},
            )
            assert qry_resp.status_code == 200
            body = qry_resp.json()

        decomp = body.get("decomposition", {})
        requires_history = decomp.get("requires_history", False)

        assert requires_history is True, (
            f"requires_history={requires_history!r} for a fluctuation query — expected True.\n"
            "Fix: add 'fluctuate', 'fluctuation', 'back down' to requires_history keywords "
            "in _decompose_rules() in decomposition.py."
        )

    def test_fluctuation_answer_not_bare_lookup(self, app_url: str):
        """After fix, the answer must not be a bare current-state lookup.

        A bare lookup only shows 'Region: $price/MWh, demand X MW, headroom Y MW.'
        and 'Dispatch interval: ...'. The fluctuation planner should show a richer response.

        GAP: Before fix, _plan_lookup was being used when routing fell through.
        """
        with httpx.Client(base_url=app_url, timeout=30.0) as client:
            sess_resp = client.post("/api/sessions", json={"region": "NSW1"})
            assert sess_resp.status_code == 201
            session_id = sess_resp.json()["id"]

            qry_resp = client.post(
                f"/api/sessions/{session_id}/query",
                json={"text": _FLUCTUATION_QUERY, "region": "NSW1"},
            )
            assert qry_resp.status_code == 200
            body = qry_resp.json()

        verdict = body.get("verdict", {})
        sections = verdict.get("answer_sections") or []
        all_items = [item for sec in sections for item in sec.get("items", [])]
        all_text = " ".join(all_items).lower()

        # A bare lookup would only say "Dispatch interval:" in evidence
        # and have no price path content in Answer
        evidence_items = next(
            (s.get("items", []) for s in sections if s.get("title") == "Evidence"),
            [],
        )
        bare_lookup_pattern = (
            len(sections) <= 2
            and any("dispatch interval" in item.lower() for item in evidence_items)
            and not any("fluctuat" in item.lower() or "price path" in item.lower() for item in all_items)
        )
        assert not bare_lookup_pattern, (
            "Answer looks like a bare _plan_lookup response — no price path analysis:\n"
            + "\n".join(f"  {s['title']}: {s['items']}" for s in sections)
            + "\n\nExpected: acknowledgment of the described price path ($143→$167→$165→$140) "
            "and driver attribution guidance."
        )

    def test_fluctuation_answer_mentions_driver_attribution_gap(self, app_url: str):
        """Answer should tell the user what's needed to confirm the driver/fuel source.

        The user asked 'which fuel source or other reasons?' — the system should
        explain that unit dispatch and bid stack evidence is required for attribution.
        """
        with httpx.Client(base_url=app_url, timeout=30.0) as client:
            sess_resp = client.post("/api/sessions", json={"region": "NSW1"})
            assert sess_resp.status_code == 201
            session_id = sess_resp.json()["id"]

            qry_resp = client.post(
                f"/api/sessions/{session_id}/query",
                json={"text": _FLUCTUATION_QUERY, "region": "NSW1"},
            )
            assert qry_resp.status_code == 200
            body = qry_resp.json()

        verdict = body.get("verdict", {})
        sections = verdict.get("answer_sections") or []
        all_text = " ".join(
            item for sec in sections for item in sec.get("items", [])
        ).lower()

        # Should mention either "dispatch" + "attribution" or "unit dispatch" guidance
        has_attribution_note = (
            "unit dispatch" in all_text
            or ("attribution" in all_text and "dispatch" in all_text)
            or "confirmed" in all_text
        )
        assert has_attribution_note, (
            "Answer does not mention the evidence needed to confirm driver/fuel attribution.\n"
            f"Full text: {all_text!r}\n"
            "Expected: reference to unit dispatch, bid/rebid, or constraint evidence for attribution."
        )


# ── UI rendering tests (Playwright browser) ───────────────────────────────────

class TestFluctuationQueryUIGap:
    """Document the visual gap between before-fix and after-fix UI rendering."""

    def test_before_fix_mock_shows_bare_snapshot(self, page, app_url: str):
        """Verify the bare-lookup format (before fix) is detectable in the UI.

        This test documents WHAT THE BUG LOOKS LIKE: the answer only shows
        current-state snapshot data with no acknowledgment of the user's price path.
        """
        _load(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')) !== undefined",
            timeout=10000,
        )

        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.auth.showLogin = false;
            const verdict = {json.dumps(_MOCK_VERDICT_BEFORE_FIX)};
            data.messages = [{{
                role: 'assistant',
                id: 'gap-before-fix',
                text: data.formatVerdictSummary(verdict),
                verdict
            }}];
        """)

        body = page.locator("#messages").inner_text(timeout=5000)

        # These are the "bare lookup" signals — just current snapshot, no price path
        assert "NSW1: $140.25" in body or "140.25" in body, (
            "Before-fix mock should show the bare price/MWh format"
        )
        assert "8106" in body or "3954" in body, (
            "Before-fix mock should show demand/headroom snapshot values"
        )
        # And critically, no price path analysis
        assert "143" not in body and "167" not in body, (
            "Before-fix mock should NOT mention the user's described price path ($143→$167)"
        )

    def test_after_fix_mock_shows_price_path_acknowledgment(self, page, app_url: str):
        """Verify the improved answer (after fix) correctly acknowledges the price path.

        This test documents WHAT THE FIX LOOKS LIKE: the system acknowledges
        the user-described sequence and explains what evidence is missing.
        """
        _load(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')) !== undefined",
            timeout=10000,
        )

        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.auth.showLogin = false;
            const verdict = {json.dumps(_MOCK_VERDICT_AFTER_FIX)};
            data.messages = [{{
                role: 'assistant',
                id: 'gap-after-fix',
                text: data.formatVerdictSummary(verdict),
                verdict
            }}];
        """)

        body = page.locator("#messages").inner_text(timeout=5000)

        # Price path must be visible
        assert "143" in body and "167" in body, (
            f"After-fix answer should mention the user-described prices $143 and $167.\nBody: {body[:500]}"
        )
        assert "140" in body, (
            "After-fix answer should mention the final price $140 in the described path."
        )

        # Attribution guidance must be visible
        assert "unit dispatch" in body.lower() or "attribution" in body.lower(), (
            "After-fix answer should explain what evidence is needed for driver attribution."
        )

        # Missing data must be visible
        assert "unit dispatch" in body.lower() or "rebid" in body.lower(), (
            "After-fix answer should list missing evidence types in the Missing section."
        )

    def test_after_fix_mock_no_js_errors(self, page, app_url: str):
        """Injecting the improved verdict must not cause any JavaScript errors."""
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))

        _load(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')) !== undefined",
            timeout=10000,
        )

        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.auth.showLogin = false;
            const verdict = {json.dumps(_MOCK_VERDICT_AFTER_FIX)};
            data.messages = [{{
                role: 'assistant',
                id: 'gap-js-errors',
                text: data.formatVerdictSummary(verdict),
                verdict
            }}];
        """)

        page.locator("#messages").inner_text(timeout=5000)
        assert not js_errors, (
            "JavaScript errors when rendering improved price-path verdict:\n"
            + "\n".join(f"  • {e}" for e in js_errors)
        )

