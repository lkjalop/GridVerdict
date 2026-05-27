"""Live Feed (Sprint Q) Playwright e2e tests.

Covers:
  - Tab presence and activation
  - Header elements (title, badge, controls)
  - Severity filter options
  - Pause / Resume toggle
  - Empty-state rendering
  - Commentary card structure (mocked API)
  - Unread badge visibility and increment
  - Dismiss removes the card
  - "Ask about this →" prefills the query textarea
  - SSE push via page.evaluate() lands in the feed

Run:
    E2E_TESTS=1 pytest tests/e2e/test_live_feed.py -v
"""
from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2E_TESTS") != "1",
    reason="Set E2E_TESTS=1 to run (requires: playwright install chromium)",
)

# app_url session fixture provided by tests/e2e/conftest.py

# ── Shared mock data ───────────────────────────────────────────────────

_MOCK_EVENT = {
    "id": "e2e-evt-001",
    "region": "NSW1",
    "valid_time": "2026-05-26T14:00:00",
    "event_type": "PRICE_SPIKE",
    "severity": "HIGH",
    "headline": "NSW1 price spiked through $300/MWh threshold",
    "contributing_factors": [
        {"label": "High afternoon demand", "tier": "confirmed"},
        {"label": "Low wind generation", "tier": "supported"},
    ],
    "missing_data": ["FCAS market depth"],
    "corroborations": {"weather": True, "notices": False},
    "confidence": 0.78,
    "next_watch": ["Watch for price normalisation below $150/MWh"],
    "counterargument": None,
}

_MOCK_COMMENTARY_RESPONSE = {
    "region": "NSW1",
    "events": [_MOCK_EVENT],
    "count": 1,
}


def _mock_commentary_route(route, request):
    """Playwright route handler: intercepts /api/commentary/recent."""
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(_MOCK_COMMENTARY_RESPONSE),
    )


def _mock_empty_commentary_route(route, request):
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps({"region": "NSW1", "events": [], "count": 0}),
    )


def _go_to_live_feed(page, app_url: str):
    """Navigate to the app, wait for Alpine, then click the Live Feed tab."""
    page.goto(app_url, wait_until="load")
    page.wait_for_function("window.Alpine !== undefined", timeout=10000)
    tab = page.locator(".vp-tab", has_text="Live Feed")
    tab.click()
    page.wait_for_selector("#vp-livefeed", state="visible", timeout=5000)


# ── Tests ──────────────────────────────────────────────────────────────

class TestLiveFeedTabPresence:
    def test_live_feed_tab_exists_in_tab_bar(self, page, app_url: str):
        """Live Feed tab button must appear in the viewport tab bar."""
        page.goto(app_url, wait_until="load")
        page.wait_for_function("window.Alpine !== undefined", timeout=10000)
        tab = page.locator(".vp-tab", has_text="Live Feed")
        assert tab.count() >= 1, "Live Feed tab not found in .vp-tab list"

    def test_live_feed_tab_click_activates_panel(self, page, app_url: str):
        """Clicking Live Feed makes #vp-livefeed visible."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        panel = page.locator("#vp-livefeed")
        assert panel.is_visible(), "#vp-livefeed panel not visible after clicking tab"

    def test_live_feed_tab_resets_unread_badge_on_click(self, page, app_url: str):
        """Switching to Live Feed tab clears the unread count to zero."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        page.goto(app_url, wait_until="load")
        page.wait_for_function("window.Alpine !== undefined", timeout=10000)

        # Inject unread count while on a different tab
        page.evaluate("""
            const data = Alpine.$data(document.querySelector('#root'));
            data.commentary.unreadCount = 3;
        """)

        # Now click the Live Feed tab — should reset to 0
        page.locator(".vp-tab", has_text="Live Feed").click()
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount === 0",
            timeout=3000,
        )
        count = page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount"
        )
        assert count == 0, f"unreadCount should be 0 after clicking tab, got {count}"


class TestLiveFeedHeader:
    def test_live_market_feed_heading_present(self, page, app_url: str):
        """Panel header must contain 'Live Market Feed'."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        assert page.locator("#vp-livefeed", has_text="Live Market Feed").count() >= 1

    def test_auto_analysis_badge_present(self, page, app_url: str):
        """The AUTO-ANALYSIS badge must be visible in the Live Feed header."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        assert page.locator("#vp-livefeed .badge", has_text="AUTO-ANALYSIS").count() >= 1

    def test_severity_filter_select_has_four_options(self, page, app_url: str):
        """Severity filter dropdown must have exactly 4 options."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        select = page.locator("#vp-livefeed select").first
        options = select.locator("option").all()
        labels = [o.inner_text() for o in options]
        assert len(labels) == 4, f"Expected 4 severity options, got {labels}"
        assert "All severity" in labels
        assert "Critical only" in labels

    def test_refresh_button_present(self, page, app_url: str):
        """A Refresh button must exist in the Live Feed header."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        assert page.locator("#vp-livefeed button", has_text="Refresh").count() >= 1


class TestLiveFeedPauseResume:
    def test_pause_button_shows_initially(self, page, app_url: str):
        """The Pause/Resume button must show 'Pause' when not paused."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)

        # Ensure not paused
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.paused = false"
        )
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.paused === false"
        )
        btn = page.locator("#vp-livefeed button", has_text="Pause")
        assert btn.count() >= 1, "Pause button not found when paused=false"

    def test_pause_button_click_toggles_to_resume(self, page, app_url: str):
        """Clicking Pause changes button text to 'Resume'."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)

        # Start in unpaused state
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.paused = false"
        )
        page.locator("#vp-livefeed button", has_text="Pause").first.click()
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.paused === true",
            timeout=3000,
        )
        assert page.locator("#vp-livefeed button", has_text="Resume").count() >= 1

    def test_resume_button_click_toggles_to_pause(self, page, app_url: str):
        """Clicking Resume changes button text back to 'Pause'."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)

        # Start in paused state
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.paused = true"
        )
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.paused === true"
        )
        page.locator("#vp-livefeed button", has_text="Resume").first.click()
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.paused === false",
            timeout=3000,
        )
        assert page.locator("#vp-livefeed button", has_text="Pause").count() >= 1


class TestLiveFeedEmptyState:
    def test_empty_state_shows_when_no_events(self, page, app_url: str):
        """'No material market changes detected yet.' must appear when events=[]."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.loading === false",
            timeout=5000,
        )
        assert page.locator(
            "#vp-livefeed", has_text="No material market changes detected yet."
        ).count() >= 1

    def test_empty_state_hidden_when_events_present(self, page, app_url: str):
        """Empty state text must not be shown when events list is non-empty."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.events.length > 0",
            timeout=5000,
        )
        assert page.locator(
            "#vp-livefeed", has_text="No material market changes detected yet."
        ).count() == 0


class TestLiveFeedCommentaryCard:
    def test_card_renders_headline(self, page, app_url: str):
        """Commentary card must show the event headline."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector(
            "#vp-livefeed .card", state="visible", timeout=5000
        )
        assert page.locator(
            "#vp-livefeed .card",
            has_text="NSW1 price spiked through $300/MWh threshold",
        ).count() >= 1

    def test_card_shows_severity_badge(self, page, app_url: str):
        """Commentary card must display the severity badge text."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator("#vp-livefeed .card .badge", has_text="HIGH").count() >= 1

    def test_card_shows_event_type_chip(self, page, app_url: str):
        """Commentary card must display the event type with underscores replaced by spaces."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator(
            "#vp-livefeed .card", has_text="PRICE SPIKE"
        ).count() >= 1

    def test_card_shows_contributing_factors(self, page, app_url: str):
        """Contributing factors section must list factor labels."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator(
            "#vp-livefeed", has_text="High afternoon demand"
        ).count() >= 1

    def test_card_shows_missing_data_line(self, page, app_url: str):
        """Missing data items must appear under 'Not explained:'."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator(
            "#vp-livefeed", has_text="FCAS market depth"
        ).count() >= 1

    def test_card_shows_confidence_percentage(self, page, app_url: str):
        """Confidence bar label must show the percentage (78%)."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator("#vp-livefeed", has_text="78%").count() >= 1

    def test_card_shows_next_watch(self, page, app_url: str):
        """Next watch line must be rendered on the card."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)
        assert page.locator(
            "#vp-livefeed", has_text="Watch for price normalisation"
        ).count() >= 1

    def test_card_dismiss_button_removes_card(self, page, app_url: str):
        """Clicking the × dismiss button must remove the card from the DOM."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)

        # Click the dismiss (×) button
        page.locator("#vp-livefeed .card button[title='Dismiss']").first.click()
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.events.length === 0",
            timeout=3000,
        )
        assert page.locator("#vp-livefeed .card").count() == 0, (
            "Card still visible after dismiss click"
        )

    def test_card_ask_about_prefills_textarea(self, page, app_url: str):
        """'Ask about this →' must prefill the query textarea with a relevant question."""
        page.route("**/api/commentary/recent**", _mock_commentary_route)
        _go_to_live_feed(page, app_url)
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=5000)

        page.locator("#vp-livefeed .card button", has_text="Ask about this").first.click()

        textarea = page.locator("textarea").first
        value = textarea.input_value()
        assert "price spike" in value.lower() or "NSW1" in value, (
            f"Textarea not prefilled correctly: {value!r}"
        )


class TestLiveFeedUnreadBadge:
    def test_unread_badge_hidden_when_count_zero(self, page, app_url: str):
        """The red unread-count badge on the Live Feed tab must not be visible at count=0."""
        page.goto(app_url, wait_until="load")
        page.wait_for_function("window.Alpine !== undefined", timeout=10000)
        # Default state: unreadCount == 0 → badge hidden via x-show
        badge = page.locator(".vp-tab", has_text="Live Feed").locator("span[style*='dc2626']")
        assert not badge.is_visible(), "Unread badge should be hidden when count=0"

    def test_unread_badge_visible_when_count_nonzero(self, page, app_url: str):
        """The unread badge must become visible when unreadCount > 0."""
        page.goto(app_url, wait_until="load")
        page.wait_for_function("window.Alpine !== undefined", timeout=10000)
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount = 2"
        )
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount === 2"
        )
        badge = page.locator(".vp-tab", has_text="Live Feed").locator("span").last
        page.wait_for_function(
            """() => {
                const tabs = Array.from(document.querySelectorAll('.vp-tab'));
                const lf = tabs.find(t => t.textContent.includes('Live Feed'));
                if (!lf) return false;
                const spans = lf.querySelectorAll('span');
                const badge = spans[spans.length - 1];
                return badge && badge.style.display !== 'none' && badge.textContent.trim() === '2';
            }""",
            timeout=3000,
        )

    def test_sse_event_increments_unread_when_not_on_tab(self, page, app_url: str):
        """Simulated SSE commentary_created event increments unreadCount when on another tab."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        page.goto(app_url, wait_until="load")
        page.wait_for_function("window.Alpine !== undefined", timeout=10000)

        # Ensure we are NOT on the livefeed tab
        is_livefeed = page.evaluate(
            "Alpine.$data(document.querySelector('#root')).viewport.activeTab === 'livefeed'"
        )
        assert not is_livefeed, "Should not be on livefeed tab initially"

        initial_count = page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount"
        )

        # Simulate SSE push by calling onCommentaryEvent directly
        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.onCommentaryEvent({json.dumps(_MOCK_EVENT)});
        """)

        page.wait_for_function(
            f"Alpine.$data(document.querySelector('#root')).commentary.unreadCount > {initial_count}",
            timeout=3000,
        )
        new_count = page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount"
        )
        assert new_count > initial_count, (
            f"unreadCount should have incremented from {initial_count}, got {new_count}"
        )

    def test_sse_event_appended_to_feed_while_not_paused(self, page, app_url: str):
        """SSE push adds event to commentary.events when feed is not paused."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)

        # Ensure unpaused
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.paused = false"
        )
        # Clear any loaded events
        page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.events = []"
        )

        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.onCommentaryEvent({json.dumps(_MOCK_EVENT)});
        """)

        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.events.length === 1",
            timeout=3000,
        )
        page.wait_for_selector("#vp-livefeed .card", state="visible", timeout=3000)
        assert page.locator(
            "#vp-livefeed", has_text="NSW1 price spiked through $300/MWh threshold"
        ).count() >= 1

    def test_sse_event_paused_increments_count_not_card(self, page, app_url: str):
        """When paused, SSE push increments unreadCount but does NOT add a card."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        _go_to_live_feed(page, app_url)

        page.evaluate("""
            const data = Alpine.$data(document.querySelector('#root'));
            data.commentary.paused = true;
            data.commentary.events = [];
            data.commentary.unreadCount = 0;
        """)

        page.evaluate(f"""
            const data = Alpine.$data(document.querySelector('#root'));
            data.onCommentaryEvent({json.dumps(_MOCK_EVENT)});
        """)

        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.unreadCount === 1",
            timeout=3000,
        )
        events_len = page.evaluate(
            "Alpine.$data(document.querySelector('#root')).commentary.events.length"
        )
        assert events_len == 0, (
            f"Paused feed should not add events to list, but events.length={events_len}"
        )


class TestLiveFeedNoJSErrors:
    def test_live_feed_tab_no_js_errors(self, page, app_url: str):
        """Navigating to Live Feed tab must not throw any JavaScript exceptions."""
        page.route("**/api/commentary/recent**", _mock_empty_commentary_route)
        js_errors: list[str] = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))

        _go_to_live_feed(page, app_url)
        page.wait_for_function(
            "Alpine.$data(document.querySelector('#root')).commentary.loading === false",
            timeout=5000,
        )

        assert not js_errors, (
            "JavaScript error(s) fired on Live Feed navigation:\n"
            + "\n".join(f"  • {e}" for e in js_errors)
        )
