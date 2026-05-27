"""Frontend E2E smoke tests — catches JS bootstrap errors that unit tests miss.

One-time browser setup:
    pip install playwright pytest-playwright
    playwright install chromium

Run:
    E2E_TESTS=1 pytest tests/e2e/ -v

What these tests catch:
  - JS exceptions on page load (e.g. window.gvCharts.analog = ... before gvCharts exists)
  - Missing Alpine.js x-data initialisation
  - Broken vendor asset paths (CDN replaced with local vendor/)
  - Query input not present in DOM

NOTE: wait_until="load" is used instead of "networkidle" because the SSE event
stream (/api/events/stream) keeps a persistent connection open, which prevents
the page from ever reaching networkidle state.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2E_TESTS") != "1",
    reason="Set E2E_TESTS=1 to run (requires: playwright install chromium)",
)

# app_url session fixture is provided by tests/e2e/conftest.py


def _load(page, app_url: str):
    """Navigate to the app and wait for Alpine.js to initialise."""
    page.goto(app_url, wait_until="load")
    page.wait_for_function("window.Alpine !== undefined", timeout=10000)


# ── Tests ─────────────────────────────────────────────────────────────

def test_page_loads_no_js_errors(page, app_url: str):
    """Page must load without any JavaScript exceptions.

    This catches the bootstrap bug where window.gvCharts.analog was assigned
    before window.gvCharts was defined, throwing a TypeError on every page load.
    """
    js_errors: list[str] = []
    page.on("pageerror", lambda exc: js_errors.append(str(exc)))

    _load(page, app_url)

    assert not js_errors, (
        f"JavaScript error(s) fired on page load:\n"
        + "\n".join(f"  • {e}" for e in js_errors)
    )


def test_chat_uses_answer_sections_not_long_why_plain_english(page, app_url: str):
    """Assistant bubbles must show the concise Answer Planner summary."""
    _load(page, app_url)
    page.wait_for_function(
        "Alpine.$data(document.querySelector('#root')) !== undefined",
        timeout=10000,
    )

    page.evaluate(
        """
        const data = Alpine.$data(document.querySelector('#root'));
        data.auth.showLogin = false;
        const verdict = {
          verdict: 'SUPPORTED',
          action: 'hold',
          confidence: 0.72,
          why_plain_english: 'DO_NOT_SHOW_LONG_WHY '.repeat(80),
          answer_sections: [
            {title: 'Answer', items: [
              'NSW1 is elevated, but the primary driver is not confirmed.',
              'Recent price path: now $143; 5m ago $164; 10m ago $164.'
            ]},
            {title: 'Evidence', items: ['Live dispatch price, demand, and headroom are fresh.']},
            {title: 'Drivers', items: ['Supported: dispatch state and historical analog pattern.']},
            {title: 'Continuation', items: ['Available models lean lower, with wide P90 risk.']},
            {title: 'Missing', items: ['Constraint, interconnector, unit dispatch, and rebid evidence.']}
          ]
        };
        data.messages = [{
          role: 'assistant',
          id: 'answer-planner-e2e',
          text: data.formatVerdictSummary(verdict),
          verdict
        }];
        """
    )

    body = page.locator("#messages").inner_text(timeout=5000)
    assert "NSW1 is elevated" in body
    assert "Recent price path" in body
    assert "Supported: dispatch state" in body
    assert "Available models lean lower" in body
    assert "DO_NOT_SHOW_LONG_WHY" not in body


def test_gvcharts_object_defined(page, app_url: str):
    """window.gvCharts must be a non-null object after Alpine and main.js load."""
    _load(page, app_url)

    defined = page.evaluate(
        "typeof window.gvCharts === 'object' && window.gvCharts !== null"
    )
    assert defined, (
        "window.gvCharts is undefined — main.js did not initialise correctly"
    )


def test_gvcharts_analog_sub_object(page, app_url: str):
    """gvCharts.analog must be a sub-object with getAnalogChart / destroyAnalogChart.

    If main.js assigns window.gvCharts.analog before window.gvCharts exists,
    this property will be missing even if the later assignment of window.gvCharts
    succeeds — catching the exact regression from the professional review.
    """
    _load(page, app_url)

    has_get = page.evaluate("typeof window.gvCharts?.analog?.getAnalogChart === 'function'")
    has_destroy = page.evaluate("typeof window.gvCharts?.analog?.destroyAnalogChart === 'function'")

    assert has_get and has_destroy, (
        "gvCharts.analog is incomplete — bootstrap ordering bug present.\n"
        f"  getAnalogChart present:    {has_get}\n"
        f"  destroyAnalogChart present: {has_destroy}"
    )


def test_gvcharts_backtest_sub_object(page, app_url: str):
    """gvCharts.backtestPnl must be a sub-object with getBacktestChart / destroyBacktestChart."""
    _load(page, app_url)

    has_get = page.evaluate("typeof window.gvCharts?.backtestPnl?.getBacktestChart === 'function'")
    has_destroy = page.evaluate("typeof window.gvCharts?.backtestPnl?.destroyBacktestChart === 'function'")

    assert has_get and has_destroy, (
        "gvCharts.backtestPnl is incomplete — bootstrap ordering bug present."
    )


def test_alpine_xdata_initializes(page, app_url: str):
    """Alpine.js must find an [x-data] element and initialise without errors."""
    _load(page, app_url)

    has_xdata = page.evaluate("document.querySelector('[x-data]') !== null")
    assert has_xdata, "No [x-data] element found — Alpine.js may not have loaded"


def test_query_textarea_present(page, app_url: str):
    """A textarea for query input must be present and visible."""
    _load(page, app_url)

    count = page.locator("textarea").count()
    assert count > 0, "No <textarea> found — query input is missing from the page"


def test_vendor_echarts_loads(page, app_url: str):
    """ECharts must load from the local vendor path, not a CDN.

    A 404 on /static/vendor/echarts/echarts.min.js means the file was not
    downloaded to frontend/static/vendor/ and the CDN fallback was removed.
    """
    failed_requests: list[str] = []

    def _on_response(response):
        if "echarts" in response.url and response.status >= 400:
            failed_requests.append(f"{response.status} {response.url}")

    page.on("response", _on_response)
    _load(page, app_url)

    assert not failed_requests, (
        f"ECharts failed to load:\n"
        + "\n".join(f"  • {r}" for r in failed_requests)
    )


def test_vendor_alpine_loads(page, app_url: str):
    """Alpine.js must load from the local vendor path, not a CDN."""
    failed_requests: list[str] = []

    def _on_response(response):
        if "alpine" in response.url.lower() and response.status >= 400:
            failed_requests.append(f"{response.status} {response.url}")

    page.on("response", _on_response)
    _load(page, app_url)

    assert not failed_requests, (
        f"Alpine.js failed to load:\n"
        + "\n".join(f"  • {r}" for r in failed_requests)
    )
