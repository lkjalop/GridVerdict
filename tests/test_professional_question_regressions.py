"""Professional question regression set for the deterministic decomposer."""
from __future__ import annotations

import pytest

from app.core.schema import IntentLabel
from app.engines.decomposition import _decompose_rules


_CASES = [
    (
        "Why is NSW price elevated right now, what evidence supports it, and is it likely to continue?",
        IntentLabel.EXPLANATION,
        "causal_explanation_with_forecast",
        {"price", "demand", "headroom", "forecast", "historical_analog"},
    ),
    (
        "Have we seen similar NSW price and headroom conditions before, and what happened afterwards?",
        IntentLabel.RETROSPECTIVE,
        "historical_analog_outcome",
        {"headroom", "historical_analog"},
    ),
    (
        "Are live weather conditions, AEMO notices, or recent RSS energy news helping explain the NSW price move?",
        IntentLabel.EXPLANATION,
        "weather_notice_news_correlation",
        {"weather", "aemo_notice", "news"},
    ),
    (
        "Does hot weather explain NSW electricity demand and price right now?",
        IntentLabel.EXPLANATION,
        "weather_notice_news_correlation",
        {"weather", "demand", "price"},
    ),
    (
        "What changed in NSW in the last dispatch interval?",
        IntentLabel.EXPLANATION,
        "causal_explanation",
        {"price", "demand", "headroom"},
    ),
    (
        "What evidence supports the QLD spike?",
        IntentLabel.EXPLANATION,
        "causal_explanation",
        {"price"},
    ),
    (
        "Compare NSW, VIC, and QLD prices right now.",
        IntentLabel.COMPARISON,
        "regional_comparison",
        set(),
    ),
    (
        "What sources are stale or missing right now?",
        IntentLabel.LOOKUP,
        "data_freshness_status",
        set(),
    ),
    (
        "Should I dispatch my NSW battery right now?",
        IntentLabel.ACTION_RECOMMENDATION,
        "portfolio_action",
        set(),
    ),
    (
        "What did GridVerdict know at the time of the previous NSW spike?",
        IntentLabel.TRACE_REPLAY,
        "trace_replay",
        set(),
    ),
    (
        "Which constraints or interconnectors are binding in NSW?",
        IntentLabel.LOOKUP,
        "current_market_state",
        {"constraint", "interconnector"},
    ),
    (
        "Did any rebids or outages contribute to the NSW price move?",
        IntentLabel.EXPLANATION,
        "causal_explanation",
        {"rebid", "outage"},
    ),
    (
        "Is FCAS tight enough that my battery should reserve capacity?",
        IntentLabel.ACTION_RECOMMENDATION,
        "portfolio_action",
        set(),
    ),
    (
        "What is the forecast for SA price over the next hour?",
        IntentLabel.LOOKUP,
        "forecast",
        {"forecast"},
    ),
    (
        "Is the QLD price move backed by notices or just market balance?",
        IntentLabel.EXPLANATION,
        "weather_notice_news_correlation",
        {"aemo_notice", "price"},
    ),
    (
        "Show me the current price, demand, availability, and headroom in TAS.",
        IntentLabel.LOOKUP,
        "current_market_state",
        {"demand", "headroom"},
    ),
    (
        "Has VIC stayed elevated after similar price spikes before?",
        IntentLabel.RETROSPECTIVE,
        "historical_analog_outcome",
        {"historical_analog"},
    ),
    (
        "What is missing before I act on this NSW price event?",
        IntentLabel.ACTION_RECOMMENDATION,
        "portfolio_action",
        set(),
    ),
    (
        "What would have happened if I discharged my SA battery 30 minutes ago?",
        IntentLabel.COUNTERFACTUAL,
        "current_market_state",
        set(),
    ),
    (
        "Is the live feed showing a real market change or just stale data?",
        IntentLabel.LOOKUP,
        "data_freshness_status",
        set(),
    ),
]


@pytest.mark.parametrize(
    "query,expected_intent,expected_output,targets",
    _CASES,
    ids=[f"pq-{i:02d}" for i in range(1, len(_CASES) + 1)],
)
def test_professional_questions_route_to_expected_output(
    query: str,
    expected_intent: IntentLabel,
    expected_output: str,
    targets: set[str],
):
    result = _decompose_rules(query, "NSW1", None)

    assert result.intent == expected_intent
    assert result.requested_output == expected_output
    assert result.intent != IntentLabel.OUT_OF_SCOPE
    assert targets.issubset(set(result.causal_targets))
