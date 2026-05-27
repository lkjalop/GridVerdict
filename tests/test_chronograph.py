"""Tests for ChronoGraph regime classifier.

Critical test: quantile-based spike labelling must never fire on low absolute
prices, even when they are at the 95th percentile of a narrow distribution.
A $58/MWh price is *not* a spike, regardless of its quantile rank.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.engines.chronograph.regime import RegimeClassifier, reset_classifiers

_NOW = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

_NSW_THRESHOLDS = {
    "elevated": 150.0,
    "spike": 300.0,
    "extreme": 5000.0,
}


def _make_classifier() -> RegimeClassifier:
    reset_classifiers()
    return RegimeClassifier("NSW1", _NSW_THRESHOLDS)


def _feed(clf: RegimeClassifier, prices: list[float]) -> None:
    """Feed a sequence of prices, one 5-min interval apart."""
    t = _NOW
    for p in prices:
        clf.observe(p, t)
        t += timedelta(minutes=5)


# ── False-positive guard ──────────────────────────────────────────────────────

def test_low_price_narrow_distribution_never_labelled_spike():
    """$55-$60 range: 95th-percentile price must not be labelled 'spike'.

    This was the reported bug: narrow low-price distribution caused the p95
    price (~$58) to be classified as 'spike' with no absolute price floor.
    """
    clf = _make_classifier()
    # Feed 48 observations of $55-$60 (2 hours at 5-min intervals) to warm digest
    prices = [55.0 + (i % 10) * 0.5 for i in range(48)]  # range $55-$59.5
    _feed(clf, prices)

    assert clf.observation_count >= 48

    # Now observe the highest price in our range — p95+ of this distribution
    state = clf.observe(59.5, _NOW + timedelta(hours=4))
    assert state.label not in ("spike", "extreme"), (
        f"$59.5 labelled '{state.label}' — quantile false-positive: "
        f"quantile_rank={state.quantile_rank}"
    )
    assert state.label in ("normal", "elevated")


def test_low_price_at_p95_is_at_most_elevated():
    """Even if quantile_rank hits 0.90+, price below elevated_threshold → at most 'elevated'."""
    clf = _make_classifier()
    prices = [50.0] * 24 + [51.0] * 24   # 48 obs, tight range
    _feed(clf, prices)

    state = clf.observe(55.0, _NOW + timedelta(hours=5))
    assert state.label in ("normal", "elevated"), (
        f"$55 (well below $150 elevated threshold) labelled '{state.label}'"
    )


# ── Absolute threshold classification ────────────────────────────────────────

def test_spike_threshold_fires_immediately():
    """Price at or above spike_threshold always → 'spike', even before digest warms."""
    clf = _make_classifier()
    state = clf.observe(350.0, _NOW)
    assert state.label == "spike"


def test_extreme_threshold_fires_immediately():
    clf = _make_classifier()
    state = clf.observe(6000.0, _NOW)
    assert state.label == "extreme"


def test_elevated_threshold_fires_immediately():
    clf = _make_classifier()
    state = clf.observe(200.0, _NOW)
    assert state.label == "elevated"


def test_normal_below_all_thresholds():
    clf = _make_classifier()
    state = clf.observe(80.0, _NOW)
    assert state.label == "normal"


# ── Quantile upgrade only when absolute price justifies it ───────────────────

def test_quantile_spike_requires_elevated_floor():
    """quantile >= 0.95 only upgrades to spike if price >= elevated_threshold."""
    clf = _make_classifier()
    # Feed 48 observations just below elevated threshold to build a distribution
    prices = [100.0 + i * 0.5 for i in range(48)]  # $100-$123.5 (below $150 elevated)
    _feed(clf, prices)

    # $148 is near the top of this distribution (high quantile) but below $150 elevated
    state = clf.observe(148.0, _NOW + timedelta(hours=5))
    assert state.label in ("normal", "elevated"), (
        f"$148 (below elevated_threshold=$150) labelled '{state.label}' "
        f"— quantile={state.quantile_rank}"
    )
    assert state.label != "spike"


def test_quantile_spike_fires_when_above_elevated_floor():
    """quantile >= 0.95 + price >= elevated_threshold → 'spike' is correct."""
    clf = _make_classifier()
    # Build a distribution mostly in the $100-$140 range (below spike $300)
    prices = [100.0 + i * 1.0 for i in range(48)]
    _feed(clf, prices)

    # Observe $160 — above elevated ($150) but below spike ($300)
    # With a high quantile rank, this should upgrade to spike
    state = clf.observe(160.0, _NOW + timedelta(hours=5))
    # quantile rank should be high (this is above everything we fed)
    if state.quantile_rank >= 0.95:
        assert state.label == "spike", (
            f"$160 at quantile={state.quantile_rank} with price>elevated should be spike"
        )
    else:
        assert state.label in ("elevated", "spike")


# ── Regime change detection ───────────────────────────────────────────────────

def test_regime_start_resets_on_label_transition():
    """regime_start should update when price crosses into a new regime."""
    clf = _make_classifier()
    t0 = _NOW

    s1 = clf.observe(80.0, t0)
    assert s1.label == "normal"
    assert s1.regime_start == t0

    t_spike = t0 + timedelta(minutes=10)
    s2 = clf.observe(400.0, t_spike)
    assert s2.label == "spike"
    assert s2.regime_start == t_spike


def test_regime_start_unchanged_within_same_regime():
    """Consecutive prices in same regime must not reset regime_start."""
    clf = _make_classifier()
    t0 = _NOW

    s1 = clf.observe(80.0, t0)
    t1 = t0 + timedelta(minutes=5)
    s2 = clf.observe(85.0, t1)

    assert s1.label == "normal"
    assert s2.label == "normal"
    assert s2.regime_start == s1.regime_start == t0


# ── Confidence sanity ─────────────────────────────────────────────────────────

def test_confidence_bounded_0_1():
    clf = _make_classifier()
    prices = [50.0 + i for i in range(100)]
    _feed(clf, prices)
    state = clf.observe(200.0, _NOW + timedelta(hours=10))
    assert 0.0 <= state.confidence <= 1.0


def test_extreme_price_has_high_confidence():
    clf = _make_classifier()
    _feed(clf, [80.0] * 100)
    state = clf.observe(10000.0, _NOW + timedelta(hours=10))
    assert state.label == "extreme"
    assert state.confidence >= 0.8


def test_quantile_rank_bounded_0_1():
    clf = _make_classifier()
    prices = [float(i) for i in range(50, 100)]
    _feed(clf, prices)
    state = clf.observe(75.0, _NOW + timedelta(hours=5))
    assert 0.0 <= state.quantile_rank <= 1.0


# ── classify_static fallback ──────────────────────────────────────────────────

def test_classify_static_matches_thresholds():
    clf = _make_classifier()
    assert clf.classify_static(6000.0) == "extreme"
    assert clf.classify_static(350.0) == "spike"
    assert clf.classify_static(200.0) == "elevated"
    assert clf.classify_static(80.0) == "normal"
    assert clf.classify_static(300.0) == "spike"   # boundary
    assert clf.classify_static(299.9) == "elevated"
