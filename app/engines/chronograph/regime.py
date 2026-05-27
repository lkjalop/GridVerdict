"""ChronoGraph regime classifier.

Composes ADWIN change-point detection with t-digest quantile estimation to
produce a RegimeState — a richer signal than the static threshold heuristic.

Classification logic:
1. Feed price into both ADWIN (change detection) and TDigest (quantile rank).
2. If ADWIN fires → note the change point and reset regime_start.
3. Classify label from (price, quantile_rank, thresholds):
   - extreme : price ≥ extreme_threshold
   - spike   : price ≥ spike_threshold  OR  (quantile ≥ 0.95 AND price ≥ elevated_threshold)
   - elevated: price ≥ elevated_threshold OR  quantile ≥ 0.90 with enough history
   - normal  : otherwise
4. Confidence is a function of data sufficiency + classification margin.

Registry: `get_classifier(region, thresholds)` returns a module-level singleton
per region so classifiers persist across API requests and warm up over time.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
Only imports from app.core.interfaces (RegimeState) and sibling modules.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.interfaces import RegimeState
from app.engines.chronograph.adwin import ADWIN
from app.engines.chronograph.tdigest import TDigest

# ── Per-region singleton registry ──────────────────────────────────��──

_registry: dict[str, "RegimeClassifier"] = {}
_registry_lock = threading.Lock()


def get_classifier(region: str, thresholds: dict[str, float]) -> "RegimeClassifier":
    """Return the module-level RegimeClassifier for a region.

    Creates one on first call. Subsequent calls return the same instance,
    which has accumulated price history since server start.
    """
    with _registry_lock:
        if region not in _registry:
            _registry[region] = RegimeClassifier(region, thresholds)
        return _registry[region]


def reset_classifiers() -> None:
    """Clear all classifiers (intended for tests only)."""
    with _registry_lock:
        _registry.clear()


# ── Classifier ────────────────────────────────────────────────────────

class RegimeClassifier:
    """Stateful per-region regime classifier.

    Call `observe(price, valid_time)` on every incoming dispatch price.
    Returns a RegimeState with:
    - label: normal | elevated | spike | extreme
    - confidence: 0-1
    - regime_start: datetime when the current regime began
    - signal_strength: ADWIN change-point strength (0-1)
    - quantile_rank: where this price sits in the rolling distribution (0-1)
    """

    # Minimum observations before t-digest quantile classification kicks in
    _MIN_DIGEST_OBS = 24   # ~2 hours at 5-min intervals

    def __init__(self, region: str, thresholds: dict[str, float]) -> None:
        self._region = region
        self._t = thresholds  # elevated, spike, extreme thresholds
        self._adwin = ADWIN(delta=0.002, min_window=5, max_window=72)
        self._digest = TDigest(compression=100.0, max_buffer=500)
        self._regime_start: datetime | None = None
        self._last_label: str | None = None
        self._obs_count: int = 0

    def observe(self, price: float, valid_time: datetime) -> RegimeState:
        """Update state with a new dispatch price and return the current regime."""
        self._adwin.update(price)
        self._digest.update(price)
        self._obs_count += 1

        quantile_rank = self._digest.cdf(price) if self._obs_count >= self._MIN_DIGEST_OBS else 0.5
        label = self._classify(price, quantile_rank)

        # Reset regime clock on first observation, regime transition, or ADWIN change-point
        if self._last_label is None or label != self._last_label or self._adwin.detected:
            self._regime_start = valid_time
            self._last_label = label

        confidence = self._confidence(price, label, quantile_rank)

        return RegimeState(
            label=label,
            confidence=confidence,
            regime_start=self._regime_start,
            signal_strength=round(self._adwin.change_strength, 3),
            quantile_rank=round(quantile_rank, 3),
        )

    def classify_static(self, price: float) -> str:
        """Threshold-only classification — no history required.

        Used as a fallback before the classifier has warmed up.
        """
        t = self._t
        if price >= t["extreme"]:
            return "extreme"
        if price >= t["spike"]:
            return "spike"
        if price >= t["elevated"]:
            return "elevated"
        return "normal"

    @property
    def observation_count(self) -> int:
        return self._obs_count

    # ── Internal ─────────────────────────────────────────────────────

    def _classify(self, price: float, quantile_rank: float) -> str:
        t = self._t
        # Hard threshold always wins for extreme/spike values
        if price >= t["extreme"]:
            return "extreme"
        if price >= t["spike"]:
            return "spike"
        # Quantile can only *upgrade* once the digest is warm, and only within
        # the range the absolute price already justifies.
        # Guard: quantile >= 0.95 escalates to "spike" only if price is already
        # at or above the elevated threshold — prevents a narrow low-price
        # distribution from producing false spike labels (e.g. $58/MWh at p95
        # when all recent prices are $50-$60).
        if self._obs_count >= self._MIN_DIGEST_OBS:
            if quantile_rank >= 0.95 and price >= t["elevated"]:
                return "spike"
            if price >= t["elevated"] or quantile_rank >= 0.90:
                return "elevated"
        else:
            if price >= t["elevated"]:
                return "elevated"
        return "normal"

    def _confidence(self, price: float, label: str, quantile_rank: float) -> float:
        """Confidence 0-1: data sufficiency × classification clarity."""
        n = self._obs_count

        # Data sufficiency component (saturates at 100 observations)
        data_score = min(1.0, n / 100.0)

        # Classification margin component
        t = self._t
        if label == "extreme":
            margin = min(1.0, (price - t["extreme"]) / max(t["extreme"] * 0.2, 1.0))
            margin_score = 0.8 + 0.2 * margin
        elif label == "spike":
            margin = min(1.0, (price - t["spike"]) / max(t["extreme"] - t["spike"], 1.0))
            margin_score = 0.65 + 0.2 * margin
        elif label == "elevated":
            margin = min(1.0, (price - t["elevated"]) / max(t["spike"] - t["elevated"], 1.0))
            margin_score = 0.55 + 0.2 * margin
        else:
            # Normal: confidence grows as price moves away from elevated threshold
            margin = min(1.0, (t["elevated"] - price) / max(t["elevated"], 1.0))
            margin_score = 0.60 + 0.25 * margin

        # Quantile confirmation bonus (once digest is warm)
        quantile_bonus = 0.0
        if self._obs_count >= self._MIN_DIGEST_OBS:
            if label in ("spike", "extreme") and quantile_rank >= 0.90:
                quantile_bonus = 0.05
            elif label == "normal" and quantile_rank <= 0.60:
                quantile_bonus = 0.05

        raw = 0.35 * data_score + 0.55 * margin_score + quantile_bonus
        return round(max(0.0, min(1.0, raw)), 3)
