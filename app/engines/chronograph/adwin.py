"""ADWIN (Adaptive Windowing) change-point detector.

Based on: Bifet & Gavalda (2007)
"Learning from Time-Changing Data with Adaptive Windowing"

ADWIN maintains a sliding window over a data stream and detects statistically
significant shifts in the mean. When a change is detected the window is cut
to the most recent stable portion, so subsequent queries reflect the new regime.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
Pure Python, no external ML dependencies.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ChangePoint:
    """Metadata for a detected distributional shift."""
    position: int        # index in the window at the cut point
    delta_mean: float    # abs(mean_recent - mean_older)
    epsilon: float       # detection threshold that was exceeded
    strength: float      # normalised strength 0-1


class ADWIN:
    """Adaptive windowing change-point detector.

    Maintains a bounded window of recent observations and tests all possible
    binary splits (W = W_recent ∪ W_older) for a statistically significant
    difference in means using the ADWIN epsilon criterion:

        ε = √(0.5 × (1/n₀ + 1/n₁) × ln(4 ln(n) / δ))

    When |mean(W_recent) - mean(W_older)| ≥ ε the window is cut to W_recent.

    Parameters
    ----------
    delta : float
        Significance level (smaller = more conservative, fewer false positives).
        Default 0.002 follows Bifet & Gavalda recommendation.
    min_window : int
        Minimum sub-window size to test (avoid noise on tiny samples).
    max_window : int
        Maximum rolling window size. At 5-min dispatch intervals:
        72 = 6 hours, 36 = 3 hours.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 5,
        max_window: int = 72,
    ) -> None:
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        self._window: deque[float] = deque(maxlen=max_window)
        self.detected: bool = False
        self.change_strength: float = 0.0
        self.last_change: ChangePoint | None = None

    # ── Public API ───────────────────────────────────────────────────

    def update(self, value: float) -> bool:
        """Add a new observation.

        Returns True if a distributional change was detected and the window
        was cut to the recent stable portion.
        """
        self._window.append(value)
        n = len(self._window)

        if n < self.min_window * 2:
            self.detected = False
            self.change_strength = 0.0
            return False

        change = self._detect_change()
        if change is not None:
            self._cut_window(change.position)
            self.detected = True
            self.change_strength = change.strength
            self.last_change = change
        else:
            self.detected = False
            self.change_strength = 0.0

        return self.detected

    @property
    def mean(self) -> float:
        """Current window mean."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def width(self) -> int:
        """Current window size."""
        return len(self._window)

    @property
    def values(self) -> list[float]:
        """Snapshot of the current window."""
        return list(self._window)

    # ── Internal ─────────────────────────────────────────────────────

    def _detect_change(self) -> ChangePoint | None:
        """Test all valid binary splits. Return the strongest change, if any."""
        w = list(self._window)
        n = len(w)
        best: ChangePoint | None = None

        # Pre-compute prefix sums for O(n) scan
        prefix = [0.0] * (n + 1)
        for i, v in enumerate(w):
            prefix[i + 1] = prefix[i] + v

        total = prefix[n]

        # Test recent windows of size k vs older window of size n-k
        for k in range(self.min_window, n - self.min_window + 1):
            n0 = k               # recent
            n1 = n - k           # older
            sum0 = prefix[n] - prefix[n - k]  # sum of last k elements
            sum1 = total - sum0

            m0 = sum0 / n0
            m1 = sum1 / n1
            eps = self._epsilon(n0, n1, n)
            diff = abs(m0 - m1)

            if diff >= eps:
                strength = min(1.0, diff / max(eps * 2.0, 1e-10))
                cp = ChangePoint(
                    position=n - k,
                    delta_mean=diff,
                    epsilon=eps,
                    strength=strength,
                )
                if best is None or strength > best.strength:
                    best = cp
                # Stop at first detection (oldest cut point that clears threshold)
                break

        return best

    def _epsilon(self, n0: int, n1: int, n: int) -> float:
        """ADWIN detection threshold with Bonferroni correction."""
        if n < 2:
            return float("inf")
        m = 1.0 / n0 + 1.0 / n1
        dd = math.log(4.0 * math.log(n + 2) / self.delta)
        return math.sqrt(0.5 * m * dd)

    def _cut_window(self, position: int) -> None:
        """Keep only elements from position onward (the recent portion)."""
        keep = list(self._window)[position:]
        self._window = deque(keep, maxlen=self.max_window)
