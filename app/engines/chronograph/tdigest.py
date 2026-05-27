"""t-digest streaming quantile estimator.

Based on: Dunning & Ertl (2019) "Computing Extremely Accurate Quantiles Using t-Digests"

Maintains a compact centroid summary of a data stream that enables accurate
quantile queries, especially at the tails (p1, p5, p95, p99).

Primary use: compute where the current NEM price sits in the recent 30-day
distribution so the regime classifier has a continuous signal alongside
the fixed threshold classification.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
Pure Python, no external ML dependencies.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


@dataclass(order=True)
class _Centroid:
    """A compressed cluster of observations."""
    mean: float
    count: int


class TDigest:
    """Streaming quantile estimator using centroid compression.

    Accumulates new values into a buffer, then periodically compresses them
    into a sorted list of (mean, count) centroids using a scale function that
    allocates more centroids to the tails.

    Parameters
    ----------
    compression : float
        Target number of centroids. Higher = more accurate, more memory.
        100–200 is typical; 100 gives sub-1% quantile error at the tails.
    max_buffer : int
        Buffer size before a compress pass is triggered.
    """

    def __init__(self, compression: float = 100.0, max_buffer: int = 500) -> None:
        self.compression = compression
        self._max_buffer = max_buffer
        self._centroids: list[_Centroid] = []
        self._buffer: list[float] = []
        self._n: int = 0          # total observations (including compressed)

    # ── Public API ───────────────────────────────────────────────────

    def update(self, value: float, count: int = 1) -> None:
        """Add a new observation to the digest."""
        for _ in range(count):
            self._buffer.append(value)
        self._n += count
        if len(self._buffer) >= self._max_buffer:
            self._compress()

    def cdf(self, value: float) -> float:
        """Return the fraction of observations ≤ value (0.0–1.0).

        This is the quantile rank of the given value in the stream seen so far.
        A result of 0.90 means 90 % of observed values were ≤ value.
        """
        self._flush()
        if not self._centroids:
            return 0.5

        total = sum(c.count for c in self._centroids)
        if total == 0:
            return 0.5

        cumulative = 0.0
        for i, c in enumerate(self._centroids):
            if value < c.mean:
                # Interpolate between c[i-1] and c[i]
                if i == 0:
                    return 0.0
                prev = self._centroids[i - 1]
                if prev.mean == c.mean:
                    return (cumulative + c.count / 2) / total
                t = (value - prev.mean) / (c.mean - prev.mean)
                return (cumulative - prev.count / 2 + t * (prev.count / 2 + c.count / 2)) / total
            cumulative += c.count

        return 1.0

    def quantile(self, q: float) -> float:
        """Return the value at quantile q (0.0–1.0).

        quantile(0.5) is the median, quantile(0.95) is the 95th percentile.
        """
        q = max(0.0, min(1.0, q))
        self._flush()
        if not self._centroids:
            return 0.0

        total = sum(c.count for c in self._centroids)
        target = q * total
        cumulative = 0.0

        for i, c in enumerate(self._centroids):
            # Each centroid contributes count/2 weight before its mean and count/2 after
            half = c.count / 2.0
            if cumulative + c.count >= target:
                # Interpolate
                if i == 0:
                    return c.mean
                delta = target - cumulative
                t = delta / c.count
                prev = self._centroids[i - 1]
                return prev.mean + t * (c.mean - prev.mean)
            cumulative += c.count

        return self._centroids[-1].mean

    @property
    def count(self) -> int:
        """Total number of observations ingested."""
        return self._n

    @property
    def is_warm(self) -> bool:
        """True once enough observations exist for reliable quantile estimates."""
        return self._n >= 30

    # ── Internal ─────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Compress any pending buffer values."""
        if self._buffer:
            self._compress()

    def _compress(self) -> None:
        """Merge buffer + existing centroids into a new compressed centroid list."""
        # Flatten existing centroids back to individual values approximation
        # (use mean as representative) and sort with new buffer
        all_vals: list[float] = list(self._buffer)
        for c in self._centroids:
            all_vals.extend([c.mean] * c.count)

        self._buffer = []
        all_vals.sort()
        n = len(all_vals)

        if n == 0:
            self._centroids = []
            return

        new_centroids: list[_Centroid] = []
        k = int(self.compression)
        target_count = max(1, n // k)

        i = 0
        while i < n:
            # Determine group size using t-digest scale function
            # Scale: more centroids near tails (q=0 or q=1)
            q_mid = (i + target_count / 2) / n
            group_limit = self._max_group_size(q_mid, n)
            j = min(n, i + max(1, group_limit))
            group = all_vals[i:j]
            centroid = _Centroid(
                mean=sum(group) / len(group),
                count=len(group),
            )
            new_centroids.append(centroid)
            i = j

        self._centroids = new_centroids

    def _max_group_size(self, q: float, n: int) -> int:
        """t-digest scale function: smaller groups at tails, larger in the middle.

        Uses the k1 scale function: k(q) = (compression / (2π)) × arcsin(2q - 1)
        Group size is proportional to dk/dq = compression / (π √(q(1-q))).
        """
        q = max(0.001, min(0.999, q))
        k = self.compression
        # Scale group size: inversely proportional to density at tails
        density = math.sqrt(q * (1.0 - q))
        size = max(1, int(2.0 * density * n / k))
        return size
