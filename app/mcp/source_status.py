"""SourceStatus — contract every MCP tool result must satisfy.

Every data source in the scatter_gather pipeline attaches a SourceStatus to its
result so the frontend can display provenance and so the claim verifier can
penalise answers that rely on stale or unavailable data.

Coverage levels:
  full        — all data fields present, freshness within threshold
  partial     — some fields missing or slightly stale
  unavailable — fetch failed; confidence = 0
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Per-source freshness thresholds in seconds.
# A source is "stale" when data age exceeds this; confidence degrades linearly.
FRESHNESS_THRESHOLDS: dict[str, int] = {
    "AEMO_DISPATCH_PRICE":       300,   # one dispatch cycle
    "AEMO_PREDISPATCH":          600,   # two dispatch cycles
    "AEMO_MARKET_NOTICES":       120,   # notice latency SLA
    "AEMO_ARCHIVE":             7200,   # hourly backfill
    "HIPPOGRAPH_ANALOGS":        300,   # one dispatch cycle
    "LNN_FORECAST":              300,
    "LIVE_QUANTILE_FORECAST":    600,
    "NEM_NEWS_RSS":              600,
    "WEATHER_CONSENSUS":         900,   # 15 min
}
_DEFAULT_THRESHOLD = 600


@dataclass
class SourceStatus:
    """Provenance contract attached to every data-source result."""

    source: str
    valid_time: str                   # ISO: timestamp of the underlying data
    system_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_ref: str = ""                 # URL or internal identifier for audit
    freshness_s: float = 0.0          # data age at query time in seconds
    confidence: float = 1.0           # 0–1 quality score
    coverage_status: str = "full"     # "full" | "partial" | "unavailable"
    error: str | None = None
    latency_ms: float = 0.0           # round-trip fetch time

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def unavailable(
        cls,
        source: str,
        error: str,
        latency_ms: float = 0.0,
    ) -> "SourceStatus":
        return cls(
            source=source,
            valid_time=datetime.now(timezone.utc).isoformat(),
            confidence=0.0,
            coverage_status="unavailable",
            error=str(error)[:200],
            latency_ms=round(latency_ms, 1),
        )

    @classmethod
    def from_data(
        cls,
        source: str,
        data_valid_time: datetime | None,
        raw_ref: str = "",
        latency_ms: float = 0.0,
        partial: bool = False,
        extra_confidence: float = 1.0,
    ) -> "SourceStatus":
        now = datetime.now(timezone.utc)
        vt = data_valid_time if data_valid_time else now
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=timezone.utc)
        freshness = (now - vt).total_seconds()
        threshold = FRESHNESS_THRESHOLDS.get(source, _DEFAULT_THRESHOLD)
        freshness_ratio = min(freshness / max(threshold, 1), 1.0)
        confidence = round(max(0.0, (1.0 - freshness_ratio) * extra_confidence), 3)
        coverage = "partial" if partial else ("full" if confidence > 0.2 else "partial")
        return cls(
            source=source,
            valid_time=vt.isoformat(),
            system_time=now.isoformat(),
            raw_ref=raw_ref[:300],
            freshness_s=round(freshness, 1),
            confidence=confidence,
            coverage_status=coverage,
            error=None,
            latency_ms=round(latency_ms, 1),
        )

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "valid_time": self.valid_time,
            "system_time": self.system_time,
            "raw_ref": self.raw_ref,
            "freshness_s": self.freshness_s,
            "confidence": self.confidence,
            "coverage_status": self.coverage_status,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def freshness_label(self) -> str:
        """Human label for UI badges: fresh / stale / unavailable."""
        if self.coverage_status == "unavailable":
            return "unavailable"
        threshold = FRESHNESS_THRESHOLDS.get(self.source, _DEFAULT_THRESHOLD)
        if self.freshness_s <= threshold * 0.5:
            return "fresh"
        if self.freshness_s <= threshold:
            return "stale"
        return "stale"


class SourceTimer:
    """Context manager that measures wall-clock latency for source fetches."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "SourceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
