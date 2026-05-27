"""TemporalRAG — data types.

Bitemporal retrieval schema. Every retrieved document carries both a
`valid_time` (when the market event occurred) and a `system_time`
(when it was recorded into the system). The no-leakage invariant is:

    doc.system_time <= query.system_time_at_query

This prevents future information from being visible in historical
reconstructions or backtest scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Recognised source types and their credibility rank (higher = more trusted)
SOURCE_CREDIBILITY: dict[str, float] = {
    "market_events": 1.0,   # AEMO metered dispatch prices — authoritative
    "notice": 0.90,         # AEMO official market notices
    "trace": 0.80,          # GridVerdict query trace (derived, not primary)
    "analog": 0.75,         # HippoGraph PPR historical analogs
    "news": 0.50,           # RSS public commentary — lowest trust
}


@dataclass
class TemporalQuery:
    """Specifies the retrieval scope for a bitemporal query.

    Attributes
    ----------
    valid_time_from : datetime
        Start of the event window (inclusive). Timezone-aware.
    valid_time_to : datetime
        End of the event window (inclusive). Timezone-aware.
    system_time_at_query : datetime
        The as-of timestamp for no-leakage enforcement. Only documents
        with system_time <= this value are returned.
    region : str | None
        If set, restricts results to this NEM region.
    source_types : list[str]
        Which sources to query. Defaults to all available sources.
    max_docs : int
        Maximum number of documents to return after ranking.
    """
    valid_time_from: datetime
    valid_time_to: datetime
    system_time_at_query: datetime
    region: str | None = None
    source_types: list[str] = field(
        default_factory=lambda: ["market_events", "notice", "trace", "analog", "news"]
    )
    max_docs: int = 20

    def __post_init__(self) -> None:
        if self.valid_time_from.tzinfo is None:
            self.valid_time_from = self.valid_time_from.replace(tzinfo=timezone.utc)
        if self.valid_time_to.tzinfo is None:
            self.valid_time_to = self.valid_time_to.replace(tzinfo=timezone.utc)
        if self.system_time_at_query.tzinfo is None:
            self.system_time_at_query = self.system_time_at_query.replace(tzinfo=timezone.utc)

    @property
    def midpoint(self) -> datetime:
        """Midpoint of the valid-time window — used as anchor for temporal scoring."""
        mid_ts = (self.valid_time_from.timestamp() + self.valid_time_to.timestamp()) / 2
        return datetime.fromtimestamp(mid_ts, tz=timezone.utc)

    @property
    def window_seconds(self) -> float:
        """Duration of the valid-time window in seconds."""
        return max(
            (self.valid_time_to - self.valid_time_from).total_seconds(),
            1.0,
        )


@dataclass
class TemporalDoc:
    """A single retrieved evidence document.

    Attributes
    ----------
    doc_id : str
        Unique identifier (DB primary key or constructed ID).
    source_type : str
        One of "market_events", "notice", "trace", "analog", "news".
    valid_time : datetime
        When the underlying event occurred.
    system_time : datetime
        When the record was written to the system (for leakage enforcement).
    content : dict[str, Any]
        The raw evidence payload — schema depends on source_type.
    relevance_score : float
        0–1 composite score (temporal proximity × source credibility).
    citation : str
        Human-readable attribution string (shown in LLM context).
    """
    doc_id: str
    source_type: str
    valid_time: datetime
    system_time: datetime
    content: dict[str, Any]
    relevance_score: float = 0.0
    citation: str = ""

    def passes_leakage_fence(self, system_time_at_query: datetime) -> bool:
        """True if this document was available at the query point in time."""
        return self.system_time <= system_time_at_query


@dataclass
class RetrievalBundle:
    """Result of a TemporalRAG retrieval pass.

    Attributes
    ----------
    query : TemporalQuery
        The originating query (for reproducibility / audit).
    docs : list[TemporalDoc]
        Ranked documents, no more than query.max_docs, all passing the
        no-leakage fence.
    source_counts : dict[str, int]
        How many documents came from each source type.
    leakage_filtered : int
        Number of candidate documents removed by the system-time fence.
    elapsed_ms : float
        Wall-clock retrieval time in milliseconds.
    """
    query: TemporalQuery
    docs: list[TemporalDoc]
    source_counts: dict[str, int]
    leakage_filtered: int
    elapsed_ms: float

    @property
    def total_docs(self) -> int:
        return len(self.docs)

    @property
    def has_market_data(self) -> bool:
        return self.source_counts.get("market_events", 0) > 0

    @property
    def has_analogs(self) -> bool:
        return self.source_counts.get("analog", 0) > 0

    def citations(self) -> list[str]:
        """Return all citation strings in relevance order."""
        return [d.citation for d in self.docs if d.citation]


def score_temporal_proximity(doc_valid_time: datetime, query: TemporalQuery) -> float:
    """Exponential decay score based on distance from window midpoint.

    Returns 1.0 when the document's valid_time equals the midpoint,
    decaying toward 0 as distance increases. Half-life = window_seconds / 2.
    """
    distance_s = abs((doc_valid_time - query.midpoint).total_seconds())
    half_life = query.window_seconds / 2.0
    return math.exp(-distance_s / max(half_life, 1.0))
