"""News / market-notice cross-correlation (second MCP).

Skin module (energy-specific sources). When ChronoGraph flags a price regime
shift or spike, this correlates it with credible explanatory events and surfaces
CITED candidate causes — correlation, not proven causation. If nothing credible
is found it says so honestly rather than inventing a reason.

Source credibility tiers:
  1 = AEMO Market Notices (gold — official trips, outages, LOR, interventions)
  2 = AER + reputable wire/energy press
  (social media / forums are never used as an explanation source)

The MCP client itself is injected (a thin wrapper over the AEMO Market Notices
feed and a credible-news MCP). This module is the correlation logic on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, Sequence

from ..types import NewsItem


class NewsMCPClient(Protocol):
    """Anything that can fetch credible explanatory items for a time/region."""

    def fetch(self, start: datetime, end: datetime, region: str | None) -> Sequence[NewsItem]:
        ...


@dataclass
class CandidateExplanation:
    item: NewsItem
    minutes_before_event: float
    relevance: float          # 0..1 region + keyword match
    score: float              # combined ranking score


@dataclass
class CorrelationResult:
    """What the why-engine consumes. Either ranked candidates or an honest null."""
    explained: bool
    candidates: list[CandidateExplanation]
    note: str

    def top_plain_english(self) -> str:
        if not self.explained:
            return self.note
        c = self.candidates[0]
        tier = "AEMO Market Notice" if c.item.credibility_tier == 1 else c.item.source
        return (
            f"A {tier} at {c.item.timestamp:%H:%M} reported: {c.item.title}. "
            f"This occurred ~{c.minutes_before_event:.0f} min before the price move "
            f"and correlates with it (correlation, not confirmed cause)."
        )


_SPIKE_KEYWORDS = (
    "trip", "outage", "lack of reserve", "lor", "intervention", "constraint",
    "forced", "derating", "transmission", "fault", "bushfire", "heatwave",
)


def _relevance(item: NewsItem, region: str | None) -> float:
    text = f"{item.title} {item.summary}".lower()
    kw = sum(1 for k in _SPIKE_KEYWORDS if k in text) / len(_SPIKE_KEYWORDS)
    region_match = 1.0 if (region and item.region and region == item.region) else 0.5
    return min(1.0, 0.5 * region_match + 0.5 * min(1.0, kw * 3))


def correlate_price_event(
    client: NewsMCPClient,
    event_time: datetime,
    region: str,
    lookback_min: int = 60,
    max_candidates: int = 3,
) -> CorrelationResult:
    """Find credible explanations for a price event at event_time/region.

    Ranks by source credibility (tier 1 first), temporal proximity (closer before
    the event is stronger), and keyword/region relevance. Returns an honest null
    when nothing credible is found.
    """
    start = event_time - timedelta(minutes=lookback_min)
    items = [i for i in client.fetch(start, event_time, region)
             if i.credibility_tier in (1, 2) and i.timestamp <= event_time]

    if not items:
        return CorrelationResult(
            explained=False,
            candidates=[],
            note="No public explanation found for this move yet — treat any "
                 "recommendation as lower-confidence until the driver is known.",
        )

    cands: list[CandidateExplanation] = []
    for it in items:
        mins_before = (event_time - it.timestamp).total_seconds() / 60.0
        proximity = max(0.0, 1.0 - mins_before / lookback_min)
        rel = _relevance(it, region)
        tier_w = 1.0 if it.credibility_tier == 1 else 0.6
        score = tier_w * (0.5 * proximity + 0.5 * rel)
        cands.append(CandidateExplanation(it, mins_before, rel, score))

    cands.sort(key=lambda c: c.score, reverse=True)
    cands = cands[:max_candidates]
    return CorrelationResult(
        explained=True,
        candidates=cands,
        note=f"{len(cands)} credible candidate explanation(s); ranked, cited, correlation only.",
    )
