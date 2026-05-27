"""Deterministic prose formatter for commentary events.

Produces the headline string and structured contributing_factors list
from a MaterialChange + WhyOutput pair. No LLM calls — all logic is
deterministic so commentary generation never blocks on an API.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.why_builder import WhyOutput
    from app.engines.commentary.detector import ChangeType, MaterialChange


def format_headline(change: "MaterialChange", why: "WhyOutput") -> str:
    """One-sentence headline for the commentary card."""
    from app.engines.commentary.detector import ChangeType

    ct = change.change_type
    r = change.region
    prev = change.prev_value
    curr = change.curr_value

    if ct == ChangeType.PRICE_SPIKE:
        delta = abs((curr or 0) - (prev or 0))
        verb = "spiked" if delta > 200 else "rose"
        return (
            f"{r} price {verb} to ${curr:.0f}/MWh "
            f"(+${delta:.0f} from ${prev:.0f})"
        )

    if ct == ChangeType.NEGATIVE_PRICE:
        return f"{r} price went negative: ${curr:.0f}/MWh (was ${prev:.0f}/MWh)"

    if ct == ChangeType.PRICE_NORMALISED:
        return (
            f"{r} price normalised to ${curr:.0f}/MWh "
            f"(was ${prev:.0f}/MWh)"
        )

    if ct == ChangeType.PRICE_MOVE_LARGE:
        delta = abs((curr or 0) - (prev or 0))
        direction = "up" if (curr or 0) > (prev or 0) else "down"
        return (
            f"{r} price moved {direction} ${delta:.0f}/MWh "
            f"(${prev:.0f} → ${curr:.0f}/MWh)"
        )

    if ct == ChangeType.PRICE_REGIME_CHANGE:
        return change.description

    if ct == ChangeType.HEADROOM_TIGHTENED:
        return (
            f"{r} supply headroom tightened to {curr:.0f}MW "
            f"(was {prev:.0f}MW)"
        )

    if ct == ChangeType.HEADROOM_RECOVERED:
        return (
            f"{r} supply headroom recovered to {curr:.0f}MW "
            f"(was {prev:.0f}MW)"
        )

    if ct == ChangeType.NOTICE_ADDED:
        return f"New AEMO market notice issued for {r}"

    if ct == ChangeType.FORECAST_RISK_INCREASED:
        return (
            f"{r} spike probability rose to {(curr or 0) * 100:.0f}% "
            f"(was {(prev or 0) * 100:.0f}%)"
        )

    if ct == ChangeType.FORECAST_RISK_DECREASED:
        return (
            f"{r} spike probability fell to {(curr or 0) * 100:.0f}% "
            f"(was {(prev or 0) * 100:.0f}%)"
        )

    # Sprint R: new ChangeType headlines
    if ct == ChangeType.CONSTRAINT_ACTIVE:
        return f"{r} new binding constraint active — dispatch price directly affected"

    if ct == ChangeType.WEATHER_PRESSURE_BUILDING:
        score = curr or 0
        label = "extreme heat" if score >= 0.9 else "heat or wind drought"
        return f"{r} weather pressure building ({label}) — demand and renewable risk elevated"

    if ct == ChangeType.DATA_STALE:
        staleness = int(curr or 0)
        return (
            f"{r} dispatch data feed stale — {staleness}s since last AEMO update; "
            f"analysis based on last known values"
        )

    if ct == ChangeType.DATA_RECOVERED:
        return f"{r} dispatch data feed recovered — live analysis resumed"

    if ct == ChangeType.WATCH_CLOSED:
        return (
            f"{r} spike watch closed — price normalised to ${curr:.0f}/MWh "
            f"from ${prev:.0f}/MWh; normal operations can resume"
        )

    return change.description


def format_factors(why: "WhyOutput") -> list[dict[str, Any]]:
    """Convert claim_map to structured contributing factors for the card.

    Only includes present claims. Sorted by tier (confirmed first).
    """
    tier_order = {"confirmed": 0, "supported": 1, "plausible": 2, "unconfirmed": 3}
    factors = []
    for item in why.claim_map:
        if not item.present:
            continue
        factors.append({
            "label": item.label,
            "tier": item.tier.value,
            "confidence": item.confidence,
            "claim_type": item.claim_type.value,
            "evidence_ref_ids": item.evidence_ref_ids,
            "note": item.note,
        })
    factors.sort(key=lambda f: tier_order.get(f["tier"], 9))
    return factors
