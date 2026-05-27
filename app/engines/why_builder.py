"""Why-builder helpers — enrich evidence tiers from behavioral and historical signals.

Takes evidence tiers already produced by the operator or live-data pipeline and
upgrades them when participant profiling or other deterministic signals support
a stronger classification.

Rules:
  - Only upgrades, never downgrades.  A 'confirmed' tier set by the operator is preserved.
  - Does not assert intent — only that the evidence is consistent with the tier.
"""
from __future__ import annotations

from typing import Any

_TIER_RANK: dict[str, int] = {
    "confirmed": 3,
    "supported": 2,
    "plausible": 1,
    "unconfirmed": 0,
}
_RANK_TIER: dict[int, str] = {v: k for k, v in _TIER_RANK.items()}


async def elevate_rebid_tier_if_habitual(
    session: Any,
    duid: str,
    region: str,
    rebid_evidence_tier: str | None,
    window_days: int = 30,
) -> str | None:
    """Elevate rebid_evidence_tier to 'confirmed' when the participant is a habitual rebidder.

    Returns the (possibly unchanged) tier string. Never downgrades a tier
    that is already 'confirmed'. Returns the original tier unchanged when
    the participant profile shows no data or a non-habitual behavioural tier.
    """
    from app.engines.participant_profiler import profile_participant

    current_rank = _TIER_RANK.get(rebid_evidence_tier or "unconfirmed", 0)
    if current_rank >= _TIER_RANK["confirmed"]:
        return rebid_evidence_tier

    profile = await profile_participant(session, duid, region, window_days=window_days)
    if not profile.data_available or profile.behavioral_tier != "habitual":
        return rebid_evidence_tier

    new_rank = max(current_rank, _TIER_RANK["confirmed"])
    return _RANK_TIER[new_rank]
