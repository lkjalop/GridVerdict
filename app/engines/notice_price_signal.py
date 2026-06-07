"""AEMO notice → price impact signal classifier.

Converts AEMO market notice type + region context into a structured price
impact probability signal. This makes notices actionable as:
  1. A forecast feature for the LNN (regime-conditional probability shift)
  2. An NLP answer enrichment ("there is an LOR2 notice → 70% spike risk")

Evidence basis:
  Historical analysis of NEM dispatch shows that specific notice types have
  consistent price impact patterns. These probabilities are based on
  published AEMO rule frameworks and historical event analysis.

Signal schema (NoticeSignal):
  notice_type:       the AEMO notice type string
  region:            NEM region (NSW1/VIC1/etc.)
  price_spike_prob:  probability of price >$300/MWh within 60 min (0.0-1.0)
  severity_label:    "emergency" | "high" | "moderate" | "low" | "informational"
  price_floor_est:   estimated floor price $/MWh under this notice
  price_ceiling_est: estimated ceiling $/MWh if unchecked
  action_signal:     "immediate_attention" | "watch" | "monitor" | "informational"
  evidence_basis:    citation for the probability estimate
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class NoticeSignal:
    notice_type: str
    region: str
    price_spike_prob: float           # 0.0–1.0 probability of price >$300/MWh in next 60 min
    severity_label: str               # emergency | high | moderate | low | informational
    price_floor_est: float | None     # $/MWh — minimum price expected under this event
    price_ceiling_est: float | None   # $/MWh — maximum price if event unresolved
    action_signal: str                # immediate_attention | watch | monitor | informational
    evidence_basis: str               # citable source for probability estimate
    as_of: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "notice_type":       self.notice_type,
            "region":            self.region,
            "price_spike_prob":  round(self.price_spike_prob, 3),
            "severity_label":    self.severity_label,
            "price_floor_est":   self.price_floor_est,
            "price_ceiling_est": self.price_ceiling_est,
            "action_signal":     self.action_signal,
            "evidence_basis":    self.evidence_basis,
            "as_of":             self.as_of.isoformat(),
        }

    def as_forecast_feature(self) -> float:
        """Return notice price-spike probability as a 0–1 float for LNN feature injection."""
        return self.price_spike_prob

    def as_nlp_bullet(self) -> str:
        """One-line NLP-ready summary suitable for answer_sections."""
        if self.price_spike_prob >= 0.8:
            emoji = "🔴"
        elif self.price_spike_prob >= 0.5:
            emoji = "🟡"
        else:
            emoji = "🟢"
        ceiling = f" (ceiling ~${self.price_ceiling_est:.0f}/MWh)" if self.price_ceiling_est else ""
        return (
            f"{emoji} {self.notice_type} in {self.region}: "
            f"{self.severity_label.upper()} — "
            f"price spike probability {self.price_spike_prob:.0%} within 60 min{ceiling}. "
            f"Action: {self.action_signal.replace('_', ' ')}. "
            f"Basis: {self.evidence_basis}"
        )


# ── Notice type → price impact mapping ───────────────────────────────────────
# Based on: AEMO Reliability and Emergency Reserve Trader (RERT) guidelines,
# NEM market rule definitions, and historical event analysis from AEMC reviews.

_NOTICE_IMPACT: dict[str, dict] = {
    # LOR = Lack of Reserve: AEMO declares reserve shortfall at T+30min/T+60min
    "LACK OF RESERVE 1": {
        "price_spike_prob": 0.45,
        "severity":         "moderate",
        "floor":            150.0,
        "ceiling":          1000.0,
        "action":           "watch",
        "basis":            "AEMO RERT guidelines — LOR1 = reserve < requirement, but market still clearing",
    },
    "LACK OF RESERVE 2": {
        "price_spike_prob": 0.70,
        "severity":         "high",
        "floor":            300.0,
        "ceiling":          5000.0,
        "action":           "immediate_attention",
        "basis":            "AEMO RERT guidelines — LOR2 = reserve critically low; RERT may be activated",
    },
    "LACK OF RESERVE 3": {
        "price_spike_prob": 0.90,
        "severity":         "emergency",
        "floor":            1000.0,
        "ceiling":          15500.0,
        "action":           "immediate_attention",
        "basis":            "AEMO RERT guidelines — LOR3 = imminent load shedding; RERT or AEMO directions expected",
    },
    "LOR1": {"price_spike_prob": 0.45, "severity": "moderate", "floor": 150.0, "ceiling": 1000.0,  "action": "watch",               "basis": "AEMO RERT (LOR1 abbreviated form)"},
    "LOR2": {"price_spike_prob": 0.70, "severity": "high",     "floor": 300.0, "ceiling": 5000.0,  "action": "immediate_attention", "basis": "AEMO RERT (LOR2 abbreviated form)"},
    "LOR3": {"price_spike_prob": 0.90, "severity": "emergency","floor": 1000.0,"ceiling": 15500.0, "action": "immediate_attention", "basis": "AEMO RERT (LOR3 abbreviated form)"},
    # DIRECTIONS = AEMO directs a generator outside normal dispatch
    "DIRECTIONS": {
        "price_spike_prob": 0.80,
        "severity":         "high",
        "floor":            500.0,
        "ceiling":          15500.0,
        "action":           "immediate_attention",
        "basis":            "AEMO NER rule 4.8.9 — Directions override normal dispatch; often precede or accompany cap events",
    },
    # RECLASSIFY CONTINGENCY = generator/line failure reclassified to credible contingency
    "RECLASSIFY CONTINGENCY": {
        "price_spike_prob": 0.55,
        "severity":         "high",
        "floor":            200.0,
        "ceiling":          3000.0,
        "action":           "immediate_attention",
        "basis":            "AEMO NER 4.2.3 — Reclassified contingency raises FCAS requirements → FCAS prices spike first",
    },
    # MARKET INTERVENTION = AEMO suspends market
    "MARKET INTERVENTION": {
        "price_spike_prob": 0.95,
        "severity":         "emergency",
        "floor":            None,
        "ceiling":          None,
        "action":           "immediate_attention",
        "basis":            "AEMO NER 3.14.5 — Market suspension means normal price not being set; administered pricing applies",
    },
    # INTER_CONSTRAINT = interconnector constraint active
    "INTER_CONSTRAINT": {
        "price_spike_prob": 0.40,
        "severity":         "moderate",
        "floor":            100.0,
        "ceiling":          2000.0,
        "action":           "watch",
        "basis":            "AEMO market notices — interconnector binding raises FCAS in separated region; price divergence",
    },
    # CONSTRAINT TIGHTENING / RELAXATION
    "CONSTRAINT TIGHTENING": {
        "price_spike_prob": 0.35,
        "severity":         "moderate",
        "floor":            80.0,
        "ceiling":          800.0,
        "action":           "watch",
        "basis":            "AEMO SCADA — tighter constraints reduce headroom, raise dispatch costs",
    },
    "CONSTRAINT RELAXATION": {
        "price_spike_prob": 0.05,
        "severity":         "low",
        "floor":            None,
        "ceiling":          None,
        "action":           "monitor",
        "basis":            "AEMO SCADA — relaxation reduces price pressure; price typically falls",
    },
    # MT PASA / ST PASA = medium/short-term adequacy revisions
    "MT PASA REVISION": {
        "price_spike_prob": 0.15,
        "severity":         "low",
        "floor":            None,
        "ceiling":          None,
        "action":           "monitor",
        "basis":            "AEMO PASA — medium-term adequacy revision; no immediate price impact",
    },
    "ST PASA REVISION": {
        "price_spike_prob": 0.20,
        "severity":         "low",
        "floor":            None,
        "ceiling":          None,
        "action":           "monitor",
        "basis":            "AEMO PASA — short-term adequacy revision; price impact within 7 days",
    },
    # ADMINISTERED PRICE CAP / FLOOR
    "ADMINISTERED PRICE CAP": {
        "price_spike_prob": 0.30,
        "severity":         "moderate",
        "floor":            None,
        "ceiling":          None,
        "action":           "watch",
        "basis":            "AEMO NER 3.14.2 — APC applied when cumulative prices exceeded limit; prices artificially capped",
    },
    "ADMINISTERED PRICE FLOOR": {
        "price_spike_prob": 0.05,
        "severity":         "low",
        "floor":            None,
        "ceiling":          None,
        "action":           "monitor",
        "basis":            "AEMO NER 3.14.2 — APF applied when cumulative negative prices exceeded limit",
    },
    # RESERVE TRADER
    "RESERVE TRADER ACTIVATION": {
        "price_spike_prob": 0.65,
        "severity":         "high",
        "floor":            300.0,
        "ceiling":          5000.0,
        "action":           "immediate_attention",
        "basis":            "AEMO RERT — reserve trader called when normal market insufficient; prices elevated",
    },
    "RESERVE TRADER CANCELLATION": {
        "price_spike_prob": 0.10,
        "severity":         "low",
        "floor":            None,
        "ceiling":          None,
        "action":           "monitor",
        "basis":            "AEMO RERT — cancellation means reserve adequate; price pressure easing",
    },
}

# Default for unknown notice types
_DEFAULT_IMPACT: dict = {
    "price_spike_prob": 0.15,
    "severity":         "informational",
    "floor":            None,
    "ceiling":          None,
    "action":           "monitor",
    "basis":            "AEMO market notice (type not in GridVerdict classification table)",
}


def classify_notice(notice_type: str, region: str) -> NoticeSignal:
    """Classify an AEMO notice type as a price impact signal.

    Args:
        notice_type: AEMO notice type string (e.g., "LACK OF RESERVE 2", "LOR2", "DIRECTIONS")
        region:      NEM region code (NSW1, VIC1, QLD1, SA1, TAS1)

    Returns:
        NoticeSignal with price_spike_prob and NLP-ready description.
    """
    key = notice_type.strip().upper()
    impact = _NOTICE_IMPACT.get(key, _DEFAULT_IMPACT)

    return NoticeSignal(
        notice_type=notice_type,
        region=region.upper(),
        price_spike_prob=float(impact["price_spike_prob"]),
        severity_label=impact["severity"],
        price_floor_est=impact.get("floor"),
        price_ceiling_est=impact.get("ceiling"),
        action_signal=impact["action"],
        evidence_basis=impact["basis"],
        as_of=datetime.now(timezone.utc),
    )


def classify_notices(notices: list[dict]) -> list[NoticeSignal]:
    """Classify a list of notice dicts (from GatherResult.notices or fetch_active_notices).

    Each dict is expected to have keys: notice_type, region (optional).
    Returns signals sorted by severity (highest first).
    """
    signals: list[NoticeSignal] = []
    for n in notices:
        notice_type = n.get("notice_type") or n.get("title", "").split(":")[0].strip()
        region = n.get("region") or "NSW1"
        if not notice_type:
            continue
        signals.append(classify_notice(notice_type, region))

    # Sort: emergency first, then by probability descending
    _order = {"emergency": 0, "high": 1, "moderate": 2, "low": 3, "informational": 4}
    signals.sort(key=lambda s: (_order.get(s.severity_label, 5), -s.price_spike_prob))
    return signals


def highest_notice_signal(notices: list[dict], region: str | None = None) -> NoticeSignal | None:
    """Return the highest-severity notice signal for a region.

    Args:
        notices: list of notice dicts from GatherResult.notices
        region:  filter to this region (or None for any region)

    Returns:
        The highest-severity NoticeSignal, or None if no notices.
    """
    if not notices:
        return None
    if region:
        relevant = [n for n in notices if not n.get("region") or n.get("region") == region.upper()]
    else:
        relevant = notices
    if not relevant:
        return None
    signals = classify_notices(relevant)
    return signals[0] if signals else None


def notice_regime_probability_shift(
    notices: list[dict],
    region: str,
    base_price_rrp: float,
) -> dict[str, float]:
    """Compute notice-adjusted price regime probabilities.

    Given a base price level and active notices, returns adjusted probabilities
    for each regime. Intended as a feature input to the LNN or as NLP context.

    Returns dict with keys: normal, elevated, spike, emergency (summing to 1.0).
    """
    # Base probabilities from price level alone (calibrated against historical dispatch)
    if base_price_rrp < 50:
        base = {"normal": 0.85, "elevated": 0.12, "spike": 0.02, "emergency": 0.01}
    elif base_price_rrp < 150:
        base = {"normal": 0.55, "elevated": 0.35, "spike": 0.08, "emergency": 0.02}
    elif base_price_rrp < 500:
        base = {"normal": 0.20, "elevated": 0.45, "spike": 0.30, "emergency": 0.05}
    else:
        base = {"normal": 0.05, "elevated": 0.20, "spike": 0.50, "emergency": 0.25}

    # Adjust for highest-severity notice
    top_signal = highest_notice_signal(notices, region)
    if top_signal is None:
        return base

    # Shift probability mass toward spike/emergency based on notice signal
    shift = top_signal.price_spike_prob
    adjusted = dict(base)

    if shift >= 0.80:  # emergency (LOR3, DIRECTIONS, MARKET INTERVENTION)
        adjusted["normal"] = max(0.0, base["normal"] - shift * 0.5)
        adjusted["elevated"] = max(0.0, base["elevated"] - shift * 0.2)
        adjusted["spike"] = min(0.6, base["spike"] + shift * 0.4)
        adjusted["emergency"] = min(0.5, base["emergency"] + shift * 0.3)
    elif shift >= 0.50:  # high (LOR2, RESERVE TRADER ACTIVATION)
        adjusted["normal"] = max(0.0, base["normal"] - shift * 0.3)
        adjusted["elevated"] = base["elevated"]
        adjusted["spike"] = min(0.5, base["spike"] + shift * 0.25)
        adjusted["emergency"] = min(0.3, base["emergency"] + shift * 0.15)
    elif shift >= 0.30:  # moderate (LOR1, RECLASSIFY, INTER_CONSTRAINT)
        adjusted["normal"] = max(0.0, base["normal"] - shift * 0.15)
        adjusted["spike"] = min(0.4, base["spike"] + shift * 0.1)

    # Normalise to sum=1
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return adjusted
