"""NEM (National Electricity Market) domain adapter.

Converts AEMO-specific IngestEvents into framework-generic types.
This is the ONLY file in the codebase that knows about NEM/AEMO concepts.
core/ and engines/ import from app.core.interfaces only.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.interfaces import (
    DomainAdapter,
    EvidenceRef,
    FeatureVector,
    IngestEvent,
    RegimeState,
)

NEM_REGIME_THRESHOLDS: dict[str, dict[str, float]] = {
    "NSW1": {"elevated": 100.0, "spike": 300.0, "extreme": 1000.0},
    "VIC1": {"elevated": 100.0, "spike": 300.0, "extreme": 1000.0},
    "QLD1": {"elevated": 100.0, "spike": 300.0, "extreme": 1000.0},
    "SA1":  {"elevated": 120.0, "spike": 400.0, "extreme": 1500.0},
    "TAS1": {"elevated":  90.0, "spike": 250.0, "extreme":  800.0},
}

_REGIME_THRESHOLDS = NEM_REGIME_THRESHOLDS  # internal alias

_REGIONS = list(_REGIME_THRESHOLDS.keys())


class NEMAdapter(DomainAdapter):
    """Plugs the NEM vertical into the GridVerdict framework."""

    def domain_name(self) -> str:
        return "nem"

    def to_feature_vector(self, event: IngestEvent) -> FeatureVector:
        md = event.metadata
        region = md.get("region", "NSW1")
        price = float(md.get("price_rrp", 0.0))
        demand = float(md.get("demand_mw", 0.0))
        avail = float(md.get("availability_mw", 0.0))
        headroom = max(avail - demand, 0.0)
        thresholds = _REGIME_THRESHOLDS.get(region, _REGIME_THRESHOLDS["NSW1"])

        return FeatureVector(
            event_id=f"fv-{uuid.uuid4().hex[:8]}",
            values={
                "price_rrp": price,
                "demand_mw": demand,
                "availability_mw": avail,
                "headroom_mw": headroom,
                "price_norm": price / thresholds["extreme"] if thresholds["extreme"] else 0.0,
                "headroom_ratio": headroom / demand if demand > 0 else 1.0,
            },
            categorical={
                "region": region,
                "regime": _classify_regime(price, region),
                "source": event.source,
            },
            valid_time=event.valid_time,
            tenant_id=event.tenant_id,
        )

    def to_evidence_ref(self, event: IngestEvent, field: str) -> EvidenceRef:
        md = event.metadata
        value_map = {
            "price_rrp": float(md.get("price_rrp", 0.0)),
            "demand_mw": float(md.get("demand_mw", 0.0)),
            "availability_mw": float(md.get("availability_mw", 0.0)),
        }
        return EvidenceRef(
            id=f"ev-{uuid.uuid4().hex[:8]}",
            source=event.source,
            region=md.get("region"),
            interval=event.valid_time,
            field=field,
            value=value_map.get(field, 0.0),
            raw_ref=event.raw_ref,
        )

    def decomposition_hints(self) -> dict[str, Any]:
        return {
            "regions": _REGIONS,
            "price_field": "price_rrp",
            "demand_field": "demand_mw",
            "availability_field": "availability_mw",
            "regime_labels": ["normal", "elevated", "spike", "extreme"],
            "intent_patterns": {
                "action_recommendation": [
                    "should i dispatch", "dispatch now", "should we bid",
                    "what action", "what should i do",
                ],
                "explanation": [
                    "why is price", "why is the price", "what is driving",
                    "explain the spike", "reason for",
                ],
                "retrospective": [
                    "what happened", "yesterday", "last week", "historical",
                    "what drove", "analogous events",
                ],
                "lookup": [
                    "current price", "what is the price", "current demand",
                    "market state", "live price",
                ],
                "forecast": [
                    "will price", "price forecast", "expected price",
                    "predispatch", "tomorrow",
                ],
            },
        }

    def why_template(self, regime: RegimeState, analog_count: int) -> str:
        label = regime.label
        if label == "extreme":
            template = (
                "The market is in an EXTREME price regime (confidence {conf:.0%}). "
                "Prices at this level occur in the top {quantile:.0%} of observed intervals. "
                "This regime began at {start}. "
            )
        elif label == "spike":
            template = (
                "The market is experiencing a price SPIKE (confidence {conf:.0%}). "
                "Current prices are in the top {quantile:.0%} of the 30-day distribution. "
            )
        elif label == "elevated":
            template = (
                "Prices are ELEVATED above normal operating range (confidence {conf:.0%}). "
                "Sitting at the {quantile:.0%} percentile of recent trading. "
            )
        else:
            template = (
                "The market is operating in a NORMAL price regime (confidence {conf:.0%}). "
                "Current prices are around the {quantile:.0%} percentile. "
            )

        base = template.format(
            conf=regime.confidence,
            quantile=regime.quantile_rank,
            start=regime.regime_start.strftime("%H:%M AEST"),
        )

        if analog_count >= 3:
            base += f"{analog_count} historical analog periods have been identified for comparison. "
        else:
            base += "Insufficient historical analogs for high-confidence comparison. "

        return base


def _classify_regime(price_rrp: float, region: str) -> str:
    thresholds = _REGIME_THRESHOLDS.get(region, _REGIME_THRESHOLDS["NSW1"])
    if price_rrp >= thresholds["extreme"]:
        return "extreme"
    if price_rrp >= thresholds["spike"]:
        return "spike"
    if price_rrp >= thresholds["elevated"]:
        return "elevated"
    return "normal"


def classify_regime(price_rrp: float, region: str) -> str:
    """Public helper — used by routes and agents."""
    return _classify_regime(price_rrp, region)
