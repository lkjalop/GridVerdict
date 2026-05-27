"""Framework contract — the only file core/ and engines/ are allowed to import.

No domain-specific imports. No AEMO. No NEM. Pure abstract types.
Any vertical (NEM, shipping, bushfire) implements DomainAdapter and the
framework stack runs unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class IngestEvent:
    """Abstract timestamped event from any domain."""
    source: str
    valid_time: datetime        # when this was true in the world
    system_time: datetime       # when GridVerdict learned about it
    raw_ref: str                # sha256 hash or canonical source URL
    tenant_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """What ChronoGraph and HippoGraph receive — no domain semantics."""
    event_id: str
    values: dict[str, float]
    categorical: dict[str, str]
    valid_time: datetime
    tenant_id: str


@dataclass
class EvidenceRef:
    """A citable source for any numeric claim in an answer."""
    id: str                     # ev-{uuid}
    source: str                 # AEMO_DISPATCH_PRICE | AEMO_NOTICE | etc.
    region: str | None
    interval: datetime
    field: str                  # price_rrp | demand_mw | etc.
    value: float
    raw_ref: str                # sha256 hash or URL


@dataclass
class RegimeState:
    """Output of ChronoGraph — the current market regime."""
    label: str                  # normal | elevated | spike | extreme
    confidence: float           # 0-1
    regime_start: datetime
    signal_strength: float      # ADWIN change-point strength
    quantile_rank: float        # where current value sits in t-digest


class DomainAdapter(ABC):
    """Every vertical implements this to plug into the framework.

    ChronoGraph, HippoGraph, and the Why Engine call these methods only.
    They never import domain-specific types directly.
    """

    @abstractmethod
    def to_feature_vector(self, event: IngestEvent) -> FeatureVector:
        """Convert a domain event into framework-compatible features."""

    @abstractmethod
    def to_evidence_ref(self, event: IngestEvent, field: str) -> EvidenceRef:
        """Produce a citable evidence ref from a domain event."""

    @abstractmethod
    def decomposition_hints(self) -> dict[str, Any]:
        """Domain-specific intent patterns for the query decomposer."""

    @abstractmethod
    def why_template(self, regime: RegimeState, analog_count: int) -> str:
        """Plain-English explanation template for the Why Engine."""

    @abstractmethod
    def domain_name(self) -> str:
        """Short identifier: 'nem', 'shipping', 'bushfire', etc."""
