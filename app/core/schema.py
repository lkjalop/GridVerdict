"""Pydantic schemas for the GridVerdict answer contract.

All API responses and inter-service payloads use these models.
The LLM may only produce text fields. Every numeric field comes from
the deterministic data layer and must have an evidence_ref.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class ActionLabel(str, Enum):
    DISPATCH_NOW = "dispatch_now"
    HOLD = "hold"
    CHARGE = "charge"
    MONITOR = "monitor"
    REFUSE = "refuse"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"


class ConfidenceBand(str, Enum):
    HIGH = "high"       # >= 0.80
    MEDIUM = "medium"   # 0.60 - 0.79
    LOW = "low"         # 0.40 - 0.59
    VERY_LOW = "very_low"  # < 0.40


class DriverConfidenceTier(str, Enum):
    CONFIRMED   = "confirmed"    # direct AEMO telemetry, single interval, no interpolation
    SUPPORTED   = "supported"    # archived dispatch evidence corroborates the claim
    PLAUSIBLE   = "plausible"    # pattern match / model signal / indirect reasoning
    UNCONFIRMED = "unconfirmed"  # mechanistic inference without backing hard data


class IntentLabel(str, Enum):
    ACTION_RECOMMENDATION = "action_recommendation"
    EXPLANATION = "explanation"
    RETROSPECTIVE = "retrospective"
    COUNTERFACTUAL = "counterfactual"
    COMPARISON = "comparison"
    LOOKUP = "lookup"
    TRACE_REPLAY = "trace_replay"
    OUT_OF_SCOPE = "out_of_scope"


class EvidenceRefSchema(BaseModel):
    id: str = Field(default_factory=lambda: f"ev-{uuid4().hex[:8]}")
    source: str
    region: str | None = None
    interval: datetime
    field: str
    value: float
    raw_ref: str


class HistoricalAnalogs(BaseModel):
    count: int
    success_count: int
    window_days: int
    method: str = "state_vector_ppr"
    outcome_summary: str | None = None


class NewsCorrelation(BaseModel):
    explained: bool
    source: str | None = None          # "AEMO Market Notice" | "AER" | None
    credibility_tier: int | None = None
    title: str | None = None
    timestamp: datetime | None = None
    correlation_note: str


class WeatherCorrelation(BaseModel):
    explained: bool
    source: str = "WEATHER_CONSENSUS"
    location: str | None = None
    confidence: float = 0.0
    consensus: dict[str, Any] = Field(default_factory=dict)
    source_count: int = 0
    relevance_tags: list[str] = Field(default_factory=list)
    correlation_note: str


class ClaimType(str, Enum):
    PRICE_ASSERTION = "price_assertion"
    DEMAND_ASSERTION = "demand_assertion"
    CAUSE_CLAIM = "cause_claim"
    FORECAST_CLAIM = "forecast_claim"
    ACTION_RECOMMENDATION = "action_recommendation"
    PROBABILITY_CLAIM = "probability_claim"
    HISTORICAL_ANALOG = "historical_analog"
    # Sprint R: extended claim types
    FCAS_CLAIM = "fcas_claim"
    REBID_EVIDENCE = "rebid_evidence"
    OUTAGE_EVIDENCE = "outage_evidence"
    WEATHER_CORRELATION = "weather_correlation"
    CONSTRAINT_BINDING = "constraint_binding"
    WATCH_CLOSED = "watch_closed"
    OTHER = "other"


class ClaimMapItem(BaseModel):
    """A single verifiable claim extracted from the answer."""
    claim_id: str = Field(default_factory=lambda: f"cl-{uuid4().hex[:8]}")
    claim_type: ClaimType
    label: str
    tier: DriverConfidenceTier
    present: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence_ref_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class QueryDecomposition(BaseModel):
    query_id: str = Field(default_factory=lambda: f"qry-{uuid4().hex[:12]}")
    raw_query: str
    intent: IntentLabel
    entities: dict[str, list[str]] = Field(default_factory=dict)
    time_range: dict[str, Any] = Field(default_factory=dict)
    scenario_params: dict[str, Any] | None = None
    # Core data requirements
    requires_why: bool = True
    requires_history: bool = False
    requires_forecast: bool = False
    requires_backtest: bool = False
    requires_portfolio: bool = False
    # Sprint O extended requirements
    requires_live_market: bool = False       # needs real-time AEMO dispatch price
    requires_incident_timeline: bool = False # needs chronological event reconstruction
    requires_bess_context: bool = False      # needs BESS portfolio state / SOC
    causal_targets: list[str] = Field(default_factory=list)   # which drivers to explain
    spike_thresholds: list[float] = Field(default_factory=list)  # e.g. [300.0, 1000.0]
    action_context: str | None = None        # e.g. "charge window", "dispatch decision"
    missing_inputs: list[str] = Field(default_factory=list)   # inputs not available at query time
    requested_output: str | None = None      # e.g. "probability", "narrative", "table"
    # Structured sub-questions extracted by the restatement layer.
    # Each item: {"type": str, "period"?: str, "entities"?: list[str]}
    # Known types: current_price_reason, fuel_source_comparison,
    #   historical_price_distribution, price_fluctuation, forecast_outlook,
    #   regime_change, market_status
    sub_questions: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    ambiguities: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    region_corrections: list[str] = Field(default_factory=list)
    output_contract: list[str] = Field(default_factory=list)


class FactualVerdict(BaseModel):
    verdict: VerdictLabel
    action: ActionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    as_of: datetime
    why_plain_english: str
    answer_sections: list[dict[str, Any]] = Field(default_factory=list)
    answer_details: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvidenceRefSchema] = Field(default_factory=list)
    evidence_manifest: list[dict[str, Any]] = Field(default_factory=list)
    historical_analogs: HistoricalAnalogs | None = None
    news_correlation: NewsCorrelation | None = None
    weather_correlation: WeatherCorrelation | None = None
    counterargument: str
    missing_data: list[str] = Field(default_factory=list)
    known_missing_before_action: list[str] = Field(default_factory=list)
    driver_tiers: list[dict] = Field(default_factory=list)   # [{label, tier, present, note}]
    claim_tiers: list[dict] = Field(default_factory=list)    # legacy: [{label, tier, present, evidence_ref_ids}]
    claim_map: list[ClaimMapItem] = Field(default_factory=list)  # typed replacement for claim_tiers
    next_watch: list[str] = Field(default_factory=list)          # actionable: what to monitor next
    disclaimer: str = (
        "Simulation and decision-support only. Not financial advice. "
        "Not a market participant. Uses public AEMO/NEMWEB data."
    )
    trace_id: str | None = None

    @model_validator(mode="after")
    def numeric_claims_need_evidence(self) -> "FactualVerdict":
        """Every non-LOW_CONFIDENCE answer must have at least one evidence ref."""
        if self.verdict == VerdictLabel.SUPPORTED and not self.evidence_refs:
            raise ValueError("SUPPORTED verdict requires at least one evidence_ref")
        return self

    @model_validator(mode="after")
    def set_confidence_band(self) -> "FactualVerdict":
        if self.confidence >= 0.80:
            object.__setattr__(self, "confidence_band", ConfidenceBand.HIGH)
        elif self.confidence >= 0.60:
            object.__setattr__(self, "confidence_band", ConfidenceBand.MEDIUM)
        elif self.confidence >= 0.40:
            object.__setattr__(self, "confidence_band", ConfidenceBand.LOW)
        else:
            object.__setattr__(self, "confidence_band", ConfidenceBand.VERY_LOW)
        return self


class MarketStateResponse(BaseModel):
    """What the frontend monitor panel renders."""
    region: str
    price_rrp: float
    demand_mw: float
    availability_mw: float
    headroom_mw: float
    regime: str                        # normal | elevated | spike | extreme
    price_percentile: float            # where this sits in rolling 30-day distribution
    valid_time: datetime
    system_time: datetime
    source: str
    is_stale: bool
    staleness_seconds: int
    active_notices: list[dict[str, Any]] = Field(default_factory=list)
