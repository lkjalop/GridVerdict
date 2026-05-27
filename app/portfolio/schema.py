"""Portfolio schema — BESS position, market snapshot, and scenario result.

All quantities are simulation-only. No real market actions are executed.
Recommendations are decision-support output, not financial advice.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ContractType(str, Enum):
    MERCHANT = "merchant"
    PPA = "ppa"
    CFD = "contract_for_difference"


class DispatchAction(str, Enum):
    DISPATCH_FULL = "dispatch_full"
    DISPATCH_PARTIAL = "dispatch_partial"
    HOLD = "hold"
    CHARGE = "charge"
    RESERVE_FCAS = "reserve_fcas"
    AVOID_INSUFFICIENT_DATA = "avoid_insufficient_data"


class BessPosition(BaseModel):
    """Operator-supplied asset position for a single BESS unit."""
    capacity_mwh: float = Field(gt=0, description="Total installed energy capacity (MWh)")
    soc_pct: float = Field(ge=0.0, le=100.0, description="Current state of charge (0–100 %)")
    max_discharge_mw: float = Field(gt=0, description="Maximum discharge power (MW)")
    max_charge_mw: float = Field(gt=0, description="Maximum charge power (MW)")
    efficiency_pct: float = Field(gt=0, le=100.0, description="Round-trip efficiency (0–100 %)")
    degradation_cost_per_mwh: float = Field(ge=0, description="Throughput degradation cost ($/MWh)")
    min_reserve_soc_pct: float = Field(ge=0.0, le=100.0, description="Minimum SOC to hold in reserve (0–100 %)")
    contract_type: ContractType = ContractType.MERCHANT
    fcas_enabled: bool = False
    risk_limit_dollar: float | None = Field(default=None, ge=0, description="Maximum acceptable net loss per action ($)")
    site_export_limit_mw: float | None = Field(default=None, gt=0, description="Grid connection export limit (MW)")

    @model_validator(mode="after")
    def min_reserve_below_soc(self) -> "BessPosition":
        if self.min_reserve_soc_pct >= 100.0:
            raise ValueError("min_reserve_soc_pct must be < 100 %")
        return self


class MarketSnapshot(BaseModel):
    """Market context fed into the scenario engine.

    Can be populated automatically from the live API or manually by the operator.
    All fields except region and price_rrp are optional — missing fields are
    surfaced in the missing_before_action checklist.
    """
    region: str
    price_rrp: float = Field(description="Current dispatch price ($/MWh)")
    price_regime: str = Field(default="normal", description="extreme | spike | elevated | normal")
    headroom_mw: float | None = Field(default=None, description="Available headroom (MW)")
    forecast_direction: str | None = Field(
        default=None,
        description="Price forecast direction: rising | falling | stable | unknown",
    )
    fcas_raise_6sec_rrp: float | None = Field(default=None, description="Raise 6-second FCAS price ($/MWh)")
    fcas_raise_reg_rrp: float | None = Field(default=None, description="Raise regulation FCAS price ($/MWh)")
    rebid_evidence_tier: str | None = Field(
        default=None,
        description="Evidence tier for intraday rebid activity: confirmed | supported | plausible | unconfirmed",
    )
    outage_evidence_tier: str | None = Field(
        default=None,
        description="Evidence tier for unit outage: confirmed | supported | plausible | unconfirmed",
    )
    evidence_quality: str = Field(
        default="insufficient",
        description="Overall evidence quality: confirmed | supported | plausible | insufficient",
    )


class BessEconomics(BaseModel):
    """Per-interval economics for a single dispatch/charge decision."""
    interval_minutes: float = 5.0
    dispatch_mw: float
    energy_mwh: float
    expected_revenue: float        # $ from energy sale
    degradation_cost: float        # $ of battery wear
    fcas_opportunity_value: float  # $ value of FCAS if dispatching (opportunity cost of not holding)
    net_expected_value: float      # revenue - degradation (excludes FCAS opp. cost)
    available_energy_mwh: float    # usable energy above the reserve floor
    usable_duration_minutes: float # how long BESS can sustain full discharge


class ScenarioResult(BaseModel):
    """Simulation output — recommendation + economics + evidence gaps.

    SIMULATION ONLY. Not financial advice. Not a market participant.
    """
    action: DispatchAction
    confidence: str               # "supported" | "low_confidence" | "insufficient_data"
    why: list[str]                # ordered rationale bullets
    economics: BessEconomics
    missing_before_action: list[str]
    risk_flags: list[str]
    caveat: str = (
        "Simulation and decision-support only. Not financial advice. "
        "Not a market participant. Uses public AEMO/NEMWEB data. "
        "All recommendations require operator validation before any action."
    )


class BessScenarioRequest(BaseModel):
    position: BessPosition
    market: MarketSnapshot


# ── Fleet coordination schema ──────────────────────────────────────────────────

class FleetAsset(BaseModel):
    """One BESS unit within a coordinated fleet."""
    asset_id: str = Field(description="Operator-assigned asset identifier (e.g. 'BESS_01')")
    position: BessPosition


class AssetDispatchResult(BaseModel):
    """Per-asset output from the fleet coordinator."""
    asset_id: str
    action: DispatchAction
    confidence: str
    dispatch_mw: float
    expected_revenue: float
    degradation_cost: float
    net_value: float
    why: list[str]
    risk_flags: list[str]
    available_energy_mwh: float
    usable_duration_minutes: float


class FleetDispatchPlan(BaseModel):
    """Aggregated simulation plan for a coordinated BESS fleet.

    SIMULATION ONLY. Not financial advice. Not a market participant.
    """
    market_region: str
    market_price_rrp: float
    market_regime: str
    assets: list[AssetDispatchResult]
    total_dispatch_mw: float
    total_expected_revenue: float
    total_degradation_cost: float
    total_net_value: float
    fleet_confidence: str   # weakest confidence across all dispatching assets
    fleet_export_limit_applied: bool
    caveat: str = (
        "Fleet dispatch plan is simulation and decision-support only. "
        "Not financial advice. Not a market participant. "
        "All recommendations require operator validation before any action."
    )


class FleetScenarioRequest(BaseModel):
    """Request body for fleet dispatch scenario.

    All assets are evaluated against the same market snapshot.
    fleet_export_limit_mw caps total fleet output (e.g. a shared grid connection).
    """
    assets: list[FleetAsset] = Field(min_length=1, max_length=20)
    market: MarketSnapshot
    fleet_export_limit_mw: float | None = Field(
        default=None, gt=0,
        description="Shared grid export limit across all fleet assets (MW)",
    )
