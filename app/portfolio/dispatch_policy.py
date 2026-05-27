"""BESS dispatch policy — translates economics into a ranked recommendation.

Rules are evaluated in priority order and short-circuit on the first match.
All logic is deterministic, logged, and fully explainable to the operator.

Priority ladder:
  0. Avoid if evidence is insufficient and price not extreme
  1. Hold if no usable energy
  2. Reserve for FCAS if FCAS market is tight and FCAS is enabled
  3. Dispatch full if price regime is extreme or spike AND profitable
  4. Dispatch partial if price elevated AND profitable AND not forecast rising
  5. Charge if price is low/normal AND SOC headroom available
  6. Hold (default — price not high enough to justify degradation cost)
"""
from __future__ import annotations

from app.portfolio.bess_engine import (
    BessEconomics,
    compute_charge_cost,
    fcas_market_is_tight,
    headroom_is_compressed,
)
from app.portfolio.schema import (
    BessPosition,
    DispatchAction,
    MarketSnapshot,
    ScenarioResult,
)

_FCAS_TIGHT_THRESHOLD = 100.0       # $/MWh — FCAS raise above this is tight
_EXTREME_FCAS_THRESHOLD = 200.0     # $/MWh — reserve for FCAS at this level
_LOW_SOC_CHARGE_THRESHOLD = 60.0    # % — charge if below this and price is low
_HIGH_CHARGE_HEADROOM = 90.0        # % — don't charge above this SOC


def evaluate(
    position: BessPosition,
    market: MarketSnapshot,
    economics: BessEconomics,
) -> ScenarioResult:
    """Apply the dispatch policy rules and return a ScenarioResult.

    All returned text is deterministic — the LLM layer narrates these facts,
    never generates numbers or asserts facts of its own.
    """
    why: list[str] = []
    risk_flags: list[str] = []
    missing: list[str] = _missing_before_action(position, market)

    # ── Context bullets (always emit) ─────────────────────────────────
    why.append(
        f"Energy price is ${market.price_rrp:.2f}/MWh "
        f"(regime: {market.price_regime})"
    )
    why.append(
        f"Available energy: {economics.available_energy_mwh:.1f} MWh "
        f"({economics.usable_duration_minutes:.0f} min at max discharge)"
    )
    if market.headroom_mw is not None:
        hstate = "compressed" if headroom_is_compressed(market) else "adequate"
        why.append(f"Headroom: {market.headroom_mw:.0f} MW ({hstate})")
    if market.forecast_direction:
        why.append(f"Price forecast direction: {market.forecast_direction}")
    if position.fcas_enabled:
        if market.fcas_raise_6sec_rrp is not None:
            why.append(
                f"FCAS Raise 6s: ${market.fcas_raise_6sec_rrp:.2f}/MWh "
                f"({'tight' if market.fcas_raise_6sec_rrp > _FCAS_TIGHT_THRESHOLD else 'normal'})"
            )
        if economics.fcas_opportunity_value > 0:
            why.append(
                f"FCAS opportunity value if dispatching: "
                f"${economics.fcas_opportunity_value:.2f} foregone"
            )
    if market.rebid_evidence_tier:
        why.append(f"Rebid evidence: {market.rebid_evidence_tier}")
    if market.outage_evidence_tier:
        why.append(f"Outage evidence: {market.outage_evidence_tier}")

    why.append(
        f"Economics: revenue ${economics.expected_revenue:.2f} "
        f"− degradation ${economics.degradation_cost:.2f} "
        f"= net ${economics.net_expected_value:.2f}"
    )

    # ── Risk flags ─────────────────────────────────────────────────────
    if position.risk_limit_dollar is not None and economics.net_expected_value < 0:
        risk_flags.append(
            f"Net value ${economics.net_expected_value:.2f} is negative — below risk limit"
        )
    if market.evidence_quality in ("insufficient", "plausible") and market.price_regime not in ("extreme", "spike"):
        risk_flags.append(
            "Evidence quality is insufficient to confirm market driver — "
            "recommendation is based on observable price only"
        )
    if economics.usable_duration_minutes < 10 and economics.available_energy_mwh > 0:
        risk_flags.append(
            f"Only {economics.usable_duration_minutes:.0f} min of dispatch remaining "
            "— consider holding for a better window"
        )

    # ── Policy rule 0: Avoid — evidence insufficient ──────────────────
    if (
        market.evidence_quality == "insufficient"
        and market.price_regime not in ("extreme", "spike")
    ):
        why.append(
            "Evidence quality is insufficient — cannot confirm market driver. "
            "Action is withheld pending data."
        )
        return _result(
            DispatchAction.AVOID_INSUFFICIENT_DATA,
            "insufficient_data",
            why, economics, missing, risk_flags,
        )

    # ── Policy rule 1: Hold — no usable energy ─────────────────────────
    if economics.available_energy_mwh <= 0:
        why.append(
            f"SOC at {position.soc_pct:.0f}% — at or below minimum reserve "
            f"({position.min_reserve_soc_pct:.0f}%). Cannot dispatch."
        )
        return _result(DispatchAction.HOLD, "supported", why, economics, missing, risk_flags)

    # ── Policy rule 2: Reserve for FCAS ──────────────────────────────
    if (
        position.fcas_enabled
        and market.fcas_raise_6sec_rrp is not None
        and market.fcas_raise_6sec_rrp > _EXTREME_FCAS_THRESHOLD
        and economics.available_energy_mwh > 0
    ):
        why.append(
            f"FCAS Raise 6s price ${market.fcas_raise_6sec_rrp:.2f}/MWh exceeds "
            f"${_EXTREME_FCAS_THRESHOLD:.0f}/MWh — FCAS enablement value "
            f"(${economics.fcas_opportunity_value:.2f}) exceeds spot dispatch economics."
        )
        confidence = _confidence_from_evidence(market.evidence_quality)
        return _result(DispatchAction.RESERVE_FCAS, confidence, why, economics, missing, risk_flags)

    # ── Policy rule 3: Dispatch full ──────────────────────────────────
    if market.price_regime in ("extreme", "spike") and economics.net_expected_value > 0:
        why.append(
            f"Price is {market.price_regime} — full dispatch is profitable "
            f"(net ${economics.net_expected_value:.2f}/interval)."
        )
        if market.forecast_direction == "falling":
            why.append("Forecast is falling — dispatch now rather than waiting.")
        confidence = _confidence_from_evidence(market.evidence_quality)
        return _result(DispatchAction.DISPATCH_FULL, confidence, why, economics, missing, risk_flags)

    # ── Policy rule 4: Dispatch partial ──────────────────────────────
    if (
        market.price_regime == "elevated"
        and economics.net_expected_value > 0
        and market.forecast_direction != "rising"
    ):
        if economics.usable_duration_minutes >= 30:
            why.append(
                f"Price is elevated — partial dispatch is profitable "
                f"(net ${economics.net_expected_value:.2f}/interval). "
                "Sufficient energy for ≥30 min of sustained output."
            )
        else:
            why.append(
                f"Price is elevated and profitable but limited duration "
                f"({economics.usable_duration_minutes:.0f} min) — partial dispatch recommended."
            )
        confidence = _confidence_from_evidence(market.evidence_quality)
        return _result(DispatchAction.DISPATCH_PARTIAL, confidence, why, economics, missing, risk_flags)

    # ── Policy rule 5: Charge ──────────────────────────────────────────
    if (
        market.price_regime == "normal"
        and position.soc_pct < _LOW_SOC_CHARGE_THRESHOLD
        and position.soc_pct < _HIGH_CHARGE_HEADROOM
    ):
        charge_cost = compute_charge_cost(position, market)
        why.append(
            f"Price is low (normal regime) and SOC is only {position.soc_pct:.0f}%. "
            f"Charging at ${market.price_rrp:.2f}/MWh costs ${charge_cost:.2f}/interval "
            "— good window to replenish for future dispatch."
        )
        confidence = _confidence_from_evidence(market.evidence_quality)
        return _result(DispatchAction.CHARGE, confidence, why, economics, missing, risk_flags)

    # ── Policy rule 6: Hold (default) ────────────────────────────────
    if economics.net_expected_value <= 0:
        why.append(
            f"Net value ${economics.net_expected_value:.2f}/interval — price does not "
            f"exceed degradation cost (${position.degradation_cost_per_mwh:.2f}/MWh). Hold."
        )
    elif market.forecast_direction == "rising":
        why.append(
            "Price forecast is rising — holding is more valuable than dispatching now."
        )
    else:
        why.append(
            "Price conditions do not meet dispatch thresholds — hold and monitor."
        )

    confidence = _confidence_from_evidence(market.evidence_quality)
    return _result(DispatchAction.HOLD, confidence, why, economics, missing, risk_flags)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _result(
    action: DispatchAction,
    confidence: str,
    why: list[str],
    economics: BessEconomics,
    missing: list[str],
    risk_flags: list[str],
) -> ScenarioResult:
    return ScenarioResult(
        action=action,
        confidence=confidence,
        why=why,
        economics=economics,
        missing_before_action=missing,
        risk_flags=risk_flags,
    )


def _confidence_from_evidence(evidence_quality: str) -> str:
    return {
        "confirmed": "supported",
        "supported": "supported",
        "plausible": "low_confidence",
        "insufficient": "insufficient_data",
    }.get(evidence_quality, "low_confidence")


def _missing_before_action(position: BessPosition, market: MarketSnapshot) -> list[str]:
    """Enumerate data gaps the operator should verify before taking action."""
    items: list[str] = []
    items.append("actual SOC telemetry (user-supplied SOC — verify against SCADA)")
    if position.site_export_limit_mw is None:
        items.append("site export limit (grid connection limit not provided)")
    if position.contract_type.value != "merchant":
        items.append(
            f"contract obligations ({position.contract_type.value} — check dispatch constraints)"
        )
    if position.fcas_enabled:
        items.append("FCAS enablement status (confirm unit is currently FCAS-enabled in AEMO)")
    if not market.forecast_direction or market.forecast_direction == "unknown":
        items.append("5-min price forecast (direction unknown — LNN not consulted)")
    if market.outage_evidence_tier is None:
        items.append("real-time outage/rebid notice (no unit dispatch evidence available)")
    if market.headroom_mw is None:
        items.append("headroom data (availability not confirmed for this interval)")
    return items
