"""BESS economic engine — pure, deterministic, no I/O.

Computes per-interval economics for a BESS unit given its position and
the current market snapshot. All maths is transparent and logged so the
narrative layer can reference exact numbers.

Interval: 5-minute dispatch interval (AEMO NEM standard).
"""
from __future__ import annotations

from app.portfolio.schema import BessEconomics, BessPosition, MarketSnapshot

_INTERVAL_HOURS = 5 / 60          # 5-minute dispatch interval


def compute_economics(position: BessPosition, market: MarketSnapshot) -> BessEconomics:
    """Compute BessEconomics for a single 5-min dispatch interval.

    Returns the economics of a full dispatch at max available power.
    The dispatch_policy layer decides whether to act on this output.
    """
    # ── Usable energy above the reserve floor ──────────────────────────
    usable_soc_pct = max(0.0, position.soc_pct - position.min_reserve_soc_pct)
    available_mwh = (usable_soc_pct / 100.0) * position.capacity_mwh

    # ── Effective discharge MW ─────────────────────────────────────────
    # Constrained by: rated power, available energy, site export limit
    power_limit = position.max_discharge_mw
    if position.site_export_limit_mw is not None:
        power_limit = min(power_limit, position.site_export_limit_mw)

    energy_at_full_power = power_limit * _INTERVAL_HOURS
    if energy_at_full_power > available_mwh and power_limit > 0:
        # Energy-constrained: reduce MW so we don't deplete below reserve
        dispatch_mw = available_mwh / _INTERVAL_HOURS
        dispatch_mw = min(dispatch_mw, power_limit)
    else:
        dispatch_mw = power_limit

    energy_mwh = dispatch_mw * _INTERVAL_HOURS

    # ── Revenue ────────────────────────────────────────────────────────
    expected_revenue = market.price_rrp * energy_mwh

    # ── Degradation cost ───────────────────────────────────────────────
    degradation_cost = position.degradation_cost_per_mwh * energy_mwh

    # ── FCAS opportunity value ─────────────────────────────────────────
    # If FCAS-enabled, holding charge has value — estimated as:
    # raise_6sec_rrp ($/MW/hr enablement) × dispatch_mw × interval_hours
    fcas_opportunity_value = 0.0
    if position.fcas_enabled and market.fcas_raise_6sec_rrp is not None:
        fcas_opportunity_value = round(
            market.fcas_raise_6sec_rrp * dispatch_mw * _INTERVAL_HOURS, 2
        )

    # ── Net value (energy only, not including FCAS) ────────────────────
    net_expected_value = expected_revenue - degradation_cost

    # ── Duration at full discharge ─────────────────────────────────────
    if position.max_discharge_mw > 0:
        usable_duration_hours = available_mwh / position.max_discharge_mw
    else:
        usable_duration_hours = 0.0

    return BessEconomics(
        interval_minutes=5.0,
        dispatch_mw=round(dispatch_mw, 2),
        energy_mwh=round(energy_mwh, 4),
        expected_revenue=round(expected_revenue, 2),
        degradation_cost=round(degradation_cost, 2),
        fcas_opportunity_value=fcas_opportunity_value,
        net_expected_value=round(net_expected_value, 2),
        available_energy_mwh=round(available_mwh, 3),
        usable_duration_minutes=round(usable_duration_hours * 60, 1),
    )


def compute_charge_cost(position: BessPosition, market: MarketSnapshot) -> float:
    """Cost to charge at max_charge_mw for one 5-min interval ($/interval).

    Accounts for round-trip efficiency — you need to buy more energy than
    you store because of losses.
    """
    energy_bought_mwh = position.max_charge_mw * _INTERVAL_HOURS
    efficiency = position.efficiency_pct / 100.0
    # You buy energy at the grid price; only (efficiency × energy_bought) reaches storage
    gross_cost = market.price_rrp * energy_bought_mwh
    return round(gross_cost, 2)


def headroom_is_compressed(market: MarketSnapshot, warn_mw: float = 500.0) -> bool:
    return market.headroom_mw is not None and market.headroom_mw < warn_mw


def fcas_market_is_tight(market: MarketSnapshot, threshold: float = 100.0) -> bool:
    r6 = market.fcas_raise_6sec_rrp or 0.0
    rreg = market.fcas_raise_reg_rrp or 0.0
    return r6 > threshold or rreg > threshold
