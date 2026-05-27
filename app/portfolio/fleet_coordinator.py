"""BESS fleet coordinator — multi-asset dispatch allocation.

Evaluates each asset against the same market snapshot, ranks by net value,
respects a fleet-level export cap, and returns an aggregated FleetDispatchPlan.

Rules:
  1. Each asset is evaluated independently via bess_engine + dispatch_policy.
  2. When a fleet_export_limit_mw is set, assets are sorted by net expected value
     (descending) and allocated in order until the limit is reached.
  3. An asset flagged AVOID_INSUFFICIENT_DATA contributes 0 MW to fleet totals.
  4. fleet_confidence = weakest confidence among assets that are dispatching.

SIMULATION ONLY. No real market actions are executed.
"""
from __future__ import annotations

from app.portfolio.bess_engine import compute_economics
from app.portfolio.dispatch_policy import evaluate
from app.portfolio.schema import (
    AssetDispatchResult,
    DispatchAction,
    FleetAsset,
    FleetDispatchPlan,
    FleetScenarioRequest,
    MarketSnapshot,
)

_DISPATCHING_ACTIONS = frozenset({
    DispatchAction.DISPATCH_FULL,
    DispatchAction.DISPATCH_PARTIAL,
})

_CONFIDENCE_RANK = {
    "supported": 0,
    "low_confidence": 1,
    "insufficient_data": 2,
}


def evaluate_fleet(request: FleetScenarioRequest) -> FleetDispatchPlan:
    """Run fleet dispatch allocation and return an aggregated plan.

    Pure function — no I/O. Deterministic.
    """
    market = request.market
    limit_mw = request.fleet_export_limit_mw

    # ── Per-asset evaluation ──────────────────────────────────────────────
    evaluated: list[tuple[FleetAsset, AssetDispatchResult]] = []
    for fa in request.assets:
        eco = compute_economics(fa.position, market)
        res = evaluate(fa.position, market, eco)
        adr = AssetDispatchResult(
            asset_id=fa.asset_id,
            action=res.action,
            confidence=res.confidence,
            dispatch_mw=eco.dispatch_mw,
            expected_revenue=eco.expected_revenue,
            degradation_cost=eco.degradation_cost,
            net_value=eco.net_expected_value,
            why=res.why,
            risk_flags=res.risk_flags,
            available_energy_mwh=eco.available_energy_mwh,
            usable_duration_minutes=eco.usable_duration_minutes,
        )
        evaluated.append((fa, adr))

    # ── Fleet export cap allocation ────────────────────────────────────────
    limit_applied = limit_mw is not None
    if limit_mw is not None:
        # Rank dispatching assets by net_value descending; non-dispatching pass through
        dispatching = [
            (fa, adr) for fa, adr in evaluated
            if adr.action in _DISPATCHING_ACTIONS
        ]
        non_dispatching = [
            (fa, adr) for fa, adr in evaluated
            if adr.action not in _DISPATCHING_ACTIONS
        ]
        dispatching.sort(key=lambda t: t[1].net_value, reverse=True)

        remaining_mw = limit_mw
        capped: list[AssetDispatchResult] = []
        for fa, adr in dispatching:
            allocated = min(adr.dispatch_mw, remaining_mw)
            if allocated <= 0.0:
                # Export limit exhausted — demote to HOLD
                capped.append(AssetDispatchResult(
                    **{**adr.model_dump(),
                       "action": DispatchAction.HOLD,
                       "dispatch_mw": 0.0,
                       "expected_revenue": 0.0,
                       "net_value": 0.0,
                       "why": adr.why + [
                           f"Fleet export limit ({limit_mw:.0f} MW) exhausted — "
                           "this asset held to respect grid connection cap."
                       ],
                    }
                ))
            else:
                scale = allocated / adr.dispatch_mw if adr.dispatch_mw > 0 else 0.0
                capped.append(AssetDispatchResult(
                    **{**adr.model_dump(),
                       "dispatch_mw": round(allocated, 2),
                       "expected_revenue": round(adr.expected_revenue * scale, 2),
                       "net_value": round(adr.net_value * scale, 2),
                    }
                ))
                remaining_mw -= allocated

        final_results = [adr for _, adr in non_dispatching] + capped
    else:
        final_results = [adr for _, adr in evaluated]

    # ── Aggregate totals ───────────────────────────────────────────────────
    total_dispatch_mw = sum(
        adr.dispatch_mw for adr in final_results
        if adr.action in _DISPATCHING_ACTIONS
    )
    total_revenue = sum(
        adr.expected_revenue for adr in final_results
        if adr.action in _DISPATCHING_ACTIONS
    )
    total_degradation = sum(
        adr.degradation_cost for adr in final_results
        if adr.action in _DISPATCHING_ACTIONS
    )
    total_net = sum(
        adr.net_value for adr in final_results
        if adr.action in _DISPATCHING_ACTIONS
    )

    # Fleet confidence = weakest confidence among dispatching assets
    dispatching_confidences = [
        adr.confidence for adr in final_results
        if adr.action in _DISPATCHING_ACTIONS
    ]
    fleet_confidence = _weakest_confidence(dispatching_confidences)

    return FleetDispatchPlan(
        market_region=market.region,
        market_price_rrp=market.price_rrp,
        market_regime=market.price_regime,
        assets=final_results,
        total_dispatch_mw=round(total_dispatch_mw, 2),
        total_expected_revenue=round(total_revenue, 2),
        total_degradation_cost=round(total_degradation, 2),
        total_net_value=round(total_net, 2),
        fleet_confidence=fleet_confidence,
        fleet_export_limit_applied=limit_applied,
    )


def _weakest_confidence(confidences: list[str]) -> str:
    if not confidences:
        return "insufficient_data"
    worst_rank = max(_CONFIDENCE_RANK.get(c, 2) for c in confidences)
    for label, rank in _CONFIDENCE_RANK.items():
        if rank == worst_rank:
            return label
    return "insufficient_data"
