"""ChangeDetector — compares successive RegionSnapshots for material changes.

Pure function: detect(prev, curr, notices) -> list[MaterialChange].
No I/O, no side effects. All thresholds are declared as module constants.

Cooldown seconds per change type — used by the engine to suppress
repeated events during sustained conditions (e.g. a 30-min price spike
should generate one commentary event, not six).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.engines.commentary.snapshot import RegionSnapshot


class ChangeType(str, Enum):
    PRICE_SPIKE = "price_spike"
    PRICE_REGIME_CHANGE = "price_regime_change"
    PRICE_NORMALISED = "price_normalised"
    NEGATIVE_PRICE = "negative_price"
    PRICE_MOVE_LARGE = "price_move_large"
    HEADROOM_TIGHTENED = "headroom_tightened"
    HEADROOM_RECOVERED = "headroom_recovered"
    NOTICE_ADDED = "notice_added"
    FORECAST_RISK_INCREASED = "forecast_risk_increased"
    FORECAST_RISK_DECREASED = "forecast_risk_decreased"
    # Sprint R: extended change types
    CONSTRAINT_ACTIVE = "constraint_active"
    WEATHER_PRESSURE_BUILDING = "weather_pressure_building"
    DATA_STALE = "data_stale"
    DATA_RECOVERED = "data_recovered"
    WATCH_CLOSED = "watch_closed"


# Seconds between repeated events of the same type per region.
# Types without an entry fire every tick they are triggered (no cooldown).
_COOLDOWN_SECONDS: dict[ChangeType, int] = {
    ChangeType.PRICE_MOVE_LARGE: 600,             # 10 min
    ChangeType.HEADROOM_TIGHTENED: 1200,          # 20 min
    ChangeType.PRICE_REGIME_CHANGE: 900,          # 15 min
    ChangeType.FORECAST_RISK_INCREASED: 900,
    ChangeType.FORECAST_RISK_DECREASED: 900,
    ChangeType.HEADROOM_RECOVERED: 300,           # 5 min
    # Sprint R: new cooldowns
    ChangeType.CONSTRAINT_ACTIVE: 600,            # 10 min
    ChangeType.WEATHER_PRESSURE_BUILDING: 1800,   # 30 min
    ChangeType.DATA_STALE: 1800,                  # 30 min — suppress during sustained outage
    ChangeType.DATA_RECOVERED: 300,               # 5 min
    ChangeType.WATCH_CLOSED: 300,                 # 5 min
}

# QueryDecomposition parameters keyed by ChangeType — injected into the
# synthetic decomposition the CommentaryEngine submits to WhyBuilder.
_CHANGE_DECOMPOSITION: dict[ChangeType, dict] = {
    ChangeType.PRICE_SPIKE: dict(
        requires_why=True,
        spike_thresholds=[300.0, 1000.0],
        causal_targets=["constraint", "demand", "headroom"],
    ),
    ChangeType.NEGATIVE_PRICE: dict(
        requires_why=True,
        spike_thresholds=[0.0],
        causal_targets=["demand", "renewable", "interconnector"],
    ),
    ChangeType.PRICE_MOVE_LARGE: dict(
        requires_why=True,
        causal_targets=["constraint", "demand", "headroom"],
    ),
    ChangeType.PRICE_REGIME_CHANGE: dict(
        requires_why=True,
        causal_targets=["demand", "constraint"],
    ),
    ChangeType.PRICE_NORMALISED: dict(
        requires_why=True,
        requires_history=True,
        causal_targets=["constraint", "demand"],
    ),
    ChangeType.HEADROOM_TIGHTENED: dict(
        requires_why=True,
        causal_targets=["headroom", "constraint", "generation"],
    ),
    ChangeType.HEADROOM_RECOVERED: dict(
        requires_why=True,
        causal_targets=["headroom"],
    ),
    ChangeType.NOTICE_ADDED: dict(
        requires_why=True,
        requires_incident_timeline=True,
        causal_targets=["notice"],
    ),
    ChangeType.FORECAST_RISK_INCREASED: dict(
        requires_why=True,
        requires_forecast=True,
        spike_thresholds=[300.0],
        causal_targets=["forecast", "constraint"],
    ),
    ChangeType.FORECAST_RISK_DECREASED: dict(
        requires_why=True,
        causal_targets=["forecast"],
    ),
    # Sprint R: extended decomposition entries
    ChangeType.CONSTRAINT_ACTIVE: dict(
        requires_why=True,
        causal_targets=["constraint", "headroom"],
    ),
    ChangeType.WEATHER_PRESSURE_BUILDING: dict(
        requires_why=True,
        causal_targets=["demand", "renewable", "weather"],
    ),
    ChangeType.DATA_STALE: dict(
        requires_why=False,
        causal_targets=[],
    ),
    ChangeType.DATA_RECOVERED: dict(
        requires_why=False,
        causal_targets=[],
    ),
    ChangeType.WATCH_CLOSED: dict(
        requires_why=True,
        requires_history=True,
        causal_targets=["constraint", "demand"],
    ),
}


@dataclass
class MaterialChange:
    change_type: ChangeType
    region: str
    valid_time: datetime
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    prev_value: float | None
    curr_value: float | None
    threshold_crossed: float | None
    description: str


def detect(
    prev: "RegionSnapshot | None",
    curr: "RegionSnapshot",
    notices: list[dict[str, Any]],
) -> list[MaterialChange]:
    """Return all material changes between prev and curr snapshots.

    When prev is None (first boot / Redis miss), only notice checks are run
    — delta-based checks require a baseline to compare against.
    """
    changes: list[MaterialChange] = []

    curr_notice_ids = {n.get("notice_id", "") for n in notices if n.get("notice_id")}

    if prev is None:
        if curr_notice_ids:
            changes.append(MaterialChange(
                change_type=ChangeType.NOTICE_ADDED,
                region=curr.region,
                valid_time=curr.valid_time,
                severity="MEDIUM",
                prev_value=None,
                curr_value=None,
                threshold_crossed=None,
                description=f"AEMO notice active for {curr.region} ({len(curr_notice_ids)} notice(s))",
            ))
        return changes

    # ── Price threshold crossings ────────────────────────────────────────────
    for threshold, sev in [(300.0, "HIGH"), (1000.0, "CRITICAL")]:
        if prev.price_rrp < threshold <= curr.price_rrp:
            delta = curr.price_rrp - prev.price_rrp
            changes.append(MaterialChange(
                change_type=ChangeType.PRICE_SPIKE,
                region=curr.region,
                valid_time=curr.valid_time,
                severity=sev,
                prev_value=prev.price_rrp,
                curr_value=curr.price_rrp,
                threshold_crossed=threshold,
                description=(
                    f"{curr.region} price crossed ${threshold:.0f}/MWh "
                    f"(${prev.price_rrp:.0f} → ${curr.price_rrp:.0f}, +${delta:.0f})"
                ),
            ))

    # ── Negative price ───────────────────────────────────────────────────────
    if prev.price_rrp >= 0.0 > curr.price_rrp:
        changes.append(MaterialChange(
            change_type=ChangeType.NEGATIVE_PRICE,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="HIGH",
            prev_value=prev.price_rrp,
            curr_value=curr.price_rrp,
            threshold_crossed=0.0,
            description=(
                f"{curr.region} price went negative "
                f"(${prev.price_rrp:.0f} → ${curr.price_rrp:.0f}/MWh)"
            ),
        ))

    # ── Price normalised ─────────────────────────────────────────────────────
    if prev.price_rrp >= 300.0 and curr.price_rrp < 150.0:
        changes.append(MaterialChange(
            change_type=ChangeType.PRICE_NORMALISED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=prev.price_rrp,
            curr_value=curr.price_rrp,
            threshold_crossed=150.0,
            description=(
                f"{curr.region} price normalised "
                f"(${prev.price_rrp:.0f} → ${curr.price_rrp:.0f}/MWh)"
            ),
        ))

    # ── Large price move (not a threshold crossing already captured) ─────────
    price_delta = abs(curr.price_rrp - prev.price_rrp)
    _already_has_crossing = any(
        c.change_type in (ChangeType.PRICE_SPIKE, ChangeType.NEGATIVE_PRICE, ChangeType.PRICE_NORMALISED)
        for c in changes
    )
    if price_delta > 100.0 and not _already_has_crossing:
        direction = "up" if curr.price_rrp > prev.price_rrp else "down"
        changes.append(MaterialChange(
            change_type=ChangeType.PRICE_MOVE_LARGE,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=prev.price_rrp,
            curr_value=curr.price_rrp,
            threshold_crossed=None,
            description=(
                f"{curr.region} price moved {direction} ${price_delta:.0f}/MWh "
                f"(${prev.price_rrp:.0f} → ${curr.price_rrp:.0f})"
            ),
        ))

    # ── Regime change ────────────────────────────────────────────────────────
    if (
        prev.regime != curr.regime
        and curr.regime not in ("unknown", "")
        and prev.regime not in ("unknown", "")
    ):
        changes.append(MaterialChange(
            change_type=ChangeType.PRICE_REGIME_CHANGE,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=None,
            curr_value=None,
            threshold_crossed=None,
            description=f"{curr.region} regime changed: {prev.regime} → {curr.regime}",
        ))

    # ── Headroom tightened ───────────────────────────────────────────────────
    for threshold, sev in [(500.0, "MEDIUM"), (200.0, "HIGH")]:
        if prev.headroom_mw >= threshold > curr.headroom_mw:
            changes.append(MaterialChange(
                change_type=ChangeType.HEADROOM_TIGHTENED,
                region=curr.region,
                valid_time=curr.valid_time,
                severity=sev,
                prev_value=prev.headroom_mw,
                curr_value=curr.headroom_mw,
                threshold_crossed=threshold,
                description=(
                    f"{curr.region} headroom tightened below {threshold:.0f}MW "
                    f"({prev.headroom_mw:.0f} → {curr.headroom_mw:.0f}MW)"
                ),
            ))
    # Sprint R: rate-of-change check — rapid decline even without threshold crossing
    _headroom_decline = prev.headroom_mw - curr.headroom_mw
    _already_headroom_tightened = any(
        c.change_type == ChangeType.HEADROOM_TIGHTENED for c in changes
    )
    if _headroom_decline > 50.0 and not _already_headroom_tightened:
        changes.append(MaterialChange(
            change_type=ChangeType.HEADROOM_TIGHTENED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=prev.headroom_mw,
            curr_value=curr.headroom_mw,
            threshold_crossed=None,
            description=(
                f"{curr.region} headroom falling rapidly "
                f"({prev.headroom_mw:.0f} → {curr.headroom_mw:.0f}MW, "
                f"{_headroom_decline:.0f}MW drop per tick)"
            ),
        ))

    # ── Headroom recovered ───────────────────────────────────────────────────
    if prev.headroom_mw < 500.0 <= curr.headroom_mw:
        changes.append(MaterialChange(
            change_type=ChangeType.HEADROOM_RECOVERED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="LOW",
            prev_value=prev.headroom_mw,
            curr_value=curr.headroom_mw,
            threshold_crossed=500.0,
            description=(
                f"{curr.region} headroom recovered to {curr.headroom_mw:.0f}MW "
                f"(was {prev.headroom_mw:.0f}MW)"
            ),
        ))

    # ── New AEMO notices ─────────────────────────────────────────────────────
    prev_notice_ids = set(prev.notice_ids)
    new_ids = curr_notice_ids - prev_notice_ids
    if new_ids:
        changes.append(MaterialChange(
            change_type=ChangeType.NOTICE_ADDED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=None,
            curr_value=None,
            threshold_crossed=None,
            description=f"New AEMO market notice for {curr.region} ({len(new_ids)} notice(s))",
        ))

    # ── Forecast risk ────────────────────────────────────────────────────────
    if prev.spike_prob_300 is not None and curr.spike_prob_300 is not None:
        prob_delta = curr.spike_prob_300 - prev.spike_prob_300
        # Sprint R: require P90 >= $500 to fire FORECAST_RISK_INCREASED
        _p90_elevated = curr.forecast_p90 is not None and curr.forecast_p90 >= 500.0
        if prob_delta > 0.15 and _p90_elevated:
            changes.append(MaterialChange(
                change_type=ChangeType.FORECAST_RISK_INCREASED,
                region=curr.region,
                valid_time=curr.valid_time,
                severity="MEDIUM",
                prev_value=prev.spike_prob_300,
                curr_value=curr.spike_prob_300,
                threshold_crossed=None,
                description=(
                    f"{curr.region} spike probability rose "
                    f"{prob_delta * 100:.0f}pp to {curr.spike_prob_300 * 100:.0f}% "
                    f"(P90 ${curr.forecast_p90:.0f}/MWh)"
                ),
            ))
        elif prob_delta < -0.15:
            changes.append(MaterialChange(
                change_type=ChangeType.FORECAST_RISK_DECREASED,
                region=curr.region,
                valid_time=curr.valid_time,
                severity="LOW",
                prev_value=prev.spike_prob_300,
                curr_value=curr.spike_prob_300,
                threshold_crossed=None,
                description=(
                    f"{curr.region} spike probability fell "
                    f"{abs(prob_delta) * 100:.0f}pp to {curr.spike_prob_300 * 100:.0f}%"
                ),
            ))

    # ── Sprint R: Constraint active ──────────────────────────────────────────
    prev_constraint_ids = set(prev.binding_constraint_ids)
    new_constraints = set(curr.binding_constraint_ids) - prev_constraint_ids
    if new_constraints:
        names = ", ".join(sorted(new_constraints)[:3])
        changes.append(MaterialChange(
            change_type=ChangeType.CONSTRAINT_ACTIVE,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=float(len(prev_constraint_ids)),
            curr_value=float(len(curr.binding_constraint_ids)),
            threshold_crossed=None,
            description=(
                f"{curr.region}: {len(new_constraints)} new binding constraint(s) — {names}"
            ),
        ))

    # ── Sprint R: Weather pressure building ──────────────────────────────────
    if curr.weather_pressure >= 0.7 and prev.weather_pressure < 0.7:
        changes.append(MaterialChange(
            change_type=ChangeType.WEATHER_PRESSURE_BUILDING,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=prev.weather_pressure,
            curr_value=curr.weather_pressure,
            threshold_crossed=0.7,
            description=(
                f"{curr.region} weather pressure building "
                f"(score {curr.weather_pressure:.2f} — heat or wind drought conditions)"
            ),
        ))

    # ── Sprint R: Data stale / recovered ─────────────────────────────────────
    _stale_threshold = 900  # 15 min
    prev_stale = prev.staleness_seconds > _stale_threshold
    curr_stale = curr.staleness_seconds > _stale_threshold
    if curr_stale and not prev_stale:
        changes.append(MaterialChange(
            change_type=ChangeType.DATA_STALE,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=float(prev.staleness_seconds),
            curr_value=float(curr.staleness_seconds),
            threshold_crossed=float(_stale_threshold),
            description=(
                f"{curr.region} dispatch data stale — "
                f"{curr.staleness_seconds}s since last AEMO update"
            ),
        ))
    elif not curr_stale and prev_stale:
        changes.append(MaterialChange(
            change_type=ChangeType.DATA_RECOVERED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="LOW",
            prev_value=float(prev.staleness_seconds),
            curr_value=float(curr.staleness_seconds),
            threshold_crossed=float(_stale_threshold),
            description=(
                f"{curr.region} dispatch data feed recovered "
                f"(was {prev.staleness_seconds}s stale)"
            ),
        ))

    # ── Sprint R: Watch closed — spike normalised ─────────────────────────────
    if prev.price_rrp >= 300.0 and curr.price_rrp < 150.0:
        changes.append(MaterialChange(
            change_type=ChangeType.WATCH_CLOSED,
            region=curr.region,
            valid_time=curr.valid_time,
            severity="MEDIUM",
            prev_value=prev.price_rrp,
            curr_value=curr.price_rrp,
            threshold_crossed=150.0,
            description=(
                f"{curr.region} spike watch closed — price normalised "
                f"(${prev.price_rrp:.0f} → ${curr.price_rrp:.0f}/MWh)"
            ),
        ))

    return changes
