"""Proactive spike alert detector (Sprint D).

Detects price-regime transitions after each dispatch refresh and returns
alert payloads for the event bus. Stateful per-region tracking so only
genuine transitions (normal→spike, spike→normal) generate events.

The detector is instantiated once per process in scheduler.py and updated
on every 5-minute dispatch cycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


_SPIKE_REGIMES: frozenset[str] = frozenset({"spike", "extreme"})
_RECOVERY_REGIMES: frozenset[str] = frozenset({"normal", "elevated"})

# Alert after price exceeds this even without a formal regime change (safety net)
_ABSOLUTE_SPIKE_THRESHOLD_RRP = 300.0


@dataclass
class SpikeAlertPayload:
    region: str
    alert_type: str        # "spike_alert" | "spike_resolved"
    regime: str
    previous_regime: str
    price_rrp: float
    demand_mw: float
    headroom_mw: float
    valid_time: str

    def to_dict(self) -> dict:
        return asdict(self)


class SpikeDetector:
    """Stateful regime-transition detector.

    Call check() once per dispatch cycle for each region.
    Returns a SpikeAlertPayload when a transition is detected, else None.
    """

    def __init__(self) -> None:
        self._last_regime: dict[str, str] = {}
        self._alert_sent: dict[str, bool] = {}  # per-region: True while spike is active

    def check(
        self,
        region: str,
        regime: str,
        price_rrp: float,
        demand_mw: float,
        headroom_mw: float,
        valid_time: datetime | None = None,
    ) -> SpikeAlertPayload | None:
        """Return a SpikeAlertPayload if a transition occurred, else None."""
        vt_str = (valid_time or datetime.now(timezone.utc)).isoformat()
        prev = self._last_regime.get(region, "unknown")
        self._last_regime[region] = regime

        if prev == "unknown":
            # Initialise silently — no alert on cold start
            if regime in _SPIKE_REGIMES:
                self._alert_sent[region] = True
            return None

        # Transition into spike/extreme
        if regime in _SPIKE_REGIMES and prev not in _SPIKE_REGIMES:
            self._alert_sent[region] = True
            return SpikeAlertPayload(
                region=region,
                alert_type="spike_alert",
                regime=regime,
                previous_regime=prev,
                price_rrp=round(price_rrp, 2),
                demand_mw=round(demand_mw, 1),
                headroom_mw=round(headroom_mw, 1),
                valid_time=vt_str,
            )

        # Recovery from spike/extreme
        if regime in _RECOVERY_REGIMES and prev in _SPIKE_REGIMES:
            self._alert_sent[region] = False
            return SpikeAlertPayload(
                region=region,
                alert_type="spike_resolved",
                regime=regime,
                previous_regime=prev,
                price_rrp=round(price_rrp, 2),
                demand_mw=round(demand_mw, 1),
                headroom_mw=round(headroom_mw, 1),
                valid_time=vt_str,
            )

        # Absolute threshold catch (price crossed $300 without regime change label)
        if (
            price_rrp >= _ABSOLUTE_SPIKE_THRESHOLD_RRP
            and not self._alert_sent.get(region, False)
        ):
            self._alert_sent[region] = True
            return SpikeAlertPayload(
                region=region,
                alert_type="spike_alert",
                regime=regime,
                previous_regime=prev,
                price_rrp=round(price_rrp, 2),
                demand_mw=round(demand_mw, 1),
                headroom_mw=round(headroom_mw, 1),
                valid_time=vt_str,
            )

        # Clear absolute-threshold flag when price recovers
        if price_rrp < _ABSOLUTE_SPIKE_THRESHOLD_RRP and self._alert_sent.get(region, False):
            if regime in _RECOVERY_REGIMES:
                self._alert_sent[region] = False

        return None

    def reset(self, region: str | None = None) -> None:
        """Reset state — used in tests and cold-start scenarios."""
        if region:
            self._last_regime.pop(region, None)
            self._alert_sent.pop(region, None)
        else:
            self._last_regime.clear()
            self._alert_sent.clear()
