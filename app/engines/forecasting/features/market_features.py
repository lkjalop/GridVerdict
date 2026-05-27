"""Market feature builder (skin / energy-specific).

Credible forecasting uses the drivers that actually CAUSE prices, not just price
history. Every feature at row t uses only information available at or before t —
no lookahead. The AEMO pre-dispatch forecast is attached as its own column so the
baseline can consume it and the harness can score skill against it.

Input is a list of canonical event dicts (the unified data model from the PRD):
    {region, metric, value, unit, valid_time, system_time, source}
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import numpy as np

# Australian public holidays (national + approximate state-level) as (month, day) pairs.
# Add future years here when the backtest window extends beyond 2027.
_AU_HOLIDAYS: frozenset[tuple[int, int]] = frozenset({
    (1, 1),   # New Year's Day
    (1, 26),  # Australia Day
    (4, 25),  # ANZAC Day
    (6, 9),   # Queen's/King's Birthday (approx Mon in June — exact varies by state)
    (12, 25), # Christmas Day
    (12, 26), # Boxing Day
})


# Feature column order is fixed so models can reference columns by index.
FEATURE_COLUMNS = [
    "last_price",          # 0  most recent dispatch price (persistence reads this)
    "seasonal_price",      # 1  price same interval one day ago (seasonal_naive reads this)
    "aemo_predispatch",    # 2  official forecast for the target interval (AEMO baseline reads this)
    "demand",              # 3  current regional demand
    "demand_forecast",     # 4  AEMO demand pre-dispatch
    "available_gen",       # 5  scheduled + semi-scheduled availability (from PASA)
    "interconnector_room", # 6  import headroom into the region (limit - flow)
    "renewable_frac",      # 7  renewable share of generation
    "roll_vol_12",         # 8  rolling std of price over last 12 intervals (1hr)
    "tod_sin", "tod_cos",  # 9,10 time-of-day cyclical encoding
    "dow",                 # 11 day of week
    "notice_lor_active",   # 12 1.0 if LOR1/LOR2/LOR3 notice active in prev 60 min, else 0.0
    "demand_ramp",         # 13 demand[t] - demand[t-1] (MW change over one dispatch interval)
    "holiday_flag",        # 14 1.0 on Australian public holidays, 0.0 otherwise
    "temp_c",              # 15 ambient temperature (°C); 0.0 when weather unavailable
    "wind_kmh",            # 16 wind speed (km/h); 0.0 when weather unavailable
    "headroom_mw",         # 17 max(available_gen - demand, 0) — supply cushion above load
    "constraint_count",    # 18 number of binding constraints reported by AEMO; 0.0 when unavailable
]


def _tod_encoding(ts: datetime) -> tuple[float, float]:
    frac = (ts.hour * 60 + ts.minute) / (24 * 60)
    return float(np.sin(2 * np.pi * frac)), float(np.cos(2 * np.pi * frac))


def build_features(
    series: Sequence[dict],
    target_times: Sequence[datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): feature matrix and the actual price for each target time.

    `series` must be sorted by valid_time. This is a reference implementation —
    the real version joins multiple canonical streams (price, demand, PASA,
    interconnector) on valid_time. Here we assume each target time has a matching
    record dict carrying the needed metrics, to keep the module self-contained.
    """
    rows, ys = [], []
    sorted_series = sorted(series, key=lambda r: r["valid_time"])
    by_time = {r["valid_time"]: r for r in sorted_series}
    # Build an index so we can look up the preceding interval for demand_ramp
    times_list = [r["valid_time"] for r in sorted_series]

    for i, ts in enumerate(target_times):
        r = by_time.get(ts)
        if r is None:
            continue
        tod_sin, tod_cos = _tod_encoding(ts)

        def _f(val, default: float = 0.0) -> float:
            return float(val) if val is not None else default

        # demand_ramp: demand[t] - demand[t-1], 0.0 when no prior row available
        demand_ramp = 0.0
        ts_idx = None
        try:
            ts_idx = times_list.index(ts)
        except ValueError:
            pass
        if ts_idx is not None and ts_idx > 0:
            prev_r = by_time.get(times_list[ts_idx - 1])
            if prev_r is not None:
                demand_ramp = _f(r.get("demand")) - _f(prev_r.get("demand"))

        holiday_flag = 1.0 if (ts.month, ts.day) in _AU_HOLIDAYS else 0.0

        avail_gen = _f(r.get("available_gen"))
        demand_mw = _f(r.get("demand"))
        headroom_mw = max(avail_gen - demand_mw, 0.0)

        rows.append([
            _f(r.get("last_price")),
            _f(r.get("seasonal_price")),
            _f(r.get("aemo_predispatch")),
            demand_mw,
            _f(r.get("demand_forecast")),
            avail_gen,
            _f(r.get("interconnector_room")),
            _f(r.get("renewable_frac")),
            _f(r.get("roll_vol_12")),
            tod_sin, tod_cos,
            float(ts.weekday()),
            _f(r.get("notice_lor_active")),
            demand_ramp,
            holiday_flag,
            _f(r.get("temp_c")),
            _f(r.get("wind_kmh")),
            headroom_mw,
            _f(r.get("constraint_count")),
        ])
        ys.append(r.get("price", 0.0))
    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float)


# Convenience: column indices the baseline models need.
COL_LAST_PRICE = FEATURE_COLUMNS.index("last_price")
COL_SEASONAL = FEATURE_COLUMNS.index("seasonal_price")
COL_AEMO = FEATURE_COLUMNS.index("aemo_predispatch")
COL_DEMAND_RAMP = FEATURE_COLUMNS.index("demand_ramp")
COL_HOLIDAY = FEATURE_COLUMNS.index("holiday_flag")
COL_TEMP_C = FEATURE_COLUMNS.index("temp_c")
COL_WIND_KMH = FEATURE_COLUMNS.index("wind_kmh")
COL_HEADROOM = FEATURE_COLUMNS.index("headroom_mw")
COL_CONSTRAINT_COUNT = FEATURE_COLUMNS.index("constraint_count")
