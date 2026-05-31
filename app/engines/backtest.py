"""Backtest engine — production entry point.

Pulls historical dispatch intervals from the DB or HippoGraph cache,
builds a feature matrix, then runs run_backtest() with the full model
ensemble (baselines + LNN).

This module crosses from engines/ into data/ intentionally — it is the
orchestration layer, not a pure algorithm.  Keep all NEM-specific details
(thresholds, regions, column mappings) here; the harness stays generic.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.engines.forecasting.evaluation.harness import run_backtest
from app.engines.forecasting.features.market_features import (
    build_features, COL_LAST_PRICE, COL_SEASONAL, COL_AEMO,
)
from app.engines.forecasting.models.baselines import (
    PersistenceModel, SeasonalNaiveModel, AEMOPredispatchModel,
)
from app.engines.forecasting.models.lear_model import LEARModel
from app.engines.forecasting.models.qra_model import QRAModel
from app.engines.forecasting.types import BacktestReport

logger = logging.getLogger(__name__)

# Default walk-forward config
_DEFAULT_HORIZON_INTERVALS = 6      # forecast 30 min ahead (6 × 5min)
_DEFAULT_STEP_INTERVALS = 12        # advance origin by 1 hour
_DEFAULT_MIN_TRAIN_INTERVALS = 288  # 1 day of training before first test origin
_DEFAULT_SPIKE_THRESHOLD = 300.0    # $/MWh


async def run_region_backtest(
    region: str,
    lookback_days: int = 7,
    horizon_intervals: int = _DEFAULT_HORIZON_INTERVALS,
    step_intervals: int = _DEFAULT_STEP_INTERVALS,
    include_lnn: bool = True,
    fast: bool = False,
    end_date: datetime | None = None,
) -> BacktestReport:
    """Run a full walk-forward backtest for a single NEM region.

    Fetches history from the DB market_events table (populated by the archive
    backfill job).  Falls back to HippoGraph if the DB is sparse.

    Args:
        region:             NEM region code (e.g. "NSW1")
        lookback_days:      how many days of history to use
        horizon_intervals:  steps ahead to forecast (default 6 = 30 min)
        include_lnn:        whether to include the LNN model (requires torch)
        fast:               if True, skip LEAR and QRA (baselines only — no HiGHS)
        end_date:           end of backtest window (default: now). Set to a past
                            date to run against historical data without needing a
                            large --lookback offset from today.

    Returns:
        BacktestReport with per-model CRPS, spike F1, calibration, skill scores
    """
    series = await _fetch_history(region, lookback_days, end_date=end_date)
    if len(series) < _DEFAULT_MIN_TRAIN_INTERVALS + horizon_intervals:
        raise ValueError(
            f"Insufficient history for {region}: have {len(series)} intervals, "
            f"need {_DEFAULT_MIN_TRAIN_INTERVALS + horizon_intervals}"
        )

    target_times = [r["valid_time"] for r in series]
    X, y = build_features(series, target_times)

    # Align for proper horizon-ahead evaluation:
    # X[t] has features known at time t; y_shifted[t] = price[t + horizon].
    # Without this shift, X[t, COL_LAST_PRICE] = price[t] = y[t], leaking the
    # answer for every test row (not just the proxy issue — the feature IS the target).
    h = horizon_intervals
    if len(X) > h:
        X = X[:-h]
        y = y[h:]
        target_times = target_times[h:]   # report target timestamps (when price realised)
    else:
        h = 0  # series too short to shift; evaluation collapses to nowcast

    # Guard: build_features may skip rows with missing valid_time
    if len(X) < _DEFAULT_MIN_TRAIN_INTERVALS + horizon_intervals:
        raise ValueError(
            f"Feature matrix too short after alignment: {len(X)} rows for {region}"
        )

    models: dict = {
        "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
        "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
        "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
    }

    if not fast:
        models["lear"] = LEARModel()
        models["qra"] = QRAModel(components={
            "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
            "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
            "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
            "lear": LEARModel(),
        })

    if include_lnn:
        try:
            from app.engines.lnn.ltc_model import LTCModel
            from app.engines.lnn.distribution import QuantileHead
            from app.engines.forecasting.models.lnn_model import LNNQuantileModel
            # Use our from-scratch LTC-backed model via an adapter
            models["experimental_lnn"] = _LTCAdapter(input_size=X.shape[1])
        except Exception as exc:
            logger.debug("LNN not included in backtest (not ready): %s", exc)

    spike_threshold = _spike_threshold_for_region(region)

    def _run():
        return run_backtest(
            models=models,
            X=X,
            y=y,
            horizon=horizon_intervals,
            step=step_intervals,
            min_train=_DEFAULT_MIN_TRAIN_INTERVALS,
            spike_threshold=spike_threshold,
            timestamps=target_times[:len(X)],
        )

    # Run in executor so we don't block the event loop during training
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, _run)
    report.region_breakdown = {
        region: {
            "intervals": len(series),
            "lookback_days": lookback_days,
            "horizon_intervals": horizon_intervals,
        }
    }
    report.spike_regime_breakdown = _spike_regime_breakdown(y, spike_threshold)
    return report


async def _fetch_history(
    region: str,
    lookback_days: int,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Fetch dispatch history from the DB, joining real predispatch when available.

    Joins AEMO_PREDISPATCH_30MIN rows to dispatch rows by rounding valid_time to
    the nearest 30-minute boundary, replacing the self-referential proxy with real
    forward-looking data where it exists.

    end_date lets callers anchor the window to a specific point in time rather than
    always computing from now(). Useful when the DB holds historical data that is
    older than the default lookback window.
    """
    anchor = end_date if end_date is not None else datetime.now(timezone.utc)
    cutoff = anchor - timedelta(days=lookback_days)
    try:
        from app.db.session import db_session as session_factory
        from sqlalchemy import text
        async with session_factory() as session:
            # Dispatch rows
            dispatch_result = await session.execute(
                text("""
                    SELECT valid_time, price_rrp, demand_mw, availability_mw
                    FROM market_events
                    WHERE region = :region
                      AND source = 'AEMO_DISPATCH_PRICE'
                      AND valid_time >= :cutoff
                      AND valid_time <= :anchor
                    ORDER BY valid_time
                """),
                {"region": region, "cutoff": cutoff, "anchor": anchor},
            )
            rows = dispatch_result.fetchall()
            if not rows:
                raise ValueError("no dispatch rows")

            # Predispatch rows (30-min ahead forecast from stored PD runs)
            pd_result = await session.execute(
                text("""
                    SELECT valid_time, price_rrp
                    FROM market_events
                    WHERE region = :region
                      AND source = 'AEMO_PREDISPATCH_30MIN'
                      AND valid_time >= :cutoff
                      AND valid_time <= :anchor
                    ORDER BY valid_time
                """),
                {"region": region, "cutoff": cutoff, "anchor": anchor},
            )
            pd_rows = pd_result.fetchall()

            # Renewable fraction: sum cleared MW by fuel type per interval.
            # NULL fuel_type rows are treated as non-renewable (conservative).
            ren_result = await session.execute(
                text("""
                    SELECT
                        valid_time,
                        SUM(CASE WHEN UPPER(fuel_type) IN ('WIND','SOLAR','HYDRO','BIOMASS')
                                 THEN COALESCE(total_cleared_mw, 0.0) ELSE 0.0 END) AS ren_mw,
                        SUM(COALESCE(total_cleared_mw, 0.0)) AS total_mw
                    FROM unit_dispatch_events
                    WHERE region = :region
                      AND valid_time >= :cutoff
                      AND valid_time <= :anchor
                    GROUP BY valid_time
                """),
                {"region": region, "cutoff": cutoff, "anchor": anchor},
            )
            ren_rows = ren_result.fetchall()

        # Build predispatch lookup: round each 30-min interval to key
        pd_by_rounded: dict[datetime, float] = {}
        for pdr in pd_rows:
            vt = pdr[0] if isinstance(pdr[0], datetime) else datetime.fromisoformat(str(pdr[0]))
            pd_by_rounded[vt] = float(pdr[1])

        # Renewable fraction lookup by valid_time
        ren_by_time: dict[datetime, float] = {}
        for rr in ren_rows:
            vt = rr[0] if isinstance(rr[0], datetime) else datetime.fromisoformat(str(rr[0]))
            total = float(rr[2]) if rr[2] else 0.0
            ren_by_time[vt] = float(rr[1]) / total if total > 0 else 0.0

        # Historical weather — fetched separately so failures don't break price data
        # (WEATHER_HISTORY rows are optional; absence gracefully degrades to temp_c=0)
        weather_by_hour: dict[datetime, tuple[float, float, float]] = {}
        try:
            from app.db.session import db_session as _wx_session_factory
            async with _wx_session_factory() as wx_session:
                wx_result = await wx_session.execute(
                    text("""
                        SELECT
                            date_trunc('hour', valid_time) AS hour_bucket,
                            AVG(price_rrp)       AS temp_c,
                            AVG(demand_mw)       AS wind_kmh,
                            AVG(availability_mw) AS cloud_frac
                        FROM market_events
                        WHERE region = :region
                          AND source = 'WEATHER_HISTORY'
                          AND valid_time >= :cutoff
                          AND valid_time <= :anchor
                        GROUP BY date_trunc('hour', valid_time)
                        ORDER BY hour_bucket
                    """),
                    {"region": region, "cutoff": cutoff, "anchor": anchor},
                )
                for wr in wx_result.fetchall():
                    bucket = wr[0] if isinstance(wr[0], datetime) else datetime.fromisoformat(str(wr[0]))
                    if not bucket.tzinfo:
                        bucket = bucket.replace(tzinfo=timezone.utc)
                    weather_by_hour[bucket] = (
                        float(wr[1]) if wr[1] is not None else 0.0,
                        float(wr[2]) if wr[2] is not None else 0.0,
                        float(wr[3]) if wr[3] is not None else 0.0,
                    )
        except Exception as wx_err:
            logger.debug("WEATHER_HISTORY fetch skipped (non-fatal): %s", wx_err)

        series = [_row_to_series_dict(r, region, pd_by_rounded, ren_by_time, weather_by_hour) for r in rows]
        _fill_derived_features(series)
        return series
    except Exception as exc:
        logger.debug("DB history fetch failed, falling back to HippoGraph: %s", exc)

    # Fallback: pull from HippoGraph in-memory nodes
    return _history_from_hippograph(region, cutoff)


def _row_to_series_dict(
    row,
    region: str,
    pd_lookup: dict[datetime, float] | None = None,
    ren_lookup: dict[datetime, float] | None = None,
    weather_by_hour: dict[datetime, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    vt = row[0]
    if not isinstance(vt, datetime):
        vt = datetime.fromisoformat(str(vt))
    price = float(row[1])
    demand = float(row[2]) if row[2] else 0.0
    avail = float(row[3]) if row[3] else 0.0

    # Look up real predispatch RRP: round valid_time to nearest 30-min boundary.
    aemo_pd: float | None = None
    if pd_lookup:
        mins = vt.minute
        rounded_min = 0 if mins < 30 else 30
        pd_key = vt.replace(minute=rounded_min, second=0, microsecond=0)
        aemo_pd = pd_lookup.get(pd_key)

    ren_frac = ren_lookup.get(vt, 0.0) if ren_lookup else 0.0

    # Weather from WEATHER_HISTORY rows (seeded by scripts/seed_training_data.py).
    # Keyed by hour bucket — each 5-min interval looks up its parent hour.
    temp_c = 0.0
    wind_kmh = 0.0
    if weather_by_hour:
        hour_key = vt.replace(minute=0, second=0, microsecond=0)
        wx = weather_by_hour.get(hour_key)
        # Also try UTC-aware key in case weather_by_hour used UTC timestamps
        if wx is None and not hour_key.tzinfo:
            wx = weather_by_hour.get(hour_key.replace(tzinfo=timezone.utc))
        if wx:
            temp_c, wind_kmh, _ = wx

    ic_room = max(0.0, avail - demand) / max(demand, 1.0) if demand > 0 else 0.0

    return {
        "valid_time": vt,
        "region": region,
        "last_price": price,
        "price": price,
        "demand": demand,
        "available_gen": avail,
        "seasonal_price": 0.0,
        "aemo_predispatch": aemo_pd,
        "demand_forecast": None,
        "interconnector_room": ic_room,
        "renewable_frac": ren_frac,
        "roll_vol_12": 0.0,
        "temp_c": temp_c,
        "wind_kmh": wind_kmh,
    }


def _history_from_hippograph(region: str, cutoff: datetime) -> list[dict[str, Any]]:
    """Extract dispatch history from HippoGraph nodes as a fallback."""
    from app.engines.hippograph.graph import get_graph
    graph = get_graph()
    nodes = [
        n for n in graph.get_region_nodes(region, limit=2016)
        if n.valid_time and n.valid_time >= cutoff
    ]
    nodes.sort(key=lambda n: n.valid_time)
    series = []
    for n in nodes:
        price = n.feature_values.get("price_rrp", 0.0)
        demand = n.feature_values.get("demand_mw", 0.0)
        avail = n.feature_values.get("availability_mw", 0.0)
        ic_room = max(0.0, avail - demand) / max(demand, 1.0) if demand > 0 else 0.0
        series.append({
            "valid_time": n.valid_time,
            "region": region,
            "last_price": price,
            "price": price,
            "demand": demand,
            "available_gen": avail,
            "seasonal_price": 0.0,
            "aemo_predispatch": None,    # filled by _fill_derived_features
            "demand_forecast": None,     # filled by _fill_derived_features
            "interconnector_room": ic_room,
            "renewable_frac": 0.0,       # no DUID data in HippoGraph fallback
            "roll_vol_12": 0.0,
        })
    _fill_derived_features(series)
    return series


def _fill_derived_features(series: list[dict]) -> None:
    """Compute seasonal_price, aemo_predispatch proxy, demand_forecast, and roll_vol_12 in-place.

    Must run AFTER the series list is fully built (needs look-behind indexing).

    aemo_predispatch: when None (no real PD data in DB), filled with price[t-6]
    (30-min lag proxy). This avoids the lookahead bug where the fallback was
    price[t] == y[t], leaking the answer into the feature matrix.

    demand_forecast: filled with demand[t-6] (30-min lag). Using current demand
    as the forecast leaks t-information into future-facing rows; the lag proxy
    provides a causally safe demand expectation at forecast time.
    """
    import numpy as _np
    prices = [r["price"] for r in series]
    demands = [r["demand"] for r in series]
    for i, r in enumerate(series):
        # seasonal_price: price same time yesterday (288 × 5-min intervals)
        lag_day = i - 288
        r["seasonal_price"] = prices[lag_day] if lag_day >= 0 else prices[0]

        # aemo_predispatch: use real PD if available, else 30-min lagged price
        if r["aemo_predispatch"] is None:
            lag_pd = i - 6
            r["aemo_predispatch"] = prices[lag_pd] if lag_pd >= 0 else prices[0]

        # demand_forecast: 30-min lagged demand as causally safe proxy
        if r.get("demand_forecast") is None:
            lag_pd = i - 6
            r["demand_forecast"] = demands[lag_pd] if lag_pd >= 0 else demands[0]

        # roll_vol_12: rolling std over 12 intervals (1 hour) of price
        lo = max(0, i - 11)
        window = prices[lo : i + 1]
        r["roll_vol_12"] = float(_np.std(window)) if len(window) > 1 else 0.0


def _spike_threshold_for_region(region: str) -> float:
    from domain.nem.adapter import NEM_REGIME_THRESHOLDS
    return NEM_REGIME_THRESHOLDS.get(region, NEM_REGIME_THRESHOLDS["NSW1"])["spike"]


def _spike_regime_breakdown(y: np.ndarray, spike_threshold: float) -> dict[str, dict]:
    total = len(y)
    spike_count = int(np.sum(y >= spike_threshold))
    return {
        "all": {"intervals": total},
        "spike": {"intervals": spike_count, "share": round(spike_count / max(total, 1), 4)},
        "non_spike": {"intervals": total - spike_count, "share": round((total - spike_count) / max(total, 1), 4)},
    }


# ── LTC adapter (wraps our from-scratch LTCModel as a ForecastModel) ─────────

class _LTCAdapter:
    """Wraps LTCModel + QuantileHead as a ForecastModel for the harness."""

    name = "experimental_lnn"
    quantiles = (0.1, 0.5, 0.9)

    def __init__(self, input_size: int, hidden_size: int = 32, seq_len: int = 12):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self._model = None
        self._head = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LTCAdapter":
        import torch
        import torch.optim as optim
        from app.engines.lnn.ltc_model import LTCModel
        from app.engines.lnn.distribution import QuantileHead, pinball_loss

        device = torch.device("cpu")
        self._model = LTCModel(self.input_size, self.hidden_size).to(device)
        self._head = QuantileHead(self.hidden_size).to(device)

        windows = self._model.build_windows(X.astype(np.float32), self.seq_len, device)
        targets = torch.tensor(y, dtype=torch.float32, device=device)

        params = list(self._model.parameters()) + list(self._head.parameters())
        opt = optim.Adam(params, lr=1e-3)

        for _ in range(40):
            opt.zero_grad()
            h = self._model(windows)
            pred = self._head(h)
            loss = pinball_loss(pred, targets, self.quantiles)
            loss.backward()
            opt.step()

        self._model.eval()
        self._head.eval()
        return self

    def predict_quantiles(self, X: np.ndarray, target_times) -> Any:
        import torch
        from app.engines.forecasting.types import QuantileForecast

        device = torch.device("cpu")
        windows = self._model.build_windows(X.astype(np.float32), self.seq_len, device)
        with torch.no_grad():
            h = self._model(windows)
            pred = self._head(h).numpy()

        pred = np.sort(pred, axis=1)
        return QuantileForecast(
            target_times=target_times,
            quantiles=self.quantiles,
            values=pred,
        )

    def _as_forecast(self, target_times, values):
        from app.engines.forecasting.types import QuantileForecast
        values = np.sort(values, axis=1)
        return QuantileForecast(target_times=target_times, quantiles=self.quantiles, values=values)
