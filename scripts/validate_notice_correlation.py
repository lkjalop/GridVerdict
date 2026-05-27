"""Validate market-notice/constraint correlation with NEM price spikes.

Two signal sources for the notice_lor_active feature:

1. BINDING CONSTRAINT PROXY (default, uses data already in DB):
   Uses T-V-MNSP1 (Basslink) binding events for TAS, and equivalent
   interconnector binding for other regions, as a proxy for LOR/stress events.
   Available for 2022-07 to 2024-02 from MMSDM backfill data.

2. AEMO MARKET NOTICES (--use-notices, requires live/current feed):
   Uses actual LOR1/LOR2/LOR3 notices. Only available for 2026-03+ from the
   current live feed — not usable for historical Jun-Jul 2024 validation.

The validation runs two backtests (with vs without the signal feature) to
measure whether the binary "constraint binding" signal improves spike F1.

Usage:
    # Default: constraint proxy, 2023-10 to 2024-02 window
    python -u scripts/validate_notice_correlation.py

    # All regions, longer window:
    python -u scripts/validate_notice_correlation.py --regions NSW1 VIC1 TAS1

    # Correlation table only (no backtest):
    python -u scripts/validate_notice_correlation.py --correlation-only

    # Live AEMO notices (needs 2026-03+ data in DB):
    python -u scripts/validate_notice_correlation.py --use-notices --end-date 2026-05-20 --lookback 30
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_notices")

NEM_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]

# Interconnector element IDs that signal supply stress per region
# Basslink = T-V-MNSP1 for TAS; Heywood = V-S-MNSP1 for SA
# Terranora (N-Q-MNSP1) for QLD; Murraylink not in MMSDM MNSP data for VIC/SA
_REGION_INTERCONNECTOR = {
    "TAS1": "T-V-MNSP1",
    "SA1": "V-S-MNSP1",
    "QLD1": "N-Q-MNSP1",
    "VIC1": "T-V-MNSP1",  # TAS→VIC direction; VIC is stressed when importing from TAS
    "NSW1": "N-Q-MNSP1",
}

_LOR_TYPES = frozenset({
    "LACK OF RESERVE 1", "LACK OF RESERVE 2", "LACK OF RESERVE 3",
    "LOR1", "LOR2", "LOR3",
})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEMO constraint/notice correlation validator")
    p.add_argument("--regions", nargs="+", default=["TAS1", "SA1", "VIC1"], metavar="REGION",
                   help="Regions to analyse (default: TAS1 SA1 VIC1)")
    p.add_argument("--lookback", type=int, default=120, metavar="DAYS",
                   help="Lookback window in days (default: 120)")
    p.add_argument("--end-date", type=str, default="2024-02-29", metavar="YYYY-MM-DD",
                   help="End of window; must be within MMSDM constraint data range "
                        "(default: 2024-02-29, the last date with constraint data)")
    p.add_argument("--step", type=int, default=288, metavar="INTERVALS",
                   help="Walk-forward step for backtest (default 288 = daily)")
    p.add_argument("--correlation-only", action="store_true",
                   help="Print correlation table only, skip backtest comparison")
    p.add_argument("--use-notices", action="store_true",
                   help="Use live AEMO market notices instead of constraint proxy")
    return p.parse_args()


async def _fetch_constraint_lor_times(
    region: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """Return datetimes when market stress is likely: interconnector near its limit.

    Uses ABS(mw_flow - export_limit) < 10 MW as a congestion proxy, since
    DISPATCHINTERCONNECTORRES MARGINALVALUE is 0 for most intervals in MMSDM.
    Falls back to high-volatility intervals from price data if no IC data.
    """
    ic_id = _REGION_INTERCONNECTOR.get(region)
    times: list[datetime] = []

    if ic_id:
        try:
            from app.db.session import db_session
            from sqlalchemy import text
            async with db_session() as session:
                result = await session.execute(
                    text("""
                        SELECT valid_time
                        FROM market_driver_events
                        WHERE driver_type = 'interconnector'
                          AND element_id = :ic_id
                          AND valid_time >= :start
                          AND valid_time <= :end
                          AND ABS(
                              (values->>'mw_flow')::float -
                              (values->>'export_limit')::float
                          ) < 10
                        ORDER BY valid_time
                    """),
                    {"ic_id": ic_id, "start": start, "end": end},
                )
                rows = result.fetchall()
            for row in rows:
                vt = row[0]
                if not isinstance(vt, datetime):
                    vt = datetime.fromisoformat(str(vt))
                if vt.tzinfo is None:
                    vt = vt.replace(tzinfo=timezone.utc)
                times.append(vt)
            logger.info("%s: %d congested intervals for %s (mw_flow~export_limit)",
                        region, len(times), ic_id)
        except Exception as exc:
            logger.warning("Could not fetch IC congestion data for %s: %s", region, exc)

    # Fallback: high price volatility as proxy (roll_vol_12 percentile)
    if not times:
        times = await _fetch_high_vol_times(region, start, end)
    return times


async def _fetch_high_vol_times(
    region: str,
    start: datetime,
    end: datetime,
    percentile_threshold: float = 0.85,
) -> list[datetime]:
    """Return intervals where rolling 1h price volatility is in the top percentile.

    This is the best available proxy when no external signal data exists.
    High price volatility is correlated with LOR-level supply stress.
    """
    try:
        from app.db.session import db_session
        from sqlalchemy import text
        async with db_session() as session:
            result = await session.execute(
                text("""
                    SELECT valid_time, price_rrp
                    FROM market_events
                    WHERE source = 'AEMO_DISPATCH_PRICE'
                      AND region = :region
                      AND valid_time >= :start
                      AND valid_time <= :end
                    ORDER BY valid_time
                """),
                {"region": region, "start": start, "end": end},
            )
            rows = result.fetchall()

        if not rows:
            return []

        import numpy as np
        prices = [float(r[1]) for r in rows]
        times_raw = [r[0] for r in rows]

        # Compute rolling std over 12 intervals (1h)
        window = 12
        vols = []
        for i in range(len(prices)):
            lo = max(0, i - window + 1)
            vols.append(float(np.std(prices[lo:i + 1])))

        threshold = float(np.percentile(vols, percentile_threshold * 100))
        stress_times = []
        for i, (vt, v) in enumerate(zip(times_raw, vols)):
            if v >= threshold:
                if not isinstance(vt, datetime):
                    vt = datetime.fromisoformat(str(vt))
                if vt.tzinfo is None:
                    vt = vt.replace(tzinfo=timezone.utc)
                stress_times.append(vt)

        logger.info(
            "%s: %d high-volatility intervals (roll_vol >= %.1f, top %.0f%%)",
            region, len(stress_times), threshold, (1 - percentile_threshold) * 100,
        )
        return stress_times
    except Exception as exc:
        logger.warning("Could not compute vol proxy for %s: %s", region, exc)
        return []


def _fetch_notice_lor_times(
    region: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """Return LOR notice timestamps from live AEMO feed (2026-03+ only)."""
    from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
    client = AEMOMarketNoticesClient()
    client._refresh_if_due()
    times = []
    for item in client._cache.values():
        n = item.item
        if n.timestamp < start or n.timestamp > end:
            continue
        if region and n.region and n.region != region:
            continue
        notice_type = (n.title or "").split(":")[0].strip().upper()
        if any(lor in notice_type for lor in _LOR_TYPES):
            times.append(n.timestamp)
    logger.info("%s: %d LOR notices from live cache", region, len(times))
    return sorted(times)


def _annotate_stress_active(series: list[dict], stress_times: list[datetime], window_min: int = 60) -> list[dict]:
    """Add notice_lor_active=1 where an interconnector binding / LOR is within prior window."""
    if not stress_times:
        return series
    stress_sorted = sorted(stress_times)
    window = timedelta(minutes=window_min)
    for r in series:
        vt = r["valid_time"]
        if vt.tzinfo is None:
            vt = vt.replace(tzinfo=timezone.utc)
        r["notice_lor_active"] = 0.0
        for st in stress_sorted:
            if st > vt:
                break
            if st >= vt - window:
                r["notice_lor_active"] = 1.0
                break
    return series


async def _run_validation(args) -> None:
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )
    start_dt = end_dt - timedelta(days=args.lookback)

    try:
        from app.db.session import init_db
        await init_db()
    except Exception as exc:
        logger.warning("DB init failed: %s", exc)

    signal_label = "AEMO LOR Notice" if args.use_notices else "Interconnector binding proxy"
    print(f"\n{'='*72}")
    print(f"  Stress Signal -> Price Spike Correlation")
    print(f"  Signal: {signal_label}")
    print(f"  Window: {start_dt.date()} to {end_dt.date()} | Lookback: 60 min")
    print(f"{'='*72}")

    print(f"\n{'Region':<8} {'Spike%_stressed':>16} {'Spike%_normal':>14} {'Stressed_n':>12} {'Total':>8}")
    print("-" * 72)

    for region in args.regions:
        try:
            from app.engines.backtest import _fetch_history, _spike_threshold_for_region
            series = await _fetch_history(region, args.lookback, end_date=end_dt)
            if not series:
                print(f"{region:<8} {'no data':>16}")
                continue

            spike_threshold = _spike_threshold_for_region(region)

            if args.use_notices:
                stress_times = _fetch_notice_lor_times(region, start_dt, end_dt)
            else:
                stress_times = await _fetch_constraint_lor_times(region, start_dt, end_dt)

            series = _annotate_stress_active(series, stress_times)

            stressed = [r for r in series if r.get("notice_lor_active", 0) > 0]
            normal = [r for r in series if r.get("notice_lor_active", 0) == 0]

            spike_stressed = sum(1 for r in stressed if r["price"] >= spike_threshold)
            spike_normal = sum(1 for r in normal if r["price"] >= spike_threshold)

            pct_stressed = spike_stressed / max(len(stressed), 1) * 100
            pct_normal = spike_normal / max(len(normal), 1) * 100
            lift = pct_stressed - pct_normal

            print(
                f"{region:<8} {pct_stressed:>15.1f}% {pct_normal:>13.1f}%"
                f" {len(stressed):>12}  {len(series):>7}"
                f"  [lift {lift:+.1f}pp]"
            )
        except Exception as exc:
            logger.error("Correlation for %s failed: %s", region, exc)
            import traceback; traceback.print_exc()

    if args.correlation_only:
        return

    # Backtest comparison
    print(f"\n{'='*72}")
    print(f"  Backtest: stress signal feature vs no-feature")
    print(f"  step={args.step} intervals | baselines only (persistence + AEMO predispatch)")
    print(f"{'='*72}")

    for region in args.regions:
        try:
            from app.engines.backtest import _fetch_history, _spike_threshold_for_region
            from app.engines.forecasting.features.market_features import (
                build_features, COL_LAST_PRICE, COL_AEMO,
            )
            from app.engines.forecasting.models.baselines import (
                PersistenceModel, AEMOPredispatchModel,
            )
            from app.engines.forecasting.evaluation.harness import run_backtest

            series = await _fetch_history(region, args.lookback, end_date=end_dt)
            if not series:
                continue

            spike_threshold = _spike_threshold_for_region(region)

            if args.use_notices:
                stress_times = _fetch_notice_lor_times(region, start_dt, end_dt)
            else:
                stress_times = await _fetch_constraint_lor_times(region, start_dt, end_dt)

            # Baseline: no stress signal
            series_plain = [dict(r, notice_lor_active=0.0) for r in series]
            target_times = [r["valid_time"] for r in series_plain]
            X_plain, y = build_features(series_plain, target_times)

            # With stress signal
            series_with = _annotate_stress_active([dict(r) for r in series], stress_times)
            X_with, _ = build_features(series_with, target_times)

            models_plain = {
                "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
                "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
            }
            models_with = {
                "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
                "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
            }

            min_train, horizon = 288, 6
            report_plain = run_backtest(
                models=models_plain, X=X_plain, y=y,
                horizon=horizon, step=args.step, min_train=min_train,
                spike_threshold=spike_threshold,
                timestamps=target_times[:len(X_plain)],
            )
            report_with = run_backtest(
                models=models_with, X=X_with, y=y,
                horizon=horizon, step=args.step, min_train=min_train,
                spike_threshold=spike_threshold,
                timestamps=target_times[:len(X_with)],
            )

            stressed_n = sum(1 for r in series_with if r.get("notice_lor_active", 0) > 0)
            print(f"\n  {region}  ({stressed_n} stressed intervals of {len(series)})")
            print(f"  {'Model':<22} {'CRPS_base':>10} {'CRPS_sig':>10} "
                  f"{'F1_base':>8} {'F1_sig':>8} {'F1_lift':>8}")
            print(f"  {'-'*70}")

            for rp in report_plain.scores:
                rw_list = [r for r in report_with.scores if r.model_name == rp.model_name]
                if not rw_list:
                    continue
                rw = rw_list[0]
                print(
                    f"  {rp.model_name:<22} "
                    f"{rp.crps:>10.3f} {rw.crps:>10.3f}  "
                    f"{rp.spike_f1:>8.3f} {rw.spike_f1:>8.3f} {rw.spike_f1 - rp.spike_f1:>+8.3f}"
                )

        except Exception as exc:
            logger.error("Backtest for %s failed: %s", region, exc)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_run_validation(args))
