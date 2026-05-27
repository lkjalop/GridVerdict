"""Generate backtest skill-score reports for all NEM regions.

Runs a walk-forward backtest (30-day lookback, 30-min horizon) for each of the
five NEM regions, serialises BacktestReport.to_dict() to
    data/backtest_report_{region}.json
and prints the summary table to stdout.

Usage:
    # Against current/live data (last 30 days):
    python -u scripts/backtest_report.py

    # Against a specific historical window (e.g. Jun-Jul 2024 MMSDM data):
    python -u scripts/backtest_report.py --end-date 2024-07-31 --lookback 61

Requires a populated market_events DB.  Set DATABASE_URL in .env or environment.
The LNN model is excluded by default (--include-lnn to enable).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to sys.path so imports resolve without package install
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

# Windows console default encoding (cp1252) can't encode ≥ and similar.
# Reconfigure stdout/stderr to UTF-8 so summary tables print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_report")

NEM_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_OUT_DIR = _repo / "data"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEM backtest skill-score report")
    p.add_argument(
        "--regions",
        nargs="+",
        default=NEM_REGIONS,
        metavar="REGION",
        help="Regions to run (default: all five)",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=30,
        metavar="DAYS",
        help="Lookback window in days (default: 30)",
    )
    p.add_argument(
        "--step",
        type=int,
        default=12,
        metavar="INTERVALS",
        help=(
            "Walk-forward step in 5-min intervals (default 12 = 1h). "
            "Use 288 for daily origins — much faster for quick smoke runs."
        ),
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "End of backtest window (default: now). Use to run against historical "
            "data without computing a large --lookback offset from today. "
            "Example: --end-date 2024-07-31 --lookback 61"
        ),
    )
    p.add_argument(
        "--include-lnn",
        action="store_true",
        default=False,
        help="Include the experimental LNN model (needs a trained state)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Quick smoke run: only persistence, seasonal_naive, aemo_predispatch. "
            "Skips LEAR and QRA (which require HiGHS solves per origin). "
            "Combine with --step 288 for sub-minute results."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        metavar="DIR",
        help="Output directory for JSON reports (default: data/)",
    )
    return p.parse_args()


def _parse_end_date(s: str | None):
    if s is None:
        return None
    from datetime import timezone
    return datetime.strptime(s, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )


async def _run_region(
    region: str,
    lookback: int,
    step: int,
    include_lnn: bool,
    fast: bool = False,
    end_date=None,
) -> dict:
    """Run backtest for one region and return serialised report dict."""
    from app.engines.backtest import run_region_backtest

    logger.info(
        "Starting backtest: region=%s lookback=%dd step=%d fast=%s end=%s",
        region, lookback, step, fast, end_date.date() if end_date else "now",
    )
    t0 = datetime.now(timezone.utc)
    report = await run_region_backtest(
        region=region,
        lookback_days=lookback,
        step_intervals=step,
        include_lnn=include_lnn,
        fast=fast,
        end_date=end_date,
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info(
        "Finished backtest: region=%s origins=%d elapsed=%.1fs",
        region, report.n_origins, elapsed,
    )
    d = report.to_dict()
    d["generated_at"] = datetime.now(timezone.utc).isoformat()
    d["elapsed_s"] = round(elapsed, 2)
    return region, report, d


async def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Initialise DB session pool (needed for _fetch_history)
    try:
        from app.db.session import init_db
        await init_db()
    except Exception as exc:
        logger.warning("DB init failed — will fall back to HippoGraph cache: %s", exc)

    end_date = _parse_end_date(args.end_date)
    all_ok = True
    for region in args.regions:
        try:
            region, report, report_dict = await _run_region(
                region, args.lookback, args.step, args.include_lnn,
                fast=args.fast, end_date=end_date,
            )
        except Exception as exc:
            logger.error("Backtest FAILED for %s: %s", region, exc)
            all_ok = False
            continue

        # Write JSON
        out_path = args.out_dir / f"backtest_report_{region}.json"
        out_path.write_text(json.dumps(report_dict, indent=2, default=str))
        logger.info("Report written: %s", out_path)

        # Print summary table
        print(f"\n{'='*96}")
        print(f"  {region}  --  {report.n_origins} origins  |  "
              f"horizon {report.horizon_min} min  |  spike >= ${report.spike_threshold:.0f}/MWh")
        print(f"{'='*96}")
        print(report.summary_table())

        best = report.best_by_crps()
        if best:
            print(f"\n  Best model by CRPS: {best.model_name}  (CRPS={best.crps:.3f})")

        # Print spike regime
        srb = report.spike_regime_breakdown
        if srb:
            spike = srb.get("spike", {})
            print(
                f"  Spike share: {spike.get('intervals', 0)} intervals "
                f"({spike.get('share', 0)*100:.1f}%)"
            )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
