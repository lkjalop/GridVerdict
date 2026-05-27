"""Data coverage audit for GridVerdict market_events DB.

Produces a human-readable and machine-readable report showing:
  - Expected vs actual interval counts per month / region / source
  - Missing interval count and percentage gap
  - Per-source status label: operational | partial | scaffolded | unavailable
  - Caveat per source
  - Acquisition run reference

Usage:
    python -u scripts/coverage_report.py
    python -u scripts/coverage_report.py --start 2023-11-01 --end 2024-02-29
    python -u scripts/coverage_report.py --json          # machine-readable only
    python -u scripts/coverage_report.py --out data/coverage_report.json

Sources tracked:
    AEMO_DISPATCH_PRICE        5-min dispatch prices (MMSDM DISPATCHPRICE)
    AEMO_PREDISPATCH_30MIN     30-min ahead predispatch forecast (P5MIN ingested live)
    market_driver/constraint   Binding dispatch constraints (MMSDM DISPATCHCONSTRAINT)
    market_driver/interconnect Interconnector flows (MMSDM DISPATCHINTERCONNECTORRES)
    unit_dispatch_events       Per-DUID 5-min dispatch (DISPATCH_UNIT_SOLUTION / DISPATCHLOAD)
    generator_units            Unit metadata (DUDETAILSUMMARY)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

NEM_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
INTERVALS_PER_DAY = 288  # 5-min dispatch

_SOURCE_CAVEATS = {
    "AEMO_DISPATCH_PRICE": (
        "operational",
        "MMSDM DISPATCHPRICE archive. Complete 5-min settlement prices for all NEM regions. "
        "Sourced from public NEMWeb archive; hashes verifiable against raw ZIP files.",
    ),
    "AEMO_PREDISPATCH_30MIN": (
        "scaffolded",
        "Live 30-min ahead predispatch ingested by scheduler since deployment. "
        "Historical window (pre-deployment) uses price[t-6] (30-min lagged price) as a "
        "non-leaking proxy -- NOT a real forward-looking forecast. "
        "P5MIN_REGIONSOLUTION backfill not yet implemented. "
        "Proxy intervals are marked in feature builder; backtest scores reflect this limitation.",
    ),
    "driver/constraint": (
        "operational",
        "MMSDM DISPATCHCONSTRAINT archive. Binding constraints only (marginal_value != 0 "
        "or violation_degree != 0). Region field is NULL for system-wide constraints. "
        "~90% storage reduction vs full constraint table — non-binding rows excluded by design.",
    ),
    "driver/interconnector": (
        "partial",
        "MMSDM DISPATCHINTERCONNECTORRES archive. Flow data present; MARGINALVALUE is 0 "
        "for all rows in this table (AEMO uses a separate constraint shadow-price mechanism). "
        "Use mw_flow vs export_limit proximity for congestion detection.",
    ),
    "unit_dispatch_events": (
        "unavailable",
        "DISPATCH_UNIT_SOLUTION / DISPATCHLOAD not included in backfill_tables setting "
        "(DISPATCHPRICE,DISPATCHINTERCONNECTORRES,DISPATCHCONSTRAINT,DUDETAILSUMMARY). "
        "Add DISPATCH_UNIT_SOLUTION to settings.backfill_tables and re-run "
        "scripts/acquire_manifest.py to populate. Platform cannot currently attribute "
        "price-setting to a specific fuel type or bidding behaviour.",
    ),
    "generator_units": (
        "operational",
        "DUDETAILSUMMARY metadata. 872 DUIDs registered with fuel type, region, max capacity. "
        "Effective-date deduplication: latest row per DUID kept. Used for fuel-type attribution "
        "in why-engine narrative.",
    ),
    "bid_offers": (
        "scaffolded",
        "BIDDAYOFFER (day-ahead bids) and BIDPEROFFER (intraday rebids) ingestion implemented. "
        "Historical backfill for Nov 2023–Feb 2024 not yet triggered. "
        "BIDPEROFFER split across BIDPEROFFER1 + BIDPEROFFER2 files on NEMWeb. "
        "Once populated: enables rebid detection for spike attribution in why-engine.",
    ),
}


def _expected_intervals(year: int, month: int) -> int:
    return monthrange(year, month)[1] * INTERVALS_PER_DAY


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GridVerdict data coverage report")
    p.add_argument("--start", default="2023-11-01", metavar="YYYY-MM-DD")
    p.add_argument("--end", default="2024-02-29", metavar="YYYY-MM-DD")
    p.add_argument("--json", action="store_true", help="Print JSON only (no table)")
    p.add_argument("--out", default=None, metavar="PATH", help="Write JSON to file")
    return p.parse_args()


async def _query(start: datetime, end: datetime) -> dict:
    from app.db.session import init_db, db_session
    from sqlalchemy import text

    try:
        await init_db()
    except Exception:
        pass

    end_incl = end.replace(hour=23, minute=59, second=59)

    async with db_session() as session:
        # --- Dispatch price per month / region ---
        r = await session.execute(text("""
            SELECT
                DATE_TRUNC('month', valid_time) AS month,
                region,
                COUNT(*) AS rows
            FROM market_events
            WHERE source = 'AEMO_DISPATCH_PRICE'
              AND valid_time >= :start AND valid_time <= :end
            GROUP BY 1, 2
            ORDER BY 1, 2
        """), {"start": start, "end": end_incl})
        price_rows = r.fetchall()

        # --- Predispatch per month ---
        r = await session.execute(text("""
            SELECT
                DATE_TRUNC('month', valid_time) AS month,
                COUNT(*) AS rows
            FROM market_events
            WHERE source = 'AEMO_PREDISPATCH_30MIN'
              AND valid_time >= :start AND valid_time <= :end
            GROUP BY 1 ORDER BY 1
        """), {"start": start, "end": end_incl})
        pd_rows = r.fetchall()

        # --- Constraints per month (all regions including NULL) ---
        r = await session.execute(text("""
            SELECT
                DATE_TRUNC('month', valid_time) AS month,
                driver_type,
                COUNT(*) AS rows
            FROM market_driver_events
            WHERE valid_time >= :start AND valid_time <= :end
            GROUP BY 1, 2 ORDER BY 1, 2
        """), {"start": start, "end": end_incl})
        driver_rows = r.fetchall()

        # --- Unit dispatch ---
        r = await session.execute(text("""
            SELECT COUNT(*) FROM unit_dispatch_events
            WHERE valid_time >= :start AND valid_time <= :end
        """), {"start": start, "end": end_incl})
        unit_count = r.scalar_one()

        # --- Generator units metadata ---
        r = await session.execute(text("SELECT COUNT(*) FROM generator_units"))
        gen_units_count = r.scalar_one()

        # --- Bid/offer data ---
        try:
            r = await session.execute(text("""
                SELECT COUNT(*) FROM bid_offers
                WHERE settlement_date >= :start AND settlement_date <= :end
            """), {"start": start, "end": end_incl})
            bid_count = r.scalar_one()
        except Exception:
            bid_count = 0   # table may not exist yet

        # --- Backfill cursor ---
        r = await session.execute(text("""
            SELECT name, last_successful_interval, status, files_completed, files_failed,
                   updated_at
            FROM backfill_cursors ORDER BY name
        """))
        cursors = r.fetchall()

    return {
        "price": price_rows,
        "predispatch": pd_rows,
        "driver": driver_rows,
        "unit_dispatch": unit_count,
        "generator_units": gen_units_count,
        "bid_offers": bid_count,
        "cursors": cursors,
    }


def _build_report(raw: dict, start: datetime, end: datetime) -> dict:
    # Build month list
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append(cur)
        year = cur.year + (cur.month // 12)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = cur.replace(year=year, month=month)

    # Index price rows
    price_index: dict[tuple, int] = {}
    for row in raw["price"]:
        month_dt = row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(str(row[0]))
        price_index[(month_dt.year, month_dt.month, row[1])] = int(row[2])

    # Index driver rows
    driver_index: dict[tuple, int] = {}
    for row in raw["driver"]:
        month_dt = row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(str(row[0]))
        driver_index[(month_dt.year, month_dt.month, str(row[1]))] = int(row[2])

    # Per-month/region price coverage
    price_coverage = []
    for m in months:
        exp = _expected_intervals(m.year, m.month)
        for region in NEM_REGIONS:
            actual = price_index.get((m.year, m.month, region), 0)
            missing = exp - actual
            pct = round(100.0 * actual / exp, 1) if exp else 0.0
            status = "operational" if missing == 0 else ("partial" if actual > 0 else "unavailable")
            price_coverage.append({
                "month": m.strftime("%Y-%m"),
                "region": region,
                "source": "AEMO_DISPATCH_PRICE",
                "expected": exp,
                "actual": actual,
                "missing": missing,
                "pct": pct,
                "status": status,
            })

    # Per-month driver coverage (system-wide, not per region)
    driver_coverage = []
    for m in months:
        for dtype in ("constraint", "interconnector"):
            actual = driver_index.get((m.year, m.month, dtype), 0)
            status_label, _ = _SOURCE_CAVEATS[f"driver/{dtype}"]
            driver_coverage.append({
                "month": m.strftime("%Y-%m"),
                "driver_type": dtype,
                "actual": actual,
                "status": status_label if actual > 0 else "unavailable",
            })

    # Predispatch
    pd_index: dict[str, int] = {}
    for row in raw["predispatch"]:
        month_dt = row[0] if isinstance(row[0], datetime) else datetime.fromisoformat(str(row[0]))
        pd_index[month_dt.strftime("%Y-%m")] = int(row[1])
    predispatch_coverage = [
        {
            "month": m.strftime("%Y-%m"),
            "source": "AEMO_PREDISPATCH_30MIN",
            "actual": pd_index.get(m.strftime("%Y-%m"), 0),
            "status": "scaffolded" if pd_index.get(m.strftime("%Y-%m"), 0) == 0 else "partial",
            "caveat": _SOURCE_CAVEATS["AEMO_PREDISPATCH_30MIN"][1],
        }
        for m in months
    ]

    # Cursors
    cursors = [
        {
            "name": str(r[0]),
            "last_successful_interval": str(r[1]) if r[1] else None,
            "status": str(r[2]),
            "files_completed": int(r[3] or 0),
            "files_failed": int(r[4] or 0),
            "updated_at": str(r[5]) if r[5] else None,
        }
        for r in raw["cursors"]
    ]

    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        },
        "sources": {
            src: {"status": status, "caveat": caveat}
            for src, (status, caveat) in _SOURCE_CAVEATS.items()
        },
        "price_coverage": price_coverage,
        "driver_coverage": driver_coverage,
        "predispatch_coverage": predispatch_coverage,
        "unit_dispatch": {
            "rows_in_window": raw["unit_dispatch"],
            "status": "unavailable",
            "caveat": _SOURCE_CAVEATS["unit_dispatch_events"][1],
        },
        "generator_units": {
            "total_duids": raw["generator_units"],
            "status": "operational",
            "caveat": _SOURCE_CAVEATS["generator_units"][1],
        },
        "bid_offers": {
            "rows_in_window": raw["bid_offers"],
            "status": "scaffolded",
            "caveat": _SOURCE_CAVEATS["bid_offers"][1],
        },
        "backfill_cursors": cursors,
    }


def _print_table(report: dict) -> None:
    w = report["window"]
    print(f"\n{'='*80}")
    print(f"  GridVerdict Data Coverage Report")
    print(f"  Window: {w['start']} to {w['end']}")
    print(f"{'='*80}")

    # Dispatch price summary
    print(f"\n  DISPATCH PRICE (AEMO_DISPATCH_PRICE) — 5-min settlement prices")
    print(f"  {'Month':<9} {'Region':<7} {'Expected':>10} {'Actual':>8} {'Missing':>8} {'Coverage':>10} {'Status':<12}")
    print(f"  {'-'*70}")
    for row in report["price_coverage"]:
        flag = "" if row["status"] == "operational" else " !"
        print(
            f"  {row['month']:<9} {row['region']:<7} "
            f"{row['expected']:>10,} {row['actual']:>8,} {row['missing']:>8,} "
            f"{row['pct']:>9.1f}% {row['status']:<12}{flag}"
        )

    # Driver coverage
    print(f"\n  DRIVER DATA (constraints + interconnectors)")
    print(f"  {'Month':<9} {'Type':<16} {'Rows':>10} {'Status':<14}")
    print(f"  {'-'*52}")
    for row in report["driver_coverage"]:
        print(f"  {row['month']:<9} {row['driver_type']:<16} {row['actual']:>10,} {row['status']:<14}")

    # Predispatch
    print(f"\n  PREDISPATCH (AEMO_PREDISPATCH_30MIN)")
    print(f"  {'Month':<9} {'Rows':>8} {'Status':<14}")
    print(f"  {'-'*36}")
    for row in report["predispatch_coverage"]:
        print(f"  {row['month']:<9} {row['actual']:>8,} {row['status']:<14}")

    # Unit dispatch
    ud = report["unit_dispatch"]
    print(f"\n  UNIT DISPATCH (unit_dispatch_events)")
    print(f"  Rows in window : {ud['rows_in_window']:,}")
    print(f"  Status         : {ud['status'].upper()}")
    print(f"  Reason         : {ud['caveat'][:100]}...")

    # Generator units
    gu = report["generator_units"]
    print(f"\n  GENERATOR METADATA (generator_units)")
    print(f"  DUIDs registered : {gu['total_duids']:,}")
    print(f"  Status           : {gu['status'].upper()}")

    # Bid/offer data
    bo = report["bid_offers"]
    print(f"\n  BID/OFFER DATA (bid_offers)")
    print(f"  Rows in window : {bo['rows_in_window']:,}")
    print(f"  Status         : {bo['status'].upper()}")
    print(f"  Note           : {bo['caveat'][:100]}...")

    # Backfill cursors
    print(f"\n  BACKFILL CURSORS")
    print(f"  {'Name':<30} {'Last interval':<24} {'Status':<10} {'OK':>5} {'Fail':>5}")
    print(f"  {'-'*76}")
    for c in report["backfill_cursors"]:
        last = c["last_successful_interval"][:19] if c["last_successful_interval"] else "never"
        print(
            f"  {c['name']:<30} {last:<24} {c['status']:<10} "
            f"{c['files_completed']:>5} {c['files_failed']:>5}"
        )

    # Source status legend
    print(f"\n  STATUS LEGEND")
    print(f"  operational  — data complete and directly from authoritative AEMO source")
    print(f"  partial      — data present but known gaps or quality limitations")
    print(f"  scaffolded   — feature exists but backed by a proxy/estimate, not real data")
    print(f"  unavailable  — source not ingested in current configuration")
    print()


async def main() -> None:
    args = _parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    raw = await _query(start, end)
    report = _build_report(raw, start, end)

    if not args.json:
        _print_table(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"Coverage report written: {args.out}")
    elif args.json:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
