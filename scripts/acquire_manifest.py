"""Reproducible MMSDM acquisition script.

Runs the full AEMO MMSDM archive backfill, then writes a provenance manifest to
data/manifest.json.  The manifest is the artifact that gets checked into source
control — it lets any reviewer reproduce the exact dataset by re-running this
script and comparing manifest hashes.

Usage:
    python scripts/acquire_manifest.py [options]

Options:
    --start-date YYYY-MM-DD   First month to acquire  (default: settings.backfill_start_date)
    --end-date   YYYY-MM-DD   Last month to acquire   (default: today)
    --tables     T1,T2,...     Comma-separated MMSDM table names (default: settings.backfill_tables)
    --max-files  N             Stop after N files (useful for smoke-tests)
    --output     PATH          Manifest output path   (default: data/manifest.json)
    --dry-run                  Discover files and print plan without fetching

The manifest file includes:
    - acquisition_command: the exact command used
    - date_range: from/to ISO dates
    - tables: list of MMSDM table names
    - files: per-file dict of {url, sha256, rows}
    - failed_files: list of {url, error}
    - row_counts: per-table counts queried from the DB after ingest
    - manifest_hash: SHA-256 of the manifest JSON (excluding this field)
    - acquired_at: ISO timestamp
    - runtime_seconds: wall-clock time

Example manifest check-in workflow:
    1. Run this script (may take several hours for 3 years of data)
    2. git add data/manifest.json
    3. Reviewer runs script with same --start-date / --tables, compares manifest_hash
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


async def _run(args: argparse.Namespace) -> None:
    from config.settings import get_settings
    from app.mcp.aemo_archive import backfill_mmsdm_archive, discover_mmsdm_files

    settings = get_settings()

    start_date = (
        datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        if args.start_date
        else datetime.fromisoformat(settings.backfill_start_date).replace(tzinfo=timezone.utc)
    )
    end_date = (
        datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
        if args.end_date
        else datetime.now(timezone.utc)
    )
    tables = (
        [t.strip().upper() for t in args.tables.split(",") if t.strip()]
        if args.tables
        else [t.strip() for t in settings.backfill_tables.split(",") if t.strip()]
    )
    max_files: int | None = args.max_files
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"GridVerdict MMSDM acquisition")
    print(f"  Date range : {start_date.date()} to {end_date.date()}")
    print(f"  Tables     : {', '.join(tables)}")
    print(f"  Max files  : {max_files or 'unlimited'}")
    print(f"  Output     : {output_path}")
    print()

    if args.dry_run:
        import httpx
        async with httpx.AsyncClient(
            base_url="https://nemweb.com.au",
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )},
            follow_redirects=True,
        ) as client:
            urls = await discover_mmsdm_files(client, start_date, end_date, tables)
        if max_files:
            urls = urls[:max_files]
        print(f"DRY RUN — {len(urls)} files would be fetched:")
        for u in urls:
            print(f"  {u}")
        return

    # Initialise DB schema if needed (deferred import — not needed for dry-run)
    from app.db.session import db_session, init_db
    await init_db()

    t0 = time.monotonic()
    counts = await backfill_mmsdm_archive(
        db_session_factory=db_session,
        start_date=start_date,
        end_date=end_date,
        max_files=max_files,
        tables=tables,
    )
    runtime = time.monotonic() - t0

    # Query final row counts from the DB
    row_counts = await _query_row_counts(db_session, start_date, end_date)

    # Build manifest
    manifest: dict = {
        "acquisition_command": " ".join(sys.argv),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime, 1),
        "date_range": {
            "from": start_date.date().isoformat(),
            "to": end_date.date().isoformat(),
        },
        "tables": tables,
        "summary": {
            "files_ok": counts["files_ok"],
            "files_failed": counts["files_failed"],
            "months_completed": counts["months_completed"],
            "months_skipped": counts["months_skipped"],
            "price_rows_ingested": counts["price_rows"],
            "driver_rows_ingested": counts["driver_rows"],
            "unit_rows_ingested": counts["unit_rows"],
            "unit_metadata_rows_ingested": counts["unit_metadata_rows"],
        },
        "row_counts_in_db": row_counts,
        "file_hashes": counts["file_hashes"],
        "failed_files": counts["failed_files"],
    }

    # Compute manifest hash over everything except the hash field itself
    manifest_body = json.dumps(
        {k: v for k, v in manifest.items()},
        sort_keys=True,
        default=str,
    ).encode()
    manifest["manifest_hash"] = hashlib.sha256(manifest_body).hexdigest()

    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(f"\nAcquisition complete in {runtime:.0f}s")
    print(f"  Files OK    : {counts['files_ok']}")
    print(f"  Files failed: {counts['files_failed']}")
    print(f"  Months done : {counts['months_completed']}")
    print(f"  Price rows  : {counts['price_rows']:,}")
    print(f"  Driver rows : {counts['driver_rows']:,}")
    print(f"  Unit rows   : {counts['unit_rows']:,}")
    print(f"  Manifest    : {output_path}  (hash: {manifest['manifest_hash'][:16]}…)")

    if counts["failed_files"]:
        print(f"\nWARNING: {counts['files_failed']} file(s) failed:")
        for f in counts["failed_files"]:
            print(f"  {f['url']}: {f['error']}")
        sys.exit(1)


async def _query_row_counts(
    db_session_factory,
    start_date: datetime,
    end_date: datetime,
) -> dict:
    """Query the DB for row counts per table within the acquired date range.

    end_date is treated as end-of-day (23:59:59) so that specifying a date like
    2024-02-29 captures the full day's intervals rather than cutting off at midnight.
    """
    from datetime import timedelta
    from sqlalchemy import func, select
    from app.db.models import MarketDriverEvent, MarketEvent, UnitDispatchEvent, GeneratorUnit

    # Ensure end_date covers the full final day regardless of what time component was passed
    end_inclusive = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    counts: dict = {}
    try:
        async with db_session_factory() as session:
            # market_events — price rows
            r = await session.execute(
                select(func.count()).select_from(MarketEvent).where(
                    MarketEvent.valid_time >= start_date,
                    MarketEvent.valid_time <= end_inclusive,
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                )
            )
            counts["market_events_price"] = r.scalar_one()

            # market_events — predispatch rows
            r = await session.execute(
                select(func.count()).select_from(MarketEvent).where(
                    MarketEvent.valid_time >= start_date,
                    MarketEvent.valid_time <= end_inclusive,
                    MarketEvent.source == "AEMO_PREDISPATCH_30MIN",
                )
            )
            counts["market_events_predispatch"] = r.scalar_one()

            # market_driver_events
            r = await session.execute(
                select(
                    MarketDriverEvent.driver_type,
                    func.count().label("n"),
                ).where(
                    MarketDriverEvent.valid_time >= start_date,
                    MarketDriverEvent.valid_time <= end_inclusive,
                ).group_by(MarketDriverEvent.driver_type)
            )
            for row in r.fetchall():
                counts[f"driver_{row.driver_type}"] = row.n

            # unit_dispatch_events
            r = await session.execute(
                select(func.count()).select_from(UnitDispatchEvent).where(
                    UnitDispatchEvent.valid_time >= start_date,
                    UnitDispatchEvent.valid_time <= end_inclusive,
                )
            )
            counts["unit_dispatch_events"] = r.scalar_one()

            # generator_units (metadata — no time filter)
            r = await session.execute(select(func.count()).select_from(GeneratorUnit))
            counts["generator_units"] = r.scalar_one()
    except Exception as exc:
        counts["_error"] = str(exc)

    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Acquire AEMO MMSDM archive data and write a reproducibility manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--start-date", metavar="YYYY-MM-DD", help="First month to acquire")
    p.add_argument("--end-date", metavar="YYYY-MM-DD", help="Last month to acquire")
    p.add_argument("--tables", metavar="T1,T2,...", help="Comma-separated MMSDM table names")
    p.add_argument("--max-files", type=int, metavar="N", help="Limit total files fetched")
    p.add_argument(
        "--output",
        metavar="PATH",
        default="data/manifest.json",
        help="Manifest output path (default: data/manifest.json)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print file list without fetching")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(_run(args))
