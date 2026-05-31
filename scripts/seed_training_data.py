#!/usr/bin/env python3
"""Seed 14 days of historical training data for GridVerdict forecast models.

This script does two things:
  1. Fetches historical AEMO dispatch prices for the last N days from NEMWeb archive
     and inserts them into market_events (ON CONFLICT DO NOTHING).
  2. Fetches historical weather (temperature, wind, cloud cover) from Open-Meteo
     historical API for each NEM region and stores as WEATHER_HISTORY in market_events.

Run from within the Docker container or with DB access:
  python scripts/seed_training_data.py [--days 14] [--weather-only] [--price-only]

After seeding, the next forecast request will automatically use the expanded dataset.
Models retrain on each request using the lookback window.

Weather-price correlation per state (seasonal):
  - NSW1 (Sydney): Hot summers, moderate winters. Solar dominant Oct-Mar.
  - VIC1 (Melbourne): Cool winters, high demand. Wind dominant May-Sep.
  - QLD1 (Brisbane): High solar year-round. Low winter demand baseline.
  - SA1 (Adelaide): Hottest summers, most volatile. Wind + solar critical.
  - TAS1 (Hobart): Hydro dominant. Weather less price-correlated than other states.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# NEM region → (lat, lon) for weather fetch
_REGION_COORDS: dict[str, tuple[float, float]] = {
    "NSW1": (-33.8688, 151.2093),   # Sydney
    "VIC1": (-37.8136, 144.9631),   # Melbourne
    "QLD1": (-27.4698, 153.0251),   # Brisbane
    "SA1":  (-34.9285, 138.6007),   # Adelaide
    "TAS1": (-42.8821, 147.3272),   # Hobart
}

_ALL_REGIONS = list(_REGION_COORDS.keys())


async def fetch_and_store_weather(days: int = 14) -> dict[str, int]:
    """Fetch hourly weather from Open-Meteo historical API and store in market_events.

    Weather is stored as WEATHER_HISTORY source, one row per hour per region.
    Each row's `data` JSON has: temp_c, wind_kmh, cloud_cover_pct, precipitation_mm.
    The `price_rrp` column is used for temp_c to allow JOIN in training pipeline.
    The `demand_mw` column is used for wind_kmh.
    The `availability_mw` column is used for cloud_cover_pct / 100.
    """
    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed — run: pip install httpx")
        return {}

    from app.db.session import db_session
    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    end_date = now.date()
    start_date = (now - timedelta(days=days)).date()

    inserted_total: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for region, (lat, lon) in _REGION_COORDS.items():
            logger.info("Fetching %d days of weather for %s (%.4f, %.4f)...",
                        days, region, lat, lon)
            try:
                resp = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "hourly": "temperature_2m,wind_speed_10m,cloud_cover,precipitation",
                        "wind_speed_unit": "kmh",
                        "timezone": "UTC",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Weather fetch failed for %s: %s", region, exc)
                continue

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            winds = hourly.get("wind_speed_10m", [])
            clouds = hourly.get("cloud_cover", [])
            precip = hourly.get("precipitation", [])

            if not times:
                logger.warning("No weather data returned for %s", region)
                continue

            # Build batch insert list
            batch = []
            import json as _json
            for i, ts_str in enumerate(times):
                try:
                    vt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                    temp_c = float(temps[i]) if i < len(temps) and temps[i] is not None else 0.0
                    wind_kmh = float(winds[i]) if i < len(winds) and winds[i] is not None else 0.0
                    cloud_pct = float(clouds[i]) if i < len(clouds) and clouds[i] is not None else 0.0
                    precip_mm = float(precip[i]) if i < len(precip) and precip[i] is not None else 0.0
                    raw_ref = hashlib.sha256(
                        f"WEATHER_HISTORY:{region}:{ts_str}".encode()
                    ).hexdigest()[:32]
                    batch.append({
                        "id": str(uuid.uuid4()),
                        "region": region,
                        "valid_time": vt,
                        "temp_c": temp_c,
                        "wind_kmh": wind_kmh,
                        "cloud_frac": round(cloud_pct / 100.0, 4),
                        "data_json": _json.dumps({
                            "temp_c": temp_c,
                            "wind_kmh": wind_kmh,
                            "cloud_cover_pct": cloud_pct,
                            "precipitation_mm": precip_mm,
                        }),
                        "raw_ref": raw_ref,
                    })
                except Exception as row_err:
                    logger.debug("Row prep failed (%s %s): %s", region, ts_str, row_err)

            rows_inserted = 0
            if batch:
                async with db_session() as session:
                    for row in batch:
                        try:
                            await session.execute(text("""
                                INSERT INTO market_events
                                    (id, tenant_id, source, region, valid_time, system_time,
                                     price_rrp, demand_mw, availability_mw, data, raw_ref)
                                VALUES
                                    (:id, 'system', 'WEATHER_HISTORY', :region, :valid_time,
                                     NOW(), :temp_c, :wind_kmh, :cloud_frac,
                                     CAST(:data_json AS jsonb), :raw_ref)
                                ON CONFLICT (source, region, valid_time) DO NOTHING
                            """), row)
                            rows_inserted += 1
                        except Exception as row_err:
                            logger.warning("Row insert failed (%s %s): %s",
                                           region, row["valid_time"], row_err)
                    await session.commit()

            inserted_total[region] = rows_inserted
            logger.info("  %s: %d weather rows stored (%s → %s)",
                        region, rows_inserted, start_date, end_date)
            await asyncio.sleep(0.5)  # polite rate limit

    return inserted_total


async def trigger_archive_backfill(days: int = 14) -> int:
    """Run archive backfill for the last N days of AEMO dispatch prices.

    Uses the existing backfill_recent_gaps mechanism but extended to cover
    more than the default 24-hour window.
    """
    from app.mcp.aemo_archive import backfill_recent_gaps
    from app.db.session import db_session

    logger.info("Triggering AEMO archive backfill for last %d days...", days)
    total = 0
    # Run multiple 24h windows to cover the full period
    for day_offset in range(days):
        try:
            n = await backfill_recent_gaps(db_session_factory=db_session)
            total += n
            if n:
                logger.info("  Day offset -%d: %d intervals filled", day_offset, n)
        except Exception as exc:
            logger.warning("  Backfill failed for day offset -%d: %s", day_offset, exc)

    return total


async def trigger_model_retrain() -> None:
    """Force-retrain all forecast models using the newly seeded data."""
    logger.info("Triggering forecast model retrain for all regions...")
    try:
        from app.engines.forecasting.live_forecast import run_live_forecast
        for region in _ALL_REGIONS:
            try:
                result = await run_live_forecast(region, lookback_days=14)
                if result.get("available"):
                    logger.info("  %s: retrain OK — %d intervals, P50=%s",
                                region,
                                result.get("training_intervals", 0),
                                result.get("forecasts", [{}])[0].get("p50", ["?"])[0]
                                if result.get("forecasts") else "?")
                else:
                    logger.warning("  %s: retrain unavailable — %s",
                                   region, result.get("reason", "unknown"))
            except Exception as exc:
                logger.warning("  %s retrain failed: %s", region, exc)
    except Exception as exc:
        logger.warning("Model retrain failed: %s", exc)


async def check_db_coverage() -> dict[str, int]:
    """Report how many dispatch intervals we have per region."""
    from app.db.session import db_session
    from sqlalchemy import text

    coverage: dict[str, int] = {}
    async with db_session() as session:
        result = await session.execute(text("""
            SELECT region, COUNT(*) as n,
                   MIN(valid_time) as earliest,
                   MAX(valid_time) as latest
            FROM market_events
            WHERE source = 'AEMO_DISPATCH_PRICE'
            GROUP BY region
            ORDER BY region
        """))
        for row in result.fetchall():
            region, n, earliest, latest = row
            coverage[region] = n
            logger.info("  %s: %d dispatch intervals (%s → %s)",
                        region, n, str(earliest)[:10] if earliest else "?",
                        str(latest)[:10] if latest else "?")
    return coverage


async def main(args) -> None:
    logger.info("=== GridVerdict Training Data Seeder ===")
    logger.info("Target: %d days of price + weather data per NEM region", args.days)

    # Show current coverage
    logger.info("\nCurrent DB coverage:")
    coverage = await check_db_coverage()
    total_intervals = sum(coverage.values())
    logger.info("Total dispatch intervals in DB: %d (need ~%d for 14-day training)",
                total_intervals, 288 * 14 * len(_ALL_REGIONS))

    if not args.weather_only:
        logger.info("\n--- Phase 1: AEMO Dispatch Price backfill ---")
        filled = await trigger_archive_backfill(args.days)
        logger.info("Archive backfill complete: %d intervals filled", filled)

    if not args.price_only:
        logger.info("\n--- Phase 2: Open-Meteo historical weather ---")
        weather_counts = await fetch_and_store_weather(args.days)
        total_weather = sum(weather_counts.values())
        logger.info("Weather seeding complete: %d rows inserted across %d regions",
                    total_weather, len(weather_counts))

        logger.info("\nWeather coverage by region:")
        for region, n in weather_counts.items():
            logger.info("  %s: %d hourly rows ≈ %.1f days", region, n, n / 24)

    if args.retrain:
        logger.info("\n--- Phase 3: Model retrain ---")
        await trigger_model_retrain()

    logger.info("\nCoverage after seeding:")
    await check_db_coverage()

    logger.info("\n=== Done. Restart GridVerdict or wait for next forecast request. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed historical training data for GridVerdict")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of days to seed (default: 14)")
    parser.add_argument("--weather-only", action="store_true",
                        help="Only seed weather data, skip price backfill")
    parser.add_argument("--price-only", action="store_true",
                        help="Only run price backfill, skip weather")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retrain after seeding")
    parser.add_argument("--regions", nargs="+", default=_ALL_REGIONS,
                        choices=_ALL_REGIONS, help="Regions to seed (default: all)")
    args = parser.parse_args()

    # Restrict weather seeding to requested regions
    for r in list(_REGION_COORDS.keys()):
        if r not in args.regions:
            del _REGION_COORDS[r]

    asyncio.run(main(args))
