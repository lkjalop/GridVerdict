"""One-time historical weather backfill using Open-Meteo ERA5 reanalysis.

Fills the weather_observations table from 2022-08-01 to yesterday for all
NEM regions. Safe to re-run — upserts on (region, observed_at).

Usage:
    python scripts/backfill_weather.py [--from 2022-08-01] [--to 2024-07-31]

Open-Meteo ERA5 API is free, no API key required, up to 10,000 calls/day.
Each region needs one API call per year of data requested.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# NEM region → approximate centroid lat/lon for weather lookup
_REGION_COORDS: dict[str, tuple[float, float]] = {
    "NSW1": (-33.87, 151.21),   # Sydney
    "VIC1": (-37.81, 144.96),   # Melbourne
    "QLD1": (-27.47, 153.03),   # Brisbane
    "SA1":  (-34.93, 138.60),   # Adelaide
    "TAS1": (-42.88, 147.33),   # Hobart
}

_OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Seasonal temperature norms per region (monthly means, degrees C)
# Used to compute deviation. Matches the values in weather_client.py.
from app.mcp.weather_client import SEASONAL_TEMP_NORMS


async def fetch_hourly_weather(
    client: httpx.AsyncClient,
    region: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch hourly ERA5 weather for one region and date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,wind_speed_10m,precipitation,cloud_cover,relative_humidity_2m",
        "timezone": "Australia/Sydney",
    }
    resp = await client.get(_OPEN_METEO_BASE, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precip = hourly.get("precipitation", [])
    cloud = hourly.get("cloud_cover", [])
    humidity = hourly.get("relative_humidity_2m", [])

    rows = []
    for i, ts_str in enumerate(times):
        try:
            # Open-Meteo returns local time strings — parse and convert to UTC
            local_dt = datetime.fromisoformat(ts_str)
            # Approximate UTC offset for AET (AEST=UTC+10, AEDT=UTC+11)
            # Good enough for weather correlation — not used for dispatch timestamps
            utc_dt = local_dt.replace(tzinfo=timezone.utc) - timedelta(hours=10)

            temp = temps[i] if i < len(temps) else None
            month = utc_dt.month
            norm = SEASONAL_TEMP_NORMS.get(region, {}).get(month)
            deviation = round(temp - norm, 2) if temp is not None and norm is not None else None

            rows.append({
                "region": region,
                "observed_at": utc_dt,
                "temperature_c": temp,
                "temp_deviation_c": deviation,
                "wind_speed_kmh": winds[i] if i < len(winds) else None,
                "precipitation_mm": precip[i] if i < len(precip) else None,
                "cloud_cover_pct": cloud[i] if i < len(cloud) else None,
                "humidity_pct": humidity[i] if i < len(humidity) else None,
                "source_count": 1,
                "raw_consensus": {"source": "open_meteo_era5"},
            })
        except Exception as exc:
            logger.debug("Skipping row %d for %s: %s", i, region, exc)
            continue
    return rows


async def upsert_weather_rows(rows: list[dict]) -> int:
    """Upsert rows into weather_observations. Returns count inserted."""
    if not rows:
        return 0

    from app.db.session import db_session
    from app.db.models import WeatherObservation
    import uuid

    try:
        from sqlalchemy.dialects.postgresql import insert as _insert
    except ImportError:
        from sqlalchemy.dialects.sqlite import insert as _insert

    BATCH = 500
    total = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        records = [
            {
                "id": str(uuid.uuid4()),
                "region": r["region"],
                "observed_at": r["observed_at"],
                "temperature_c": r.get("temperature_c"),
                "temp_deviation_c": r.get("temp_deviation_c"),
                "wind_speed_kmh": r.get("wind_speed_kmh"),
                "precipitation_mm": r.get("precipitation_mm"),
                "cloud_cover_pct": r.get("cloud_cover_pct"),
                "humidity_pct": r.get("humidity_pct"),
                "source_count": r.get("source_count", 1),
                "raw_consensus": r.get("raw_consensus", {}),
            }
            for r in batch
        ]
        async with db_session() as session:
            stmt = _insert(WeatherObservation).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["region", "observed_at"],
                set_={
                    "temperature_c": stmt.excluded.temperature_c,
                    "temp_deviation_c": stmt.excluded.temp_deviation_c,
                    "wind_speed_kmh": stmt.excluded.wind_speed_kmh,
                    "precipitation_mm": stmt.excluded.precipitation_mm,
                    "cloud_cover_pct": stmt.excluded.cloud_cover_pct,
                    "humidity_pct": stmt.excluded.humidity_pct,
                },
            )
            await session.execute(stmt)
            await session.commit()
        total += len(batch)
    return total


async def run_backfill(start: date, end: date) -> None:
    logger.info("Weather backfill: %s to %s for %d regions",
                start, end, len(_REGION_COORDS))

    async with httpx.AsyncClient() as client:
        for region, (lat, lon) in _REGION_COORDS.items():
            logger.info("Fetching %s (%s, %s)...", region, lat, lon)
            try:
                rows = await fetch_hourly_weather(client, region, lat, lon, start, end)
                inserted = await upsert_weather_rows(rows)
                logger.info("%s: fetched %d hourly rows, upserted %d", region, len(rows), inserted)
            except Exception as exc:
                logger.error("%s: failed — %s", region, exc)
            # Polite delay between regions
            await asyncio.sleep(1.0)

    logger.info("Backfill complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical weather from Open-Meteo ERA5")
    parser.add_argument("--from", dest="start", default="2022-08-01",
                        help="Start date YYYY-MM-DD (default: 2022-08-01)")
    parser.add_argument("--to", dest="end", default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)

    if start > end:
        logger.error("Start date must be before end date")
        sys.exit(1)

    asyncio.run(run_backfill(start, end))


if __name__ == "__main__":
    main()
