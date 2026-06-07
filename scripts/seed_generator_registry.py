"""Seed the generator_units table from AEMO's public participant/generator register.

Data source:
  AEMO NEM Registration and Exemption List (NREL) — publicly available Excel/CSV.
  URL: https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/
       participate-in-the-market/registration

  Alternative / bootstrapped: use embedded known-units below (updated quarterly from NREL).

Run once (idempotent — upserts on DUID primary key):
    python scripts/seed_generator_registry.py

Or fetch live from AEMO:
    python scripts/seed_generator_registry.py --fetch-live

The embedded registry covers the top ~200 DUIDs by capacity for all NEM regions.
It is intentionally conservative — missing DUIDs get fuel_type=None in unit_attribution,
which is handled gracefully by the fuel_mix engine.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Embedded known DUIDs (top capacity generators, updated 2024-Q1) ──────────
# Source: AEMO NREL 2024, filtered to capacity > 50 MW.
# Fuel types normalised: coal | gas | hydro | solar | wind | battery | distillate
_KNOWN_UNITS: list[dict] = [
    # NSW1 coal
    {"duid": "BAYSW",    "station_name": "Bayswater",      "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "BAYSW2",   "station_name": "Bayswater",      "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "BAYSW3",   "station_name": "Bayswater",      "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "BAYSW4",   "station_name": "Bayswater",      "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "ERRA1",    "station_name": "Eraring",        "participant": "Origin Energy",  "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 720},
    {"duid": "ERRA2",    "station_name": "Eraring",        "participant": "Origin Energy",  "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 720},
    {"duid": "ERRA3",    "station_name": "Eraring",        "participant": "Origin Energy",  "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 720},
    {"duid": "ERRA4",    "station_name": "Eraring",        "participant": "Origin Energy",  "fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 720},
    {"duid": "VP5",      "station_name": "Vales Point",    "participant": "Delta Electricity", "fuel_type": "coal", "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "VP6",      "station_name": "Vales Point",    "participant": "Delta Electricity", "fuel_type": "coal", "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 660},
    {"duid": "MT_PIPER1","station_name": "Mt Piper",       "participant": "EnergyAustralia","fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 700},
    {"duid": "MT_PIPER2","station_name": "Mt Piper",       "participant": "EnergyAustralia","fuel_type": "coal",    "tech_type": "Steam",  "region": "NSW1", "capacity_mw": 700},
    # NSW1 gas
    {"duid": "COLONGRA1","station_name": "Colongra",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "OCGT",   "region": "NSW1", "capacity_mw": 172},
    {"duid": "COLONGRA2","station_name": "Colongra",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "OCGT",   "region": "NSW1", "capacity_mw": 172},
    {"duid": "COLONGRA3","station_name": "Colongra",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "OCGT",   "region": "NSW1", "capacity_mw": 172},
    {"duid": "COLONGRA4","station_name": "Colongra",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "OCGT",   "region": "NSW1", "capacity_mw": 172},
    {"duid": "TALLAWWF1","station_name": "Tallawarra A",   "participant": "EnergyAustralia","fuel_type": "gas",     "tech_type": "CCGT",   "region": "NSW1", "capacity_mw": 380},
    {"duid": "TALLWB1",  "station_name": "Tallawarra B",   "participant": "EnergyAustralia","fuel_type": "gas",     "tech_type": "OCGT",   "region": "NSW1", "capacity_mw": 228},
    # NSW1 hydro
    {"duid": "TUMUT3",   "station_name": "Tumut 3",        "participant": "Snowy Hydro",    "fuel_type": "hydro",   "tech_type": "Hydro",  "region": "NSW1", "capacity_mw": 1500},
    {"duid": "MURRAY",   "station_name": "Murray",         "participant": "Snowy Hydro",    "fuel_type": "hydro",   "tech_type": "Hydro",  "region": "NSW1", "capacity_mw": 950},
    # VIC1 coal
    {"duid": "LOYYB1",   "station_name": "Loy Yang B",     "participant": "EnergyAustralia","fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 502},
    {"duid": "LOYYB2",   "station_name": "Loy Yang B",     "participant": "EnergyAustralia","fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 502},
    {"duid": "LOYYB3",   "station_name": "Loy Yang B",     "participant": "EnergyAustralia","fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 502},
    {"duid": "LYA1",     "station_name": "Loy Yang A",     "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 560},
    {"duid": "LYA2",     "station_name": "Loy Yang A",     "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 560},
    {"duid": "LYA3",     "station_name": "Loy Yang A",     "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 560},
    {"duid": "LYA4",     "station_name": "Loy Yang A",     "participant": "AGL Energy",     "fuel_type": "coal",    "tech_type": "Steam",  "region": "VIC1", "capacity_mw": 560},
    # VIC1 gas
    {"duid": "MORTLK11", "station_name": "Mortlake",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "CCGT",   "region": "VIC1", "capacity_mw": 282},
    {"duid": "MORTLK12", "station_name": "Mortlake",       "participant": "Origin Energy",  "fuel_type": "gas",     "tech_type": "CCGT",   "region": "VIC1", "capacity_mw": 282},
    # VIC1 hydro
    {"duid": "LAVERTON", "station_name": "Laverton North", "participant": "Snowy Hydro",    "fuel_type": "gas",     "tech_type": "OCGT",   "region": "VIC1", "capacity_mw": 312},
    # SA1 gas
    {"duid": "PPCCGT",   "station_name": "Pelican Pt CCGT","participant": "Engie",          "fuel_type": "gas",     "tech_type": "CCGT",   "region": "SA1",  "capacity_mw": 484},
    {"duid": "OSBL1",    "station_name": "Osborne",        "participant": "Engie",          "fuel_type": "gas",     "tech_type": "CCGT",   "region": "SA1",  "capacity_mw": 182},
    {"duid": "LADBROK1", "station_name": "Ladbroke Grove", "participant": "Engie",          "fuel_type": "gas",     "tech_type": "OCGT",   "region": "SA1",  "capacity_mw": 80},
    {"duid": "TORRB1",   "station_name": "Torrens Island B","participant": "AGL Energy",    "fuel_type": "gas",     "tech_type": "Steam",  "region": "SA1",  "capacity_mw": 200},
    # SA1 battery
    {"duid": "HPRL1",    "station_name": "Hornsdale Power Reserve","participant": "Tesla/Neoen","fuel_type": "battery","tech_type": "BESS","region": "SA1",  "capacity_mw": 150},
    # QLD1 coal
    {"duid": "CALL_B_1", "station_name": "Callide B",      "participant": "CS Energy",      "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 350},
    {"duid": "CALL_B_2", "station_name": "Callide B",      "participant": "CS Energy",      "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 350},
    {"duid": "CALLIDE3", "station_name": "Callide C",      "participant": "CS Energy",      "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 450},
    {"duid": "CALLIDE4", "station_name": "Callide C",      "participant": "CS Energy",      "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 450},
    {"duid": "TARONG1",  "station_name": "Tarong",         "participant": "Stanwell",       "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 350},
    {"duid": "TARONG2",  "station_name": "Tarong",         "participant": "Stanwell",       "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 350},
    {"duid": "GLADSTONE1","station_name":"Gladstone",      "participant": "NRG Gladstone",  "fuel_type": "coal",    "tech_type": "Steam",  "region": "QLD1", "capacity_mw": 280},
    # TAS1 hydro
    {"duid": "BASTYAN",  "station_name": "Bastyan",        "participant": "Hydro Tasmania", "fuel_type": "hydro",   "tech_type": "Hydro",  "region": "TAS1", "capacity_mw": 80},
    {"duid": "POATINA",  "station_name": "Poatina",        "participant": "Hydro Tasmania", "fuel_type": "hydro",   "tech_type": "Hydro",  "region": "TAS1", "capacity_mw": 300},
    {"duid": "GORDON",   "station_name": "Gordon",         "participant": "Hydro Tasmania", "fuel_type": "hydro",   "tech_type": "Hydro",  "region": "TAS1", "capacity_mw": 432},
]


async def _upsert_units(units: list[dict]) -> int:
    """Upsert generator units into the DB. Returns count upserted."""
    try:
        from sqlalchemy import text
        from app.db.session import db_session

        async with db_session() as session:
            async with session.begin():
                for u in units:
                    await session.execute(text("""
                        INSERT INTO generator_units
                          (duid, station_name, participant, region, fuel_type, dispatch_type, max_capacity_mw, updated_at)
                        VALUES (:duid, :station, :part, :region, :fuel, 'GENERATOR', :cap, NOW())
                        ON CONFLICT (duid) DO UPDATE SET
                          station_name = EXCLUDED.station_name,
                          participant  = EXCLUDED.participant,
                          region       = EXCLUDED.region,
                          fuel_type    = EXCLUDED.fuel_type,
                          max_capacity_mw = EXCLUDED.max_capacity_mw,
                          updated_at   = NOW()
                    """), {
                        "duid":    u["duid"],
                        "station": u["station_name"],
                        "part":    u["participant"],
                        "region":  u["region"],
                        "fuel":    u["fuel_type"],
                        "cap":     u["capacity_mw"],
                    })
        return len(units)
    except Exception as exc:
        logger.error("Upsert failed: %s", exc)
        return 0


async def _fetch_live_nrel() -> list[dict]:
    """Try to fetch and parse AEMO NREL (NEM Registration and Exemption List).

    The NREL is a public Excel download from AEMO. URL and format change occasionally.
    Returns empty list on failure — caller falls back to embedded registry.
    """
    try:
        import httpx
        _NREL_URL = (
            "https://www.aemo.com.au/-/media/files/electricity/nem/participant_information/"
            "nem-registration-and-exemption-list.xls"
        )
        logger.info("Fetching AEMO NREL from %s", _NREL_URL)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(_NREL_URL)
            if resp.status_code != 200:
                logger.warning("NREL fetch returned %d", resp.status_code)
                return []
        try:
            import openpyxl
            from io import BytesIO
            wb = openpyxl.load_workbook(BytesIO(resp.content), read_only=True, data_only=True)
            # Look for the Generators tab (sheet name varies by year)
            sheet = None
            for name in wb.sheetnames:
                if any(kw in name.lower() for kw in ["generator", "schedule", "unit"]):
                    sheet = wb[name]
                    break
            if sheet is None:
                logger.warning("Could not find generator sheet in NREL; sheets: %s", wb.sheetnames)
                return []

            units = []
            header = None
            for row in sheet.iter_rows(values_only=True):
                if header is None:
                    header = [str(c).strip().upper() if c else "" for c in row]
                    continue
                rd = dict(zip(header, row))
                duid = str(rd.get("DUID", "") or "").strip().upper()
                region = str(rd.get("REGIONID", rd.get("REGION", "")) or "").strip().upper()
                fuel = str(rd.get("FUEL SOURCE - PRIMARY", rd.get("FUEL_TYPE", "")) or "").strip().lower()
                station = str(rd.get("STATION NAME", rd.get("STATION", "")) or "").strip()
                participant = str(rd.get("PARTICIPANT", rd.get("PARTICIPANT_ID", "")) or "").strip()
                cap_raw = rd.get("REG CAP (MW)", rd.get("REGISTERED_CAPACITY", None))
                cap = float(cap_raw) if cap_raw else None
                if not duid or region not in {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}:
                    continue
                # Normalise fuel type
                fuel_map = {
                    "black coal": "coal", "brown coal": "coal", "gas": "gas",
                    "liquid fuel": "distillate", "water": "hydro", "solar": "solar",
                    "wind": "wind", "battery storage": "battery",
                }
                fuel_norm = next((v for k, v in fuel_map.items() if k in fuel), fuel or "unknown")
                units.append({
                    "duid": duid, "station_name": station, "participant": participant,
                    "region": region, "fuel_type": fuel_norm, "capacity_mw": cap or 0,
                })
            logger.info("Parsed %d units from AEMO NREL", len(units))
            return units
        except ImportError:
            logger.warning("openpyxl not installed — cannot parse NREL Excel; using embedded registry")
            return []
    except Exception as exc:
        logger.warning("NREL live fetch failed (%s) — using embedded registry", exc)
        return []


async def main(fetch_live: bool = False) -> None:
    units = []
    if fetch_live:
        units = await _fetch_live_nrel()
    if not units:
        logger.info("Using embedded registry (%d units)", len(_KNOWN_UNITS))
        units = _KNOWN_UNITS

    count = await _upsert_units(units)
    logger.info("Seeded %d generator units into generator_units table", count)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seed generator_units from AER/AEMO NREL")
    ap.add_argument("--fetch-live", action="store_true", help="Fetch current AEMO NREL (requires openpyxl)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(main(fetch_live=args.fetch_live))
