"""AEMO ST PASA (Short Term PASA) client — 7-day system adequacy forecast.

ST PASA = Short Term Projected Assessment of System Adequacy.
Published by AEMO every 30 minutes, covering the next 7 days by trading interval.
Available at NEMWeb without registration — same access as dispatch data.

Data provides:
  - Forecast demand by region (HH/LL scenarios)
  - Available generation (by region, by fuel type at trading interval level)
  - Reserve margin: how much headroom above demand
  - Generator outage schedule (planned + forced, aggregated by region)
  - LOR risk assessment: when reserve is expected to fall below threshold

Why this matters for >4h forecasting:
  The LNN/LEAR ensemble covers 0-30 min ahead using recent dispatch patterns.
  For 30 min - 7 days ahead, ST PASA provides AEMO's own forecast of supply/demand
  balance — which is more accurate than any statistical model for scheduled outages
  and demand forecasts beyond the LNN's training horizon.

Architecture for >4h forecast (not yet implemented — scaffold only):

  Horizon 0-30min:   LNN + LEAR + QRA ensemble (current implementation)
  Horizon 30min-4h:  Extended LEAR (24-lag) + AEMO predispatch intervals
  Horizon 4h-7days:  ST PASA region supply/demand + BOM 7-day weather + analogs
  Horizon >7 days:   ISP scenario price bands (Step Change / Slow Change)

  Each horizon has decreasing precision — this is by design and must be disclosed.

Data source:
  NEMWeb ST PASA reports: https://nemweb.com.au/Reports/Current/STPASA_SOD_Reports/
  Format: ZIP containing CSV with rows for each region × trading interval.

Key tables in ST PASA:
  STPASA_REGIONSOLUTION — per-region supply/demand forecast
  STPASA_CASESOLUTION — metadata (run datetime, case type)

AEMO publishes ST PASA every 30 minutes. Each report covers the next 7 days
at 30-minute trading interval resolution (336 intervals).
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# NEMWeb ST PASA directory (public, no credentials)
_ST_PASA_BASE = "https://nemweb.com.au/Reports/Current/STPASA_SOD_Reports/"

_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}

# Column names in STPASA_REGIONSOLUTION (may vary by AEMO version)
_COL_REGIONID = "REGIONID"
_COL_INTERVAL  = "INTERVAL_DATETIME"
_COL_DEMAND_LOW = "DEMAND10"       # low-demand scenario (10th percentile)
_COL_DEMAND_HIGH = "DEMAND50"      # central-demand scenario (50th percentile)
_COL_DEMAND_HH = "DEMAND90"        # high-demand scenario (90th percentile)
_COL_AVAIL = "AVAILABLEGENERATION" # total available generation
_COL_LOR1   = "LOR1FORECAST"       # LOR1 threshold exceedance probability
_COL_LOR2   = "LOR2FORECAST"
_COL_LOR3   = "LOR3FORECAST"


@dataclass
class StPasaInterval:
    """Supply/demand forecast for one region at one trading interval."""
    region: str
    interval_datetime: datetime
    demand_p10_mw: float | None    # low demand (10th percentile)
    demand_p50_mw: float | None    # central demand forecast
    demand_p90_mw: float | None    # high demand (90th percentile)
    available_gen_mw: float | None # available generation
    reserve_mw: float | None       # = available_gen - demand_p50 (headroom)
    lor1_risk: float | None        # 0-1 probability of LOR1 in this interval
    lor2_risk: float | None        # 0-1 probability of LOR2
    lor3_risk: float | None        # 0-1 probability of LOR3


@dataclass
class StPasaForecast:
    """Complete 7-day ST PASA forecast for one region."""
    region: str
    run_datetime: datetime                          # when AEMO published this run
    intervals: list[StPasaInterval] = field(default_factory=list)
    available: bool = False
    error: str | None = None

    @property
    def tight_intervals(self) -> list[StPasaInterval]:
        """Intervals where reserve is below 1000 MW or LOR1 risk > 0.2."""
        return [
            i for i in self.intervals
            if (i.reserve_mw is not None and i.reserve_mw < 1000)
            or (i.lor1_risk is not None and i.lor1_risk > 0.2)
        ]

    @property
    def next_lor_risk_interval(self) -> StPasaInterval | None:
        """Next interval with any LOR risk in the 7-day window."""
        for i in sorted(self.intervals, key=lambda x: x.interval_datetime):
            if (i.lor1_risk or 0) > 0.1 or (i.lor2_risk or 0) > 0.05:
                return i
        return None

    def to_dict(self) -> dict[str, Any]:
        tight = self.tight_intervals
        next_risk = self.next_lor_risk_interval
        return {
            "region":           self.region,
            "run_datetime":     self.run_datetime.isoformat(),
            "available":        self.available,
            "error":            self.error,
            "interval_count":   len(self.intervals),
            "tight_interval_count": len(tight),
            "next_lor_risk_interval": {
                "datetime": next_risk.interval_datetime.isoformat(),
                "demand_mw": next_risk.demand_p50_mw,
                "available_mw": next_risk.available_gen_mw,
                "reserve_mw": next_risk.reserve_mw,
                "lor1_risk": next_risk.lor1_risk,
                "lor2_risk": next_risk.lor2_risk,
            } if next_risk else None,
        }


async def fetch_st_pasa(region: str) -> StPasaForecast:
    """Fetch the latest AEMO ST PASA forecast for a region.

    Returns StPasaForecast with available=False on any error.
    Never raises — always returns a result.

    NOTE (2026-06): AEMO changed the ST PASA URL format in late 2024.
    The _ST_PASA_BASE URL above may need to be updated if the index
    directory listing format changes. This function uses directory-listing
    scraping — check NEMWeb if data is unavailable.
    """
    region = region.upper()
    if region not in _REGIONS:
        return StPasaForecast(
            region=region,
            run_datetime=datetime.now(timezone.utc),
            available=False,
            error=f"Unknown region: {region}",
        )

    try:
        # Step 1: Get the latest report file from the directory listing
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": _BROWSER_UA},
            follow_redirects=True,
        ) as client:
            index_resp = await client.get(_ST_PASA_BASE)
            if index_resp.status_code != 200:
                raise RuntimeError(f"ST PASA index returned {index_resp.status_code}")

            # Find the latest PUBLIC_STPASA_SOD_SOLUTION_*.zip file
            import re
            zips = re.findall(
                r'href="(PUBLIC_STPASA_SOD_SOLUTION_\d{14}_\d{14}\.zip)"',
                index_resp.text,
                re.IGNORECASE,
            )
            if not zips:
                raise RuntimeError("No ST PASA ZIP files found in index listing")

            latest_zip = sorted(zips)[-1]   # highest timestamp = most recent
            zip_url = _ST_PASA_BASE + latest_zip

            # Step 2: Download and parse
            report_resp = await client.get(zip_url)
            report_resp.raise_for_status()

        intervals = _parse_stpasa_zip(report_resp.content, region)
        run_dt = _extract_run_datetime(latest_zip)

        return StPasaForecast(
            region=region,
            run_datetime=run_dt,
            intervals=intervals,
            available=True,
        )

    except Exception as exc:
        logger.info("ST PASA fetch failed for %s (non-fatal): %s", region, exc)
        return StPasaForecast(
            region=region,
            run_datetime=datetime.now(timezone.utc),
            available=False,
            error=str(exc)[:200],
        )


def _parse_stpasa_zip(content: bytes, region: str) -> list[StPasaInterval]:
    """Parse ST PASA ZIP content for the specified region."""
    intervals: list[StPasaInterval] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if "STPASA" in name.upper() and name.endswith(".CSV"):
                    csv_text = zf.read(name).decode("utf-8", errors="replace")
                    intervals.extend(_parse_stpasa_csv(csv_text, region))
    except Exception as exc:
        logger.debug("ST PASA ZIP parse failed: %s", exc)
    return intervals


def _parse_stpasa_csv(text: str, region: str) -> list[StPasaInterval]:
    """Parse STPASA_REGIONSOLUTION CSV rows for the specified region."""
    intervals: list[StPasaInterval] = []
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None

    for row in reader:
        if not row:
            continue
        tag = row[0].strip().upper()
        if tag == "I" and "REGIONID" in " ".join(row).upper():
            header = [c.strip().upper() for c in row]
            continue
        if tag != "D" or header is None:
            continue
        try:
            rd = dict(zip(header, row))
            if rd.get(_COL_REGIONID, "").strip().upper() != region:
                continue

            interval_str = rd.get(_COL_INTERVAL, "").strip()
            if not interval_str:
                continue
            interval_dt = _parse_stpasa_dt(interval_str)

            def _f(col: str) -> float | None:
                v = rd.get(col, "").strip()
                return float(v) if v and v not in ("", "NULL") else None

            demand_p50 = _f(_COL_DEMAND_HIGH)
            avail = _f(_COL_AVAIL)
            reserve = round(avail - demand_p50, 1) if avail is not None and demand_p50 is not None else None

            intervals.append(StPasaInterval(
                region=region,
                interval_datetime=interval_dt,
                demand_p10_mw=_f(_COL_DEMAND_LOW),
                demand_p50_mw=demand_p50,
                demand_p90_mw=_f(_COL_DEMAND_HH),
                available_gen_mw=avail,
                reserve_mw=reserve,
                lor1_risk=_f(_COL_LOR1),
                lor2_risk=_f(_COL_LOR2),
                lor3_risk=_f(_COL_LOR3),
            ))
        except Exception:
            continue

    # Sort by interval ascending
    intervals.sort(key=lambda i: i.interval_datetime)
    return intervals


def _parse_stpasa_dt(s: str) -> datetime:
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse ST PASA datetime: {s!r}")


def _extract_run_datetime(filename: str) -> datetime:
    """Extract run datetime from ST PASA filename (PUBLIC_STPASA_SOD_SOLUTION_YYYYMMDDHHMMSS_*.zip)."""
    import re
    m = re.search(r"(\d{14})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
