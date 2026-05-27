"""AEMO NEMWeb live data client — no credentials, no stubs.

Fetches real 5-minute dispatch prices from public NEMWeb endpoints.
Parses DISPATCHPRICE CSV, extracts RRP, demand, availability per region.

NEMWeb directory pattern:
  https://nemweb.com.au/Reports/Current/DispatchIS_Reports/
  File: PUBLIC_DISPATCHIS_{YYYYMMDD}_{HHMM}_{seq}.zip
  Inner CSV: PUBLIC_DISPATCHPRICE_{YYYYMMDD}_{HHMM}_{seq}.CSV

Dispatch cycle: every 5 minutes. Fresh data is <=180s old.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}

_DISPATCH_DIR = "/Reports/Current/DispatchIS_Reports/"
_PREDISPATCH_DIR = "/Reports/Current/PredispatchIS_Reports/"

# Column indices in DISPATCHPRICE CSV (0-indexed, after stripping the 'I' header row)
# Row type D: D,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,
#             INTERVENTION,RRP,EEPSRMW,...,TOTALDEMAND,AVAILABLEGENERATION,...
_COL_SETTLEMENTDATE = 4
_COL_REGIONID = 6
_COL_RRP = 9
_COL_TOTALDEMAND = 27
_COL_AVAILABLEGENERATION = 28

# Some versions shift columns; we detect by header row
_HEADER_RRP = "RRP"
_HEADER_DEMAND = "TOTALDEMAND"
_HEADER_AVAIL = "AVAILABLEGENERATION"


@dataclass
class DispatchPrice:
    region: str
    valid_time: datetime          # SETTLEMENTDATE in UTC
    system_time: datetime         # when GridVerdict fetched this
    price_rrp: float              # $/MWh
    demand_mw: float
    availability_mw: float
    raw_ref: str                  # sha256 of zip content


@dataclass
class LiveMarketSnapshot:
    """All NEM regions in one dispatch interval."""
    interval: datetime
    fetched_at: datetime
    regions: dict[str, DispatchPrice] = field(default_factory=dict)
    raw_ref: str = ""

    def get(self, region: str) -> DispatchPrice | None:
        return self.regions.get(region.upper())

    def staleness_seconds(self, region: str) -> int:
        dp = self.get(region)
        if not dp:
            return 9999
        return max(0, int((datetime.now(timezone.utc) - dp.valid_time).total_seconds()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "regions": {r: _dp_to_dict(dp) for r, dp in self.regions.items()},
        }


def _dp_to_dict(dp: DispatchPrice) -> dict[str, Any]:
    return {
        "region": dp.region,
        "valid_time": dp.valid_time.isoformat(),
        "price_rrp": dp.price_rrp,
        "demand_mw": dp.demand_mw,
        "availability_mw": dp.availability_mw,
    }


class AEMOLiveClient:
    """Async client for live NEMWeb dispatch price data."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base = settings.nemweb_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                headers={"User-Agent": "GridVerdict/1.0 (research; public data only)"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def fetch_latest_snapshot(self) -> LiveMarketSnapshot:
        """Download and parse the most recent dispatch price file."""
        async with self._lock:
            client = await self._get_client()
            zip_url, zip_content = await self._fetch_latest_zip(client, _DISPATCH_DIR)
            raw_ref = hashlib.sha256(zip_content).hexdigest()
            prices = _parse_dispatch_zip(zip_content, raw_ref)
            now = datetime.now(timezone.utc)
            if not prices:
                raise RuntimeError("No dispatch prices parsed from NEMWeb zip")
            interval = max(p.valid_time for p in prices.values())
            return LiveMarketSnapshot(
                interval=interval,
                fetched_at=now,
                regions=prices,
                raw_ref=raw_ref,
            )

    async def fetch_predispatch(self) -> dict[str, list["PredispatchInterval"]]:
        """Download and parse the latest PREDISPATCH run from NEMWeb.

        Returns a dict of region → list[PredispatchInterval] sorted by interval.
        Each interval covers 30-min settlement periods up to ~2 hours ahead.
        Returns {} on failure — never raises (pre-dispatch is supplementary data).
        """
        try:
            async with self._lock:
                client = await self._get_client()
                _, zip_content = await self._fetch_latest_zip(client, _PREDISPATCH_DIR)
                return _parse_predispatch_zip(zip_content)
        except Exception as exc:
            logger.debug("Pre-dispatch fetch failed (non-fatal): %s", exc)
            return {}

    async def _fetch_latest_zip(
        self, client: httpx.AsyncClient, directory: str
    ) -> tuple[str, bytes]:
        """List the directory index, find the most recent zip, download it."""
        resp = await client.get(directory)
        resp.raise_for_status()
        zip_url = _extract_latest_zip_url(resp.text, directory)
        if not zip_url:
            raise RuntimeError(f"Could not locate dispatch zip in {directory}")
        logger.debug("Fetching %s", zip_url)
        zip_resp = await client.get(zip_url)
        zip_resp.raise_for_status()
        return zip_url, zip_resp.content


def _extract_latest_zip_url(html: str, directory: str) -> str | None:
    """Parse the NEMWeb directory listing HTML and return the most recent zip href."""
    # NEMWeb directory lists files as <a href="PUBLIC_DISPATCHIS_*.zip">
    pattern = re.compile(r'href="([^"]+\.zip)"', re.IGNORECASE)
    matches = pattern.findall(html)
    if not matches:
        return None
    # Sort lexicographically — filenames encode datetime, so latest = last
    matches.sort()
    latest = matches[-1]
    if latest.startswith("/"):
        return latest
    return directory + latest


def _parse_dispatch_zip(content: bytes, raw_ref: str) -> dict[str, DispatchPrice]:
    """Unzip and parse DISPATCHPRICE + REGIONSUM tables into one DispatchPrice per region.

    NEMWeb DispatchIS zips contain a combined CSV with multiple tables identified
    by their row prefix: I,DISPATCH,PRICE,... and I,DISPATCH,REGIONSUM,...
    RRP lives in the PRICE table; TOTALDEMAND and AVAILABLEGENERATION live in REGIONSUM.
    D rows carry the table name at parts[2] so we route each row to the right parser.
    """
    system_time = datetime.now(timezone.utc)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next((n for n in zf.namelist() if n.endswith(".CSV")), None)
        if csv_name is None:
            raise RuntimeError("No CSV found inside dispatch zip")
        raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

    # Per-table column mappings (detected from I header rows)
    _price_cols: dict[str, int | None] = {
        "date": None, "region": None, "rrp": None, "demand": None, "avail": None
    }
    _regionsum_cols: dict[str, int | None] = {
        "date": None, "region": None, "demand": None, "avail": None
    }

    # Raw parsed rows before joining
    # price_rows: region → (valid_time, rrp, demand_from_price, avail_from_price)
    price_rows: dict[str, tuple] = {}
    regionsum_rows: dict[str, tuple[float, float]] = {}   # region → (demand, avail)

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if not parts:
            continue
        row_type = parts[0].upper()
        table = parts[2].upper() if len(parts) > 2 else ""

        if row_type == "I":
            upper = [p.upper().strip() for p in parts]
            if table == "PRICE":
                _price_cols["date"] = upper.index("SETTLEMENTDATE") if "SETTLEMENTDATE" in upper else _COL_SETTLEMENTDATE
                _price_cols["region"] = upper.index("REGIONID") if "REGIONID" in upper else _COL_REGIONID
                _price_cols["rrp"] = upper.index(_HEADER_RRP) if _HEADER_RRP in upper else _COL_RRP
                # Older format (v4) embeds TOTALDEMAND/AVAILABLEGENERATION in PRICE
                _price_cols["demand"] = upper.index(_HEADER_DEMAND) if _HEADER_DEMAND in upper else None
                _price_cols["avail"] = upper.index(_HEADER_AVAIL) if _HEADER_AVAIL in upper else None
            elif table == "REGIONSUM":
                _regionsum_cols["date"] = upper.index("SETTLEMENTDATE") if "SETTLEMENTDATE" in upper else _COL_SETTLEMENTDATE
                _regionsum_cols["region"] = upper.index("REGIONID") if "REGIONID" in upper else _COL_REGIONID
                _regionsum_cols["demand"] = upper.index(_HEADER_DEMAND) if _HEADER_DEMAND in upper else _COL_TOTALDEMAND
                _regionsum_cols["avail"] = upper.index(_HEADER_AVAIL) if _HEADER_AVAIL in upper else _COL_AVAILABLEGENERATION
            continue

        if row_type != "D":
            continue

        try:
            if table == "PRICE":
                ci = _price_cols
                if ci["rrp"] is None:
                    continue
                region = parts[ci["region"]].strip().strip('"').upper()
                if region not in _REGIONS:
                    continue
                valid_time = _parse_aemo_dt(parts[ci["date"]].strip().strip('"'))
                rrp = float(parts[ci["rrp"]])
                # Extract demand/avail from PRICE row if columns exist (v4 format)
                demand_in_price = 0.0
                avail_in_price = 0.0
                if ci["demand"] is not None and ci["demand"] < len(parts):
                    d_raw = parts[ci["demand"]].strip()
                    demand_in_price = float(d_raw) if d_raw else 0.0
                if ci["avail"] is not None and ci["avail"] < len(parts):
                    a_raw = parts[ci["avail"]].strip()
                    avail_in_price = float(a_raw) if a_raw else 0.0
                existing = price_rows.get(region)
                if existing is None or valid_time >= existing[0]:
                    price_rows[region] = (valid_time, rrp, demand_in_price, avail_in_price)

            elif table == "REGIONSUM":
                ci = _regionsum_cols
                if ci["demand"] is None:
                    continue
                region = parts[ci["region"]].strip().strip('"').upper()
                if region not in _REGIONS:
                    continue
                demand_raw = parts[ci["demand"]].strip()
                avail_raw = parts[ci["avail"]].strip()
                demand = float(demand_raw) if demand_raw else 0.0
                avail = float(avail_raw) if avail_raw else 0.0
                regionsum_rows[region] = (demand, avail)

        except (ValueError, IndexError):
            continue

    # Join: REGIONSUM takes priority for demand/avail (v5); fall back to PRICE values (v4)
    prices: dict[str, DispatchPrice] = {}
    for region, row in price_rows.items():
        valid_time, rrp, d_price, a_price = row
        demand, avail = regionsum_rows.get(region, (d_price, a_price))
        prices[region] = DispatchPrice(
            region=region,
            valid_time=valid_time,
            system_time=system_time,
            price_rrp=rrp,
            demand_mw=demand,
            availability_mw=avail,
            raw_ref=raw_ref,
        )

    return prices


@dataclass
class PredispatchInterval:
    """One 30-minute pre-dispatch interval from AEMO PREDISPATCH run."""
    region: str
    interval_datetime: datetime   # settlement interval in UTC
    rrp: float                    # pre-dispatch RRP ($/MWh)
    demand_mw: float
    raw_ref: str                  # sha256 of the predispatch zip


def _parse_predispatch_zip(content: bytes) -> dict[str, list[PredispatchInterval]]:
    """Parse PREDISPATCH CSV from a NEMWeb predispatch zip.

    PREDISPATCH CSV has 30-min intervals, row type D, table PREDISPATCHPRICE.
    Key columns: REGIONID, INTERVAL_DATETIME, RRP, TOTALDEMAND.
    Returns region → intervals sorted by interval_datetime (earliest first).
    """
    raw_ref = hashlib.sha256(content).hexdigest()
    result: dict[str, list[PredispatchInterval]] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next(
                (n for n in zf.namelist() if "PREDISPATCH" in n.upper() and n.endswith(".CSV")),
                None,
            )
            if csv_name is None:
                return {}
            text = zf.read(csv_name).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        return {}

    col_region: int | None = None
    col_interval: int | None = None
    col_rrp: int | None = None
    col_demand: int | None = None

    for raw_line in text.splitlines():
        parts = raw_line.strip().split(",")
        if not parts:
            continue
        row_type = parts[0].upper()

        if row_type == "I" and "PREDISPATCHPRICE" in raw_line.upper():
            header = [p.strip().upper() for p in parts]
            try:
                col_region = header.index("REGIONID")
                col_interval = header.index("INTERVAL_DATETIME")
                col_rrp = header.index("RRP")
                col_demand = header.index("TOTALDEMAND")
            except ValueError:
                pass
            continue

        if row_type != "D" or col_rrp is None:
            continue

        try:
            region = parts[col_region].strip().upper()
            if region not in _REGIONS:
                continue
            interval_dt = _parse_aemo_dt(parts[col_interval].strip())
            rrp = float(parts[col_rrp])
            demand = float(parts[col_demand]) if parts[col_demand].strip() else 0.0
            interval = PredispatchInterval(
                region=region,
                interval_datetime=interval_dt,
                rrp=rrp,
                demand_mw=demand,
                raw_ref=raw_ref,
            )
            result.setdefault(region, []).append(interval)
        except (ValueError, IndexError):
            continue

    # Sort each region's intervals chronologically
    for region in result:
        result[region].sort(key=lambda x: x.interval_datetime)

    return result


def _parse_aemo_dt(s: str) -> datetime:
    """Parse AEMO settlement date string to UTC datetime.

    AEMO dates are in AEST (UTC+10) or AEDT (UTC+11) without explicit offset.
    We store in UTC; AEMO uses the convention that the date rolls at midnight AEST.
    We use +10 (AEST) year-round — acceptable for evidence references.
    """
    from datetime import timedelta

    # Format: "2026/05/23 14:30:00" or "2026-05-23 14:30:00"
    s = s.replace("/", "-")
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    # AEMO SETTLEMENTDATE is AEST = UTC+10
    return dt.replace(tzinfo=timezone.utc) - timedelta(hours=10)


# Module-level singleton
_client: AEMOLiveClient | None = None


def get_aemo_client() -> AEMOLiveClient:
    global _client
    if _client is None:
        _client = AEMOLiveClient()
    return _client
