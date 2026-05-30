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
    fcas_prices: dict | None = None  # {raise6sec, lower6sec, raise60sec, lower60sec,
                                     #  raise5min, lower5min, raisereg, lowerreg} all $/MWh


@dataclass
class LiveMarketSnapshot:
    """All NEM regions in one dispatch interval."""
    interval: datetime
    fetched_at: datetime
    regions: dict[str, DispatchPrice] = field(default_factory=dict)
    raw_ref: str = ""
    # Sprint T: extended tables from the same DISPATCHIS zip
    unit_dispatch_rows: list[dict] = field(default_factory=list)   # DISPATCHLOAD — DUID level
    constraint_rows: list[dict] = field(default_factory=list)       # DISPATCHCONSTRAINT — binding only
    interconnector_rows: list[dict] = field(default_factory=list)   # DISPATCHINTERCONNECTORRES

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
    d: dict[str, Any] = {
        "region": dp.region,
        "valid_time": dp.valid_time.isoformat(),
        "price_rrp": dp.price_rrp,
        "demand_mw": dp.demand_mw,
        "availability_mw": dp.availability_mw,
    }
    if dp.fcas_prices:
        d["fcas_prices"] = dp.fcas_prices
    return d


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
        """Download and parse the most recent dispatch price file.

        Also parses DISPATCHLOAD (unit-level), DISPATCHCONSTRAINT (binding constraints),
        and DISPATCHINTERCONNECTORRES from the same zip — these populate snapshot fields
        used by the scheduler to write causal attribution tables (UnitDispatchEvent,
        MarketDriverEvent) without a second network round-trip.
        """
        async with self._lock:
            client = await self._get_client()
            zip_url, zip_content = await self._fetch_latest_zip(client, _DISPATCH_DIR)
            raw_ref = hashlib.sha256(zip_content).hexdigest()
            prices = _parse_dispatch_zip(zip_content, raw_ref)
            now = datetime.now(timezone.utc)
            if not prices:
                raise RuntimeError("No dispatch prices parsed from NEMWeb zip")
            interval = max(p.valid_time for p in prices.values())

            # Parse extended tables (never raises — failures produce empty lists)
            try:
                unit_rows = _parse_dispatch_unit_load(zip_content, raw_ref)
            except Exception:
                unit_rows = []
            try:
                constraint_rows = _parse_dispatch_constraints(zip_content, raw_ref)
            except Exception:
                constraint_rows = []
            try:
                interconnector_rows = _parse_dispatch_interconnectors(zip_content, raw_ref)
            except Exception:
                interconnector_rows = []

            return LiveMarketSnapshot(
                interval=interval,
                fetched_at=now,
                regions=prices,
                raw_ref=raw_ref,
                unit_dispatch_rows=unit_rows,
                constraint_rows=constraint_rows,
                interconnector_rows=interconnector_rows,
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
        "date": None, "region": None, "rrp": None, "demand": None, "avail": None,
        # FCAS service prices (8 markets)
        "raise6sec": None, "lower6sec": None,
        "raise60sec": None, "lower60sec": None,
        "raise5min": None, "lower5min": None,
        "raisereg": None, "lowerreg": None,
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
                # FCAS market clearing prices (8 services)
                for _hdr, _key in [
                    ("RAISE6SECRRP", "raise6sec"), ("LOWER6SECRRP", "lower6sec"),
                    ("RAISE60SECRRP", "raise60sec"), ("LOWER60SECRRP", "lower60sec"),
                    ("RAISE5MINRRP", "raise5min"), ("LOWER5MINRRP", "lower5min"),
                    ("RAISEREGRRP", "raisereg"), ("LOWERREGRRP", "lowerreg"),
                ]:
                    _price_cols[_key] = upper.index(_hdr) if _hdr in upper else None
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
                # Extract all 8 FCAS prices (None when column absent)
                fcas: dict[str, float | None] = {}
                for _fk in ("raise6sec", "lower6sec", "raise60sec", "lower60sec",
                            "raise5min", "lower5min", "raisereg", "lowerreg"):
                    _idx = ci.get(_fk)
                    if _idx is not None and _idx < len(parts):
                        _v = parts[_idx].strip()
                        fcas[_fk] = float(_v) if _v else None
                    else:
                        fcas[_fk] = None

                existing = price_rows.get(region)
                if existing is None or valid_time >= existing[0]:
                    price_rows[region] = (valid_time, rrp, demand_in_price, avail_in_price, fcas)

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
        valid_time, rrp, d_price, a_price, fcas = row if len(row) == 5 else (*row, None)
        demand, avail = regionsum_rows.get(region, (d_price, a_price))
        # Only attach FCAS dict when at least one service price was parsed
        fcas_out = fcas if fcas and any(v is not None for v in fcas.values()) else None
        prices[region] = DispatchPrice(
            region=region,
            valid_time=valid_time,
            system_time=system_time,
            price_rrp=rrp,
            demand_mw=demand,
            availability_mw=avail,
            raw_ref=raw_ref,
            fcas_prices=fcas_out,
        )

    return prices


def _parse_dispatch_unit_load(content: bytes, raw_ref: str) -> list[dict]:
    """Extract DUID-level dispatch from the DISPATCH,LOAD table in a DISPATCHIS zip.

    Same zip as _parse_dispatch_zip — just reads the LOAD table rows.
    Returns one dict per DUID with: duid, region (from DISPATCHLOAD), valid_time,
    initialmw, totalcleared, dispatchedgeneration, availability, rampdownrate, rampuprate.

    Region is the CONNECTIONPOINTID prefix (e.g. "NSW1") — not always present.
    Callers that need region-filtered rows should join against the GeneratorUnit table.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".CSV")), None)
            if csv_name is None:
                return []
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
    except Exception:
        return []

    cols: dict[str, int | None] = {
        "date": None, "duid": None, "initialmw": None, "totalcleared": None,
        "availability": None, "rampdownrate": None, "rampuprate": None,
        "dispatchedgeneration": None, "dispatchedload": None,
        "semi_dispatch_cap": None, "lowerreg": None, "raisereg": None,
    }

    def _idx(name: str) -> int | None:
        return cols.get(name)

    results: list[dict] = []
    system_time = datetime.now(timezone.utc)

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        row_type = parts[0].upper()
        table = parts[2].upper() if len(parts) > 2 else ""

        if row_type == "I" and table == "LOAD":
            upper = [p.upper().strip() for p in parts]
            for name, header in [
                ("date",                "SETTLEMENTDATE"),
                ("duid",                "DUID"),
                ("initialmw",           "INITIALMW"),
                ("totalcleared",        "TOTALCLEARED"),
                ("availability",        "AVAILABILITY"),
                ("rampdownrate",        "RAMPDOWNRATE"),
                ("rampuprate",          "RAMPUPRATE"),
                ("dispatchedgeneration","DISPATCHEDGENERATION"),
                ("dispatchedload",      "DISPATCHEDLOAD"),
                ("semi_dispatch_cap",   "SEMIDISPATCH"),
                ("lowerreg",            "LOWER5MIN"),
                ("raisereg",            "RAISE5MIN"),
            ]:
                cols[name] = upper.index(header) if header in upper else None
            continue

        if row_type != "D" or table != "LOAD":
            continue

        try:
            duid_idx = _idx("duid")
            date_idx = _idx("date")
            if duid_idx is None or date_idx is None or len(parts) <= max(duid_idx, date_idx):
                continue
            duid = parts[duid_idx].strip().strip('"').upper()
            if not duid:
                continue
            raw_date = parts[date_idx].strip().strip('"')
            try:
                valid_time = datetime.strptime(raw_date, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            def _f(name: str) -> float | None:
                idx = _idx(name)
                if idx is None or idx >= len(parts):
                    return None
                v = parts[idx].strip()
                return float(v) if v else None

            results.append({
                "source": "DISPATCHLOAD",
                "duid": duid,
                "valid_time": valid_time,
                "system_time": system_time,
                "initialmw": _f("initialmw"),
                "totalcleared": _f("totalcleared"),
                "availability": _f("availability"),
                "rampdownrate": _f("rampdownrate"),
                "rampuprate": _f("rampuprate"),
                "dispatchedgeneration": _f("dispatchedgeneration"),
                "dispatchedload": _f("dispatchedload"),
                "semi_dispatch_cap": _f("semi_dispatch_cap"),
                "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue

    return results


def _parse_dispatch_constraints(content: bytes, raw_ref: str) -> list[dict]:
    """Extract binding constraints from DISPATCH,CONSTRAINT table in a DISPATCHIS zip.

    Only returns rows where MARGINALVALUE > 0 (constraint is binding).
    Returns: constraintid, marginalvalue, violationdegree, rhs, valid_time.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".CSV")), None)
            if csv_name is None:
                return []
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
    except Exception:
        return []

    cols: dict[str, int | None] = {
        "date": None, "constraintid": None, "marginalvalue": None,
        "violationdegree": None, "rhs": None,
    }
    results: list[dict] = []
    system_time = datetime.now(timezone.utc)

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        row_type = parts[0].upper()
        table = parts[2].upper() if len(parts) > 2 else ""

        if row_type == "I" and table == "CONSTRAINT":
            upper = [p.upper().strip() for p in parts]
            for name, header in [
                ("date",           "SETTLEMENTDATE"),
                ("constraintid",   "CONSTRAINTID"),
                ("marginalvalue",  "MARGINALVALUE"),
                ("violationdegree","VIOLATIONDEGREE"),
                ("rhs",            "RHS"),
            ]:
                cols[name] = upper.index(header) if header in upper else None
            continue

        if row_type != "D" or table != "CONSTRAINT":
            continue

        try:
            cid_idx = cols.get("constraintid")
            mv_idx  = cols.get("marginalvalue")
            dt_idx  = cols.get("date")
            if cid_idx is None or mv_idx is None or dt_idx is None:
                continue
            if max(cid_idx, mv_idx, dt_idx) >= len(parts):
                continue

            mv_str = parts[mv_idx].strip()
            if not mv_str:
                continue
            marginal_value = float(mv_str)
            if marginal_value <= 0.0:
                continue   # non-binding — skip

            constraint_id = parts[cid_idx].strip().strip('"')
            if not constraint_id:
                continue

            raw_date = parts[dt_idx].strip().strip('"')
            try:
                valid_time = datetime.strptime(raw_date, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            vd_idx = cols.get("violationdegree")
            rhs_idx = cols.get("rhs")
            violation = float(parts[vd_idx].strip()) if vd_idx and vd_idx < len(parts) and parts[vd_idx].strip() else 0.0
            rhs = float(parts[rhs_idx].strip()) if rhs_idx and rhs_idx < len(parts) and parts[rhs_idx].strip() else None

            results.append({
                "constraint_id": constraint_id,
                "marginal_value": marginal_value,
                "violation_degree": violation,
                "rhs": rhs,
                "valid_time": valid_time,
                "system_time": system_time,
                "raw_ref": raw_ref,
                "source": "DISPATCHCONSTRAINT",
            })
        except (ValueError, IndexError):
            continue

    return results


def _parse_dispatch_interconnectors(content: bytes, raw_ref: str) -> list[dict]:
    """Extract interconnector flows from DISPATCH,INTERCONNECTORRES in a DISPATCHIS zip.

    Returns all interconnectors with: id, metered_mw_flow, mw_flow, mw_losses,
    export_limit, import_limit, violation_degree, at_export_limit, at_import_limit.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".CSV")), None)
            if csv_name is None:
                return []
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
    except Exception:
        return []

    cols: dict[str, int | None] = {
        "date": None, "interconnectorid": None, "meteredmwflow": None,
        "mwflow": None, "mwlosses": None, "exportlimit": None,
        "importlimit": None, "violationdegree": None,
    }
    results: list[dict] = []
    system_time = datetime.now(timezone.utc)

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        row_type = parts[0].upper()
        table = parts[2].upper() if len(parts) > 2 else ""

        if row_type == "I" and table == "INTERCONNECTORRES":
            upper = [p.upper().strip() for p in parts]
            for name, header in [
                ("date",             "SETTLEMENTDATE"),
                ("interconnectorid", "INTERCONNECTORID"),
                ("meteredmwflow",    "METEREDMWFLOW"),
                ("mwflow",           "MWFLOW"),
                ("mwlosses",         "MWLOSSES"),
                ("exportlimit",      "EXPORTLIMIT"),
                ("importlimit",      "IMPORTLIMIT"),
                ("violationdegree",  "VIOLATIONDEGREE"),
            ]:
                cols[name] = upper.index(header) if header in upper else None
            continue

        if row_type != "D" or table != "INTERCONNECTORRES":
            continue

        try:
            ic_idx = cols.get("interconnectorid")
            dt_idx = cols.get("date")
            mf_idx = cols.get("meteredmwflow")
            if ic_idx is None or dt_idx is None or max(filter(None, [ic_idx, dt_idx])) >= len(parts):
                continue

            ic_id = parts[ic_idx].strip().strip('"').upper()
            if not ic_id:
                continue

            raw_date = parts[dt_idx].strip().strip('"')
            try:
                valid_time = datetime.strptime(raw_date, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            def _f2(name: str) -> float | None:
                idx = cols.get(name)
                if idx is None or idx >= len(parts):
                    return None
                v = parts[idx].strip()
                return float(v) if v else None

            metered = _f2("meteredmwflow")
            export_lim = _f2("exportlimit")
            import_lim = _f2("importlimit")

            results.append({
                "interconnector_id": ic_id,
                "metered_mw_flow": metered,
                "mw_flow": _f2("mwflow"),
                "mw_losses": _f2("mwlosses"),
                "export_limit": export_lim,
                "import_limit": import_lim,
                "violation_degree": _f2("violationdegree") or 0.0,
                "at_export_limit": (
                    abs((metered or 0) - (export_lim or 0)) < 5.0
                    if metered is not None and export_lim is not None else False
                ),
                "at_import_limit": (
                    abs((metered or 0) - (import_lim or 0)) < 5.0
                    if metered is not None and import_lim is not None else False
                ),
                "valid_time": valid_time,
                "system_time": system_time,
                "raw_ref": raw_ref,
                "source": "DISPATCHINTERCONNECTORRES",
            })
        except (ValueError, IndexError):
            continue

    return results


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
