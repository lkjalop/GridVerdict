"""MMSDM CSV parsing functions — extracted from aemo_archive.py.

All functions here are pure (no async, no DB).  They transform raw bytes or
text into list[dict] row payloads suitable for the upsert layer in aemo_db.py.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import zipfile
from datetime import datetime, timezone
from typing import Any

from app.data.aemo_live_client import (
    _COL_AVAILABLEGENERATION,
    _COL_REGIONID,
    _COL_RRP,
    _COL_SETTLEMENTDATE,
    _COL_TOTALDEMAND,
    _REGIONS,
)

# ── MMSDM archive format constants ───────────────────────────────────

# AEMO uses abbreviated names in MMSDM DVD filenames for some tables (pre-2024-08).
_TABLE_FILENAME_ALIASES: dict[str, list[str]] = {
    "DISPATCH_UNIT_SOLUTION": ["DISPATCH_UNIT_SOLUTION", "DISPATCH_UNIT_SOLN"],
    "BIDPEROFFER": ["BIDPEROFFER1", "BIDPEROFFER2"],
}

# AEMO switched from PUBLIC_DVD_* to PUBLIC_ARCHIVE#*#FILE01#* naming in August 2024.
_ARCHIVE_FORMAT_START = datetime(2024, 8, 1, tzinfo=timezone.utc)

# Table name changes in the ARCHIVE format (2024-08+).
_TABLE_ARCHIVE_ALIASES: dict[str, list[str]] = {
    "BIDPEROFFER": ["BIDOFFERPERIOD"],
}

_FCAS_COLUMNS = [
    "RAISE6SECRRP", "RAISE60SECRRP", "RAISE5MINRRP", "RAISEREGRRP",
    "LOWER6SECRRP", "LOWER60SECRRP", "LOWER5MINRRP", "LOWERREGRRP",
]
_FCAS_FIELD_MAP = {
    "RAISE6SECRRP":  "raise_6sec_rrp",
    "RAISE60SECRRP": "raise_60sec_rrp",
    "RAISE5MINRRP":  "raise_5min_rrp",
    "RAISEREGRRP":   "raise_reg_rrp",
    "LOWER6SECRRP":  "lower_6sec_rrp",
    "LOWER60SECRRP": "lower_60sec_rrp",
    "LOWER5MINRRP":  "lower_5min_rrp",
    "LOWERREGRRP":   "lower_reg_rrp",
}


# ── Public content parsers ────────────────────────────────────────────

def parse_mmsdm_dispatchprice_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM archive content containing DISPATCHPRICE rows."""
    texts: list[str] = []
    if content[:2] == b"\x1f\x8b":
        texts.append(gzip.decompress(content).decode("utf-8", errors="replace"))
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    if "DISPATCHPRICE" in name.upper() and name.upper().endswith((".CSV", ".CSV.GZ")):
                        data = zf.read(name)
                        if name.upper().endswith(".GZ"):
                            data = gzip.decompress(data)
                        texts.append(data.decode("utf-8", errors="replace"))
        except zipfile.BadZipFile:
            texts.append(content.decode("utf-8", errors="replace"))
    rows: list[dict[str, Any]] = []
    for text in texts:
        rows.extend(_parse_mmsdm_text(text, raw_ref))
    return rows


def parse_mmsdm_driver_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM driver tables: interconnector flows and dispatch constraints."""
    rows: list[dict[str, Any]] = []
    for name, text in _extract_archive_texts(content):
        upper_name = name.upper()
        upper_text = text[:2000].upper()
        if "DISPATCHINTERCONNECTORRES" in upper_name or "INTERCONNECTORRES" in upper_name:
            rows.extend(_parse_interconnector_text(text, raw_ref))
        elif "DISPATCHCONSTRAINT" in upper_name:
            rows.extend(_parse_constraint_text(text, raw_ref))
        else:
            if "DISPATCHINTERCONNECTORRES" in upper_text or "INTERCONNECTORRES" in upper_text:
                rows.extend(_parse_interconnector_text(text, raw_ref))
            if "DISPATCHCONSTRAINT" in upper_text or ",CONSTRAINT," in upper_text:
                rows.extend(_parse_constraint_text(text, raw_ref))
    return rows


def parse_mmsdm_bid_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM bid/offer tables: BIDDAYOFFER and BIDPEROFFER."""
    rows: list[dict[str, Any]] = []
    for name, text in _extract_archive_texts(content):
        upper_name = name.upper()
        upper_text = text[:2000].upper()
        if "BIDDAYOFFER" in upper_name or "BIDDAYOFFER" in upper_text:
            rows.extend(_parse_biddayoffer_text(text, raw_ref))
        if "BIDPEROFFER" in upper_name or "BIDOFFERPERIOD" in upper_text:
            rows.extend(_parse_bidperoffer_text(text, raw_ref))
    return rows


def parse_mmsdm_unit_content(content: bytes, raw_ref: str = "mmsdm") -> dict[str, list[dict[str, Any]]]:
    """Parse DUID-level MMSDM unit dispatch and metadata tables."""
    unit_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for name, text in _extract_archive_texts(content):
        upper_name = name.upper()
        upper_text = text[:2000].upper()
        if "DISPATCH_UNIT_SOLUTION" in upper_name or "DISPATCH_UNIT_SOLUTION" in upper_text:
            unit_rows.extend(_parse_dispatch_unit_solution_text(text, raw_ref))
        if "DISPATCHLOAD" in upper_name or "DISPATCHLOAD" in upper_text:
            unit_rows.extend(_parse_dispatchload_text(text, raw_ref))
        if "DUDETAILSUMMARY" in upper_name or "DUDETAILSUMMARY" in upper_text:
            metadata_rows.extend(_parse_dudetail_summary_text(text, raw_ref))
    return {"unit_rows": unit_rows, "metadata_rows": metadata_rows}


def parse_mmsdm_fcas_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM archive content and extract FCAS ancillary service prices."""
    rows: list[dict[str, Any]] = []
    for _name, text in _extract_archive_texts(content):
        upper_text = text[:3000].upper()
        if any(col in upper_text for col in _FCAS_COLUMNS):
            rows.extend(_parse_mmsdm_fcas_text(text, raw_ref))
    return rows


# ── Streaming bid-row iterator (large files) ─────────────────────────

def _iter_bid_rows_from_file(path: "pathlib.Path", raw_ref: str):  # type: ignore[name-defined]
    """Generator: stream-parse bid rows from a ZIP on disk without full RAM load.

    Opens the ZIP without decompressing the whole file, iterates the inner
    CSV line-by-line via csv.reader, and yields bid_offer dicts one at a time.
    """
    import pathlib  # noqa: F401 — used by callers via type annotation

    system_time = datetime.now(timezone.utc)

    def _detect_and_yield(inner_name: str, csv_file):
        upper = inner_name.upper()
        is_dayoffer = "BIDDAYOFFER" in upper and "BIDPEROFFER" not in upper
        is_peroffer = "BIDPEROFFER" in upper or "BIDOFFERPERIOD" in upper

        header: list[str] | None = None
        for row_raw in csv.reader(csv_file):
            if not row_raw:
                continue
            parts = [p.strip().strip('"') for p in row_raw]
            row0 = parts[0].upper() if parts else ""
            joined = ",".join(parts).upper()

            if row0 == "I":
                if "BIDDAYOFFER" in joined and "BIDPEROFFER" not in joined:
                    header = [p.upper() for p in parts]
                    is_dayoffer, is_peroffer = True, False
                elif "BIDOFFERPERIOD" in joined or "BIDPEROFFER" in joined:
                    header = [p.upper() for p in parts]
                    is_peroffer, is_dayoffer = True, False
                continue

            if row0 != "D" or header is None:
                continue

            try:
                if is_dayoffer and "BIDDAYOFFER" in joined:
                    duid = parts[_idx(header, "DUID")].upper()
                    bid_type = _first_text(parts, header, ["BIDTYPE"]) or "ENERGY"
                    settlement_date = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
                    offer_date_raw = _first_text(parts, header, ["OFFERDATE"])
                    offer_date = _parse_archive_dt(offer_date_raw) if offer_date_raw else None
                    price_bands = {
                        i: _float_or_none(parts, header, f"PRICEBAND{i}")
                        for i in range(1, 11)
                        if _float_or_none(parts, header, f"PRICEBAND{i}") is not None
                    }
                    row_id = hashlib.sha256(
                        f"BIDDAYOFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{raw_ref}".encode()
                    ).hexdigest()[:32]
                    yield {
                        "id": row_id, "tenant_id": "system", "source": "BIDDAYOFFER",
                        "duid": duid, "region": _infer_region_from_duid(duid),
                        "bid_type": bid_type, "settlement_date": settlement_date,
                        "period_id": None, "offer_date": offer_date,
                        "max_avail_mw": None,
                        "minimum_load_mw": _float_or_none(parts, header, "MINIMUMLOAD"),
                        "ramp_up_mw_per_min": None, "ramp_down_mw_per_min": None,
                        "price_bands": price_bands, "avail_bands": {},
                        "data": {"raw_ref": raw_ref}, "raw_ref": raw_ref,
                    }

                elif is_peroffer and ("BIDOFFERPERIOD" in joined or "BIDPEROFFER" in joined):
                    duid = parts[_idx(header, "DUID")].upper()
                    bid_type = _first_text(parts, header, ["BIDTYPE"]) or "ENERGY"
                    settlement_date = _parse_archive_dt(
                        parts[_idx(header, _first_col(header, ["TRADINGDATE", "SETTLEMENTDATE"]))]
                    )
                    period_id_raw = _first_text(parts, header, ["PERIODID"])
                    period_id = int(period_id_raw) if period_id_raw else None
                    max_avail = _first_float(parts, header, ["MAXAVAIL", "MAXAVAILABILITY"])
                    ramp_up = _first_float(parts, header, ["RAMPUPRATE", "ROCUP"])
                    ramp_dn = _first_float(parts, header, ["RAMPDOWNRATE", "ROCDOWN"])
                    avail_bands = {
                        i: _float_or_none(parts, header, f"BANDAVAIL{i}")
                        for i in range(1, 11)
                        if _float_or_none(parts, header, f"BANDAVAIL{i}") is not None
                    }
                    row_id = hashlib.sha256(
                        f"BIDPEROFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{period_id}|{raw_ref}".encode()
                    ).hexdigest()[:32]
                    yield {
                        "id": row_id, "tenant_id": "system", "source": "BIDPEROFFER",
                        "duid": duid, "region": _infer_region_from_duid(duid),
                        "bid_type": bid_type, "settlement_date": settlement_date,
                        "period_id": period_id, "offer_date": None,
                        "max_avail_mw": max_avail, "minimum_load_mw": None,
                        "ramp_up_mw_per_min": ramp_up, "ramp_down_mw_per_min": ramp_dn,
                        "price_bands": {}, "avail_bands": avail_bands,
                        "data": {"raw_ref": raw_ref}, "raw_ref": raw_ref,
                    }
            except (ValueError, IndexError):
                continue

    with zipfile.ZipFile(path) as zf:
        for inner_name in zf.namelist():
            if not inner_name.upper().endswith((".CSV", ".CSV.GZ")):
                continue
            with zf.open(inner_name) as raw_fh:
                if inner_name.upper().endswith(".GZ"):
                    import io as _io
                    raw_fh = _io.TextIOWrapper(gzip.open(raw_fh), encoding="utf-8", errors="replace")
                else:
                    import io as _io
                    raw_fh = _io.TextIOWrapper(raw_fh, encoding="utf-8", errors="replace")
                yield from _detect_and_yield(inner_name, raw_fh)


# ── Internal text parsers ─────────────────────────────────────────────

def _extract_archive_texts(content: bytes) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    if content[:2] == b"\x1f\x8b":
        texts.append(("archive.csv.gz", gzip.decompress(content).decode("utf-8", errors="replace")))
        return texts
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if not name.upper().endswith((".CSV", ".CSV.GZ")):
                    continue
                data = zf.read(name)
                if name.upper().endswith(".GZ"):
                    data = gzip.decompress(data)
                texts.append((name, data.decode("utf-8", errors="replace")))
    except zipfile.BadZipFile:
        texts.append(("archive.csv", content.decode("utf-8", errors="replace")))
    return texts


def _parse_mmsdm_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        if parts[0].upper() == "I" and "DISPATCHPRICE" in ",".join(parts).upper():
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D":
            continue
        try:
            if header:
                date_i = header.index("SETTLEMENTDATE")
                region_i = header.index("REGIONID")
                rrp_i = header.index("RRP")
                demand_i = header.index("TOTALDEMAND")
                avail_i = header.index("AVAILABLEGENERATION")
            else:
                date_i, region_i, rrp_i = _COL_SETTLEMENTDATE, _COL_REGIONID, _COL_RRP
                demand_i, avail_i = _COL_TOTALDEMAND, _COL_AVAILABLEGENERATION

            region = parts[region_i].upper()
            if region not in _REGIONS:
                continue
            valid_time = _parse_archive_dt(parts[date_i])
            price = float(parts[rrp_i])
            demand = float(parts[demand_i]) if demand_i < len(parts) and parts[demand_i] else 0.0
            avail = float(parts[avail_i]) if avail_i < len(parts) and parts[avail_i] else 0.0
            rows.append({
                "id": f"{raw_ref}-{region}-{valid_time.strftime('%Y%m%d%H%M')}",
                "tenant_id": "system",
                "source": "AEMO_DISPATCH_PRICE",
                "region": region,
                "valid_time": valid_time,
                "system_time": system_time,
                "price_rrp": price,
                "demand_mw": demand,
                "availability_mw": avail,
                "data": {"raw_ref": raw_ref, "headroom_mw": max(avail - demand, 0.0)},
                "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_mmsdm_fcas_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None

    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        if parts[0].upper() == "I":
            upper_parts = [p.upper() for p in parts]
            if any(col in upper_parts for col in _FCAS_COLUMNS):
                header = upper_parts
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            date_i = header.index("SETTLEMENTDATE")
            region_i = header.index("REGIONID")
            region = parts[region_i].upper()
            if region not in _REGIONS:
                continue
            valid_time = _parse_archive_dt(parts[date_i])
            fcas_values: dict[str, float | None] = {}
            for col in _FCAS_COLUMNS:
                field = _FCAS_FIELD_MAP[col]
                try:
                    idx = header.index(col)
                    val = parts[idx] if idx < len(parts) else ""
                    fcas_values[field] = float(val) if val else None
                except (ValueError, IndexError):
                    fcas_values[field] = None
            if all(v is None for v in fcas_values.values()):
                continue
            rows.append({
                "id": hashlib.sha256(f"fcas|{region}|{valid_time.isoformat()}".encode()).hexdigest()[:32],
                "tenant_id": "system",
                "source": "AEMO_DISPATCH_PRICE",
                "region": region,
                "valid_time": valid_time,
                "system_time": system_time,
                "raw_ref": raw_ref,
                **fcas_values,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_dispatch_unit_solution_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and "DISPATCH_UNIT_SOLUTION" in joined:
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            valid_time = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            duid = parts[_idx(header, "DUID")].upper()
            rows.append(_unit_row(
                source="AEMO_DISPATCH_UNIT_SOLUTION", duid=duid, valid_time=valid_time,
                initial_mw=_float_or_none(parts, header, "INITIALMW"),
                total_cleared_mw=_float_or_none(parts, header, "TOTALCLEARED"),
                availability_mw=_float_or_none(parts, header, "AVAILABILITY"),
                target_mw=_first_float(parts, header, ["TARGETMW", "TARGET"]),
                ramp_rate=_first_float(parts, header, ["RAMPUPRATE", "RAMP_RATE", "RAMPDOWNRATE"]),
                semi_dispatch_cap=_float_or_none(parts, header, "SEMIDISPATCHCAP"),
                raw_ref=raw_ref, system_time=system_time, data=_row_payload(parts, header),
            ))
        except (ValueError, IndexError):
            continue
    return rows


def _parse_dispatchload_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and "DISPATCHLOAD" in joined:
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            valid_time = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            duid = parts[_idx(header, "DUID")].upper()
            rows.append(_unit_row(
                source="AEMO_DISPATCHLOAD", duid=duid, valid_time=valid_time,
                initial_mw=_float_or_none(parts, header, "INITIALMW"),
                total_cleared_mw=_first_float(parts, header, ["TOTALCLEARED", "DISPATCHTARGET"]),
                availability_mw=_float_or_none(parts, header, "AVAILABILITY"),
                target_mw=_first_float(parts, header, ["DISPATCHTARGET", "TARGETMW"]),
                ramp_rate=_first_float(parts, header, ["RAMPUPRATE", "RAMPDOWNRATE"]),
                semi_dispatch_cap=_float_or_none(parts, header, "SEMIDISPATCHCAP"),
                raw_ref=raw_ref, system_time=system_time, data=_row_payload(parts, header),
            ))
        except (ValueError, IndexError):
            continue
    return rows


def _parse_dudetail_summary_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and "DUDETAILSUMMARY" in joined:
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            duid = parts[_idx(header, "DUID")].upper()
        except (ValueError, IndexError):
            continue
        metadata = _row_payload(parts, header)
        rows.append({
            "duid": duid,
            "station_name": _first_text(parts, header, ["STATIONNAME", "STATION_NAME"]),
            "participant": _first_text(parts, header, ["PARTICIPANTID", "PARTICIPANT", "OWNER"]),
            "region": _first_text(parts, header, ["REGIONID", "REGION"]),
            "fuel_type": _normalise_fuel_type(
                _first_text(parts, header, ["FUELTYPE", "FUEL_SOURCE", "TECHNOLOGYTYPE"]) or duid
            ),
            "dispatch_type": _first_text(parts, header, ["DISPATCHTYPE", "SCHEDULE_TYPE"]),
            "max_capacity_mw": _first_float(parts, header, ["REGISTEREDCAPACITY", "MAXCAPACITY", "NAMEPLATECAPACITY"]),
            "metadata_json": {**metadata, "raw_ref": raw_ref},
        })
    return rows


def _parse_biddayoffer_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    """Parse BIDDAYOFFER CSV text into bid_offer row dicts (period_id=None)."""
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and "BIDDAYOFFER" in joined and "BIDPEROFFER" not in joined:
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        if "BIDDAYOFFER" not in joined:
            continue
        try:
            duid = parts[_idx(header, "DUID")].upper()
            bid_type = _first_text(parts, header, ["BIDTYPE"]) or "ENERGY"
            settlement_date = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            offer_date_raw = _first_text(parts, header, ["OFFERDATE"])
            offer_date = _parse_archive_dt(offer_date_raw) if offer_date_raw else None
            price_bands = {
                i: _float_or_none(parts, header, f"PRICEBAND{i}")
                for i in range(1, 11)
                if _float_or_none(parts, header, f"PRICEBAND{i}") is not None
            }
            row_id = hashlib.sha256(
                f"BIDDAYOFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{raw_ref}".encode()
            ).hexdigest()[:32]
            rows.append({
                "id": row_id, "tenant_id": "system", "source": "BIDDAYOFFER",
                "duid": duid, "region": _infer_region_from_duid(duid),
                "bid_type": bid_type, "settlement_date": settlement_date,
                "period_id": None, "offer_date": offer_date,
                "max_avail_mw": None, "minimum_load_mw": _float_or_none(parts, header, "MINIMUMLOAD"),
                "ramp_up_mw_per_min": None, "ramp_down_mw_per_min": None,
                "price_bands": price_bands, "avail_bands": {},
                "data": {"raw_ref": raw_ref}, "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_bidperoffer_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    """Parse BIDPEROFFER (BIDOFFERPERIOD) CSV into bid_offer rows (period_id 1-48)."""
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and ("BIDOFFERPERIOD" in joined or "BIDPEROFFER" in joined):
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            duid = parts[_idx(header, "DUID")].upper()
            bid_type = _first_text(parts, header, ["BIDTYPE"]) or "ENERGY"
            settlement_date = _parse_archive_dt(
                parts[_idx(header, "TRADINGDATE")] if "TRADINGDATE" in header
                else parts[_idx(header, "SETTLEMENTDATE")]
            )
            period_id_raw = _first_text(parts, header, ["PERIODID"])
            period_id = int(period_id_raw) if period_id_raw and period_id_raw.isdigit() else None
            offer_date_raw = _first_text(parts, header, ["OFFERDATETIME", "OFFERDATE"])
            offer_date = _parse_archive_dt(offer_date_raw) if offer_date_raw else None
            avail_bands = {
                i: _float_or_none(parts, header, f"BANDAVAIL{i}")
                for i in range(1, 11)
                if _float_or_none(parts, header, f"BANDAVAIL{i}") is not None
            }
            row_id = hashlib.sha256(
                f"BIDPEROFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{period_id}|{raw_ref}".encode()
            ).hexdigest()[:32]
            rows.append({
                "id": row_id, "tenant_id": "system", "source": "BIDPEROFFER",
                "duid": duid, "region": _infer_region_from_duid(duid),
                "bid_type": bid_type, "settlement_date": settlement_date,
                "period_id": period_id, "offer_date": offer_date,
                "max_avail_mw": _first_float(parts, header, ["MAXAVAIL"]),
                "minimum_load_mw": None,
                "ramp_up_mw_per_min": _first_float(parts, header, ["RAMPUPRATE", "ROCUP"]),
                "ramp_down_mw_per_min": _first_float(parts, header, ["RAMPDOWNRATE", "ROCDOWN"]),
                "price_bands": {}, "avail_bands": avail_bands,
                "data": {"raw_ref": raw_ref}, "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_interconnector_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        if parts[0].upper() == "I" and "INTERCONNECTORRES" in ",".join(parts).upper():
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            valid_time = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            interconnector_id = parts[_idx(header, "INTERCONNECTORID")]
            values = {
                "metered_mw_flow": _float_or_none(parts, header, "METEREDMWFLOW"),
                "mw_flow": _float_or_none(parts, header, "MWFLOW"),
                "export_limit": _float_or_none(parts, header, "EXPORTLIMIT"),
                "import_limit": _float_or_none(parts, header, "IMPORTLIMIT"),
                "marginal_value": _float_or_none(parts, header, "MARGINALVALUE"),
            }
            rows.append(_driver_row(
                source="AEMO_DISPATCHINTERCONNECTORRES", driver_type="interconnector",
                element_id=interconnector_id, valid_time=valid_time,
                values={k: v for k, v in values.items() if v is not None},
                raw_ref=raw_ref, system_time=system_time,
            ))
        except (ValueError, IndexError):
            continue
    return rows


def _parse_constraint_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    system_time = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts:
            continue
        joined = ",".join(parts).upper()
        if parts[0].upper() == "I" and ("DISPATCHCONSTRAINT" in joined or ",CONSTRAINT," in joined):
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            valid_time = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            constraint_id = parts[_idx(header, "CONSTRAINTID")]
            marginal_value = _float_or_none(parts, header, "MARGINALVALUE")
            violation_degree = _float_or_none(parts, header, "VIOLATIONDEGREE")
            if not marginal_value and not violation_degree:
                continue
            values = {
                "rhs": _float_or_none(parts, header, "RHS"),
                "marginal_value": marginal_value,
                "violation_degree": violation_degree,
            }
            rows.append(_driver_row(
                source="AEMO_DISPATCHCONSTRAINT", driver_type="constraint",
                element_id=constraint_id, valid_time=valid_time,
                values={k: v for k, v in values.items() if v is not None},
                raw_ref=raw_ref, system_time=system_time,
            ))
        except (ValueError, IndexError):
            continue
    return rows


# ── Row builder helpers ───────────────────────────────────────────────

def _unit_row(
    source: str, duid: str, valid_time: datetime,
    initial_mw: float | None, total_cleared_mw: float | None,
    availability_mw: float | None, target_mw: float | None,
    ramp_rate: float | None, semi_dispatch_cap: float | None,
    raw_ref: str, system_time: datetime, data: dict[str, Any],
) -> dict[str, Any]:
    row_id = hashlib.sha256(
        f"{source}|{duid}|{valid_time.isoformat()}|{raw_ref}".encode()
    ).hexdigest()[:32]
    return {
        "id": row_id, "tenant_id": "system", "source": source,
        "duid": duid, "station_name": None, "participant": None,
        "region": _infer_region_from_duid(duid),
        "fuel_type": _normalise_fuel_type(duid),
        "valid_time": valid_time, "system_time": system_time,
        "initial_mw": initial_mw, "total_cleared_mw": total_cleared_mw,
        "availability_mw": availability_mw, "target_mw": target_mw,
        "ramp_rate": ramp_rate, "semi_dispatch_cap": semi_dispatch_cap,
        "data": {**data, "raw_ref": raw_ref}, "raw_ref": raw_ref,
    }


def _driver_row(
    source: str, driver_type: str, element_id: str,
    valid_time: datetime, values: dict[str, Any],
    raw_ref: str, system_time: datetime,
) -> dict[str, Any]:
    row_id = hashlib.sha256(
        f"{source}|{driver_type}|{element_id}|{valid_time.isoformat()}|{raw_ref}".encode()
    ).hexdigest()[:32]
    return {
        "id": row_id, "tenant_id": "system", "source": source,
        "driver_type": driver_type, "element_id": element_id,
        "region": _infer_region_from_element(element_id),
        "valid_time": valid_time, "system_time": system_time,
        "values": values, "raw_ref": raw_ref,
    }


# ── Low-level field helpers ───────────────────────────────────────────

def _idx(header: list[str], name: str) -> int:
    return header.index(name)


def _float_or_none(parts: list[str], header: list[str], name: str) -> float | None:
    if name not in header:
        return None
    idx = header.index(name)
    if idx >= len(parts) or parts[idx] == "":
        return None
    return float(parts[idx])


def _first_float(parts: list[str], header: list[str], names: list[str]) -> float | None:
    for name in names:
        try:
            value = _float_or_none(parts, header, name)
        except ValueError:
            continue
        if value is not None:
            return value
    return None


def _first_text(parts: list[str], header: list[str], names: list[str]) -> str | None:
    for name in names:
        if name not in header:
            continue
        idx = header.index(name)
        if idx < len(parts) and parts[idx].strip():
            return parts[idx].strip().upper()
    return None


def _first_col(header: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in header:
            return c
    return candidates[0]


def _row_payload(parts: list[str], header: list[str]) -> dict[str, Any]:
    return {name.lower(): parts[i] for i, name in enumerate(header) if i < len(parts)}


def _infer_region_from_element(element_id: str) -> str | None:
    upper = element_id.upper()
    if upper.startswith("N-") or "NSW" in upper:
        return "NSW1"
    if upper.startswith("V-") or "VIC" in upper:
        return "VIC1"
    if upper.startswith("Q-") or "QLD" in upper:
        return "QLD1"
    if upper.startswith("S-") or "SA" in upper:
        return "SA1"
    if upper.startswith("T-") or "TAS" in upper:
        return "TAS1"
    return None


def _infer_region_from_duid(duid: str) -> str | None:
    upper = duid.upper()
    if upper.startswith(("N", "ER", "BW", "VP")) or "NSW" in upper:
        return "NSW1"
    if upper.startswith(("V", "YW", "LY", "MURRAY")) or "VIC" in upper:
        return "VIC1"
    if upper.startswith(("Q", "TARONG", "STAN")) or "QLD" in upper:
        return "QLD1"
    if upper.startswith(("S", "TORR", "LADB", "SNOWTOWN")) or "SA" in upper:
        return "SA1"
    if upper.startswith(("T", "GORDON", "POAT", "REECE")) or "TAS" in upper:
        return "TAS1"
    return None


def _normalise_fuel_type(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    checks = [
        ("battery",          ("BAT", "BESS", "BATT")),
        ("hydro",            ("HYDRO", "WATER", "GORDON", "POAT", "REECE", "MURRAY", "TUMUT", "GUTHEGA", "UPPTUMUT")),
        ("coal",             ("COAL", "BROWN", "BLACK", "ERARING", "BAYS", "LIDDELL", "TARONG", "STANWELL", "LOY", "YWPS")),
        ("gas",              ("GAS", "OCGT", "CCGT", "TURBINE", "TORR", "LADB")),
        ("wind",             ("WIND", "WF", "SNOWTOWN", "MACARTHUR")),
        ("solar",            ("SOLAR", "SF", "PV")),
        ("biomass",          ("BIO", "BAGASSE")),
        ("demand_response",  ("DR", "DSP", "LOAD")),
    ]
    for fuel, tokens in checks:
        if any(token in upper for token in tokens):
            return fuel
    return "unknown"


def _parse_archive_dt(value: str) -> datetime:
    from app.data.aemo_live_client import _parse_aemo_dt
    return _parse_aemo_dt(value.strip().replace('"', ""))


# ── Rooftop solar parsers ─────────────────────────────────────────────

_ROOFTOP_ACTUAL_COLS = {"INTERVAL_DATETIME", "REGIONID", "POWER"}
_ROOFTOP_FORECAST_COLS = {"INTERVAL_DATETIME", "REGIONID", "POWERMEAN"}


def parse_mmsdm_rooftop_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict]:
    """Parse MMSDM ROOFTOP_PV_ACTUAL_SCADA and ROOFTOP_PV_FORECAST_SCADA content.

    Returns list of dicts with keys: region, interval_datetime, actual_mw,
    forecast_mw, delta_mw. Both table types are merged by (region, interval_datetime)
    so partial files (actual-only or forecast-only) are handled gracefully.
    """
    actuals: dict[tuple, float] = {}
    forecasts: dict[tuple, float] = {}

    for _name, text in _extract_archive_texts(content):
        upper = text[:2000].upper()
        if "ROOFTOP_PV_ACTUAL" in upper or ("POWER" in upper and "REGIONID" in upper and "INTERVAL_DATETIME" in upper):
            _parse_rooftop_actual_text(text, actuals)
        elif "ROOFTOP_PV_FORECAST" in upper or ("POWERMEAN" in upper and "REGIONID" in upper):
            _parse_rooftop_forecast_text(text, forecasts)

    # Merge: union of all (region, interval_datetime) keys
    all_keys = set(actuals) | set(forecasts)
    rows = []
    for key in sorted(all_keys):
        region, interval_dt = key
        actual = actuals.get(key)
        forecast = forecasts.get(key)
        delta = round(actual - forecast, 2) if actual is not None and forecast is not None else None
        rows.append({
            "region": region,
            "interval_datetime": interval_dt,
            "actual_mw": actual,
            "forecast_mw": forecast,
            "delta_mw": delta,
        })
    return rows


def _parse_rooftop_actual_text(text: str, out: dict) -> None:
    """Extract (region, interval_datetime) → actual_mw from ROOFTOP_PV_ACTUAL_SCADA CSV."""
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    for row in reader:
        if not row:
            continue
        tag = row[0].strip().upper()
        if tag == "C":
            break
        if tag == "I":
            header = [c.strip().upper() for c in row]
            continue
        if tag != "D" or header is None:
            continue
        try:
            rd = dict(zip(header, row))
            region = rd.get("REGIONID", "").strip().upper()
            dt_str = rd.get("INTERVAL_DATETIME", "").strip()
            power_str = rd.get("POWER", "").strip()
            if not region or not dt_str or not power_str:
                continue
            dt = _parse_archive_dt(dt_str)
            out[(region, dt)] = round(float(power_str), 2)
        except Exception:
            continue


def _parse_rooftop_forecast_text(text: str, out: dict) -> None:
    """Extract (region, interval_datetime) → forecast_mw from ROOFTOP_PV_FORECAST_SCADA CSV."""
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    for row in reader:
        if not row:
            continue
        tag = row[0].strip().upper()
        if tag == "C":
            break
        if tag == "I":
            header = [c.strip().upper() for c in row]
            continue
        if tag != "D" or header is None:
            continue
        try:
            rd = dict(zip(header, row))
            region = rd.get("REGIONID", "").strip().upper()
            dt_str = rd.get("INTERVAL_DATETIME", "").strip()
            power_str = rd.get("POWERMEAN", "").strip()
            if not region or not dt_str or not power_str:
                continue
            dt = _parse_archive_dt(dt_str)
            out[(region, dt)] = round(float(power_str), 2)
        except Exception:
            continue
