"""AEMO NEMWeb archive fetcher — historical interval backfill.

Fetches historical DISPATCHPRICE CSV files from NEMWeb and populates the
market_events table so HippoGraph can retrieve analogs from history.

NEMWeb archive path:
  /Reports/Archive/DispatchIS_Reports/PUBLIC_DISPATCHIS_YYYYMMDD_HHMM_SEQ.zip

This module runs as a background job (once per hour) and also exposes
`backfill_recent_gaps()` called by the scheduler to fill any missed intervals.

Rate: NEMWeb allows public access; we add 1s delay between requests.
Deduplication: we track fetched intervals in the market_events table.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import zipfile
import gzip
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.data.aemo_live_client import (
    _COL_AVAILABLEGENERATION,
    _COL_REGIONID,
    _COL_RRP,
    _COL_SETTLEMENTDATE,
    _COL_TOTALDEMAND,
    _HEADER_AVAIL,
    _HEADER_DEMAND,
    _HEADER_RRP,
    _REGIONS,
)

logger = logging.getLogger(__name__)

_NEMWEB_BASE = "https://nemweb.com.au"
_MMSDM_ARCHIVE_DIR = "/Data_Archive/Wholesale_Electricity/MMSDM/"
_ARCHIVE_DIR = "/Reports/Archive/DispatchIS_Reports/"
_CURRENT_DIR = "/Reports/Current/DispatchIS_Reports/"
_REQUEST_DELAY_S = 1.0          # fallback — settings.backfill_request_delay_s takes priority
_MAX_INTERVALS_PER_BACKFILL = 288   # 1 day of 5-min intervals
_TIMEOUT = httpx.Timeout(60.0)  # archive files can be large; 60s is safer than 30s
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 2.0       # doubles each attempt: 2s, 4s, 8s
_RATE_LIMIT_BACKOFF_S = 60.0    # wait before retrying a 403 (rate-limited, not auth)
# NEMWeb blocks automated user-agents; use a browser-like UA for all requests
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_MMSDM_TABLES = {
    "DISPATCHPRICE": "price",
    "DISPATCHINTERCONNECTORRES": "interconnector",
    "DISPATCHCONSTRAINT": "constraint",
    "DISPATCH_UNIT_SOLUTION": "unit_dispatch",
    "DISPATCHLOAD": "unit_dispatch",
    "DUDETAILSUMMARY": "unit_metadata",
    "BIDDAYOFFER": "bid_offer",
    "BIDPEROFFER": "bid_offer",
}


# ── HTTP retry helper ────────────────────────────────────────────────

async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _RETRY_BASE_DELAY_S,
) -> bytes:
    """GET url with exponential backoff.

    Retries on connection errors, timeouts, and 5xx responses.
    Raises the last exception after max_retries exhausted.
    """
    last_exc: Exception = RuntimeError(f"no attempts made for {url}")
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
        ) as exc:
            last_exc = exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 403:
                # 403 from NEMWeb = transient rate-limiting; back off and retry
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "403 rate-limited on %s (attempt %d/%d) — waiting %.0fs",
                        url, attempt + 1, max_retries, _RATE_LIMIT_BACKOFF_S,
                    )
                    await asyncio.sleep(_RATE_LIMIT_BACKOFF_S)
                continue
            elif status >= 500:
                last_exc = exc
            else:
                raise  # 404, 401, etc. — permanent, don't retry
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.debug(
                "Retry %d/%d for %s in %.1fs: %s",
                attempt + 1, max_retries, url, delay, last_exc,
            )
            await asyncio.sleep(delay)
    raise last_exc


# ── Public interface ──────────────────────────────────────────────────

async def backfill_recent_gaps(db_session_factory) -> int:
    """Fill missing market_events rows for the last 24 hours.

    Called by the APScheduler archive_backfill job.
    Returns the number of intervals inserted.
    """
    async with db_session_factory() as session:
        existing = await _fetch_existing_intervals(session)
        await _ensure_cursor(session, "recent_gap_backfill", "AEMO_DISPATCH_PRICE")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    to_fetch = _identify_gaps(existing, cutoff)

    if not to_fetch:
        logger.debug("Archive backfill: no gaps found")
        return 0

    logger.info("Archive backfill: %d intervals to fetch", len(to_fetch))
    inserted = 0

    async with httpx.AsyncClient(
        base_url=_NEMWEB_BASE,
        timeout=_TIMEOUT,
        headers={"User-Agent": _BROWSER_UA},
        follow_redirects=True,
    ) as client:
        for ts in to_fetch[:_MAX_INTERVALS_PER_BACKFILL]:
            try:
                rows = await _fetch_interval(client, ts)
                if rows:
                    async with db_session_factory() as session:
                        await _upsert_rows(session, rows)
                        await _update_cursor(session, "recent_gap_backfill", ts, "ok", None)
                        await session.commit()
                    inserted += len(rows)
                await asyncio.sleep(_REQUEST_DELAY_S)
            except Exception as exc:
                logger.warning("Archive backfill failed for %s: %s", ts.isoformat(), exc)
                async with db_session_factory() as session:
                    await _update_cursor(session, "recent_gap_backfill", ts, "error", str(exc))
                    await session.commit()

    logger.info("Archive backfill: inserted %d rows", inserted)
    return inserted


async def fetch_interval(timestamp: datetime) -> list[dict[str, Any]]:
    """Fetch and parse a single 5-minute dispatch interval from NEMWeb.

    Returns a list of per-region row dicts ready for market_events insertion.
    """
    async with httpx.AsyncClient(
        base_url=_NEMWEB_BASE,
        timeout=_TIMEOUT,
        headers={"User-Agent": _BROWSER_UA},
        follow_redirects=True,
    ) as client:
        return await _fetch_interval(client, timestamp)


async def backfill_mmsdm_archive(
    db_session_factory,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    max_files: int | None = None,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    """Bulk backfill historical MMSDM monthly archive files.

    Idempotency contract:
    - Rows are upserted against deterministic uniqueness keys (no double-counting).
    - The cursor advances at month granularity: all table files for a given month
      must succeed before the cursor moves forward.  On resume, months <= the
      cursor are skipped without re-downloading, saving significant bandwidth.
    - Partial-retry: within an incomplete month, files that already succeeded on a
      prior interrupted run are tracked in the cursor's error field as JSON and
      skipped on resume.  This avoids re-downloading large files after a transient
      failure partway through a month.
    - Failed files are collected and returned so acquire_manifest.py can write a
      complete provenance record.

    Returns a rich stats dict including per-file hashes and the failed-file list.
    """
    import json as _json
    from itertools import groupby
    from config.settings import get_settings

    settings = get_settings()
    start = start_date or datetime.fromisoformat(settings.backfill_start_date).replace(tzinfo=timezone.utc)
    end = end_date or datetime.now(timezone.utc)
    requested_tables = [t.upper() for t in (tables or [t.strip() for t in settings.backfill_tables.split(",")])]
    request_delay = settings.backfill_request_delay_s

    counts: dict[str, Any] = {
        "files_ok": 0,
        "files_failed": 0,
        "price_rows": 0,
        "driver_rows": 0,
        "unit_rows": 0,
        "unit_metadata_rows": 0,
        "file_hashes": {},          # url → sha256 hex
        "failed_files": [],         # list of {"url": ..., "error": ...}
        "months_skipped": 0,
        "months_completed": 0,
        "files_skipped_partial": 0,
    }

    # Read existing cursor so we can skip already-processed months and
    # resume partial months without re-downloading completed files.
    last_completed_month: datetime | None = None
    partial_month_key: str | None = None
    partial_ok_urls: set[str] = set()
    async with db_session_factory() as session:
        await _ensure_cursor(session, "mmsdm_archive_backfill", "AEMO_MMSDM")
        from sqlalchemy import select
        from app.db.models import BackfillCursor
        result = await session.execute(
            select(BackfillCursor).where(BackfillCursor.name == "mmsdm_archive_backfill")
        )
        cursor_row = result.scalar_one_or_none()
        if cursor_row and cursor_row.last_successful_interval:
            last_completed_month = cursor_row.last_successful_interval
        if cursor_row and cursor_row.error:
            try:
                partial_meta = _json.loads(cursor_row.error)
                partial_month_key = partial_meta.get("partial_month")
                partial_ok_urls = set(partial_meta.get("partial_ok", []))
                if partial_month_key:
                    logger.info(
                        "MMSDM backfill: resuming partial month %s (%d files already done)",
                        partial_month_key, len(partial_ok_urls),
                    )
            except (_json.JSONDecodeError, AttributeError):
                pass
        await session.commit()

    if last_completed_month:
        logger.info(
            "MMSDM backfill resuming from %s (months up to this date already complete)",
            last_completed_month.strftime("%Y-%m"),
        )

    async with httpx.AsyncClient(
        base_url=_NEMWEB_BASE,
        timeout=_TIMEOUT,
        headers={"User-Agent": _BROWSER_UA},
        follow_redirects=True,
    ) as client:
        all_urls = await discover_mmsdm_files(client, start, end, requested_tables)
        if max_files is not None:
            all_urls = all_urls[:max_files]

        # Group by archive month so we can advance the cursor only after a full
        # month completes.  URLs without a parseable month are processed individually
        # and never advance the monthly cursor.
        def _month_sort_key(url: str) -> str:
            m = _archive_month_from_url(url)
            return m.isoformat() if m else f"zz_{url}"

        for month_key, month_iter in groupby(sorted(all_urls, key=_month_sort_key), key=_month_sort_key):
            month_urls = list(month_iter)
            month_dt = _archive_month_from_url(month_urls[0])

            # Skip months the cursor has already confirmed complete
            if (
                last_completed_month is not None
                and month_dt is not None
                and month_dt <= last_completed_month
            ):
                logger.debug("Skipping %s (already in cursor)", month_key)
                counts["months_skipped"] += 1
                continue

            # For the current partial month, seed the in-progress ok set from cursor
            current_partial_ok: set[str] = (
                partial_ok_urls if month_key == partial_month_key else set()
            )

            month_all_ok = True
            for url in month_urls:
                # Skip files that already completed in a previous interrupted run
                if url in current_partial_ok:
                    logger.debug("Partial-retry skip: %s", url.split("/")[-1])
                    counts["files_skipped_partial"] += 1
                    counts["files_ok"] += 1
                    continue

                try:
                    content = await _fetch_with_retry(client, url)
                    raw_ref = hashlib.sha256(content).hexdigest()
                    counts["file_hashes"][url] = raw_ref
                    short_ref = raw_ref[:16]

                    price_rows = parse_mmsdm_dispatchprice_content(content, short_ref)
                    fcas_rows = parse_mmsdm_fcas_content(content, short_ref)
                    driver_rows = parse_mmsdm_driver_content(content, short_ref)
                    unit_payload = parse_mmsdm_unit_content(content, short_ref)
                    bid_rows = parse_mmsdm_bid_content(content, short_ref)

                    async with db_session_factory() as session:
                        await _upsert_rows(session, price_rows)
                        await _upsert_fcas_rows(session, fcas_rows)
                        await _upsert_driver_rows(session, driver_rows)
                        await _upsert_generator_units(session, unit_payload["metadata_rows"])
                        await _upsert_unit_rows(session, unit_payload["unit_rows"])
                        await _upsert_bid_rows(session, bid_rows)
                        await session.commit()

                    current_partial_ok.add(url)
                    # Persist partial progress so an interruption here is resumable
                    async with db_session_factory() as session:
                        await _update_cursor_partial(
                            session, "mmsdm_archive_backfill", month_key, current_partial_ok
                        )
                        await session.commit()

                    counts["files_ok"] += 1
                    counts["price_rows"] += len(price_rows)
                    counts["driver_rows"] += len(driver_rows)
                    counts["unit_rows"] += len(unit_payload["unit_rows"])
                    counts["unit_metadata_rows"] += len(unit_payload["metadata_rows"])
                    counts.setdefault("bid_rows", 0)
                    counts["bid_rows"] += len(bid_rows)
                    counts.setdefault("fcas_rows", 0)
                    counts["fcas_rows"] += len(fcas_rows)
                    logger.info(
                        "MMSDM %s: +%d price +%d fcas +%d driver +%d unit +%d bid rows",
                        url.split("/")[-1], len(price_rows), len(fcas_rows),
                        len(driver_rows), len(unit_payload["unit_rows"]), len(bid_rows),
                    )
                    await asyncio.sleep(request_delay)

                except Exception as exc:
                    month_all_ok = False
                    counts["files_failed"] += 1
                    counts["failed_files"].append({"url": url, "error": str(exc)})
                    logger.warning("MMSDM backfill failed for %s: %s", url, exc)
                    await asyncio.sleep(request_delay)

            # Advance cursor only after all files in the month succeed.
            # Clears partial-tracking JSON (error=None) on success.
            if month_all_ok and month_dt is not None:
                async with db_session_factory() as session:
                    await _update_cursor(
                        session, "mmsdm_archive_backfill", month_dt, "ok", None
                    )
                    await _increment_cursor_counts(
                        session, "mmsdm_archive_backfill",
                        files_ok=len(month_urls), files_failed=0,
                    )
                    await session.commit()
                last_completed_month = month_dt
                partial_month_key = None
                partial_ok_urls = set()
                counts["months_completed"] += 1
                logger.info("MMSDM month %s complete", month_key)

    return counts


async def discover_mmsdm_files(
    client: httpx.AsyncClient,
    start_date: datetime,
    end_date: datetime,
    tables: list[str] | None = None,
) -> list[str]:
    """Return MMSDM archive file URLs for the requested months/tables.

    Strategy:
    1. Probe deterministic candidate URLs via HEAD (fast, covers most months).
    2. For months where no file was found by step 1, fall back to the directory
       walker (slower but handles AEMO archive path changes between years).
    This two-phase approach avoids re-walking the full archive for months that
    are already fully covered, while not silently missing months where AEMO
    changed their directory structure.
    """
    requested = [t.upper() for t in (tables or list(_MMSDM_TABLES))]
    all_months = _iter_months(start_date, end_date)
    found_urls: list[str] = []
    hit_months: set[str] = set()

    for url in _candidate_mmsdm_urls(start_date, end_date, requested):
        try:
            # HEAD probe: single attempt, no retry — 404 = file doesn't exist
            # Small delay between probes avoids triggering NEMWeb rate limiting
            await asyncio.sleep(0.25)
            resp = await client.head(url, follow_redirects=True)
            if resp.status_code < 400:
                found_urls.append(url)
                month_dt = _archive_month_from_url(url)
                if month_dt:
                    hit_months.add(month_dt.strftime("%Y%m"))
        except (httpx.ConnectError, httpx.TimeoutException):
            for _ in range(2):
                try:
                    await asyncio.sleep(1.0)
                    resp = await client.head(url, follow_redirects=True)
                    if resp.status_code < 400:
                        found_urls.append(url)
                        month_dt = _archive_month_from_url(url)
                        if month_dt:
                            hit_months.add(month_dt.strftime("%Y%m"))
                    break
                except Exception:
                    continue
        except Exception:
            continue

    # Walk the directory listing for months where deterministic probes found nothing
    missed_months = [m for m in all_months if m.strftime("%Y%m") not in hit_months]
    if missed_months:
        walk_start = min(missed_months)
        walk_end = max(missed_months)
        logger.debug(
            "Walking MMSDM listing for %d months with no deterministic hit (%s to %s)",
            len(missed_months), walk_start.strftime("%Y-%m"), walk_end.strftime("%Y-%m"),
        )
        walked = await _walk_mmsdm_listing(client, walk_start, walk_end, requested)
        found_urls.extend(walked)

    return sorted(dict.fromkeys(found_urls))


# ── Internal fetch logic ──────────────────────────────────────────────

async def _fetch_interval(
    client: httpx.AsyncClient,
    ts: datetime,
) -> list[dict[str, Any]]:
    """Fetch the DISPATCHPRICE file for a given UTC timestamp."""
    # NEMWeb uses AEST (UTC+10) in filenames
    aest = ts.astimezone(timezone(timedelta(hours=10)))
    date_str = aest.strftime("%Y%m%d")
    time_str = aest.strftime("%H%M")

    # Try current directory first, then archive
    for base_dir in (_CURRENT_DIR, _ARCHIVE_DIR):
        try:
            index_url = f"{base_dir}"
            resp = await client.get(index_url)
            resp.raise_for_status()

            # Find the zip file for this date/time
            zip_name = _find_zip_in_index(resp.text, date_str, time_str)
            if not zip_name:
                continue

            zip_resp = await client.get(f"{base_dir}{zip_name}")
            zip_resp.raise_for_status()

            return _parse_zip(zip_resp.content, ts)
        except Exception as exc:
            logger.debug("Failed from %s for %s: %s", base_dir, ts.isoformat(), exc)

    return []


def _find_zip_in_index(html: str, date_str: str, time_str: str) -> str | None:
    """Scan directory listing HTML for the matching zip filename."""
    pattern = re.compile(
        rf'PUBLIC_DISPATCHIS_{date_str}_{time_str}_\d+\.zip',
        re.IGNORECASE,
    )
    matches = pattern.findall(html)
    if not matches:
        return None
    # Return the highest sequence number
    return sorted(matches)[-1]


def _parse_zip(content: bytes, valid_time: datetime) -> list[dict[str, Any]]:
    """Extract and parse DISPATCHPRICE+REGIONSUM rows from a NEMWeb DispatchIS zip.

    Delegates to the table-aware parser in aemo_live_client so demand/availability
    come from REGIONSUM (v5) rather than hardcoded PRICE columns (v4).
    """
    from app.data.aemo_live_client import _parse_dispatch_zip

    raw_ref = hashlib.sha256(content).hexdigest()[:16]
    system_time = datetime.now(timezone.utc)
    try:
        prices = _parse_dispatch_zip(content, raw_ref)
    except Exception:
        return []

    rows = []
    for region, dp in prices.items():
        rows.append({
            "id": f"{raw_ref}-{region}",
            "tenant_id": "system",
            "source": "AEMO_DISPATCH_PRICE",
            "region": region,
            "valid_time": valid_time,
            "system_time": system_time,
            "price_rrp": dp.price_rrp,
            "demand_mw": dp.demand_mw,
            "availability_mw": dp.availability_mw,
            "data": {
                "raw_ref": raw_ref,
                "headroom_mw": max(dp.availability_mw - dp.demand_mw, 0.0),
            },
            "raw_ref": raw_ref,
        })
    return rows


def parse_mmsdm_dispatchprice_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM archive content containing DISPATCHPRICE rows.

    Handles plain CSV, gzip-compressed CSV, and zip containers from the AEMO
    MMSDM data archive. This is separate from live DispatchIS parsing because
    monthly archive files use different containers and can contain many
    intervals in one file.
    """
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
        if "DISPATCHINTERCONNECTORRES" in upper_name or "INTERCONNECTORRES" in upper_name:
            rows.extend(_parse_interconnector_text(text, raw_ref))
        elif "DISPATCHCONSTRAINT" in upper_name:
            rows.extend(_parse_constraint_text(text, raw_ref))
        else:
            # Plain/gzip content may not carry a useful filename; inspect headers.
            upper_text = text[:2000].upper()
            if "DISPATCHINTERCONNECTORRES" in upper_text or "INTERCONNECTORRES" in upper_text:
                rows.extend(_parse_interconnector_text(text, raw_ref))
            if "DISPATCHCONSTRAINT" in upper_text or ",CONSTRAINT," in upper_text:
                rows.extend(_parse_constraint_text(text, raw_ref))
    return rows


def parse_mmsdm_bid_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM bid/offer tables: BIDDAYOFFER and BIDPEROFFER.

    BIDDAYOFFER: day-ahead offers submitted before trading day (period_id=None).
    BIDPEROFFER (split across BIDPEROFFER1 / BIDPEROFFER2): intraday rebids
    for specific 30-min trading periods (period_id 1-48).
    """
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


_FCAS_COLUMNS = [
    "RAISE6SECRRP",
    "RAISE60SECRRP",
    "RAISE5MINRRP",
    "RAISEREGRRP",
    "LOWER6SECRRP",
    "LOWER60SECRRP",
    "LOWER5MINRRP",
    "LOWERREGRRP",
]

_FCAS_FIELD_MAP = {
    "RAISE6SECRRP": "raise_6sec_rrp",
    "RAISE60SECRRP": "raise_60sec_rrp",
    "RAISE5MINRRP": "raise_5min_rrp",
    "RAISEREGRRP": "raise_reg_rrp",
    "LOWER6SECRRP": "lower_6sec_rrp",
    "LOWER60SECRRP": "lower_60sec_rrp",
    "LOWER5MINRRP": "lower_5min_rrp",
    "LOWERREGRRP": "lower_reg_rrp",
}


def parse_mmsdm_fcas_content(content: bytes, raw_ref: str = "mmsdm") -> list[dict[str, Any]]:
    """Parse MMSDM archive content and extract FCAS ancillary service prices.

    Reads the same DISPATCHPRICE CSV as parse_mmsdm_dispatchprice_content but
    extracts the 8 FCAS service price columns instead of spot price.
    Returns rows suitable for upsert into fcas_price_events.
    """
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
        rows.extend(_parse_mmsdm_fcas_text(text, raw_ref))
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
            # Accept this as the DISPATCHPRICE header only if FCAS columns are present
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

            # Only emit a row if at least one FCAS price was parsed
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
                source="AEMO_DISPATCH_UNIT_SOLUTION",
                duid=duid,
                valid_time=valid_time,
                initial_mw=_float_or_none(parts, header, "INITIALMW"),
                total_cleared_mw=_float_or_none(parts, header, "TOTALCLEARED"),
                availability_mw=_float_or_none(parts, header, "AVAILABILITY"),
                target_mw=_first_float(parts, header, ["TARGETMW", "TARGET"]),
                ramp_rate=_first_float(parts, header, ["RAMPUPRATE", "RAMP_RATE", "RAMPDOWNRATE"]),
                semi_dispatch_cap=_float_or_none(parts, header, "SEMIDISPATCHCAP"),
                raw_ref=raw_ref,
                system_time=system_time,
                data=_row_payload(parts, header),
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
                source="AEMO_DISPATCHLOAD",
                duid=duid,
                valid_time=valid_time,
                initial_mw=_float_or_none(parts, header, "INITIALMW"),
                total_cleared_mw=_first_float(parts, header, ["TOTALCLEARED", "DISPATCHTARGET"]),
                availability_mw=_float_or_none(parts, header, "AVAILABILITY"),
                target_mw=_first_float(parts, header, ["DISPATCHTARGET", "TARGETMW"]),
                ramp_rate=_first_float(parts, header, ["RAMPUPRATE", "RAMPDOWNRATE"]),
                semi_dispatch_cap=_float_or_none(parts, header, "SEMIDISPATCHCAP"),
                raw_ref=raw_ref,
                system_time=system_time,
                data=_row_payload(parts, header),
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
            "fuel_type": _normalise_fuel_type(_first_text(parts, header, ["FUELTYPE", "FUEL_SOURCE", "TECHNOLOGYTYPE"]) or duid),
            "dispatch_type": _first_text(parts, header, ["DISPATCHTYPE", "SCHEDULE_TYPE"]),
            "max_capacity_mw": _first_float(parts, header, ["REGISTEREDCAPACITY", "MAXCAPACITY", "NAMEPLATECAPACITY"]),
            "metadata_json": {**metadata, "raw_ref": raw_ref},
        })
    return rows


def _parse_biddayoffer_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    """Parse BIDDAYOFFER CSV text into bid_offer row dicts (period_id=None).

    BIDDAYOFFER header: I,BIDS,BIDDAYOFFER,1,DUID,BIDTYPE,SETTLEMENTDATE,...,PRICEBAND1-10
    Contains price bands only — BANDAVAIL (availability per period) is in BIDPEROFFER.
    """
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
            min_load = _float_or_none(parts, header, "MINIMUMLOAD")

            row_id = hashlib.sha256(
                f"BIDDAYOFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{raw_ref}".encode()
            ).hexdigest()[:32]
            rows.append({
                "id": row_id,
                "tenant_id": "system",
                "source": "BIDDAYOFFER",
                "duid": duid,
                "region": _infer_region_from_duid(duid),
                "bid_type": bid_type,
                "settlement_date": settlement_date,
                "period_id": None,
                "offer_date": offer_date,
                "max_avail_mw": None,        # not in BIDDAYOFFER; use BIDPEROFFER
                "minimum_load_mw": min_load,
                "ramp_up_mw_per_min": None,  # not in BIDDAYOFFER v1
                "ramp_down_mw_per_min": None,
                "price_bands": price_bands,
                "avail_bands": {},            # availability is in BIDPEROFFER per period
                "data": {"raw_ref": raw_ref},
                "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _parse_bidperoffer_text(text: str, raw_ref: str) -> list[dict[str, Any]]:
    """Parse BIDPEROFFER (BIDOFFERPERIOD) CSV into bid_offer rows (period_id 1-48).

    Actual section name in MMSDM: BIDOFFERPERIOD.
    Columns: DUID, BIDTYPE, TRADINGDATE, OFFERDATETIME, PERIODID, MAXAVAIL,
             FIXEDLOAD, RAMPUPRATE, RAMPDOWNRATE, BANDAVAIL1-10, PASAAVAILABILITY.
    """
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
            # BIDOFFERPERIOD uses TRADINGDATE; fall back to SETTLEMENTDATE for older formats
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
            max_avail = _first_float(parts, header, ["MAXAVAIL"])
            ramp_up = _first_float(parts, header, ["RAMPUPRATE", "ROCUP"])
            ramp_dn = _first_float(parts, header, ["RAMPDOWNRATE", "ROCDOWN"])

            row_id = hashlib.sha256(
                f"BIDPEROFFER|{duid}|{bid_type}|{settlement_date.isoformat()}|{period_id}|{raw_ref}".encode()
            ).hexdigest()[:32]
            rows.append({
                "id": row_id,
                "tenant_id": "system",
                "source": "BIDPEROFFER",
                "duid": duid,
                "region": _infer_region_from_duid(duid),
                "bid_type": bid_type,
                "settlement_date": settlement_date,
                "period_id": period_id,
                "offer_date": offer_date,
                "max_avail_mw": max_avail,
                "minimum_load_mw": None,
                "ramp_up_mw_per_min": ramp_up,
                "ramp_down_mw_per_min": ramp_dn,
                "price_bands": {},
                "avail_bands": avail_bands,
                "data": {"raw_ref": raw_ref},
                "raw_ref": raw_ref,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _unit_row(
    source: str,
    duid: str,
    valid_time: datetime,
    initial_mw: float | None,
    total_cleared_mw: float | None,
    availability_mw: float | None,
    target_mw: float | None,
    ramp_rate: float | None,
    semi_dispatch_cap: float | None,
    raw_ref: str,
    system_time: datetime,
    data: dict[str, Any],
) -> dict[str, Any]:
    row_id = hashlib.sha256(f"{source}|{duid}|{valid_time.isoformat()}|{raw_ref}".encode()).hexdigest()[:32]
    fuel_type = _normalise_fuel_type(duid)
    return {
        "id": row_id,
        "tenant_id": "system",
        "source": source,
        "duid": duid,
        "station_name": None,
        "participant": None,
        "region": _infer_region_from_duid(duid),
        "fuel_type": fuel_type,
        "valid_time": valid_time,
        "system_time": system_time,
        "initial_mw": initial_mw,
        "total_cleared_mw": total_cleared_mw,
        "availability_mw": availability_mw,
        "target_mw": target_mw,
        "ramp_rate": ramp_rate,
        "semi_dispatch_cap": semi_dispatch_cap,
        "data": {**data, "raw_ref": raw_ref},
        "raw_ref": raw_ref,
    }


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
                source="AEMO_DISPATCHINTERCONNECTORRES",
                driver_type="interconnector",
                element_id=interconnector_id,
                valid_time=valid_time,
                values={k: v for k, v in values.items() if v is not None},
                raw_ref=raw_ref,
                system_time=system_time,
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
        if parts[0].upper() == "I" and ("DISPATCHCONSTRAINT" in ",".join(parts).upper() or ",CONSTRAINT," in ",".join(parts).upper()):
            header = [p.upper() for p in parts]
            continue
        if parts[0].upper() != "D" or header is None:
            continue
        try:
            valid_time = _parse_archive_dt(parts[_idx(header, "SETTLEMENTDATE")])
            constraint_id = parts[_idx(header, "CONSTRAINTID")]
            marginal_value = _float_or_none(parts, header, "MARGINALVALUE")
            violation_degree = _float_or_none(parts, header, "VIOLATIONDEGREE")
            # Only store binding constraints — non-binding rows (marginal_value=0,
            # violation_degree=0) had no effect on dispatch or prices and have no
            # attribution value. This reduces storage by ~90%.
            if not marginal_value and not violation_degree:
                continue
            values = {
                "rhs": _float_or_none(parts, header, "RHS"),
                "marginal_value": marginal_value,
                "violation_degree": violation_degree,
            }
            rows.append(_driver_row(
                source="AEMO_DISPATCHCONSTRAINT",
                driver_type="constraint",
                element_id=constraint_id,
                valid_time=valid_time,
                values={k: v for k, v in values.items() if v is not None},
                raw_ref=raw_ref,
                system_time=system_time,
            ))
        except (ValueError, IndexError):
            continue
    return rows


def _driver_row(
    source: str,
    driver_type: str,
    element_id: str,
    valid_time: datetime,
    values: dict[str, Any],
    raw_ref: str,
    system_time: datetime,
) -> dict[str, Any]:
    row_id = hashlib.sha256(
        f"{source}|{driver_type}|{element_id}|{valid_time.isoformat()}|{raw_ref}".encode()
    ).hexdigest()[:32]
    return {
        "id": row_id,
        "tenant_id": "system",
        "source": source,
        "driver_type": driver_type,
        "element_id": element_id,
        "region": _infer_region_from_element(element_id),
        "valid_time": valid_time,
        "system_time": system_time,
        "values": values,
        "raw_ref": raw_ref,
    }


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


def _row_payload(parts: list[str], header: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for i, name in enumerate(header):
        if i < len(parts):
            payload[name.lower()] = parts[i]
    return payload


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
        ("battery", ("BAT", "BESS", "BATT")),
        ("hydro", ("HYDRO", "WATER", "GORDON", "POAT", "REECE", "MURRAY", "TUMUT", "GUTHEGA", "UPPTUMUT")),
        ("coal", ("COAL", "BROWN", "BLACK", "ERARING", "BAYS", "LIDDELL", "TARONG", "STANWELL", "LOY", "YWPS")),
        ("gas", ("GAS", "OCGT", "CCGT", "TURBINE", "TORR", "LADB")),
        ("wind", ("WIND", "WF", "SNOWTOWN", "MACARTHUR")),
        ("solar", ("SOLAR", "SF", "PV")),
        ("biomass", ("BIO", "BAGASSE")),
        ("demand_response", ("DR", "DSP", "LOAD")),
    ]
    for fuel, tokens in checks:
        if any(token in upper for token in tokens):
            return fuel
    return "unknown"


def _parse_archive_dt(value: str) -> datetime:
    from app.data.aemo_live_client import _parse_aemo_dt
    return _parse_aemo_dt(value.strip().replace('"', ""))


# AEMO uses abbreviated names in MMSDM DVD filenames for some tables (pre-2024-08).
_TABLE_FILENAME_ALIASES: dict[str, list[str]] = {
    # DISPATCH_UNIT_SOLUTION is shortened in older DVD exports
    "DISPATCH_UNIT_SOLUTION": ["DISPATCH_UNIT_SOLUTION", "DISPATCH_UNIT_SOLN"],
    # BIDPEROFFER is split across two files (large table)
    "BIDPEROFFER": ["BIDPEROFFER1", "BIDPEROFFER2"],
}

# AEMO switched from PUBLIC_DVD_* to PUBLIC_ARCHIVE#*#FILE01#* naming in August 2024.
# The # is double-percent-encoded (%2523) because the filenames use literal # characters.
_ARCHIVE_FORMAT_START = datetime(2024, 8, 1, tzinfo=timezone.utc)

# Table name changes in the ARCHIVE format (2024-08+).
_TABLE_ARCHIVE_ALIASES: dict[str, list[str]] = {
    # BIDPEROFFER1/2 (DVD split) → BIDOFFERPERIOD (single ARCHIVE file)
    "BIDPEROFFER": ["BIDOFFERPERIOD"],
}


def _candidate_mmsdm_urls(
    start_date: datetime,
    end_date: datetime,
    tables: list[str],
) -> list[str]:
    urls: list[str] = []
    for month in _iter_months(start_date, end_date):
        yyyy = month.strftime("%Y")
        mm = month.strftime("%m")
        yyyymm = month.strftime("%Y%m")
        data_base = (
            f"{_MMSDM_ARCHIVE_DIR}{yyyy}/MMSDM_{yyyy}_{mm}"
            f"/MMSDM_Historical_Data_SQLLoader/DATA/"
        )
        if month >= _ARCHIVE_FORMAT_START:
            # New format: PUBLIC_ARCHIVE%2523{TABLE}%2523FILE01%2523{YYYYMM}010000.zip
            for table in tables:
                for variant in _TABLE_ARCHIVE_ALIASES.get(table, [table]):
                    urls.append(
                        f"{data_base}PUBLIC_ARCHIVE%2523{variant}%2523FILE01%2523{yyyymm}010000.zip"
                    )
        else:
            # Old format: PUBLIC_DVD_{TABLE}_{YYYYMM}010000.zip
            for table in tables:
                for variant in _TABLE_FILENAME_ALIASES.get(table, [table]):
                    filename = f"PUBLIC_DVD_{variant}_{yyyymm}010000.zip"
                    urls.extend([
                        f"{data_base}{filename}",
                        f"{_MMSDM_ARCHIVE_DIR}{yyyy}/MMSDM_{yyyy}_{mm}/{filename}",
                        f"{_MMSDM_ARCHIVE_DIR}{yyyy}/{filename}",
                    ])
    return urls


async def _walk_mmsdm_listing(
    client: httpx.AsyncClient,
    start_date: datetime,
    end_date: datetime,
    tables: list[str],
    max_depth: int = 5,
) -> list[str]:
    wanted = tuple(t.upper() for t in tables)
    seen: set[str] = set()
    found: list[str] = []
    queue: list[tuple[str, int]] = [(_MMSDM_ARCHIVE_DIR, 0)]
    month_tokens = {m.strftime("%Y%m") for m in _iter_months(start_date, end_date)}

    while queue:
        path, depth = queue.pop(0)
        if path in seen or depth > max_depth:
            continue
        seen.add(path)
        try:
            resp = await client.get(path)
            resp.raise_for_status()
        except Exception:
            continue
        for href in _extract_links(resp.text):
            child = _join_url_path(path, href)
            upper = child.upper()
            if upper.endswith((".ZIP", ".CSV", ".CSV.GZ")):
                if any(table in upper for table in wanted) and any(token in upper for token in month_tokens):
                    found.append(child)
            elif href.endswith("/") and not href.startswith("../"):
                queue.append((child, depth + 1))
    return sorted(dict.fromkeys(found))


def _extract_links(html: str) -> list[str]:
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)


def _join_url_path(base_path: str, href: str) -> str:
    if href.startswith("http"):
        return href.replace(_NEMWEB_BASE, "")
    if href.startswith("/"):
        return href
    if not base_path.endswith("/"):
        base_path += "/"
    return base_path + href


def _iter_months(start_date: datetime, end_date: datetime) -> list[datetime]:
    cur = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[datetime] = []
    while cur <= end:
        months.append(cur)
        year = cur.year + (cur.month // 12)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = cur.replace(year=year, month=month)
    return months


def _archive_month_from_url(url: str) -> datetime | None:
    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])010000", url)
    if not match:
        return None
    return datetime(int(match.group(1)), int(match.group(2)), 1, tzinfo=timezone.utc)


# ── DB helpers ────────────────────────────────────────────────────────

async def _fetch_existing_intervals(session) -> set[datetime]:
    """Return the set of valid_times already in market_events."""
    from sqlalchemy import select, text
    from app.db.models import MarketEvent
    result = await session.execute(
        select(MarketEvent.valid_time).where(
            MarketEvent.source == "AEMO_DISPATCH_PRICE"
        )
    )
    return {row[0] for row in result.fetchall()}


async def _ensure_cursor(session, name: str, source: str) -> None:
    from sqlalchemy import select
    from app.db.models import BackfillCursor
    existing = await session.execute(select(BackfillCursor).where(BackfillCursor.name == name))
    if existing.scalar_one_or_none() is None:
        session.add(BackfillCursor(name=name, source=source, status="idle"))


async def _update_cursor(
    session,
    name: str,
    interval: datetime,
    status: str,
    error: str | None,
) -> None:
    from sqlalchemy import select
    from app.db.models import BackfillCursor
    result = await session.execute(select(BackfillCursor).where(BackfillCursor.name == name))
    cursor = result.scalar_one_or_none()
    if cursor is None:
        cursor = BackfillCursor(name=name, source="AEMO_DISPATCH_PRICE")
        session.add(cursor)
    if status == "ok":
        cursor.last_successful_interval = interval
    cursor.status = status
    cursor.error = error


async def _update_cursor_partial(
    session,
    name: str,
    month_key: str,
    completed_urls: set[str],
) -> None:
    """Store partial-month progress in the cursor's error field as JSON.

    Called after each successful file within an incomplete month so that
    interrupted runs can skip already-done files on resume.  Cleared by
    _update_cursor(..., status="ok", error=None) when the month completes.
    """
    import json as _json
    from sqlalchemy import select
    from app.db.models import BackfillCursor
    result = await session.execute(select(BackfillCursor).where(BackfillCursor.name == name))
    cursor = result.scalar_one_or_none()
    if cursor is None:
        return
    cursor.error = _json.dumps({
        "partial_month": month_key,
        "partial_ok": sorted(completed_urls),
    })
    cursor.status = "partial"


async def _increment_cursor_counts(
    session,
    name: str,
    files_ok: int = 0,
    files_failed: int = 0,
) -> None:
    from sqlalchemy import select
    from app.db.models import BackfillCursor
    result = await session.execute(select(BackfillCursor).where(BackfillCursor.name == name))
    cursor = result.scalar_one_or_none()
    if cursor is None:
        return
    cursor.files_completed = (cursor.files_completed or 0) + files_ok
    cursor.files_failed = (cursor.files_failed or 0) + files_failed


def _identify_gaps(existing: set[datetime], cutoff: datetime) -> list[datetime]:
    """Return 5-minute intervals in the last 24h that are absent from DB."""
    now = datetime.now(timezone.utc)
    # Round now down to nearest 5-min
    minute = (now.minute // 5) * 5
    now_rounded = now.replace(minute=minute, second=0, microsecond=0)

    intervals = []
    t = cutoff.replace(second=0, microsecond=0)
    minute_rounded = (t.minute // 5) * 5
    t = t.replace(minute=minute_rounded)

    while t <= now_rounded:
        if t not in existing:
            intervals.append(t)
        t += timedelta(minutes=5)

    return intervals


_BULK_BATCH = 2000   # rows per INSERT statement
# asyncpg hard limit: 32767 placeholders per statement.
# BidOffer has 17 columns → max 1927 rows per batch.
_BID_BULK_BATCH = 32767 // 17  # 1927


def _bulk_insert_stmt(model, rows: list[dict[str, Any]], on_conflict: str = "nothing"):
    """Build a dialect-aware bulk INSERT statement.

    PostgreSQL: INSERT ... ON CONFLICT DO NOTHING / DO UPDATE
    SQLite:     INSERT OR IGNORE ...
    """
    from sqlalchemy import inspect as sa_inspect
    dialect = sa_inspect(model).mapper.persist_selectable.bind
    # dialect binding may not be available in all contexts; fall back safely
    try:
        dialect_name = dialect.dialect.name
    except Exception:
        dialect_name = "postgresql"

    if dialect_name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _insert
        stmt = _insert(model).values(rows)
        return stmt.on_conflict_do_nothing()
    else:
        from sqlalchemy.dialects.postgresql import insert as _insert
        stmt = _insert(model).values(rows)
        if on_conflict == "nothing":
            return stmt.on_conflict_do_nothing()
        return stmt  # caller handles update


async def _upsert_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert market_event rows, skipping duplicates via ON CONFLICT DO NOTHING."""
    if not rows:
        return
    from app.db.models import MarketEvent
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for i in range(0, len(rows), _BULK_BATCH):
        batch = rows[i: i + _BULK_BATCH]
        stmt = pg_insert(MarketEvent).values(batch).on_conflict_do_nothing(
            index_elements=["source", "region", "valid_time"]
        )
        await session.execute(stmt)


async def _upsert_driver_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert market_driver_event rows, skipping duplicates."""
    if not rows:
        return
    from app.db.models import MarketDriverEvent
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for i in range(0, len(rows), _BULK_BATCH):
        batch = rows[i: i + _BULK_BATCH]
        stmt = pg_insert(MarketDriverEvent).values(batch).on_conflict_do_nothing(
            index_elements=["source", "driver_type", "element_id", "valid_time"]
        )
        await session.execute(stmt)


async def _upsert_generator_units(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-upsert generator metadata keyed by DUID.

    DUDETAILSUMMARY contains multiple rows per DUID (one per effective date).
    We deduplicate within the batch and use ON CONFLICT DO UPDATE to keep the
    most complete metadata per DUID.
    """
    if not rows:
        return
    from app.db.models import GeneratorUnit
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    # Deduplicate within batch — last row per DUID wins (latest effective date)
    deduped: dict[str, dict] = {}
    for row in rows:
        deduped[row["duid"]] = row
    unique_rows = list(deduped.values())

    for i in range(0, len(unique_rows), _BULK_BATCH):
        batch = unique_rows[i: i + _BULK_BATCH]
        stmt = pg_insert(GeneratorUnit).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["duid"],
            set_={
                col: stmt.excluded[col]
                for col in ("station_name", "participant", "region",
                            "fuel_type", "dispatch_type", "max_capacity_mw", "metadata_json")
                if col in stmt.excluded
            },
        )
        await session.execute(stmt)


async def _upsert_unit_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert unit dispatch event rows, skipping duplicates.

    Generator metadata enrichment (station_name, fuel_type, region) is handled
    by a separate pass after all DUDETAILSUMMARY data is loaded, rather than
    per-row lookup here, to avoid N+1 queries.
    """
    if not rows:
        return
    from app.db.models import UnitDispatchEvent
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for i in range(0, len(rows), _BULK_BATCH):
        batch = rows[i: i + _BULK_BATCH]
        stmt = pg_insert(UnitDispatchEvent).values(batch).on_conflict_do_nothing(
            index_elements=["source", "duid", "valid_time"]
        )
        await session.execute(stmt)


async def _upsert_bid_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert bid/offer rows, skipping duplicates."""
    if not rows:
        return
    from app.db.models import BidOffer
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for i in range(0, len(rows), _BID_BULK_BATCH):
        batch = rows[i: i + _BID_BULK_BATCH]
        # on_conflict_do_nothing() without index_elements generates bare
        # ON CONFLICT DO NOTHING — catches both PK and composite-index conflicts
        # (needed for retries where partial rows from a failed run already exist)
        stmt = pg_insert(BidOffer).values(batch).on_conflict_do_nothing()
        await session.execute(stmt)


async def _upsert_fcas_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert FCAS price rows, skipping duplicates on (region, valid_time)."""
    if not rows:
        return
    from app.db.models import FcasPriceEvent
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    for i in range(0, len(rows), _BULK_BATCH):
        batch = rows[i: i + _BULK_BATCH]
        stmt = pg_insert(FcasPriceEvent).values(batch).on_conflict_do_nothing(
            index_elements=["region", "valid_time"]
        )
        await session.execute(stmt)
