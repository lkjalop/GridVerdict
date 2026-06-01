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
from app.mcp.aemo_parsers import (
    _iter_bid_rows_from_file,
    _first_col,
    parse_mmsdm_dispatchprice_content,
    parse_mmsdm_driver_content,
    parse_mmsdm_bid_content,
    parse_mmsdm_unit_content,
    parse_mmsdm_fcas_content,
    _TABLE_FILENAME_ALIASES,
    _TABLE_ARCHIVE_ALIASES,
    _ARCHIVE_FORMAT_START,
)
from app.mcp.aemo_db import (
    _fetch_existing_intervals,
    _identify_gaps,
    _ensure_cursor,
    _update_cursor,
    _update_cursor_partial,
    _increment_cursor_counts,
    _upsert_rows,
    _upsert_driver_rows,
    _upsert_generator_units,
    _upsert_unit_rows,
    _upsert_bid_rows,
    _upsert_fcas_rows,
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


# ── Streaming download + bid parser ──────────────────────────────────

# Files larger than this threshold are streamed to disk rather than held in RAM.
# BIDDAYOFFER is ~115–182 MB compressed; threshold set well below that.
_STREAM_THRESHOLD_BYTES = 10 * 1024 * 1024   # 10 MB

# Streaming upsert batch size: small enough to stay under asyncpg 32767-param limit
# (BidOffer has 17 cols → 32767 // 17 = 1927 max; 500 gives headroom for safety).
_BID_STREAM_BATCH = 500


async def _stream_download(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = _MAX_RETRIES,
) -> "tuple[pathlib.Path, str]":
    """Stream a large file to a named temp file; return (path, sha256_hex).

    Hashes during write so a second pass is not needed.
    Avoids loading 100–200 MB into RAM.  Caller is responsible for unlinking.
    """
    import pathlib, tempfile
    last_exc: Exception = RuntimeError(f"no attempts for {url}")
    for attempt in range(max_retries + 1):
        tmp = pathlib.Path(tempfile.mktemp(suffix=".zip"))
        try:
            file_hash = hashlib.sha256()
            async with client.stream("GET", url) as resp:
                if resp.status_code == 403:
                    tmp.unlink(missing_ok=True)
                    last_exc = httpx.HTTPStatusError(
                        f"403 rate-limited", request=resp.request, response=resp
                    )
                    if attempt < max_retries:
                        logger.warning(
                            "403 streaming %s (attempt %d/%d) — waiting %.0fs",
                            url, attempt + 1, max_retries, _RATE_LIMIT_BACKOFF_S,
                        )
                        await asyncio.sleep(_RATE_LIMIT_BACKOFF_S)
                    continue
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                        fh.write(chunk)
                        file_hash.update(chunk)
            return tmp, file_hash.hexdigest()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            tmp.unlink(missing_ok=True)
            last_exc = exc
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise
        if attempt < max_retries:
            delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
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
                    is_bid_file = (
                        "BIDDAYOFFER" in url.upper() or "BIDPEROFFER" in url.upper()
                    )

                    if is_bid_file:
                        # Streaming path: download to disk and batch-upsert, avoids
                        # loading 115–182 MB compressed BIDDAYOFFER files into RAM.
                        tmp_path, raw_ref = await _stream_download(client, url)
                        short_ref = raw_ref[:16]
                        counts["file_hashes"][url] = raw_ref
                        total_bid = 0
                        try:
                            batch: list[dict] = []
                            for row in _iter_bid_rows_from_file(tmp_path, short_ref):
                                batch.append(row)
                                if len(batch) >= _BID_STREAM_BATCH:
                                    async with db_session_factory() as session:
                                        await _upsert_bid_rows(session, batch)
                                        await session.commit()
                                    total_bid += len(batch)
                                    batch.clear()
                            if batch:
                                async with db_session_factory() as session:
                                    await _upsert_bid_rows(session, batch)
                                    await session.commit()
                                total_bid += len(batch)
                        finally:
                            tmp_path.unlink(missing_ok=True)

                        current_partial_ok.add(url)
                        # Persist partial progress so an interruption here is resumable
                        async with db_session_factory() as session:
                            await _update_cursor_partial(
                                session, "mmsdm_archive_backfill", month_key, current_partial_ok
                            )
                            await session.commit()

                        counts["files_ok"] += 1
                        counts.setdefault("bid_rows", 0)
                        counts["bid_rows"] += total_bid
                        logger.info(
                            "MMSDM %s: streamed %d bid rows",
                            url.split("/")[-1], total_bid,
                        )

                    else:
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

