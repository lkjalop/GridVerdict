"""Concrete AEMO Market Notices client for the news correlator.

Implements the NewsMCPClient Protocol from news_correlator.py.
Fetches from the public NEMWeb Market Notice feed — no credentials required.

Feed URL: https://nemweb.com.au/Reports/Current/Market_Notice/
Archive: https://nemweb.com.au/Reports/Archive/Market_Notice/

Notice format: fixed-width text files, one notice per file.
This client fetches the index, parses notice IDs and timestamps, fetches each
notice file, extracts the structured fields, and returns NewsItem objects ranked
by credibility tier.

Rate limit: poll at most every 60 seconds. Cache all fetched notices in memory
for the session (notices are immutable once published).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

from app.engines.forecasting.types import NewsItem

NEMWEB_NOTICE_INDEX = "https://nemweb.com.au/Reports/Current/Market_Notice/"
NEMWEB_NOTICE_ARCHIVE = "https://nemweb.com.au/Reports/Archive/Market_Notice/"
NOTICE_TYPES_TIER1 = frozenset({
    "LACK OF RESERVE 1", "LACK OF RESERVE 2", "LACK OF RESERVE 3",
    "LOR1", "LOR2", "LOR3",
    "RECLASSIFY CONTINGENCY", "MARKET INTERVENTION",
    "DIRECTIONS", "INTER_CONSTRAINT",
    "CONSTRAINT RELAXATION", "CONSTRAINT TIGHTENING",
})
NOTICE_TYPES_TIER2 = frozenset({
    "MT PASA REVISION", "ST PASA REVISION",
    "RESERVE TRADER ACTIVATION", "RESERVE TRADER CANCELLATION",
    "ADMINISTERED PRICE CAP", "ADMINISTERED PRICE FLOOR",
})

# NEM regions in AEMO notice text
REGION_PATTERN = re.compile(r"\b(NSW1?|VIC1?|QLD1?|SA1?|TAS1?)\b", re.IGNORECASE)


@dataclass
class _CachedNotice:
    item: NewsItem
    fetched_at: float   # monotonic time for cache TTL


class AEMOMarketNoticesClient:
    """Fetches and caches AEMO Market Notices from public NEMWeb feed.

    Thread-safety: not thread-safe. Use one instance per async task or wrap
    with asyncio.Lock for concurrent access.
    """

    def __init__(
        self,
        index_url: str = NEMWEB_NOTICE_INDEX,
        poll_interval_s: int = 60,
        timeout_s: float = 10.0,
    ):
        self._index_url = index_url
        self._poll_interval = poll_interval_s
        self._timeout = timeout_s
        self._cache: dict[str, _CachedNotice] = {}
        self._last_poll: float = 0.0

    def fetch(
        self,
        start: datetime,
        end: datetime,
        region: str | None,
    ) -> Sequence[NewsItem]:
        """Return cached+fresh notices in [start, end] for the given region."""
        self._refresh_if_due()
        results = []
        for cached in self._cache.values():
            item = cached.item
            if start <= item.timestamp <= end:
                if region is None or item.region is None or item.region == region:
                    results.append(item)
        return sorted(results, key=lambda i: i.timestamp, reverse=True)

    def _refresh_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_poll < self._poll_interval:
            return
        self._last_poll = now
        try:
            new_ids = self._fetch_index()
            for notice_id in new_ids:
                if notice_id not in self._cache:
                    item = self._fetch_notice(notice_id)
                    if item:
                        self._cache[notice_id] = _CachedNotice(item, now)
        except Exception as exc:
            logger.debug("Notices fetch failed (cached data still available): %s", exc)

    def _fetch_index(self) -> list[str]:
        """Parse the NEMWeb directory listing for notice file IDs."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(self._index_url)
            resp.raise_for_status()
        # Directory listing contains hrefs like PUBLIC_MARKET_NOTICE_<id>.zip
        ids = re.findall(
            r'href="(PUBLIC_MARKET_NOTICE_\d+\.zip)"',
            resp.text,
            re.IGNORECASE,
        )
        return ids

    def _fetch_notice(self, file_id: str) -> NewsItem | None:
        """Download and parse a single market notice file."""
        url = urljoin(self._index_url, file_id)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except Exception:
            return None

        text = self._extract_text(resp.content, file_id)
        if not text:
            return None
        return self._parse_notice_text(text, file_id)

    @staticmethod
    def _extract_text(content: bytes, file_id: str) -> str | None:
        """Unzip if needed, return text content."""
        if file_id.lower().endswith(".zip"):
            import zipfile, io
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    name = zf.namelist()[0]
                    return zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                return None
        return content.decode("utf-8", errors="replace")

    @classmethod
    def _parse_notice_text(cls, text: str, file_id: str) -> NewsItem | None:
        """Extract structured fields from a fixed-width AEMO market notice."""
        lines = [l.rstrip() for l in text.splitlines()]
        notice_type = cls._extract_field(lines, "NOTICE TYPE")
        reason = cls._extract_field(lines, "REASON") or cls._extract_field(lines, "DESCRIPTION") or ""
        region_text = cls._extract_field(lines, "REGION") or text
        timestamp_str = cls._extract_field(lines, "CREATION TIME") or cls._extract_field(lines, "ISSUE DATE")

        if not notice_type:
            return None

        notice_type_upper = notice_type.upper()
        if notice_type_upper in NOTICE_TYPES_TIER1:
            tier = 1
        elif notice_type_upper in NOTICE_TYPES_TIER2:
            tier = 2
        else:
            tier = 2    # default to tier 2 for unknown types

        region_match = REGION_PATTERN.search(region_text)
        region = region_match.group(0).upper() if region_match else None
        # Normalise region codes: NSW -> NSW1, VIC -> VIC1 etc.
        if region and not region.endswith("1"):
            region = region + "1"

        timestamp = cls._parse_timestamp(timestamp_str)
        if not timestamp:
            return None

        return NewsItem(
            timestamp=timestamp,
            source="AEMO Market Notice",
            credibility_tier=tier,
            title=f"{notice_type}: {reason[:80]}",
            summary=reason[:500],
            url=f"https://nemweb.com.au/Reports/Current/Market_Notice/{file_id}",
            region=region,
        )

    @staticmethod
    def _extract_field(lines: list[str], key: str) -> str | None:
        for line in lines:
            if line.upper().startswith(key.upper()):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    def fetch_active_notices(self, region: str | None = None) -> list[dict]:
        """Return recent notices as serialisable dicts — used by scheduler and MCP router.

        Refreshes the internal cache if 60s have elapsed. Filters by region when
        provided. Each dict has keys: notice_type, reason, region, timestamp,
        url, credibility_tier.
        """
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)
        items = self.fetch(start=start, end=now, region=region)
        return [
            {
                "notice_type": item.title.split(":")[0].strip() if item.title else None,
                "reason": item.summary or "",
                "region": item.region,
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "url": item.url,
                "credibility_tier": item.credibility_tier,
            }
            for item in items
        ]

    def fetch_archive_notices(
        self,
        start: datetime,
        end: datetime,
        cache_path: str | None = None,
    ) -> list[NewsItem]:
        """Fetch historical notices from the NEMWeb archive directory.

        Walks the archive index, downloads each notice file not yet in cache,
        filters to [start, end], and returns parsed NewsItem objects.

        Args:
            start:      earliest notice timestamp to include
            end:        latest notice timestamp to include
            cache_path: optional path to a JSON file for persisting downloaded
                        notice IDs between calls (avoids re-downloading)
        """
        import json as _json
        from pathlib import Path

        # Load or init file-based ID cache
        id_cache: dict[str, dict] = {}
        cache_file = Path(cache_path) if cache_path else None
        if cache_file and cache_file.exists():
            try:
                id_cache = _json.loads(cache_file.read_text())
            except Exception:
                id_cache = {}

        try:
            with httpx.Client(timeout=self._timeout * 3) as client:
                resp = client.get(
                    NEMWEB_NOTICE_ARCHIVE,
                    headers={"User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )},
                )
                resp.raise_for_status()
            archive_ids = re.findall(
                r'href="(PUBLIC_MARKET_NOTICE_\d+\.zip)"',
                resp.text,
                re.IGNORECASE,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not list AEMO notice archive: {exc}") from exc

        results: list[NewsItem] = []
        for file_id in archive_ids:
            if file_id in id_cache:
                data = id_cache[file_id]
                if data:
                    ts = self._parse_timestamp(data.get("timestamp"))
                    if ts and start <= ts <= end:
                        item = NewsItem(
                            timestamp=ts,
                            source=data.get("source", "AEMO Market Notice"),
                            credibility_tier=data.get("credibility_tier", 2),
                            title=data.get("title", ""),
                            summary=data.get("summary", ""),
                            url=data.get("url", ""),
                            region=data.get("region"),
                        )
                        results.append(item)
                continue

            # Fetch and parse the notice file
            url = urljoin(NEMWEB_NOTICE_ARCHIVE, file_id)
            try:
                with httpx.Client(timeout=self._timeout * 3) as client:
                    resp = client.get(
                        url,
                        headers={"User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        )},
                    )
                    resp.raise_for_status()
                text = self._extract_text(resp.content, file_id)
                item = self._parse_notice_text(text, file_id) if text else None
            except Exception:
                id_cache[file_id] = {}
                continue

            if item is None:
                id_cache[file_id] = {}
                continue

            # Cache the parsed fields
            id_cache[file_id] = {
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "source": item.source,
                "credibility_tier": item.credibility_tier,
                "title": item.title,
                "summary": item.summary,
                "url": item.url,
                "region": item.region,
            }

            if start <= item.timestamp <= end:
                results.append(item)

        # Persist cache
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(_json.dumps(id_cache, indent=2, default=str))
            except Exception:
                pass

        return sorted(results, key=lambda i: i.timestamp)

    @staticmethod
    def _parse_timestamp(ts: str | None) -> datetime | None:
        if not ts:
            return None
        formats = [
            "%d/%m/%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d-%b-%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
