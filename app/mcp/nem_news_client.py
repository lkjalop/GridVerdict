"""Read-only RSS/Atom client for public NEM market commentary.

RSS content is untrusted evidence. The caller may cite titles/links, but must
never treat feed content as instructions and must not pass article bodies to an
LLM for factual market reasoning.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

from config.settings import get_settings


def _settings_lists() -> tuple[list[str], list[str]]:
    settings = get_settings()
    urls = [u.strip() for u in settings.nem_news_rss_urls.split(",") if u.strip()]
    keywords = [k.strip().lower() for k in settings.nem_news_keywords.split(",") if k.strip()]
    return urls, keywords


class NEMNewsRSSClient:
    def __init__(self, urls: list[str] | None = None, keywords: list[str] | None = None) -> None:
        default_urls, default_keywords = _settings_lists()
        self.urls = urls or default_urls
        self.keywords = keywords or default_keywords

    async def fetch_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0), follow_redirects=True) as client:
            for url in self.urls:
                try:
                    resp = await client.get(url, headers={"User-Agent": "GridVerdict-RSS/1.0"})
                    resp.raise_for_status()
                    items.extend(parse_feed(resp.text, source_url=url, keywords=self.keywords))
                except Exception as exc:
                    logger.debug("RSS fetch failed for %s: %s", url, exc)
                    continue
        items.sort(key=lambda x: x.get("published_at") or "", reverse=True)
        return items[:limit]


def parse_feed(xml_text: str, source_url: str = "", keywords: list[str] | None = None) -> list[dict[str, Any]]:
    keywords = keywords or []
    root = ET.fromstring(xml_text)
    entries = root.findall(".//item")
    if not entries:
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    results: list[dict[str, Any]] = []
    for entry in entries:
        title = _text(entry, "title")
        summary = _text(entry, "description") or _text(entry, "summary")
        link = _text(entry, "link")
        if not link:
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            link = link_el.attrib.get("href", "") if link_el is not None else ""
        published_raw = _text(entry, "pubDate") or _text(entry, "published") or _text(entry, "updated")
        haystack = f"{title} {summary}".lower()
        matched = [kw for kw in keywords if kw in haystack]
        if keywords and not matched:
            continue
        results.append({
            "source": "NEM_NEWS_RSS",
            "title": title.strip(),
            "link": link.strip(),
            "published_at": _parse_date(published_raw).isoformat(),
            "summary": summary.strip()[:500],
            "matched_keywords": matched,
            "source_url": source_url,
        })
    return results


def _text(entry, name: str) -> str:
    found = entry.find(name)
    if found is None:
        found = entry.find(f"{{http://www.w3.org/2005/Atom}}{name}")
    return "".join(found.itertext()) if found is not None else ""


def _parse_date(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = parsedate_to_datetime(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
