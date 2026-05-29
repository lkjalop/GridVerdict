"""Calendar utilities for deterministic NEM seasonal query handling."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_SEASON_MONTHS = {
    "summer": (12, 1, 2),
    "autumn": (3, 4, 5),
    "winter": (6, 7, 8),
    "spring": (9, 10, 11),
}


def resolve_season_to_range(season: str, year: int) -> dict:
    """Resolve Southern Hemisphere season label to UTC date range.

    Summer is labelled by the year containing January and February. For example
    summer 2024 means 2023-12-01 through 2024-02-29 inclusive.
    """
    season_l = season.lower()
    if season_l not in _SEASON_MONTHS:
        raise ValueError(f"Unknown season: {season}")

    if season_l == "summer":
        from_dt = datetime(year - 1, 12, 1, tzinfo=timezone.utc)
        to_dt = datetime(year, 3, 1, tzinfo=timezone.utc)
    elif season_l == "autumn":
        from_dt = datetime(year, 3, 1, tzinfo=timezone.utc)
        to_dt = datetime(year, 6, 1, tzinfo=timezone.utc)
    elif season_l == "winter":
        from_dt = datetime(year, 6, 1, tzinfo=timezone.utc)
        to_dt = datetime(year, 9, 1, tzinfo=timezone.utc)
    else:
        from_dt = datetime(year, 9, 1, tzinfo=timezone.utc)
        to_dt = datetime(year, 12, 1, tzinfo=timezone.utc)

    return {
        "label": f"{season_l} {year}",
        "season": season_l,
        "year": year,
        "from_dt": from_dt.isoformat(),
        "to_dt": to_dt.isoformat(),
    }


def extract_season_buckets(text: str, now: datetime | None = None) -> list[dict]:
    """Extract simple seasonal ranges such as 'last three autumns'."""
    now = now or datetime.now(timezone.utc)
    lower = text.lower()
    match = re.search(r"last\s+(?:(\d+)|two|three|four|five|six)\s+(summer|autumn|winter|spring)s?", lower)
    if match:
        raw_count, season = match.groups()
        word_counts = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        count = int(raw_count) if raw_count and raw_count.isdigit() else word_counts.get(match.group(1), 3)
        return [resolve_season_to_range(season, now.year - i) for i in range(1, count + 1)]

    match = re.search(r"\b(summer|autumn|winter|spring)\s+(20\d{2})\b", lower)
    if match:
        season, year = match.groups()
        return [resolve_season_to_range(season, int(year))]

    return []


# ── Time anchor resolution ───────────────────────────────────────────────────

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_AEST_UTC_OFFSET = 10   # NEM operates in AEST = UTC+10 year-round (no DST adjustment)
_MIN_ARCHIVE_DATE = datetime(2022, 8, 1, tzinfo=timezone.utc)


def resolve_time_anchor(
    from_offset: str | None,
    raw_query: str = "",
    now: datetime | None = None,
) -> datetime | None:
    """Convert a natural-language time offset to an absolute UTC datetime.

    Handles the common patterns the LLM returns in `time_range.from_offset`:
      "N hours/minutes/days/weeks ago", "yesterday [time]",
      "last [weekday] [time]", "this morning", "earlier today", ISO strings.
    All clock times are interpreted as AEST (UTC+10) — the NEM timezone.

    Returns None when the offset cannot be parsed or resolves to a future time.
    Never raises.
    """
    now = now or datetime.now(timezone.utc)

    # Try each candidate source in priority order: from_offset first, then raw_query
    sources = [s for s in (from_offset, raw_query) if s]
    for text in sources:
        result = _try_parse(text.strip().lower(), now)
        if result is not None:
            # Only return if within archive window and at least 15 min in the past
            if result < now - timedelta(minutes=15) and result >= _MIN_ARCHIVE_DATE:
                return result
    return None


def _try_parse(text: str, now: datetime) -> datetime | None:
    # ISO datetime (LLM sometimes provides these directly)
    iso_m = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)", text)
    if iso_m:
        try:
            s = iso_m.group(1).replace(" ", "T")
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # "N hours ago" / "N minutes ago" / "N days ago" / "N weeks ago"
    delta_m = re.search(r"(\d+)\s+(hour|minute|min|day|week)s?\s+ago", text)
    if delta_m:
        n = int(delta_m.group(1))
        unit = delta_m.group(2)
        if unit in ("hour",):
            return now - timedelta(hours=n)
        if unit in ("minute", "min"):
            return now - timedelta(minutes=n)
        if unit in ("day",):
            return now - timedelta(days=n)
        if unit in ("week",):
            return now - timedelta(weeks=n)

    # "last [weekday] [time]" or "last [weekday]"
    last_wd_m = re.search(
        r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
        text,
    )
    if last_wd_m:
        day_name = last_wd_m.group(1)
        base = _last_weekday_utc(day_name, now)
        if last_wd_m.group(2):
            base = _set_aest_time(base, last_wd_m.group(2), last_wd_m.group(3), last_wd_m.group(4))
        return base

    # "yesterday [time]" or "yesterday"
    yest_m = re.search(
        r"yesterday(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?", text
    )
    if yest_m:
        base = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if yest_m.group(1):
            base = _set_aest_time(base, yest_m.group(1), yest_m.group(2), yest_m.group(3))
        return base

    # "this morning" → today 08:00 AEST
    if "this morning" in text:
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return _set_aest_time(base, "8", None, None)

    # "earlier today" or "today at [time]"
    today_m = re.search(
        r"(?:earlier\s+today|today\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)", text
    )
    if today_m:
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_m.group(1):
            base = _set_aest_time(base, today_m.group(1), today_m.group(2), today_m.group(3))
        else:
            base = now - timedelta(hours=2)   # "earlier today" ≈ 2h ago
        return base

    return None


def _last_weekday_utc(day_name: str, now: datetime) -> datetime:
    """Return the most recent past occurrence of day_name at midnight AEST."""
    target = _WEEKDAYS.get(day_name, 0)
    current = now.weekday()
    days_back = (current - target) % 7
    if days_back == 0:
        days_back = 7   # "last" means the previous week, not today
    base_aest_midnight = now - timedelta(days=days_back)
    # Midnight AEST = UTC 14:00 previous day
    return base_aest_midnight.replace(
        hour=(24 - _AEST_UTC_OFFSET) % 24,
        minute=0, second=0, microsecond=0,
    ) - timedelta(days=1)


def _set_aest_time(
    base_utc: datetime, hour_str: str, minute_str: str | None, ampm: str | None
) -> datetime:
    """Set a specific AEST clock time on a base UTC date."""
    hour = int(hour_str)
    minute = int(minute_str or 0)
    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
    # AEST hour → UTC: subtract 10 hours
    utc_hour = hour - _AEST_UTC_OFFSET
    day_offset = 0
    if utc_hour < 0:
        utc_hour += 24
        day_offset = -1
    result = base_utc + timedelta(days=day_offset)
    return result.replace(hour=utc_hour, minute=minute, second=0, microsecond=0)
