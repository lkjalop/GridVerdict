"""Calendar utilities for deterministic NEM seasonal query handling."""
from __future__ import annotations

import re
from datetime import datetime, timezone

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
