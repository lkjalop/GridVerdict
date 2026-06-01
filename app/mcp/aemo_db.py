"""AEMO archive — database upsert and cursor management layer.

Extracted from aemo_archive.py to keep that file focused on orchestration.
All functions here are async and depend on SQLAlchemy sessions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

_BULK_BATCH = 2000        # rows per INSERT statement
# asyncpg hard limit: 32767 placeholders per statement.
# BidOffer has 17 columns → max 1927 rows per batch.
_BID_BULK_BATCH = 32767 // 17  # 1927


# ── Interval gap detection ────────────────────────────────────────────

async def _fetch_existing_intervals(session) -> set[datetime]:
    """Return the set of valid_times already in market_events."""
    from sqlalchemy import select
    from app.db.models import MarketEvent
    result = await session.execute(
        select(MarketEvent.valid_time).where(
            MarketEvent.source == "AEMO_DISPATCH_PRICE"
        )
    )
    return {row[0] for row in result.fetchall()}


def _identify_gaps(existing: set[datetime], cutoff: datetime) -> list[datetime]:
    """Return 5-minute intervals in the last 24h absent from the DB."""
    now = datetime.now(timezone.utc)
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


# ── Bulk INSERT helpers ───────────────────────────────────────────────

def _bulk_insert_stmt(model, rows: list[dict[str, Any]], on_conflict: str = "nothing"):
    """Build a dialect-aware bulk INSERT statement.

    PostgreSQL: INSERT ... ON CONFLICT DO NOTHING / DO UPDATE
    SQLite:     INSERT OR IGNORE ...
    """
    from sqlalchemy import inspect as sa_inspect
    dialect = sa_inspect(model).mapper.persist_selectable.bind
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
        return stmt


async def _upsert_rows(session, rows: list[dict[str, Any]]) -> None:
    """Bulk-insert market_event rows, skipping duplicates."""
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
    """Bulk-insert unit dispatch event rows, skipping duplicates."""
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


# ── Backfill cursor management ────────────────────────────────────────

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

    Called after each successful file within an incomplete month so interrupted
    runs can skip already-done files on resume. Cleared by _update_cursor(status="ok")
    when the month completes.
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
