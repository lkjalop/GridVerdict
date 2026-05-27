"""Quick DB row count check."""
import asyncio
from sqlalchemy import text


async def main() -> None:
    from app.db.session import db_session
    tables = ("market_events", "market_driver_events", "generator_units", "backfill_cursors")
    async with db_session() as s:
        for tbl in tables:
            r = await s.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
            print(f"  {tbl}: {r.scalar():,} rows")


if __name__ == "__main__":
    asyncio.run(main())
