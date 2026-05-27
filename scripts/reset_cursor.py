"""Reset the backfill cursor so acquisition starts fresh."""
import asyncio
from sqlalchemy import text


async def main() -> None:
    from app.db.session import db_session
    async with db_session() as s:
        r = await s.execute(text("SELECT name, last_successful_interval, status FROM backfill_cursors"))
        rows = r.fetchall()
        print("Current cursors:")
        for row in rows:
            print(f"  {row[0]}: last={row[1]}, status={row[2]}")
        await s.execute(text("DELETE FROM backfill_cursors"))
        await s.commit()
        print("Cursors cleared.")


if __name__ == "__main__":
    asyncio.run(main())
