"""HippoGraph cold-start loader.

Reads the most recent N days of dispatch price rows from market_events and
populates the in-memory MarketStateGraph singleton so analog retrieval works
immediately on server startup rather than waiting 24h for live accumulation.

Design:
- HippoGraph is a rolling 30-day window (max_nodes=8640 default).
- Cold-start seeds it with the same window from historical data.
- Deep-history queries (>30 days) bypass HippoGraph and hit the DB directly
  via retrieve_seasonal_summary() — this script does NOT try to load 2 years.

Usage:
    python -u scripts/hippo_coldstart.py [--days 30] [--region NSW1]

Called by the FastAPI startup hook (app.api.main.lifespan) if
GRIDVERDICT_HIPPOGRAPH_COLDSTART=true.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_coldstart(days: int = 30, region: str | None = None) -> dict:
    from sqlalchemy import select, func
    from app.db.session import db_session, init_db
    from app.db.models import MarketEvent
    from app.engines.hippograph.graph import get_graph

    await init_db()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    graph = get_graph()

    stats: dict = {
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "days_window": days,
        "region_filter": region,
        "nodes_inserted": 0,
        "nodes_skipped": 0,
        "nodes_in_graph": 0,
        "regions": {},
    }

    t0 = time.monotonic()

    async with db_session() as session:
        stmt = (
            select(MarketEvent)
            .where(MarketEvent.valid_time >= cutoff)
            .where(MarketEvent.source == "AEMO_DISPATCH_PRICE")
            .order_by(MarketEvent.valid_time.asc())
        )
        if region:
            stmt = stmt.where(MarketEvent.region == region.upper())

        result = await session.execute(stmt)
        rows = result.scalars().all()

    logger.info("Loaded %d rows from DB (last %d days)", len(rows), days)

    for row in rows:
        row_dict = {
            "id": str(row.id),
            "price_rrp": row.price_rrp,
            "demand_mw": row.demand_mw,
            "availability_mw": row.availability_mw,
            "region": row.region,
            "source": row.source,
            "valid_time": row.valid_time,
            "tenant_id": row.tenant_id,
            "data": row.data or {},
        }
        node = graph.insert_from_dict(row_dict)
        if node is not None:
            stats["nodes_inserted"] += 1
            stats["regions"][row.region] = stats["regions"].get(row.region, 0) + 1
        else:
            stats["nodes_skipped"] += 1

    stats["nodes_in_graph"] = graph.node_count()
    stats["runtime_seconds"] = round(time.monotonic() - t0, 1)

    logger.info(
        "HippoGraph cold-start complete: %d nodes in graph (%d inserted, %d skipped) in %.1fs",
        stats["nodes_in_graph"], stats["nodes_inserted"],
        stats["nodes_skipped"], stats["runtime_seconds"],
    )
    for r, count in sorted(stats["regions"].items()):
        logger.info("  %s: %d nodes", r, count)

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Seed HippoGraph from market_events DB")
    p.add_argument("--days", type=int, default=30, help="How many days of history to load")
    p.add_argument("--region", default=None, help="Restrict to a single NEM region")
    p.add_argument("--output", default="data/hippo_coldstart.json", help="Stats output path")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    stats = asyncio.run(run_coldstart(days=args.days, region=args.region))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"Stats written to {out}")


if __name__ == "__main__":
    main()
