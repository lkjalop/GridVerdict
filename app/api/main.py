"""FastAPI application factory — wires all routes, CORS, lifespan."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import routes_auth, routes_backtest, routes_commentary, routes_compliance, routes_constraints, routes_events, routes_health, routes_incidents, routes_market, routes_metrics, routes_models, routes_portfolio, routes_query, routes_rebid, routes_security, routes_sessions, routes_temporalrag, routes_trace
from app.data.aemo_live_client import get_aemo_client
from app.data.cache import get_cache
from app.data.scheduler import start_scheduler, stop_scheduler
from config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

logging.basicConfig(level=_settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GridVerdict starting up — dev_no_auth=%s", _settings.gridverdict_dev_no_auth)
    # Auto-create tables for SQLite (dev / test) — Alembic handles Postgres
    if _settings.database_url.startswith("sqlite"):
        from app.db.session import init_db
        await init_db()
    elif _settings.gridverdict_dev_no_auth:
        from app.db.session import ensure_runtime_schema_compat
        await ensure_runtime_schema_compat()
    # Warm the AEMO cache on startup so first request is fast
    try:
        client = get_aemo_client()
        cache = get_cache()
        snapshot = await client.fetch_latest_snapshot()
        await cache.set("dispatch_snapshot", snapshot.to_dict())
        logger.info("AEMO snapshot warmed: %d regions", len(snapshot.regions))
    except Exception as exc:
        logger.warning("Startup AEMO fetch failed (will retry on demand): %s", exc)
        try:
            cache = get_cache()
            warmed = await _warm_dispatch_snapshot_from_db(cache)
            if warmed:
                logger.info("AEMO snapshot warmed from persisted DB fallback: %d regions", warmed)
        except Exception as db_exc:
            logger.warning("Persisted dispatch fallback failed: %s", db_exc)
    # Rebuild HippoGraph from DB (last 30 days of dispatch prices)
    try:
        from app.engines.hippograph.graph import rebuild_from_db
        n = await rebuild_from_db(lookback_days=30)
        logger.info("HippoGraph rebuilt from DB: %d nodes", n)
    except Exception as exc:
        logger.warning("HippoGraph rebuild failed (cold start, no analogs yet): %s", exc)

    # Start background scheduler (dispatch + notices + archive backfill)
    await start_scheduler()

    # Start Redis event bus listener (no-op when Redis is not configured)
    from app.data.event_bus import start_redis_listener
    await start_redis_listener()

    yield

    # Shutdown
    from app.data.event_bus import stop_redis_listener
    from app.data.redis_client import close_redis
    await stop_redis_listener()
    await stop_scheduler()
    await close_redis()
    client = get_aemo_client()
    await client.close()
    logger.info("GridVerdict shut down cleanly")


def create_app() -> FastAPI:
    app = FastAPI(
        title="GridVerdict",
        description="Evidence-grounded NEM decision-support cockpit",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if _settings.gridverdict_dev_no_auth else ["http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.middleware import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware)

    # API routes under /api prefix
    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_auth.router, prefix="/api")
    app.include_router(routes_market.router, prefix="/api")
    app.include_router(routes_sessions.router, prefix="/api")
    app.include_router(routes_query.router, prefix="/api")
    app.include_router(routes_trace.router, prefix="/api")
    app.include_router(routes_backtest.router, prefix="/api")
    app.include_router(routes_constraints.router, prefix="/api")
    app.include_router(routes_models.router, prefix="/api")
    app.include_router(routes_rebid.router, prefix="/api")
    app.include_router(routes_security.router, prefix="/api")
    app.include_router(routes_events.router, prefix="/api")
    app.include_router(routes_temporalrag.router, prefix="/api")
    app.include_router(routes_incidents.router, prefix="/api")
    app.include_router(routes_portfolio.router, prefix="/api")
    app.include_router(routes_compliance.router, prefix="/api")
    app.include_router(routes_commentary.router, prefix="/api")
    app.include_router(routes_metrics.router, prefix="/api")

    @app.get("/api/tools", include_in_schema=True, tags=["mcp"])
    async def list_tools():
        """Return registered MCP tools from config/tools.yaml."""
        from app.mcp.router import get_all_tools
        return get_all_tools()

    # Serve frontend static files
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static")
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_app():
        index = os.path.join(frontend_dir, "app.html")
        if os.path.isfile(index):
            return FileResponse(index)
        return {"message": "GridVerdict API", "docs": "/api/docs"}

    return app


app = create_app()


async def _warm_dispatch_snapshot_from_db(cache) -> int:
    """Use latest persisted real AEMO rows when live NEMWeb is unavailable."""
    from sqlalchemy import select
    from app.data.aemo_live_client import DispatchPrice, LiveMarketSnapshot
    from app.db.models import MarketEvent
    from app.db.session import db_session
    from datetime import datetime, timezone

    regions = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
    prices = {}
    async with db_session() as session:
        for region in regions:
            row = (await session.execute(
                select(MarketEvent)
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                )
                .order_by(MarketEvent.valid_time.desc())
                .limit(1)
            )).scalar_one_or_none()
            if row is None:
                continue
            prices[region] = DispatchPrice(
                region=region,
                valid_time=row.valid_time,
                system_time=row.system_time,
                price_rrp=float(row.price_rrp or 0.0),
                demand_mw=float(row.demand_mw or 0.0),
                availability_mw=float(row.availability_mw or 0.0),
                raw_ref=row.raw_ref,
            )

    if not prices:
        return 0

    interval = max(dp.valid_time for dp in prices.values())
    snapshot = LiveMarketSnapshot(
        interval=interval,
        fetched_at=datetime.now(timezone.utc),
        regions=prices,
        raw_ref="persisted_db_fallback",
    )
    await cache.set("dispatch_snapshot", snapshot.to_dict())
    return len(prices)
