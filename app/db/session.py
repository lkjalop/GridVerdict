"""Async SQLAlchemy engine, session factory, and base dependency."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import inspect, text

from config.settings import get_settings

_settings = get_settings()

_is_sqlite = _settings.database_url.startswith("sqlite")
_engine_kwargs: dict = {"echo": _settings.db_echo}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["poolclass"] = StaticPool
else:
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(_settings.database_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session and commits on clean exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Use outside FastAPI (background tasks, CLI, Alembic seed scripts)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all ORM tables if they do not yet exist.

    Safe to call multiple times (CREATE TABLE IF NOT EXISTS semantics).
    Used by CLI scripts that run outside the Alembic migration chain.
    """
    from app.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_runtime_schema_compat() -> None:
    """Non-destructive local/dev schema repair for older demo databases.

    Alembic is still the production migration path. This helper exists so a
    Docker/Postgres database with valuable historical market rows but no
    alembic_version table can still run the current dev app without dropping
    or recreating anything. It only creates missing tables and adds nullable
    or defaulted columns expected by the runtime.
    """
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        dialect = conn.dialect.name

        def _has_column(sync_conn, table: str, column: str) -> bool:
            try:
                cols = inspect(sync_conn).get_columns(table)
            except Exception:
                return False
            return any(c["name"] == column for c in cols)

        async def add_column_if_missing(table: str, column: str, ddl: str) -> None:
            exists = await conn.run_sync(lambda sc: _has_column(sc, table, column))
            if exists:
                return
            if dialect == "postgresql":
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {ddl}"))
            else:
                try:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
                except Exception:
                    # SQLite has no IF NOT EXISTS for ADD COLUMN on older builds.
                    pass

        tz_type = "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "DATETIME"
        json_type = "JSONB" if dialect == "postgresql" else "JSON"

        await add_column_if_missing("sessions", "deleted_at", f"deleted_at {tz_type} NULL")
        await add_column_if_missing("observer_events", "control_ref", "control_ref VARCHAR(20) NULL")
        await add_column_if_missing("decision_audit_log", "model_version", "model_version VARCHAR(80) NULL")
        await add_column_if_missing("decision_audit_log", "training_data_ref", "training_data_ref VARCHAR(200) NULL")
        await add_column_if_missing("backfill_cursors", "files_completed", "files_completed INTEGER NOT NULL DEFAULT 0")
        await add_column_if_missing("backfill_cursors", "files_failed", "files_failed INTEGER NOT NULL DEFAULT 0")
        await add_column_if_missing("commentary_events", "claim_map", f"claim_map {json_type} NULL")

        if dialect == "postgresql":
            await conn.execute(text(
                "CREATE TABLE IF NOT EXISTS alembic_version "
                "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
            ))
            await conn.execute(text(
                "INSERT INTO alembic_version(version_num) VALUES ('0004') "
                "ON CONFLICT (version_num) DO NOTHING"
            ))
