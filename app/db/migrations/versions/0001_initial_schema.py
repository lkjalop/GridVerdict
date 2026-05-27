"""Initial schema — all tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=False), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=False), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=False), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("region", sa.String(10), nullable=False, server_default="NSW1"),
        sa.Column("title", sa.String(200)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_sessions_tenant_id", "sessions", ["tenant_id"])
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "queries",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=False), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("session_id", UUID(as_uuid=False), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("raw_query", sa.Text, nullable=False),
        sa.Column("decomposition", JSONB),
        sa.Column("answer", JSONB),
        sa.Column("trace_id", sa.String(50)),
        sa.Column("intent", sa.String(50)),
        sa.Column("verdict", sa.String(30)),
        sa.Column("region", sa.String(10)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_queries_tenant_id", "queries", ["tenant_id"])
    op.create_index("ix_queries_session_id", "queries", ["session_id"])
    op.create_index("ix_queries_trace_id", "queries", ["trace_id"])

    op.create_table(
        "traces",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=False), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("query_id", sa.String(40), sa.ForeignKey("queries.id")),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_profile", sa.String(30), nullable=False, server_default="cost_optimized"),
        sa.Column("source_manifest", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("tool_calls", JSONB, nullable=False, server_default=sa.text("'[]'")),
        sa.Column("decomposition", JSONB),
        sa.Column("prefill", JSONB),
        sa.Column("answer", JSONB),
        sa.Column("validator_result", JSONB),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_traces_tenant_id", "traces", ["tenant_id"])
    op.create_index("ix_traces_query_id", "traces", ["query_id"])

    op.create_table(
        "market_events",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False, server_default="system"),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("region", sa.String(10)),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price_rrp", sa.Float),
        sa.Column("demand_mw", sa.Float),
        sa.Column("availability_mw", sa.Float),
        sa.Column("data", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("raw_ref", sa.String(200), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_market_events_tenant_id", "market_events", ["tenant_id"])
    op.create_index("ix_market_events_region", "market_events", ["region"])
    op.create_index("ix_market_events_valid_time", "market_events", ["valid_time"])
    op.create_index("ix_market_events_region_valid_time", "market_events", ["region", "valid_time"])
    op.create_unique_constraint(
        "uq_market_events_source_region_valid_time",
        "market_events",
        ["source", "region", "valid_time"],
    )

    op.create_table(
        "market_driver_events",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False, server_default="system"),
        sa.Column("source", sa.String(80), nullable=False),
        sa.Column("driver_type", sa.String(40), nullable=False),
        sa.Column("element_id", sa.String(120), nullable=False),
        sa.Column("region", sa.String(10)),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("values", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("raw_ref", sa.String(200), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_market_driver_type_time", "market_driver_events", ["driver_type", "valid_time"])
    op.create_index("ix_market_driver_region_time", "market_driver_events", ["region", "valid_time"])
    op.create_unique_constraint(
        "uq_market_driver_source_type_element_time",
        "market_driver_events",
        ["source", "driver_type", "element_id", "valid_time"],
    )

    op.create_table(
        "generator_units",
        sa.Column("duid", sa.String(40), primary_key=True),
        sa.Column("station_name", sa.String(160)),
        sa.Column("participant", sa.String(160)),
        sa.Column("region", sa.String(10)),
        sa.Column("fuel_type", sa.String(40)),
        sa.Column("dispatch_type", sa.String(40)),
        sa.Column("max_capacity_mw", sa.Float),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_generator_units_region", "generator_units", ["region"])
    op.create_index("ix_generator_units_fuel_type", "generator_units", ["fuel_type"])

    op.create_table(
        "unit_dispatch_events",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False, server_default="system"),
        sa.Column("source", sa.String(80), nullable=False),
        sa.Column("duid", sa.String(40), nullable=False),
        sa.Column("station_name", sa.String(160)),
        sa.Column("participant", sa.String(160)),
        sa.Column("region", sa.String(10)),
        sa.Column("fuel_type", sa.String(40)),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("initial_mw", sa.Float),
        sa.Column("total_cleared_mw", sa.Float),
        sa.Column("availability_mw", sa.Float),
        sa.Column("target_mw", sa.Float),
        sa.Column("ramp_rate", sa.Float),
        sa.Column("semi_dispatch_cap", sa.Float),
        sa.Column("data", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("raw_ref", sa.String(200), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_unit_dispatch_tenant_id", "unit_dispatch_events", ["tenant_id"])
    op.create_index("ix_unit_dispatch_region", "unit_dispatch_events", ["region"])
    op.create_index("ix_unit_dispatch_fuel_type", "unit_dispatch_events", ["fuel_type"])
    op.create_index("ix_unit_dispatch_valid_time", "unit_dispatch_events", ["valid_time"])
    op.create_index("ix_unit_dispatch_region_time", "unit_dispatch_events", ["region", "valid_time"])
    op.create_index("ix_unit_dispatch_fuel_time", "unit_dispatch_events", ["fuel_type", "valid_time"])
    op.create_index("ix_unit_dispatch_duid_time", "unit_dispatch_events", ["duid", "valid_time"])
    op.create_unique_constraint(
        "uq_unit_dispatch_source_duid_time",
        "unit_dispatch_events",
        ["source", "duid", "valid_time"],
    )

    op.create_table(
        "backfill_cursors",
        sa.Column("name", sa.String(80), primary_key=True),
        sa.Column("source", sa.String(80), nullable=False),
        sa.Column("last_successful_interval", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(30), nullable=False, server_default="idle"),
        sa.Column("error", sa.Text),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "observer_events",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False),
        sa.Column("query_id", sa.String(40)),
        sa.Column("trace_id", sa.String(50)),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("relevance_class", sa.String(30)),
        sa.Column("risk_score", sa.Integer),
        sa.Column("risk_band", sa.String(10)),
        sa.Column("verdict", sa.String(20)),
        sa.Column("signals", JSONB, nullable=False, server_default=sa.text("'[]'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_observer_events_tenant_id", "observer_events", ["tenant_id"])
    op.create_index("ix_observer_events_query_id", "observer_events", ["query_id"])

    # Seed the default local tenant
    op.execute(
        sa.text(
            "INSERT INTO tenants (id, name) VALUES "
            "('00000000-0000-0000-0000-000000000001', 'local')"
        )
    )


def downgrade() -> None:
    op.drop_table("observer_events")
    op.drop_table("backfill_cursors")
    op.drop_table("unit_dispatch_events")
    op.drop_table("generator_units")
    op.drop_table("market_driver_events")
    op.drop_table("market_events")
    op.drop_table("traces")
    op.drop_table("queries")
    op.drop_table("sessions")
    op.drop_table("users")
    op.drop_table("tenants")
