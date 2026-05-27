"""Add tables and columns from Sprints B–K.

Sprints B-K added: decision_audit_log, bid_offers, fcas_price_events,
sessions.deleted_at, backfill_cursors.files_completed/files_failed.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-26
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── sessions.deleted_at (soft-delete support) ─────────────────────────────
    op.add_column(
        "sessions",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── backfill_cursors: progress counters ───────────────────────────────────
    op.add_column(
        "backfill_cursors",
        sa.Column("files_completed", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "backfill_cursors",
        sa.Column("files_failed", sa.Integer, nullable=False, server_default="0"),
    )

    # ── decision_audit_log ────────────────────────────────────────────────────
    op.create_table(
        "decision_audit_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False),
        sa.Column("user_id", sa.String(36)),
        sa.Column("trace_id", sa.String(50)),
        sa.Column("decision_type", sa.String(40), nullable=False,
                  comment="bess_dispatch | fleet_dispatch"),
        sa.Column("region", sa.String(10), nullable=False),
        sa.Column("asset_id", sa.String(100)),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("confidence", sa.String(30), nullable=False),
        sa.Column("price_rrp", sa.Float),
        sa.Column("price_regime", sa.String(20)),
        sa.Column("economics", JSONB),
        sa.Column("evidence_quality", sa.String(30)),
        sa.Column("risk_flags", JSONB),
        sa.Column("why_summary", JSONB),
        sa.Column("simulation_only", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_dal_tenant_id", "decision_audit_log", ["tenant_id"])
    op.create_index("ix_dal_tenant_created", "decision_audit_log", ["tenant_id", "created_at"])
    op.create_index("ix_dal_region_created", "decision_audit_log", ["region", "created_at"])
    op.create_index("ix_dal_user_id", "decision_audit_log", ["user_id"])
    op.create_index("ix_dal_trace_id", "decision_audit_log", ["trace_id"])

    # ── bid_offers ────────────────────────────────────────────────────────────
    op.create_table(
        "bid_offers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False, server_default="system"),
        sa.Column("source", sa.String(40), nullable=False),
        sa.Column("duid", sa.String(40), nullable=False),
        sa.Column("region", sa.String(10)),
        sa.Column("bid_type", sa.String(20), nullable=False),
        sa.Column("settlement_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_id", sa.Integer),
        sa.Column("offer_date", sa.DateTime(timezone=True)),
        sa.Column("max_avail_mw", sa.Float),
        sa.Column("minimum_load_mw", sa.Float),
        sa.Column("ramp_up_mw_per_min", sa.Float),
        sa.Column("ramp_down_mw_per_min", sa.Float),
        sa.Column("price_bands", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("avail_bands", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("data", JSONB, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("raw_ref", sa.String(200), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_bid_offer_tenant_id", "bid_offers", ["tenant_id"])
    op.create_index("ix_bid_offer_duid", "bid_offers", ["duid"])
    op.create_index("ix_bid_offer_region", "bid_offers", ["region"])
    op.create_index("ix_bid_offer_duid_date", "bid_offers", ["duid", "settlement_date"])
    op.create_index("ix_bid_offer_region_date", "bid_offers", ["region", "settlement_date"])
    op.create_unique_constraint(
        "uq_bid_offer_source_duid_type_date_period",
        "bid_offers",
        ["source", "duid", "bid_type", "settlement_date", "period_id"],
    )

    # ── fcas_price_events ─────────────────────────────────────────────────────
    op.create_table(
        "fcas_price_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(40), nullable=False, server_default="system"),
        sa.Column("source", sa.String(80), nullable=False),
        sa.Column("region", sa.String(10), nullable=False),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raise_6sec_rrp", sa.Float),
        sa.Column("raise_60sec_rrp", sa.Float),
        sa.Column("raise_5min_rrp", sa.Float),
        sa.Column("raise_reg_rrp", sa.Float),
        sa.Column("lower_6sec_rrp", sa.Float),
        sa.Column("lower_60sec_rrp", sa.Float),
        sa.Column("lower_5min_rrp", sa.Float),
        sa.Column("lower_reg_rrp", sa.Float),
        sa.Column("raw_ref", sa.String(200), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_fcas_price_tenant_id", "fcas_price_events", ["tenant_id"])
    op.create_index("ix_fcas_price_region", "fcas_price_events", ["region"])
    op.create_index("ix_fcas_price_valid_time", "fcas_price_events", ["valid_time"])
    op.create_index("ix_fcas_price_region_time", "fcas_price_events", ["region", "valid_time"])
    op.create_unique_constraint(
        "uq_fcas_price_region_valid_time",
        "fcas_price_events",
        ["region", "valid_time"],
    )


def downgrade() -> None:
    op.drop_table("fcas_price_events")
    op.drop_table("bid_offers")
    op.drop_table("decision_audit_log")
    op.drop_column("backfill_cursors", "files_failed")
    op.drop_column("backfill_cursors", "files_completed")
    op.drop_column("sessions", "deleted_at")
