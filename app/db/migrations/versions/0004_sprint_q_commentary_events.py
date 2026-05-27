"""Sprint Q — Rolling Market Commentary: commentary_events table.

Each row is one auto-generated Why Engine analysis triggered by a material
market change (price spike, headroom tightened, new AEMO notice, etc.).
Stores the full structured output for both SSE push and RAG retrieval.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-26
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "commentary_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("region", sa.String(10), nullable=False),
        sa.Column("valid_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("system_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("headline", sa.Text, nullable=False),
        sa.Column("contributing_factors", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("missing_data", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("evidence_refs", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("claim_map", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("corroborations", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("next_watch", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("counterargument", sa.Text, nullable=True),
        sa.Column("snapshot_before", sa.JSON, nullable=True),
        sa.Column("snapshot_after", sa.JSON, nullable=True),
        sa.Column("trace_id", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_commentary_region_valid_time", "commentary_events", ["region", "valid_time"])
    op.create_index("ix_commentary_event_type", "commentary_events", ["event_type"])
    op.create_index("ix_commentary_severity", "commentary_events", ["severity"])
    op.create_index("ix_commentary_created_at", "commentary_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_commentary_created_at", table_name="commentary_events")
    op.drop_index("ix_commentary_severity", table_name="commentary_events")
    op.drop_index("ix_commentary_event_type", table_name="commentary_events")
    op.drop_index("ix_commentary_region_valid_time", table_name="commentary_events")
    op.drop_table("commentary_events")
