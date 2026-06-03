"""Sprint Z — RooftopSolarInterval: AEMO rooftop PV actual vs forecast.

Stores 30-minute actual and forecast rooftop solar generation per NEM region.
The delta (actual - forecast) is the key signal for price movement:
  positive delta = more solar than expected => price suppressing
  negative delta = less solar than expected => price supporting

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-03
"""
from alembic import op
import sqlalchemy as sa


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rooftop_solar_intervals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("region", sa.String(10), nullable=False),
        sa.Column("interval_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual_mw", sa.Float(), nullable=True,
                  comment="ROOFTOP_PV_ACTUAL_SCADA power_mw"),
        sa.Column("forecast_mw", sa.Float(), nullable=True,
                  comment="ROOFTOP_PV_FORECAST_SCADA power_mw"),
        sa.Column("delta_mw", sa.Float(), nullable=True,
                  comment="actual_mw - forecast_mw; positive = more solar than expected"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_rooftop_region_time", "rooftop_solar_intervals",
                    ["region", "interval_datetime"])
    op.create_unique_constraint("uq_rooftop_region_time", "rooftop_solar_intervals",
                                ["region", "interval_datetime"])


def downgrade() -> None:
    op.drop_table("rooftop_solar_intervals")
