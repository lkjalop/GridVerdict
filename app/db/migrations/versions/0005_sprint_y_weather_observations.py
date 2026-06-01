"""Sprint Y — WeatherObservation: persist BOM/consensus weather snapshots.

Enables live feed enrichment to match weather at historical event valid_times
rather than relying on the live cache (which is wrong for old events).
Upserts on (region, observed_at) so re-polling the same BOM interval is safe.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-01
"""
from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "weather_observations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("region", sa.String(10), nullable=False),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Observation timestamp from BOM/consensus — the time the weather IS FOR",
        ),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("temperature_c", sa.Float, nullable=True),
        sa.Column(
            "temp_deviation_c",
            sa.Float,
            nullable=True,
            comment="Degrees above/below monthly seasonal norm for this region",
        ),
        sa.Column("humidity_pct",     sa.Float, nullable=True),
        sa.Column("wind_speed_kmh",   sa.Float, nullable=True),
        sa.Column("wind_gust_kmh",    sa.Float, nullable=True),
        sa.Column("precipitation_mm", sa.Float, nullable=True),
        sa.Column("cloud_cover_pct",  sa.Float, nullable=True),
        sa.Column(
            "source_count",
            sa.Integer,
            nullable=True,
            comment="Number of sources contributing to consensus",
        ),
        sa.Column("raw_consensus", sa.JSON, nullable=False, server_default="{}"),
    )
    op.create_index(
        "ix_weather_obs_region_time",
        "weather_observations",
        ["region", "observed_at"],
    )
    op.create_unique_constraint(
        "uq_weather_obs_region_time",
        "weather_observations",
        ["region", "observed_at"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_weather_obs_region_time", "weather_observations", type_="unique")
    op.drop_index("ix_weather_obs_region_time", table_name="weather_observations")
    op.drop_table("weather_observations")
