"""Add offer_track table for DISPATCHOFFERTRK real-time rebid detection.

Revision: 0007_sprint_aa_bid_offers
Note: generator_units exists since 0001, bid_offers since 0002.

Table added:
  offer_track — DISPATCHOFFERTRK: available in 5-min dispatch reports on NEMWeb
                Shows WHEN a generator's offer was last modified per settlement period.
                Enables real-time rebid detection (no 30-day confidentiality delay).
                Key signal: OFFERDATE changes between dispatch intervals = rebid occurred.

Also adds missing columns to bid_offers: reason text (rebid justification from BIDPEROFFER).
"""
from alembic import op
import sqlalchemy as sa


revision = "0007_sprint_aa_bid_offers"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── offer_track: DISPATCHOFFERTRK ─────────────────────────────────────────
    # Available from NEMWeb DispatchIS reports — no confidentiality delay.
    # Each row: when the energy offer for a DUID was last modified for a given period.
    op.create_table(
        "offer_track",
        sa.Column("id",               sa.String(36),              primary_key=True),
        sa.Column("duid",             sa.String(40),              nullable=False),
        sa.Column("settlement_date",  sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_id",        sa.Integer,                 nullable=False),
        sa.Column("offer_date",       sa.DateTime(timezone=True), nullable=True),   # last offer modification time
        sa.Column("version_no",       sa.Integer,                 nullable=True),
        sa.Column("energy_offer_date",sa.DateTime(timezone=True), nullable=True),
        sa.Column("raise_6s_offer_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("region",           sa.String(10),              nullable=True),
        sa.Column("source",           sa.String(30),              nullable=True,   server_default="DISPATCHOFFERTRK"),
        sa.Column("ingested_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_offer_track_duid_date",   "offer_track", ["duid", "settlement_date"])
    op.create_index("ix_offer_track_region_date", "offer_track", ["region", "settlement_date"])
    op.create_unique_constraint(
        "uq_offer_track_duid_date_period",
        "offer_track",
        ["duid", "settlement_date", "period_id"],
    )

    # ── Extend bid_offers with rebid reason text ───────────────────────────────
    # BIDPEROFFER includes a REASON field explaining the rebid. Currently not stored.
    # Adding here so rebid_engine can surface the reason in NLP answers.
    try:
        op.add_column("bid_offers", sa.Column("reason", sa.Text, nullable=True))
    except Exception:
        pass  # column may already exist in some deployments


def downgrade() -> None:
    op.drop_constraint("uq_offer_track_duid_date_period", "offer_track", type_="unique")
    op.drop_index("ix_offer_track_region_date", table_name="offer_track")
    op.drop_index("ix_offer_track_duid_date", table_name="offer_track")
    op.drop_table("offer_track")
