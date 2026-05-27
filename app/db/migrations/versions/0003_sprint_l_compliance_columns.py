"""Sprint L — ISO 42001 / 27001 compliance columns.

Adds model traceability columns to decision_audit_log and ISO 27001
Annex A control reference to observer_events.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-26
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ISO/IEC 42001:2023 §8.4 — AI system documentation (model traceability)
    op.add_column(
        "decision_audit_log",
        sa.Column(
            "model_version",
            sa.String(80),
            nullable=True,
            comment="ISO 42001 §8.4 — model identifier and version that produced this decision",
        ),
    )
    op.add_column(
        "decision_audit_log",
        sa.Column(
            "training_data_ref",
            sa.String(200),
            nullable=True,
            comment="ISO 42001 §8.4 — training data window or provenance reference",
        ),
    )

    # ISO/IEC 27001:2022 Annex A — primary control ref for the highest-risk signal
    op.add_column(
        "observer_events",
        sa.Column(
            "control_ref",
            sa.String(20),
            nullable=True,
            comment="ISO 27001:2022 Annex A control reference (e.g. 'A.8.28') for primary signal",
        ),
    )


def downgrade() -> None:
    op.drop_column("observer_events", "control_ref")
    op.drop_column("decision_audit_log", "training_data_ref")
    op.drop_column("decision_audit_log", "model_version")
