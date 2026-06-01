"""SQLAlchemy ORM models — every table has tenant_id from day one."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship(back_populates="tenant")
    sessions: Mapped[list["Session"]] = relationship(back_populates="tenant")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tenant: Mapped["Tenant"] = relationship(back_populates="users")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    region: Mapped[str] = mapped_column(String(10), nullable=False, default="NSW1")
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    tenant: Mapped["Tenant"] = relationship(back_populates="sessions")
    user: Mapped["User"] = relationship(back_populates="sessions")
    queries: Mapped[list["Query"]] = relationship(back_populates="session")


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)   # qry-{uuid}
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    raw_query: Mapped[str] = mapped_column(Text, nullable=False)
    decomposition: Mapped[dict | None] = mapped_column(JSON)
    answer: Mapped[dict | None] = mapped_column(JSON)
    trace_id: Mapped[str | None] = mapped_column(String(50), index=True)
    intent: Mapped[str | None] = mapped_column(String(50))
    verdict: Mapped[str | None] = mapped_column(String(30))
    region: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="queries")


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)   # trace-{uuid}
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    query_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id"), index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_profile: Mapped[str] = mapped_column(String(30), default="cost_optimized")
    source_manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    decomposition: Mapped[dict | None] = mapped_column(JSON)
    prefill: Mapped[dict | None] = mapped_column(JSON)
    answer: Mapped[dict | None] = mapped_column(JSON)
    validator_result: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MarketEvent(Base):
    """Ingested market data — persisted for archive queries and analog retrieval."""
    __tablename__ = "market_events"
    __table_args__ = (
        Index("ix_market_events_region_valid_time", "region", "valid_time"),
        UniqueConstraint("source", "region", "valid_time", name="uq_market_events_source_region_valid_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True, default="system")
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str | None] = mapped_column(String(10), index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price_rrp: Mapped[float | None] = mapped_column(Float)
    demand_mw: Mapped[float | None] = mapped_column(Float)
    availability_mw: Mapped[float | None] = mapped_column(Float)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MarketDriverEvent(Base):
    """Historical non-price market drivers such as interconnector and constraint rows."""
    __tablename__ = "market_driver_events"
    __table_args__ = (
        Index("ix_market_driver_type_time", "driver_type", "valid_time"),
        Index("ix_market_driver_region_time", "region", "valid_time"),
        UniqueConstraint(
            "source",
            "driver_type",
            "element_id",
            "valid_time",
            name="uq_market_driver_source_type_element_time",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True, default="system")
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    driver_type: Mapped[str] = mapped_column(String(40), nullable=False)
    element_id: Mapped[str] = mapped_column(String(120), nullable=False)
    region: Mapped[str | None] = mapped_column(String(10), index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    values: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DecisionAuditLog(Base):
    """Immutable record of every BESS dispatch recommendation (simulation only).

    Written once per decision; never updated. Used for regulatory export and
    post-hoc analysis. simulation_only=True is hardcoded — this system never
    executes real market actions.
    """
    __tablename__ = "decision_audit_log"
    __table_args__ = (
        Index("ix_dal_tenant_created", "tenant_id", "created_at"),
        Index("ix_dal_region_created", "region", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(50), index=True)

    decision_type: Mapped[str] = mapped_column(
        String(40), nullable=False,
        comment="bess_dispatch | fleet_dispatch",
    )
    region: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    asset_id: Mapped[str | None] = mapped_column(String(100))

    action: Mapped[str] = mapped_column(String(40), nullable=False)
    confidence: Mapped[str] = mapped_column(String(30), nullable=False)
    price_rrp: Mapped[float | None] = mapped_column(Float)
    price_regime: Mapped[str | None] = mapped_column(String(20))

    economics: Mapped[dict | None] = mapped_column(JSON)
    evidence_quality: Mapped[str | None] = mapped_column(String(30))
    risk_flags: Mapped[list | None] = mapped_column(JSON)
    why_summary: Mapped[list | None] = mapped_column(JSON)

    simulation_only: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # ISO/IEC 42001:2023 §8.4 — AI system documentation (model traceability)
    model_version: Mapped[str | None] = mapped_column(
        String(80),
        comment="ISO 42001 §8.4 — model identifier and version that produced this decision",
    )
    training_data_ref: Mapped[str | None] = mapped_column(
        String(200),
        comment="ISO 42001 §8.4 — training data window or provenance reference",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class GeneratorUnit(Base):
    """Unit metadata used to map DUID-level dispatch into technology evidence."""
    __tablename__ = "generator_units"

    duid: Mapped[str] = mapped_column(String(40), primary_key=True)
    station_name: Mapped[str | None] = mapped_column(String(160))
    participant: Mapped[str | None] = mapped_column(String(160))
    region: Mapped[str | None] = mapped_column(String(10), index=True)
    fuel_type: Mapped[str | None] = mapped_column(String(40), index=True)
    dispatch_type: Mapped[str | None] = mapped_column(String(40))
    max_capacity_mw: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UnitDispatchEvent(Base):
    """DUID-level dispatch evidence from DISPATCH_UNIT_SOLUTION/DISPATCHLOAD."""
    __tablename__ = "unit_dispatch_events"
    __table_args__ = (
        Index("ix_unit_dispatch_region_time", "region", "valid_time"),
        Index("ix_unit_dispatch_fuel_time", "fuel_type", "valid_time"),
        Index("ix_unit_dispatch_duid_time", "duid", "valid_time"),
        UniqueConstraint("source", "duid", "valid_time", name="uq_unit_dispatch_source_duid_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True, default="system")
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    duid: Mapped[str] = mapped_column(String(40), nullable=False)
    station_name: Mapped[str | None] = mapped_column(String(160))
    participant: Mapped[str | None] = mapped_column(String(160))
    region: Mapped[str | None] = mapped_column(String(10), index=True)
    fuel_type: Mapped[str | None] = mapped_column(String(40), index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_mw: Mapped[float | None] = mapped_column(Float)
    total_cleared_mw: Mapped[float | None] = mapped_column(Float)
    availability_mw: Mapped[float | None] = mapped_column(Float)
    target_mw: Mapped[float | None] = mapped_column(Float)
    ramp_rate: Mapped[float | None] = mapped_column(Float)
    semi_dispatch_cap: Mapped[float | None] = mapped_column(Float)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BackfillCursor(Base):
    """Progress cursor for idempotent archive backfill jobs.

    last_successful_interval is set to the first datetime of the last month that
    completed all table files without error.  On resume, any month whose start
    datetime is <= this value is skipped entirely — the upsert layer protects
    against duplicates should a partial month ever need to be re-run manually.
    """
    __tablename__ = "backfill_cursors"

    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    last_successful_interval: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="idle")
    error: Mapped[str | None] = mapped_column(Text)
    files_completed: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class BidOffer(Base):
    """DUID-level bid/offer data from BIDDAYOFFER and BIDPEROFFER MMSDM tables.

    BIDDAYOFFER: day-ahead offer submitted before the trading day (period_id=None).
    BIDPEROFFER: intraday rebid for a specific 30-min period (period_id 1-48).
    Comparing BIDDAYOFFER vs subsequent BIDPEROFFER rows reveals strategic rebidding
    patterns that explain price spikes.
    """
    __tablename__ = "bid_offers"
    __table_args__ = (
        Index("ix_bid_offer_duid_date", "duid", "settlement_date"),
        Index("ix_bid_offer_region_date", "region", "settlement_date"),
        UniqueConstraint(
            "source", "duid", "bid_type", "settlement_date", "period_id",
            name="uq_bid_offer_source_duid_type_date_period",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True, default="system")
    source: Mapped[str] = mapped_column(String(40), nullable=False)   # BIDDAYOFFER | BIDPEROFFER
    duid: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    region: Mapped[str | None] = mapped_column(String(10), index=True)
    bid_type: Mapped[str] = mapped_column(String(20), nullable=False)  # ENERGY, L5RE, etc.
    settlement_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_id: Mapped[int | None] = mapped_column(Integer)   # None for BIDDAYOFFER; 1-48 for BIDPEROFFER
    offer_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    max_avail_mw: Mapped[float | None] = mapped_column(Float)
    minimum_load_mw: Mapped[float | None] = mapped_column(Float)
    ramp_up_mw_per_min: Mapped[float | None] = mapped_column(Float)
    ramp_down_mw_per_min: Mapped[float | None] = mapped_column(Float)
    price_bands: Mapped[dict] = mapped_column(JSON, default=dict)   # {1: $/MWh, 2: $/MWh, ...}
    avail_bands: Mapped[dict] = mapped_column(JSON, default=dict)   # {1: MW, 2: MW, ...}
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    raw_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FcasPriceEvent(Base):
    """FCAS ancillary service prices extracted from MMSDM DISPATCHPRICE.

    Stores all 8 FCAS service prices per region per dispatch interval.
    Used to detect tight FCAS markets and attribute price spikes to
    contingency reserve exhaustion.
    """
    __tablename__ = "fcas_price_events"
    __table_args__ = (
        Index("ix_fcas_price_region_time", "region", "valid_time"),
        UniqueConstraint("region", "valid_time", name="uq_fcas_price_region_valid_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True, default="system")
    source: Mapped[str] = mapped_column(String(80), nullable=False)
    region: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raise_6sec_rrp: Mapped[float | None] = mapped_column(Float)
    raise_60sec_rrp: Mapped[float | None] = mapped_column(Float)
    raise_5min_rrp: Mapped[float | None] = mapped_column(Float)
    raise_reg_rrp: Mapped[float | None] = mapped_column(Float)
    lower_6sec_rrp: Mapped[float | None] = mapped_column(Float)
    lower_60sec_rrp: Mapped[float | None] = mapped_column(Float)
    lower_5min_rrp: Mapped[float | None] = mapped_column(Float)
    lower_reg_rrp: Mapped[float | None] = mapped_column(Float)
    raw_ref: Mapped[str] = mapped_column(String(200), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CommentaryEvent(Base):
    """Auto-generated market commentary event — one per material market change.

    Produced by the CommentaryEngine when the scheduler detects a material
    change in dispatch price, headroom, regime, notices, or forecast risk.
    Stores the full Why Engine output so the event is self-contained and
    queryable for RAG retrieval.
    """
    __tablename__ = "commentary_events"
    __table_args__ = (
        Index("ix_commentary_region_valid_time", "region", "valid_time"),
        Index("ix_commentary_event_type", "event_type"),
        Index("ix_commentary_severity", "severity"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    region: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    system_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    contributing_factors: Mapped[list] = mapped_column(JSON, default=list)
    missing_data: Mapped[list] = mapped_column(JSON, default=list)
    evidence_refs: Mapped[list] = mapped_column(JSON, default=list)
    claim_map: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    corroborations: Mapped[dict] = mapped_column(JSON, default=dict)
    next_watch: Mapped[list] = mapped_column(JSON, default=list)
    counterargument: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    snapshot_after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class ObserverEvent(Base):
    __tablename__ = "observer_events"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)   # obs-evt-{uuid}
    tenant_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    query_id: Mapped[str | None] = mapped_column(String(40), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(50))
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    relevance_class: Mapped[str | None] = mapped_column(String(30))
    risk_score: Mapped[int | None] = mapped_column(Integer)
    risk_band: Mapped[str | None] = mapped_column(String(10))
    verdict: Mapped[str | None] = mapped_column(String(20))
    signals: Mapped[list] = mapped_column(JSON, default=list)
    # ISO/IEC 27001:2022 Annex A — primary control ref for the highest-risk signal in this event
    control_ref: Mapped[str | None] = mapped_column(
        String(20),
        comment="ISO 27001:2022 Annex A control reference (e.g. 'A.8.28') for primary signal",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WeatherObservation(Base):
    """Persisted weather consensus snapshot per NEM region, written every poll cycle.

    Allows live feed enrichment to JOIN weather at a historical event's valid_time
    rather than only the live consensus (which would be wrong for old events).
    Upserts on (region, observed_at) so re-polling the same BOM interval is safe.
    """
    __tablename__ = "weather_observations"
    __table_args__ = (
        Index("ix_weather_obs_region_time", "region", "observed_at"),
        UniqueConstraint("region", "observed_at", name="uq_weather_obs_region_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    region: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
        comment="Observation timestamp from BOM/consensus — the time the weather IS FOR",
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    temperature_c: Mapped[float | None] = mapped_column(Float)
    temp_deviation_c: Mapped[float | None] = mapped_column(
        Float, comment="Degrees above/below monthly seasonal norm for this region"
    )
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float)
    wind_gust_kmh: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)
    source_count: Mapped[int | None] = mapped_column(Integer, comment="Number of sources contributing to consensus")
    raw_consensus: Mapped[dict] = mapped_column(JSON, default=dict)
