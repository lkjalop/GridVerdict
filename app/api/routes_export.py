"""Export routes — OData feed (PowerBI), CSV, and daily report.

Endpoints:
  GET /export/odata/market_events   — OData v4 feed for PowerBI / Tableau
  GET /export/odata/fcas_prices     — FCAS ancillary service prices (OData)
  GET /export/csv/market_events     — Flat CSV download for Excel / Pandas
  GET /export/report/daily          — HTML daily market report (print-ready PDF)

PowerBI connection:
  Get Data → OData feed → https://yourhost/api/export/odata/market_events
  PowerBI will honour $top/$skip/$filter/$orderby automatically.

Tableau connection:
  Web Data Connector or JSON file import (Tableau 2023+).
  Use the CSV endpoint for broadest compatibility.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from fastapi.responses import Response, HTMLResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])

_ODATA_CONTEXT_BASE = "/api/export/odata"
_SUPPORTED_REGIONS = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
_PAGE_SIZE = 1000   # default OData page size


# ── OData v4 helper ───────────────────────────────────────────────────

def _odata_response(entity: str, rows: list[dict], next_skip: int | None = None, host: str = "") -> dict:
    payload: dict = {
        "@odata.context": f"{host}{_ODATA_CONTEXT_BASE}/$metadata#{entity}",
        "value": rows,
    }
    if next_skip is not None:
        payload["@odata.nextLink"] = (
            f"{host}{_ODATA_CONTEXT_BASE}/{entity}?$skip={next_skip}&$top={_PAGE_SIZE}"
        )
    return payload


# ── OData — market_events ─────────────────────────────────────────────

@router.get("/odata/market_events")
async def odata_market_events(
    region: str | None = Query(default=None, description="NEM region (NSW1/VIC1/QLD1/SA1/TAS1)"),
    start: str | None = Query(default=None, description="ISO date start (e.g. 2026-01-01)"),
    end: str | None = Query(default=None, description="ISO date end (e.g. 2026-06-01)"),
    top: int = Query(default=_PAGE_SIZE, ge=1, le=10000, alias="$top"),
    skip: int = Query(default=0, ge=0, alias="$skip"),
    orderby: str = Query(default="valid_time desc", alias="$orderby"),
):
    """OData v4 market price feed — connect PowerBI via 'Get Data → OData'.

    Returns tidy rows: valid_time, region, price_rrp, demand_mw,
    availability_mw, headroom_mw, raw_ref.
    """
    try:
        from sqlalchemy import select, desc, asc
        from app.db.session import db_session
        from app.db.models import MarketEvent

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=30))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)

        order_col = MarketEvent.valid_time
        order_fn = desc if "desc" in orderby.lower() else asc

        async with db_session() as session:
            q = (
                select(MarketEvent)
                .where(MarketEvent.source == "AEMO_DISPATCH_PRICE")
                .where(MarketEvent.valid_time >= start_dt)
                .where(MarketEvent.valid_time <= end_dt)
                .order_by(order_fn(order_col))
                .offset(skip)
                .limit(top + 1)   # fetch one extra to detect next page
            )
            if region and region.upper() in _SUPPORTED_REGIONS:
                q = q.where(MarketEvent.region == region.upper())

            result = await session.execute(q)
            rows_raw = result.scalars().all()

        has_next = len(rows_raw) > top
        rows_raw = rows_raw[:top]

        rows = [
            {
                "valid_time":      r.valid_time.isoformat(),
                "region":          r.region,
                "price_rrp":       round(float(r.price_rrp or 0), 4),
                "demand_mw":       round(float(r.demand_mw or 0), 1),
                "availability_mw": round(float(r.availability_mw or 0), 1),
                "headroom_mw":     round(max(float(r.availability_mw or 0) - float(r.demand_mw or 0), 0), 1),
                "source":          r.source,
                "raw_ref":         r.raw_ref or "",
            }
            for r in rows_raw
        ]

        next_skip = skip + top if has_next else None
        return _odata_response("market_events", rows, next_skip)

    except Exception as exc:
        logger.warning("OData market_events failed: %s", exc)
        return _odata_response("market_events", [])


# ── OData — fcas_prices ───────────────────────────────────────────────

@router.get("/odata/fcas_prices")
async def odata_fcas_prices(
    region: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    top: int = Query(default=_PAGE_SIZE, ge=1, le=10000, alias="$top"),
    skip: int = Query(default=0, ge=0, alias="$skip"),
):
    """OData v4 FCAS ancillary service prices — all 8 services per interval."""
    try:
        from sqlalchemy import select, desc
        from app.db.session import db_session
        from app.db.models import FcasPriceEvent

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=30))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)

        async with db_session() as session:
            q = (
                select(FcasPriceEvent)
                .where(FcasPriceEvent.valid_time >= start_dt)
                .where(FcasPriceEvent.valid_time <= end_dt)
                .order_by(desc(FcasPriceEvent.valid_time))
                .offset(skip)
                .limit(top + 1)
            )
            if region and region.upper() in _SUPPORTED_REGIONS:
                q = q.where(FcasPriceEvent.region == region.upper())

            result = await session.execute(q)
            rows_raw = result.scalars().all()

        has_next = len(rows_raw) > top
        rows_raw = rows_raw[:top]

        def _f(v):
            return round(float(v), 4) if v is not None else None

        rows = [
            {
                "valid_time":      r.valid_time.isoformat(),
                "region":          r.region,
                "raise_6sec_rrp":  _f(r.raise_6sec_rrp),
                "raise_60sec_rrp": _f(r.raise_60sec_rrp),
                "raise_5min_rrp":  _f(r.raise_5min_rrp),
                "raise_reg_rrp":   _f(r.raise_reg_rrp),
                "lower_6sec_rrp":  _f(r.lower_6sec_rrp),
                "lower_60sec_rrp": _f(r.lower_60sec_rrp),
                "lower_5min_rrp":  _f(r.lower_5min_rrp),
                "lower_reg_rrp":   _f(r.lower_reg_rrp),
            }
            for r in rows_raw
        ]

        next_skip = skip + top if has_next else None
        return _odata_response("fcas_prices", rows, next_skip)

    except Exception as exc:
        logger.warning("OData fcas_prices failed: %s", exc)
        return _odata_response("fcas_prices", [])


# ── OData — $metadata stub (required by PowerBI) ──────────────────────

@router.get("/odata/$metadata", include_in_schema=False)
async def odata_metadata():
    """Full OData v4 EDMX metadata document — required for PowerBI/Tableau to parse schema.

    Entity sets:
      market_events  — 5-min dispatch prices (price, demand, availability, headroom)
      fcas_prices    — all 8 FCAS ancillary service prices per interval
      queries        — NLP query history (raw_query, intent, verdict, region, answer)
      traces         — bitemporal decision traces (valid_time vs system_time)
    """
    edmx = """<?xml version="1.0" encoding="utf-8"?>
<edmx:Edmx Version="4.0" xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx">
  <edmx:DataServices>
    <Schema Namespace="GridVerdict" xmlns="http://docs.oasis-open.org/odata/ns/edm">

      <EntityType Name="MarketEvent">
        <Key><PropertyRef Name="valid_time"/><PropertyRef Name="region"/></Key>
        <Property Name="valid_time"      Type="Edm.DateTimeOffset" Nullable="false"/>
        <Property Name="region"          Type="Edm.String"/>
        <Property Name="price_rrp"       Type="Edm.Double"/>
        <Property Name="demand_mw"       Type="Edm.Double"/>
        <Property Name="availability_mw" Type="Edm.Double"/>
        <Property Name="headroom_mw"     Type="Edm.Double"/>
        <Property Name="source"          Type="Edm.String"/>
        <Property Name="raw_ref"         Type="Edm.String"/>
      </EntityType>

      <EntityType Name="FcasPrice">
        <Key><PropertyRef Name="valid_time"/><PropertyRef Name="region"/></Key>
        <Property Name="valid_time"      Type="Edm.DateTimeOffset" Nullable="false"/>
        <Property Name="region"          Type="Edm.String"/>
        <Property Name="raise_6sec_rrp"  Type="Edm.Double"/>
        <Property Name="raise_60sec_rrp" Type="Edm.Double"/>
        <Property Name="raise_5min_rrp"  Type="Edm.Double"/>
        <Property Name="raise_reg_rrp"   Type="Edm.Double"/>
        <Property Name="lower_6sec_rrp"  Type="Edm.Double"/>
        <Property Name="lower_60sec_rrp" Type="Edm.Double"/>
        <Property Name="lower_5min_rrp"  Type="Edm.Double"/>
        <Property Name="lower_reg_rrp"   Type="Edm.Double"/>
      </EntityType>

      <EntityType Name="Query">
        <Key><PropertyRef Name="id"/></Key>
        <Property Name="id"           Type="Edm.String"          Nullable="false"/>
        <Property Name="tenant_id"    Type="Edm.String"/>
        <Property Name="session_id"   Type="Edm.String"/>
        <Property Name="raw_query"    Type="Edm.String"/>
        <Property Name="intent"       Type="Edm.String"/>
        <Property Name="verdict"      Type="Edm.String"/>
        <Property Name="region"       Type="Edm.String"/>
        <Property Name="trace_id"     Type="Edm.String"/>
        <Property Name="created_at"   Type="Edm.DateTimeOffset"/>
      </EntityType>

      <EntityType Name="Trace">
        <Key><PropertyRef Name="id"/></Key>
        <Property Name="id"             Type="Edm.String"        Nullable="false"/>
        <Property Name="tenant_id"      Type="Edm.String"/>
        <Property Name="query_id"       Type="Edm.String"/>
        <Property Name="valid_time"     Type="Edm.DateTimeOffset"/>
        <Property Name="system_time"    Type="Edm.DateTimeOffset"/>
        <Property Name="model_profile"  Type="Edm.String"/>
        <Property Name="created_at"     Type="Edm.DateTimeOffset"/>
      </EntityType>

      <EntityContainer Name="GridVerdictService">
        <EntitySet Name="market_events" EntityType="GridVerdict.MarketEvent"/>
        <EntitySet Name="fcas_prices"   EntityType="GridVerdict.FcasPrice"/>
        <EntitySet Name="queries"       EntityType="GridVerdict.Query"/>
        <EntitySet Name="traces"        EntityType="GridVerdict.Trace"/>
      </EntityContainer>

    </Schema>
  </edmx:DataServices>
</edmx:Edmx>"""
    return Response(content=edmx, media_type="application/xml")


# ── OData — queries ──────────────────────────────────────────────────
# Enables Power BI / Tableau to load full NLP query history + verdicts.

@router.get("/odata/queries")
async def odata_queries(
    intent: str | None = Query(default=None, description="Filter by intent (e.g. explanation)"),
    verdict: str | None = Query(default=None, description="Filter by verdict (e.g. SUPPORTED)"),
    region: str | None = Query(default=None),
    start: str | None = Query(default=None, description="ISO start date"),
    end: str | None = Query(default=None),
    top: int = Query(default=_PAGE_SIZE, ge=1, le=5000, alias="$top"),
    skip: int = Query(default=0, ge=0, alias="$skip"),
):
    """OData v4 NLP query history — connect to Power BI for query analytics."""
    try:
        from sqlalchemy import select, desc
        from app.db.session import db_session
        from app.db.models import Query as QueryModel

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=90))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)

        async with db_session() as session:
            q = (
                select(QueryModel)
                .where(QueryModel.created_at >= start_dt, QueryModel.created_at <= end_dt)
                .order_by(desc(QueryModel.created_at))
                .offset(skip)
                .limit(top + 1)
            )
            if intent:
                q = q.where(QueryModel.intent == intent)
            if verdict:
                q = q.where(QueryModel.verdict == verdict)
            if region and region.upper() in _SUPPORTED_REGIONS:
                q = q.where(QueryModel.region == region.upper())

            result = await session.execute(q)
            rows_raw = result.scalars().all()

        has_next = len(rows_raw) > top
        rows_raw = rows_raw[:top]

        rows = [
            {
                "id":         r.id,
                "tenant_id":  r.tenant_id,
                "session_id": r.session_id,
                "raw_query":  r.raw_query,
                "intent":     r.intent,
                "verdict":    r.verdict,
                "region":     r.region,
                "trace_id":   r.trace_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows_raw
        ]

        next_skip = skip + top if has_next else None
        return _odata_response("queries", rows, next_skip)

    except Exception as exc:
        logger.warning("OData queries failed: %s", exc)
        return _odata_response("queries", [])


@router.get("/odata/traces")
async def odata_traces(
    start: str | None = Query(default=None, description="ISO start date"),
    end: str | None = Query(default=None),
    top: int = Query(default=_PAGE_SIZE, ge=1, le=5000, alias="$top"),
    skip: int = Query(default=0, ge=0, alias="$skip"),
):
    """OData v4 bitemporal decision traces — valid_time vs system_time for replay audit."""
    try:
        from sqlalchemy import select, desc
        from app.db.session import db_session
        from app.db.models import Trace

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=90))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)

        async with db_session() as session:
            q = (
                select(Trace)
                .where(Trace.created_at >= start_dt, Trace.created_at <= end_dt)
                .order_by(desc(Trace.created_at))
                .offset(skip)
                .limit(top + 1)
            )
            result = await session.execute(q)
            rows_raw = result.scalars().all()

        has_next = len(rows_raw) > top
        rows_raw = rows_raw[:top]

        rows = [
            {
                "id":            r.id,
                "tenant_id":     r.tenant_id,
                "query_id":      r.query_id,
                "valid_time":    r.valid_time.isoformat() if r.valid_time else None,
                "system_time":   r.system_time.isoformat() if r.system_time else None,
                "model_profile": r.model_profile,
                "created_at":    r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows_raw
        ]

        next_skip = skip + top if has_next else None
        return _odata_response("traces", rows, next_skip)

    except Exception as exc:
        logger.warning("OData traces failed: %s", exc)
        return _odata_response("traces", [])


# ── CSV export ────────────────────────────────────────────────────────

@router.get("/csv/market_events")
async def csv_market_events(
    region: str = Query(default="NSW1"),
    start: str | None = Query(default=None, description="ISO date e.g. 2026-01-01"),
    end: str | None = Query(default=None),
    limit: int = Query(default=8640, ge=1, le=52560),   # default ~30 days of 5-min
):
    """Download market price data as a flat CSV for Excel, Pandas, or Tableau."""
    try:
        from sqlalchemy import select, desc
        from app.db.session import db_session
        from app.db.models import MarketEvent

        region = region.upper()
        if region not in _SUPPORTED_REGIONS:
            return Response(status_code=400, content=f"Invalid region: {region}")

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=30))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)

        async with db_session() as session:
            result = await session.execute(
                select(MarketEvent)
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                    MarketEvent.valid_time >= start_dt,
                    MarketEvent.valid_time <= end_dt,
                )
                .order_by(desc(MarketEvent.valid_time))
                .limit(limit)
            )
            rows = result.scalars().all()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["valid_time", "region", "price_rrp", "demand_mw", "availability_mw", "headroom_mw"])
        for r in rows:
            avail = float(r.availability_mw or 0)
            demand = float(r.demand_mw or 0)
            writer.writerow([
                r.valid_time.isoformat(),
                r.region,
                round(float(r.price_rrp or 0), 4),
                round(demand, 1),
                round(avail, 1),
                round(max(avail - demand, 0), 1),
            ])

        date_str = start_dt.strftime("%Y%m%d")
        filename = f"gridverdict_{region}_{date_str}.csv"
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as exc:
        logger.warning("CSV export failed: %s", exc)
        return Response(status_code=500, content="Export failed")


# ── Daily HTML report ─────────────────────────────────────────────────

@router.get("/report/daily", response_class=HTMLResponse)
async def daily_report(
    region: str = Query(default="NSW1"),
    date: str | None = Query(default=None, description="YYYY-MM-DD, default today"),
):
    """Print-ready daily market report — open in browser and use Print → Save as PDF."""
    region = region.upper()
    if region not in _SUPPORTED_REGIONS:
        return HTMLResponse(content="<p>Invalid region</p>", status_code=400)

    report_date = _parse_dt(date) or datetime.now(timezone.utc)
    day_start = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + timedelta(days=1)

    prices = []
    commentary = []
    fcas_summary = {}

    try:
        from sqlalchemy import select, desc, func
        from app.db.session import db_session
        from app.db.models import MarketEvent, CommentaryEvent, FcasPriceEvent

        async with db_session() as session:
            # Price data for the day
            pr = await session.execute(
                select(MarketEvent)
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.region == region,
                    MarketEvent.valid_time >= day_start,
                    MarketEvent.valid_time < day_end,
                )
                .order_by(MarketEvent.valid_time)
            )
            prices = pr.scalars().all()

            # Commentary events
            ce = await session.execute(
                select(CommentaryEvent)
                .where(
                    CommentaryEvent.region == region,
                    CommentaryEvent.valid_time >= day_start,
                    CommentaryEvent.valid_time < day_end,
                )
                .order_by(desc(CommentaryEvent.valid_time))
                .limit(10)
            )
            commentary = ce.scalars().all()

            # FCAS daily averages
            fa = await session.execute(
                select(
                    func.avg(FcasPriceEvent.raise_reg_rrp).label("avg_raise_reg"),
                    func.avg(FcasPriceEvent.raise_6sec_rrp).label("avg_raise_fast"),
                    func.max(FcasPriceEvent.raise_reg_rrp).label("max_raise_reg"),
                )
                .where(
                    FcasPriceEvent.region == region,
                    FcasPriceEvent.valid_time >= day_start,
                    FcasPriceEvent.valid_time < day_end,
                )
            )
            fa_row = fa.fetchone()
            if fa_row:
                fcas_summary = {
                    "avg_raise_reg":  round(fa_row.avg_raise_reg or 0, 2),
                    "avg_raise_fast": round(fa_row.avg_raise_fast or 0, 2),
                    "max_raise_reg":  round(fa_row.max_raise_reg or 0, 2),
                }
    except Exception as exc:
        logger.warning("Daily report DB query failed: %s", exc)

    # Compute summary stats
    price_vals = [float(r.price_rrp or 0) for r in prices]
    avg_price  = round(sum(price_vals) / len(price_vals), 2) if price_vals else None
    max_price  = round(max(price_vals), 2) if price_vals else None
    min_price  = round(min(price_vals), 2) if price_vals else None
    intervals  = len(prices)

    # Sparkline data (hourly average for the chart)
    hourly: dict[int, list[float]] = {}
    for r in prices:
        h = r.valid_time.hour
        hourly.setdefault(h, []).append(float(r.price_rrp or 0))
    hourly_avg = {h: round(sum(v)/len(v), 1) for h, v in sorted(hourly.items())}
    chart_data = [hourly_avg.get(h, 0) for h in range(24)]
    chart_max  = max(chart_data) if chart_data else 1
    bars = "".join(
        f'<div style="display:inline-block;width:3.8%;height:{int(v/chart_max*60)}px;background:{"#e07c3a" if v==chart_max else "#2d4a5a"};margin:0 0.1%;vertical-align:bottom;border-radius:1px 1px 0 0;" title="{h:02d}:00 ${v:.0f}/MWh"></div>'
        for h, v in enumerate(chart_data)
    )

    # Commentary rows
    comm_rows = "".join(
        f'<tr><td style="color:#888;white-space:nowrap;">{c.valid_time.strftime("%H:%M")}</td>'
        f'<td><span style="font-size:10px;padding:1px 5px;border-radius:3px;background:{"#fee2e2" if c.severity=="CRITICAL" else "#fef9c3" if c.severity=="HIGH" else "#f0fdf4" if c.severity=="MEDIUM" else "#f8fafc"};color:#374151;">{c.severity}</span></td>'
        f'<td>{c.headline}</td></tr>'
        for c in commentary
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>GridVerdict Daily Report — {region} {report_date.strftime('%Y-%m-%d')}</title>
<style>
  @page {{ margin: 20mm; }}
  body {{ font-family: "Inter", system-ui, sans-serif; font-size: 12px; color: #1c1a17; background: #fff; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; color: #1c1a17; }}
  .sub {{ font-size: 11px; color: #888; margin-bottom: 20px; }}
  .kpi-row {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .kpi {{ flex: 1; min-width: 100px; padding: 10px 14px; border: 1px solid #e5e7eb; border-radius: 6px; }}
  .kpi-label {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: .05em; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; color: #1c1a17; margin-top: 2px; }}
  .kpi-sub {{ font-size: 10px; color: #aaa; }}
  .section {{ margin-bottom: 20px; }}
  .section-title {{ font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #888; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; margin-bottom: 10px; }}
  .chart-bar {{ height: 60px; background: #f9fafb; padding: 4px; border-radius: 4px; border: 1px solid #e5e7eb; display: flex; align-items: flex-end; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  th {{ text-align: left; font-weight: 600; color: #888; padding: 4px 8px; border-bottom: 1px solid #e5e7eb; }}
  td {{ padding: 4px 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }}
  .footer {{ margin-top: 24px; font-size: 10px; color: #bbb; border-top: 1px solid #e5e7eb; padding-top: 8px; }}
  .brand {{ color: #e07c3a; font-weight: 700; }}
  @media print {{ body {{ -webkit-print-color-adjust: exact; }} }}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">
  <div>
    <h1><span class="brand">Grid</span>Verdict Daily Brief</h1>
    <div class="sub">{region} · {report_date.strftime('%A, %d %B %Y')} · Generated {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div>
  </div>
  <div style="font-size:10px;color:#bbb;text-align:right;">gridverdict.io<br/>Confidential — not investment advice</div>
</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Avg Spot Price</div>
    <div class="kpi-value">${avg_price or "—"}/MWh</div>
    <div class="kpi-sub">{intervals} intervals</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Daily High</div>
    <div class="kpi-value" style="color:#c0392b;">${max_price or "—"}/MWh</div>
    <div class="kpi-sub">peak interval</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Daily Low</div>
    <div class="kpi-value" style="color:#16a34a;">${min_price or "—"}/MWh</div>
    <div class="kpi-sub">trough interval</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg FCAS Raise Reg</div>
    <div class="kpi-value">${fcas_summary.get('avg_raise_reg', '—')}/MWh</div>
    <div class="kpi-sub">peak ${fcas_summary.get('max_raise_reg', '—')}</div>
  </div>
</div>

<div class="section">
  <div class="section-title">Hourly Average Spot Price ({region})</div>
  <div class="chart-bar">{bars}</div>
  <div style="display:flex;justify-content:space-between;font-size:9px;color:#bbb;margin-top:2px;padding:0 2px;">
    <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span>
  </div>
</div>

<div class="section">
  <div class="section-title">Market Events</div>
  {'<table><thead><tr><th>Time</th><th>Severity</th><th>Event</th></tr></thead><tbody>' + comm_rows + '</tbody></table>' if comm_rows else '<p style="color:#bbb;font-size:11px;">No material events detected for this period.</p>'}
</div>

<div class="footer">
  Generated by GridVerdict · Data sourced from AEMO NEMWeb and BOM ·
  Covers {day_start.strftime('%Y-%m-%d')} 00:00–23:59 AEST ·
  <strong>Not financial advice.</strong> Past prices do not predict future prices.
</div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Rich REST query API ───────────────────────────────────────────────

@router.get("/query/dispatch")
async def query_dispatch(
    region: str | None = Query(default=None, description="NEM region filter (NSW1/VIC1/QLD1/SA1/TAS1)"),
    start: str | None = Query(default=None, description="ISO start e.g. 2026-01-01"),
    end: str | None = Query(default=None),
    price_min: float | None = Query(default=None, description="Minimum spot price filter ($/MWh)"),
    price_max: float | None = Query(default=None, description="Maximum spot price filter ($/MWh)"),
    include: str = Query(default="", description="Comma-separated extras: fcas,interconnector,weather"),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
):
    """Rich multi-table query API for researchers and analysts.

    Returns dispatch price rows with optional joins for FCAS, interconnector,
    and weather data at each interval.  Designed for pandas/polars/R data loading.

    Example:
        GET /api/export/query/dispatch?region=NSW1&price_min=300&include=fcas,interconnector
    """
    try:
        from sqlalchemy import select, desc
        from app.db.session import db_session
        from app.db.models import MarketEvent, FcasPriceEvent, MarketDriverEvent, WeatherObservation

        start_dt = _parse_dt(start) or (datetime.now(timezone.utc) - timedelta(days=30))
        end_dt   = _parse_dt(end)   or datetime.now(timezone.utc)
        extras = {e.strip().lower() for e in include.split(",") if e.strip()}

        async with db_session() as session:
            q = (
                select(MarketEvent)
                .where(
                    MarketEvent.source == "AEMO_DISPATCH_PRICE",
                    MarketEvent.valid_time >= start_dt,
                    MarketEvent.valid_time <= end_dt,
                )
                .order_by(desc(MarketEvent.valid_time))
                .offset(offset)
                .limit(limit + 1)
            )
            if region:
                q = q.where(MarketEvent.region == region.upper())
            if price_min is not None:
                q = q.where(MarketEvent.price_rrp >= price_min)
            if price_max is not None:
                q = q.where(MarketEvent.price_rrp <= price_max)

            result = await session.execute(q)
            rows = result.scalars().all()
            has_more = len(rows) > limit
            rows = rows[:limit]

            # Build base response
            base = []
            for r in rows:
                avail = float(r.availability_mw or 0)
                demand = float(r.demand_mw or 0)
                base.append({
                    "valid_time":      r.valid_time.isoformat(),
                    "region":          r.region,
                    "price_rrp":       round(float(r.price_rrp or 0), 4),
                    "demand_mw":       round(demand, 1),
                    "availability_mw": round(avail, 1),
                    "headroom_mw":     round(max(avail - demand, 0), 1),
                })

            # Batch-join extras if requested
            if rows and extras:
                vt_list = [r.valid_time for r in rows]

                if "fcas" in extras:
                    fcas_q = select(FcasPriceEvent).where(FcasPriceEvent.valid_time.in_(vt_list))
                    if region:
                        fcas_q = fcas_q.where(FcasPriceEvent.region == region.upper())
                    fcas_result = await session.execute(fcas_q)
                    fcas_map: dict[str, dict] = {}
                    for f in fcas_result.scalars().all():
                        key = f.valid_time.isoformat()
                        fcas_map[key] = {
                            "raise_reg_rrp":   round(f.raise_reg_rrp or 0, 4),
                            "raise_6sec_rrp":  round(f.raise_6sec_rrp or 0, 4),
                            "lower_reg_rrp":   round(f.lower_reg_rrp or 0, 4),
                            "raise_5min_rrp":  round(f.raise_5min_rrp or 0, 4),
                        }
                    for row in base:
                        row["fcas"] = fcas_map.get(row["valid_time"])

                if "interconnector" in extras:
                    ic_result = await session.execute(
                        select(MarketDriverEvent.valid_time, MarketDriverEvent.element_id, MarketDriverEvent.values)
                        .where(
                            MarketDriverEvent.driver_type == "interconnector",
                            MarketDriverEvent.valid_time.in_(vt_list),
                        )
                    )
                    ic_map: dict[str, list] = {}
                    for ic in ic_result.fetchall():
                        key = ic.valid_time.isoformat()
                        flow = ic.values.get("metered_mw_flow") or ic.values.get("mw_flow")
                        ic_map.setdefault(key, []).append({
                            "id":      ic.element_id,
                            "flow_mw": round(flow, 1) if flow is not None else None,
                        })
                    for row in base:
                        row["interconnectors"] = ic_map.get(row["valid_time"], [])

                if "weather" in extras:
                    wx_result = await session.execute(
                        select(WeatherObservation)
                        .where(WeatherObservation.region == (region or "NSW1").upper())
                        .where(WeatherObservation.observed_at >= start_dt - timedelta(hours=1))
                        .where(WeatherObservation.observed_at <= end_dt + timedelta(hours=1))
                    )
                    wx_rows = sorted(wx_result.scalars().all(), key=lambda w: w.observed_at)
                    for row in base:
                        try:
                            from datetime import datetime as _dt
                            row_dt = _dt.fromisoformat(row["valid_time"].replace("Z", "+00:00"))
                            best = min(
                                wx_rows,
                                key=lambda w: abs(
                                    (w.observed_at.replace(tzinfo=timezone.utc)
                                     if w.observed_at.tzinfo is None else w.observed_at) - row_dt
                                ),
                                default=None,
                            )
                            if best:
                                row["weather"] = {
                                    "temp_c":     best.temperature_c,
                                    "temp_dev_c": best.temp_deviation_c,
                                    "wind_kmh":   best.wind_speed_kmh,
                                    "cloud_pct":  best.cloud_cover_pct,
                                }
                        except Exception:
                            pass

        return {
            "region": region,
            "start": start_dt.isoformat(),
            "end":   end_dt.isoformat(),
            "count": len(base),
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
            "includes": sorted(extras),
            "rows": base,
        }
    except Exception as exc:
        logger.warning("Rich query failed: %s", exc)
        return {"rows": [], "count": 0, "error": str(exc)}


# ── Utilities ─────────────────────────────────────────────────────────

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    return None
