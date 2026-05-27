"""Shared rebid detection engine.

Extracts the BIDDAYOFFER vs BIDPEROFFER comparison algorithm from
routes_rebid.py so it can be used by both the API route and the
incident timeline engine without duplication.

Public API:
    detect_rebids(session, region, settlement_dt, threshold_mw, min_price)
        → list[RebidEvent]
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any

_PERIOD_MINUTES = 30


@dataclass
class RebidEvent:
    duid: str
    station_name: str | None
    fuel_type: str | None
    period_id: int | None
    period_start_utc: datetime | None
    day_avail_mw: float
    intra_avail_mw: float
    withdrawal_mw: float
    cheap_band_withdrawn_mw: float
    spot_price_context: float | None
    offer_delta_minutes: float | None
    rebid_flag: str    # "strategic" | "availability"
    severity: str      # "high" | "medium" | "low"


async def detect_rebids(
    session: Any,
    region: str,
    settlement_dt: datetime | None,
    threshold_mw: float = 50.0,
    min_price: float = 100.0,
) -> list[RebidEvent]:
    """Compare BIDDAYOFFER vs BIDPEROFFER to detect availability withdrawals.

    Returns RebidEvent instances sorted by withdrawal_mw descending.
    Returns an empty list (never raises) when bid data is absent.
    """
    from sqlalchemy import text

    if settlement_dt is None:
        res = await session.execute(text("""
            SELECT MAX(settlement_date) FROM bid_offers
            WHERE region = :region AND source = 'BIDPEROFFER'
        """), {"region": region})
        val = res.scalar()
        if val is None:
            return []
        settlement_dt = val if isinstance(val, datetime) else datetime.fromisoformat(str(val))

    date_str = settlement_dt.date().isoformat()

    day_res = await session.execute(text("""
        SELECT b.duid, b.bid_type, b.max_avail_mw, b.offer_date,
               b.price_bands, b.avail_bands,
               g.fuel_type, g.station_name
        FROM bid_offers b
        LEFT JOIN generator_units g ON g.duid = b.duid
        WHERE b.region = :region
          AND b.source = 'BIDDAYOFFER'
          AND DATE(b.settlement_date) = :date
          AND b.bid_type = 'ENERGY'
    """), {"region": region, "date": date_str})
    day_rows: dict[str, Any] = {row[0]: row for row in day_res.fetchall()}

    if not day_rows:
        return []

    intra_res = await session.execute(text("""
        SELECT b.duid, b.period_id, b.max_avail_mw, b.offer_date,
               b.price_bands, b.avail_bands
        FROM bid_offers b
        WHERE b.region = :region
          AND b.source = 'BIDPEROFFER'
          AND DATE(b.settlement_date) = :date
          AND b.bid_type = 'ENERGY'
        ORDER BY b.duid, b.period_id
    """), {"region": region, "date": date_str})
    intra_rows = intra_res.fetchall()

    price_res = await session.execute(text("""
        SELECT valid_time, price_rrp
        FROM market_events
        WHERE region = :region
          AND source = 'AEMO_DISPATCH_PRICE'
          AND DATE(valid_time) = :date
        ORDER BY valid_time
    """), {"region": region, "date": date_str})
    price_by_period = _prices_by_period(price_res.fetchall(), settlement_dt)

    events: list[RebidEvent] = []
    for row in intra_rows:
        duid, period_id, intra_avail, offer_date, price_bands_intra, avail_bands_intra = row
        if duid not in day_rows:
            continue

        _, _, day_avail, day_offer_date, price_bands_day, avail_bands_day, fuel_type, station = day_rows[duid]

        if day_avail is None or intra_avail is None:
            continue

        withdrawal_mw = float(day_avail) - float(intra_avail)
        if withdrawal_mw < threshold_mw:
            continue

        spot = price_by_period.get(period_id)
        if spot is not None and spot < min_price:
            continue

        offer_delta_min: float | None = None
        if offer_date and day_offer_date:
            if not isinstance(offer_date, datetime):
                offer_date = datetime.fromisoformat(str(offer_date))
            if not isinstance(day_offer_date, datetime):
                day_offer_date = datetime.fromisoformat(str(day_offer_date))
            offer_delta_min = round((offer_date - day_offer_date).total_seconds() / 60, 0)

        day_cheap = _low_band_mw(price_bands_day or {}, avail_bands_day or {})
        intra_cheap = _low_band_mw(price_bands_intra or {}, avail_bands_intra or {})
        cheap_withdrawn = max(0.0, day_cheap - intra_cheap)

        period_start: datetime | None = None
        if period_id is not None:
            period_start = _period_to_utc(settlement_dt, period_id)

        events.append(RebidEvent(
            duid=duid,
            station_name=station,
            fuel_type=_norm_fuel(fuel_type),
            period_id=period_id,
            period_start_utc=period_start,
            day_avail_mw=round(float(day_avail), 1),
            intra_avail_mw=round(float(intra_avail), 1),
            withdrawal_mw=round(withdrawal_mw, 1),
            cheap_band_withdrawn_mw=round(cheap_withdrawn, 1),
            spot_price_context=round(spot, 2) if spot is not None else None,
            offer_delta_minutes=offer_delta_min,
            rebid_flag="strategic" if cheap_withdrawn > threshold_mw / 2 else "availability",
            severity=_severity(withdrawal_mw, spot),
        ))

    events.sort(key=lambda e: e.withdrawal_mw, reverse=True)
    return events


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prices_by_period(price_rows: list, base_dt: datetime) -> dict[int, float]:
    by_period: dict[int, list[float]] = {}
    base_date = base_dt.date()
    for vt, price in price_rows:
        if vt is None or price is None:
            continue
        if not isinstance(vt, datetime):
            vt = datetime.fromisoformat(str(vt))
        if vt.date() != base_date:
            continue
        minutes = vt.hour * 60 + vt.minute
        period = minutes // _PERIOD_MINUTES + 1
        by_period.setdefault(period, []).append(float(price))
    return {p: mean(prices) for p, prices in by_period.items() if prices}


def _low_band_mw(price_bands: dict, avail_bands: dict) -> float:
    total = 0.0
    for band_id, price in price_bands.items():
        try:
            if float(price) <= 300.0:
                total += float(avail_bands.get(band_id, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _period_to_utc(settlement_dt: datetime, period_id: int) -> datetime:
    minutes = (int(period_id) - 1) * _PERIOD_MINUTES
    return settlement_dt.replace(
        hour=minutes // 60, minute=minutes % 60, second=0, microsecond=0,
        tzinfo=timezone.utc,
    )


def _severity(withdrawal_mw: float, spot: float | None) -> str:
    score = withdrawal_mw / 100.0
    if spot:
        score += spot / 1000.0
    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _norm_fuel(raw: str | None) -> str:
    from app.engines.fuel_mix import _FUEL_GROUPS
    return _FUEL_GROUPS.get(raw or "", "other")
