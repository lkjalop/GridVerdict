"""Participant behavior profiler — aggregate rebid patterns across a rolling window.

Queries bid_offers and market_events to build a behavioral profile for a
generator (DUID) showing how frequently they withdraw availability intraday,
whether those withdrawals coincide with high-price periods, and whether the
pattern is consistent enough to classify as "habitual" strategic behaviour.

The profiler is deterministic: it summarises observed BIDDAYOFFER/BIDPEROFFER
divergence. It does not infer intent or assert strategic motivation — forced
outages and scheduled maintenance can produce identical patterns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

_SPIKE_PRICE = 300.0
_HABITUAL_MIN_REBIDS = 5
_HABITUAL_MIN_STRATEGIC_FRACTION = 0.40
_OCCASIONAL_MIN_REBIDS = 2
_PERIOD_MINUTES = 30


@dataclass
class ParticipantProfile:
    duid: str
    station_name: str | None
    participant: str | None
    fuel_type: str | None
    region: str | None
    window_days: int
    rebid_count: int
    days_with_rebids: int
    avg_mw_withdrawn: float
    max_mw_withdrawn: float
    strategic_fraction: float  # fraction of rebids where cheap-band capacity was withdrawn
    spike_correlation: float   # fraction of priced rebid events at or above _SPIKE_PRICE
    behavioral_tier: str       # "habitual" | "occasional" | "low_activity"
    data_available: bool
    caveat: str = (
        "Behavioral profile based on observed BIDDAYOFFER/BIDPEROFFER availability divergence. "
        "Does not assert strategic intent — forced outages and scheduled maintenance "
        "can produce identical patterns. Requires corroborating constraint and SCADA evidence."
    )


async def profile_participant(
    session: Any,
    duid: str,
    region: str | None = None,
    window_days: int = 30,
    threshold_mw: float = 50.0,
    min_price: float = 100.0,
) -> ParticipantProfile:
    """Build a behavioral rebid profile for a single market participant (DUID).

    Returns a ParticipantProfile with data_available=False and behavioral_tier
    "low_activity" when no bid data has been ingested for this DUID in the window.
    """
    from sqlalchemy import text

    start_dt = datetime.now(timezone.utc) - timedelta(days=window_days)
    duid_upper = duid.upper()

    meta = await _fetch_generator_meta(session, duid_upper)
    station_name = meta.get("station_name") if meta else None
    participant_name = meta.get("participant") if meta else None
    fuel_type = meta.get("fuel_type") if meta else None
    eff_region = region or (meta.get("region") if meta else None)

    region_filter = "AND b.region = :region" if region else ""
    base_params = _build_params(duid_upper, start_dt, region)

    day_res = await session.execute(text(f"""
        SELECT b.settlement_date, b.max_avail_mw, b.price_bands, b.avail_bands
        FROM bid_offers b
        WHERE b.duid = :duid AND b.source = 'BIDDAYOFFER'
          AND b.bid_type = 'ENERGY' AND b.settlement_date >= :start_dt
          {region_filter}
    """), base_params)

    day_rows: dict[str, dict] = {}
    for row in day_res.fetchall():
        sdate_raw, max_avail, price_bands_raw, avail_bands_raw = row
        sdate_str = _to_date_str(sdate_raw)
        if sdate_str and max_avail is not None:
            day_rows[sdate_str] = {
                "max_avail_mw": float(max_avail),
                "price_bands": _parse_json(price_bands_raw),
                "avail_bands": _parse_json(avail_bands_raw),
            }

    if not day_rows:
        return _empty_profile(
            duid_upper, station_name, participant_name, fuel_type, eff_region,
            window_days, data_available=False,
        )

    intra_res = await session.execute(text(f"""
        SELECT b.settlement_date, b.period_id, b.max_avail_mw, b.price_bands, b.avail_bands
        FROM bid_offers b
        WHERE b.duid = :duid AND b.source = 'BIDPEROFFER'
          AND b.bid_type = 'ENERGY' AND b.settlement_date >= :start_dt
          {region_filter}
        ORDER BY b.settlement_date, b.period_id
    """), base_params)
    intra_rows = intra_res.fetchall()

    price_by_date_period: dict[tuple[str, int], float] = {}
    if eff_region:
        price_by_date_period = await _fetch_price_context(session, eff_region, start_dt)

    rebid_events: list[dict] = []
    days_with_rebids: set[str] = set()

    for row in intra_rows:
        sdate_raw, period_id, intra_avail, intra_pb_raw, intra_ab_raw = row
        sdate_str = _to_date_str(sdate_raw)
        if not sdate_str or intra_avail is None:
            continue
        day = day_rows.get(sdate_str)
        if day is None:
            continue

        withdrawal_mw = day["max_avail_mw"] - float(intra_avail)
        if withdrawal_mw < threshold_mw:
            continue

        spot_price: float | None = None
        if period_id is not None:
            spot_price = price_by_date_period.get((sdate_str, int(period_id)))
        if spot_price is not None and spot_price < min_price:
            continue

        day_cheap = _low_band_mw(day["price_bands"], day["avail_bands"])
        intra_cheap = _low_band_mw(_parse_json(intra_pb_raw), _parse_json(intra_ab_raw))
        cheap_withdrawn = max(0.0, day_cheap - intra_cheap)
        is_strategic = cheap_withdrawn > threshold_mw / 2

        rebid_events.append({
            "sdate": sdate_str,
            "withdrawal_mw": withdrawal_mw,
            "is_strategic": is_strategic,
            "spot_price": spot_price,
        })
        days_with_rebids.add(sdate_str)

    if not rebid_events:
        return _empty_profile(
            duid_upper, station_name, participant_name, fuel_type, eff_region,
            window_days, data_available=bool(day_rows),
        )

    rebid_count = len(rebid_events)
    withdrawals = [e["withdrawal_mw"] for e in rebid_events]
    strategic_count = sum(1 for e in rebid_events if e["is_strategic"])
    strategic_frac = round(strategic_count / rebid_count, 3)
    priced = [e for e in rebid_events if e["spot_price"] is not None]
    spike_corr = (
        round(sum(1 for e in priced if (e["spot_price"] or 0) >= _SPIKE_PRICE) / len(priced), 3)
        if priced else 0.0
    )

    return ParticipantProfile(
        duid=duid_upper,
        station_name=station_name,
        participant=participant_name,
        fuel_type=fuel_type,
        region=eff_region,
        window_days=window_days,
        rebid_count=rebid_count,
        days_with_rebids=len(days_with_rebids),
        avg_mw_withdrawn=round(mean(withdrawals), 1),
        max_mw_withdrawn=round(max(withdrawals), 1),
        strategic_fraction=strategic_frac,
        spike_correlation=spike_corr,
        behavioral_tier=_classify_tier(rebid_count, strategic_frac),
        data_available=True,
    )


def _classify_tier(rebid_count: int, strategic_fraction: float) -> str:
    if rebid_count >= _HABITUAL_MIN_REBIDS and strategic_fraction >= _HABITUAL_MIN_STRATEGIC_FRACTION:
        return "habitual"
    if rebid_count >= _OCCASIONAL_MIN_REBIDS:
        return "occasional"
    return "low_activity"


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _fetch_generator_meta(session: Any, duid: str) -> dict | None:
    from sqlalchemy import text
    res = await session.execute(
        text("SELECT station_name, participant, region, fuel_type FROM generator_units WHERE duid = :duid"),
        {"duid": duid},
    )
    row = res.fetchone()
    if row is None:
        return None
    return {"station_name": row[0], "participant": row[1], "region": row[2], "fuel_type": row[3]}


async def _fetch_price_context(
    session: Any, region: str, start_dt: datetime
) -> dict[tuple[str, int], float]:
    from sqlalchemy import text
    res = await session.execute(text("""
        SELECT valid_time, price_rrp
        FROM market_events
        WHERE region = :region
          AND source = 'AEMO_DISPATCH_PRICE'
          AND valid_time >= :start_dt
    """), {"region": region, "start_dt": start_dt})
    by_dp: dict[tuple[str, int], list[float]] = {}
    for vt_raw, price in res.fetchall():
        if vt_raw is None or price is None:
            continue
        if not isinstance(vt_raw, datetime):
            try:
                vt_raw = datetime.fromisoformat(str(vt_raw))
            except ValueError:
                continue
        vdate_str = vt_raw.date().isoformat()
        minutes = vt_raw.hour * 60 + vt_raw.minute
        period = minutes // _PERIOD_MINUTES + 1
        by_dp.setdefault((vdate_str, period), []).append(float(price))
    return {k: mean(vs) for k, vs in by_dp.items()}


def _low_band_mw(price_bands: dict, avail_bands: dict) -> float:
    total = 0.0
    for band_id, price in price_bands.items():
        try:
            if float(price) <= 300.0:
                total += float(avail_bands.get(band_id, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _parse_json(v: Any) -> dict:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _to_date_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    s = str(v)
    return s[:10] if len(s) >= 10 else None


def _build_params(duid: str, start_dt: datetime, region: str | None) -> dict:
    p: dict = {"duid": duid, "start_dt": start_dt}
    if region:
        p["region"] = region
    return p


def _empty_profile(
    duid: str,
    station_name: str | None,
    participant: str | None,
    fuel_type: str | None,
    region: str | None,
    window_days: int,
    data_available: bool,
) -> ParticipantProfile:
    return ParticipantProfile(
        duid=duid,
        station_name=station_name,
        participant=participant,
        fuel_type=fuel_type,
        region=region,
        window_days=window_days,
        rebid_count=0,
        days_with_rebids=0,
        avg_mw_withdrawn=0.0,
        max_mw_withdrawn=0.0,
        strategic_fraction=0.0,
        spike_correlation=0.0,
        behavioral_tier="low_activity",
        data_available=data_available,
    )
