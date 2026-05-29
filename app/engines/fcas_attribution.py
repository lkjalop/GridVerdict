"""FCAS (Frequency Control Ancillary Services) attribution engine.

Queries persisted FcasPriceEvent rows and computes opportunity context
for BESS dispatch decisions. FCAS revenue is 20-40% of typical BESS
income — omitting it gives an incomplete dispatch recommendation.

Eight NEM FCAS markets:
  Contingency: Raise/Lower × 6s, 60s, 5min
  Regulation:  Raise Reg, Lower Reg

Framework boundary: no imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class FcasOpportunityContext:
    region: str
    valid_time: datetime
    raise_6sec_rrp: float | None = None
    lower_6sec_rrp: float | None = None
    raise_60sec_rrp: float | None = None
    lower_60sec_rrp: float | None = None
    raise_5min_rrp: float | None = None
    lower_5min_rrp: float | None = None
    raise_reg_rrp: float | None = None
    lower_reg_rrp: float | None = None
    # Derived convenience fields
    max_raise_rrp: float | None = None    # highest raise service price
    max_lower_rrp: float | None = None    # highest lower service price
    best_raise_service: str | None = None # "6s" | "60s" | "5min" | "reg"
    best_lower_service: str | None = None
    total_opportunity_mwh: float | None = None  # max_raise + max_lower (signal for BESS)
    tight_markets: list[str] = field(default_factory=list)  # services with RRP > $50/MWh
    available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "valid_time": self.valid_time.isoformat(),
            "raise_6sec_rrp": self.raise_6sec_rrp,
            "lower_6sec_rrp": self.lower_6sec_rrp,
            "raise_60sec_rrp": self.raise_60sec_rrp,
            "lower_60sec_rrp": self.lower_60sec_rrp,
            "raise_5min_rrp": self.raise_5min_rrp,
            "lower_5min_rrp": self.lower_5min_rrp,
            "raise_reg_rrp": self.raise_reg_rrp,
            "lower_reg_rrp": self.lower_reg_rrp,
            "max_raise_rrp": self.max_raise_rrp,
            "max_lower_rrp": self.max_lower_rrp,
            "best_raise_service": self.best_raise_service,
            "best_lower_service": self.best_lower_service,
            "total_opportunity_mwh": self.total_opportunity_mwh,
            "tight_markets": self.tight_markets,
            "available": self.available,
        }


_TIGHT_THRESHOLD = 50.0   # $/MWh — FCAS price above this signals tight reserve margins

_RAISE_SERVICES = [
    ("6s",    "raise_6sec_rrp"),
    ("60s",   "raise_60sec_rrp"),
    ("5min",  "raise_5min_rrp"),
    ("reg",   "raise_reg_rrp"),
]
_LOWER_SERVICES = [
    ("6s",    "lower_6sec_rrp"),
    ("60s",   "lower_60sec_rrp"),
    ("5min",  "lower_5min_rrp"),
    ("reg",   "lower_reg_rrp"),
]


def _derive(ctx: FcasOpportunityContext) -> FcasOpportunityContext:
    """Compute derived fields from the raw service prices."""
    raise_prices = {lbl: getattr(ctx, attr) for lbl, attr in _RAISE_SERVICES
                    if getattr(ctx, attr) is not None}
    lower_prices = {lbl: getattr(ctx, attr) for lbl, attr in _LOWER_SERVICES
                    if getattr(ctx, attr) is not None}

    if raise_prices:
        best_r = max(raise_prices, key=raise_prices.__getitem__)
        ctx.max_raise_rrp = round(raise_prices[best_r], 2)
        ctx.best_raise_service = best_r

    if lower_prices:
        best_l = max(lower_prices, key=lower_prices.__getitem__)
        ctx.max_lower_rrp = round(lower_prices[best_l], 2)
        ctx.best_lower_service = best_l

    if ctx.max_raise_rrp is not None and ctx.max_lower_rrp is not None:
        ctx.total_opportunity_mwh = round(ctx.max_raise_rrp + ctx.max_lower_rrp, 2)

    # Identify tight markets
    all_services = {**{f"raise_{k}": v for k, v in raise_prices.items()},
                    **{f"lower_{k}": v for k, v in lower_prices.items()}}
    ctx.tight_markets = [svc for svc, price in all_services.items() if price >= _TIGHT_THRESHOLD]

    ctx.available = bool(raise_prices or lower_prices)
    return ctx


async def get_fcas_context(session: Any, region: str, valid_time: datetime) -> FcasOpportunityContext:
    """Return the most recent FCAS prices for a region near valid_time.

    Looks back up to 10 minutes to handle slight dispatch timing mismatches.
    Returns an unavailable context when no DB rows are found.
    """
    try:
        from sqlalchemy import select, and_
        from app.db.models import FcasPriceEvent

        cutoff = valid_time - timedelta(minutes=10)
        result = await session.execute(
            select(FcasPriceEvent)
            .where(
                and_(
                    FcasPriceEvent.region == region.upper(),
                    FcasPriceEvent.valid_time >= cutoff,
                    FcasPriceEvent.valid_time <= valid_time,
                )
            )
            .order_by(FcasPriceEvent.valid_time.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return FcasOpportunityContext(region=region, valid_time=valid_time, available=False)

        ctx = FcasOpportunityContext(
            region=region,
            valid_time=row.valid_time,
            raise_6sec_rrp=row.raise_6sec_rrp,
            lower_6sec_rrp=row.lower_6sec_rrp,
            raise_60sec_rrp=row.raise_60sec_rrp,
            lower_60sec_rrp=row.lower_60sec_rrp,
            raise_5min_rrp=row.raise_5min_rrp,
            lower_5min_rrp=row.lower_5min_rrp,
            raise_reg_rrp=row.raise_reg_rrp,
            lower_reg_rrp=row.lower_reg_rrp,
        )
        return _derive(ctx)

    except Exception:
        return FcasOpportunityContext(region=region, valid_time=valid_time, available=False)


def fcas_opportunity_summary(ctx: FcasOpportunityContext) -> str:
    """One-line human summary of the FCAS opportunity for the answer planner."""
    if not ctx.available:
        return "FCAS prices unavailable for this interval."
    parts = []
    if ctx.max_raise_rrp is not None:
        parts.append(f"Best raise: {ctx.best_raise_service} @ ${ctx.max_raise_rrp:.0f}/MWh")
    if ctx.max_lower_rrp is not None:
        parts.append(f"best lower: {ctx.best_lower_service} @ ${ctx.max_lower_rrp:.0f}/MWh")
    if ctx.tight_markets:
        parts.append(f"tight markets: {', '.join(ctx.tight_markets)}")
    return "; ".join(parts) if parts else "FCAS data present but no material prices."
