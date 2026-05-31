"""AEMO Gas Bulletin Board + STTM gas hub price client.

Fetches east coast domestic gas prices from AEMO's Short Term Trading Market (STTM)
and Gas Bulletin Board (GBB). These prices feed directly into the LNG→gas→electricity
causal chain that explains NEM price events during gas-constrained periods.

Data sources (all free, public):
  1. AEMO STTM hub prices — Sydney, Adelaide, Brisbane hub daily settlement prices
     URL: AEMO MIBB report STTM_TRADING_HUB (STTM daily trading results)
  2. ACCC Gas Inquiry quarterly data — LNG netback, sector prices (parsed quarterly)
  3. AEMO GSOO annual gas outlook (parsed annually)

The LNG → domestic gas → NEM electricity causal chain:
  JKM (Asian LNG spot)  →  east coast domestic gas price floor (via export netback)
  East coast gas price  ×  generator heat rate  →  gas SRMC
  Gas SRMC  →  NEM spot price (gas is marginal setter ~40% of peak hours)

This chain is why the 2022 energy crisis occurred: JKM hit $70/MMBtu →
domestic gas hit $30/GJ → gas SRMCs hit $400-600/MWh → NEM caps triggered.
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# AEMO STTM hub trading results endpoint (public MIBB report)
_STTM_BASE = "https://www.aemo.com.au/aemo/apps/api/report/STTM_TRADING_HUB"

# STTM hub codes → NEM region mapping
_HUB_TO_REGION: dict[str, str] = {
    "SYDNEY":    "NSW1",
    "BRISBANE":  "QLD1",
    "ADELAIDE":  "SA1",
    # Wallumbilla is QLD-adjacent but feeds entire east coast
    "WALLUMBILLA": "QLD1",
    "MOOMBA":    "SA1",
    "VIC HUB":   "VIC1",
    "SYDNEY HUB": "NSW1",
    "BRISBANE HUB": "QLD1",
    "ADELAIDE HUB": "SA1",
}

# Approximate heat rates for major gas generator classes (GJ/MWh)
_HEAT_RATE_GJ_MWH: dict[str, float] = {
    "CCGT":   6.5,   # combined-cycle gas turbine (most efficient)
    "OCGT":   10.0,  # open-cycle gas turbine (peakers)
    "STEAM":  11.5,  # older gas steam turbines
}

# Variable O&M adder for gas generators ($/MWh)
_GAS_VARIABLE_OM: float = 4.0


@dataclass
class GasHubPrice:
    settlement_date: date
    hub: str
    region: str
    price_gj: float          # $/GJ domestic gas price
    volume_gj: float | None  # traded volume
    raw_ref: str = ""

    @property
    def srmc_ccgt(self) -> float:
        """Short-run marginal cost for a CCGT at this gas price ($/MWh)."""
        return self.price_gj * _HEAT_RATE_GJ_MWH["CCGT"] + _GAS_VARIABLE_OM

    @property
    def srmc_ocgt(self) -> float:
        """Short-run marginal cost for an OCGT peaker at this gas price ($/MWh)."""
        return self.price_gj * _HEAT_RATE_GJ_MWH["OCGT"] + _GAS_VARIABLE_OM

    def to_dict(self) -> dict[str, Any]:
        return {
            "settlement_date": self.settlement_date.isoformat(),
            "hub": self.hub,
            "region": self.region,
            "price_gj": round(self.price_gj, 4),
            "volume_gj": self.volume_gj,
            "srmc_ccgt_mwh": round(self.srmc_ccgt, 2),
            "srmc_ocgt_mwh": round(self.srmc_ocgt, 2),
            "raw_ref": self.raw_ref,
        }


@dataclass
class GasMarketState:
    """Current gas market state for a NEM region — ready for scatter-gather injection."""
    region: str
    as_of: datetime
    latest_hub_price_gj: float | None
    hub_name: str | None
    srmc_ccgt: float | None
    srmc_ocgt: float | None
    recent_prices: list[GasHubPrice] = field(default_factory=list)
    lng_netback_gj: float | None = None        # from ACCC quarterly (may be lagged)
    price_trend: str = "unknown"               # "rising" | "falling" | "stable"
    high_price_alert: bool = False             # True when > $15/GJ (above normal range)
    crisis_alert: bool = False                 # True when > $25/GJ (2022-style crisis)
    source: str = "AEMO_STTM"
    raw_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "as_of": self.as_of.isoformat(),
            "latest_hub_price_gj": self.latest_hub_price_gj,
            "hub_name": self.hub_name,
            "srmc_ccgt_mwh": round(self.srmc_ccgt, 2) if self.srmc_ccgt else None,
            "srmc_ocgt_mwh": round(self.srmc_ocgt, 2) if self.srmc_ocgt else None,
            "lng_netback_gj": self.lng_netback_gj,
            "price_trend": self.price_trend,
            "high_price_alert": self.high_price_alert,
            "crisis_alert": self.crisis_alert,
            "source": self.source,
            "raw_ref": self.raw_ref,
            "recent_prices": [p.to_dict() for p in self.recent_prices[-7:]],
        }


# ── ACCC Gas Inquiry static context (updated quarterly by parsing ACCC Excel) ──
# These are from ACCC Gas Inquiry 2024 Q1 data. Kept as static fallback so the
# gas-electricity nexus explanation is always available even when live STTM is down.
_ACCC_STATIC_CONTEXT = {
    "lng_netback_gj_2024_q1": 10.20,    # $/GJ — ACCC Gas Inquiry Q1 2024
    "domestic_spot_range_2024": (7.0, 18.0),    # (min, max) $/GJ typical range
    "crisis_peak_2022": 30.0,            # $/GJ — peak domestic price during 2022 crisis
    "jkm_peak_2022_mmbtu": 70.0,        # $/MMBtu — JKM peak that drove 2022 crisis
    "normal_range_pre_2021": (5.0, 8.0),  # $/GJ — pre-LNG-export-parity range
    "accc_report_quarter": "2024-Q1",
    "accc_source": "ACCC Gas Inquiry 2024 (accc.gov.au/gas-inquiry)",
}

# Historical context for causal chain explanations
_GAS_ELECTRICITY_CAUSAL_CHAIN = """
The LNG export-domestic gas-electricity price chain:

1. JKM (Japan-Korea Marker) sets the LNG export netback price — what
   a producer earns by liquefying gas and shipping it to Asia.

2. East coast domestic gas producers price to the netback (why sell
   domestically at $8/GJ when you can export for equivalent of $15+/GJ?).

3. Gas-fired generators (CCGT, OCGT) use gas as fuel. Their Short-Run
   Marginal Cost (SRMC) = gas price × heat rate + variable O&M.
   At $10/GJ: CCGT SRMC ≈ $69/MWh; OCGT peaker ≈ $104/MWh.
   At $25/GJ (2022 crisis): CCGT SRMC ≈ $167/MWh; OCGT ≈ $254/MWh.

4. Gas generators set the NEM spot price approximately 40% of peak
   demand hours (when coal is fully dispatched and renewables are low).

5. Transmission: JKM spike → domestic gas spike → SRMC spike → NEM cap events.

This chain explains why the 2022 global LNG market shock (Russia-Ukraine,
JKM to $70/MMBtu) caused Australian household electricity bills to rise
despite most electricity being generated from coal and renewables.
"""


async def fetch_sttm_hub_prices(days_back: int = 30) -> list[GasHubPrice]:
    """Fetch recent STTM hub prices from AEMO MIBB. Returns empty list on failure."""
    prices: list[GasHubPrice] = []
    cutoff = date.today() - timedelta(days=days_back)

    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": _BROWSER_UA},
        ) as client:
            resp = await client.get(_STTM_BASE)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            # AEMO MIBB returns CSV or ZIP depending on the report type
            if "zip" in content_type or resp.content[:2] == b"PK":
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
                if csv_name:
                    raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
                else:
                    return prices
            else:
                raw_text = resp.text

            prices = _parse_sttm_csv(raw_text, cutoff, _STTM_BASE)

    except Exception as exc:
        logger.info("STTM fetch failed (%s) — gas nexus explanation still available from static context", exc)

    return prices


def _parse_sttm_csv(raw_text: str, cutoff: date, raw_ref: str) -> list[GasHubPrice]:
    """Parse AEMO STTM CSV format. Tolerant of different column orderings."""
    prices: list[GasHubPrice] = []
    reader = csv.DictReader(io.StringIO(raw_text))
    if reader.fieldnames is None:
        return prices

    # Normalise column names: strip whitespace, lower
    norm_fields = {f.strip().lower(): f for f in (reader.fieldnames or [])}

    def _col(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in norm_fields:
                return norm_fields[c.lower()]
        return None

    date_col    = _col("gas_date", "settlement_date", "gas day", "gasday", "date")
    hub_col     = _col("hub", "hub_name", "location")
    price_col   = _col("price", "hub_price", "price_gj", "schedule_price", "ex_ante_price")
    volume_col  = _col("volume", "traded_qty", "quantity_gj")

    if not (date_col and hub_col and price_col):
        logger.debug("STTM CSV columns not recognised: %s", list(reader.fieldnames or []))
        return prices

    for row in reader:
        try:
            d = date.fromisoformat(row[date_col].strip().split("T")[0])
            if d < cutoff:
                continue
            hub = row[hub_col].strip().upper()
            price = float(row[price_col].strip())
            vol = float(row[volume_col].strip()) if volume_col and row.get(volume_col) else None
            region = _HUB_TO_REGION.get(hub, "UNKNOWN")
            prices.append(GasHubPrice(
                settlement_date=d,
                hub=hub,
                region=region,
                price_gj=price,
                volume_gj=vol,
                raw_ref=raw_ref,
            ))
        except (ValueError, KeyError):
            continue

    return prices


async def get_gas_market_state(region: str) -> GasMarketState:
    """Return current gas market state for a NEM region.

    Always returns a GasMarketState — falls back to ACCC static context
    when live STTM data is unavailable, so the causal chain explanation
    is always possible.
    """
    region = region.upper()
    prices = await fetch_sttm_hub_prices(days_back=14)

    # Filter to this region's hub prices, sorted by date desc
    regional = sorted(
        [p for p in prices if p.region == region],
        key=lambda p: p.settlement_date,
        reverse=True,
    )

    # If we have no regional data, try Wallumbilla (feeds all east coast)
    if not regional:
        regional = sorted(
            [p for p in prices if p.hub in ("WALLUMBILLA", "VIC HUB")],
            key=lambda p: p.settlement_date,
            reverse=True,
        )

    now = datetime.now(timezone.utc)

    if not regional:
        # No live data — return static context with ACCC lagged prices
        return GasMarketState(
            region=region,
            as_of=now,
            latest_hub_price_gj=_ACCC_STATIC_CONTEXT["lng_netback_gj_2024_q1"],
            hub_name="ACCC_LAGGED",
            srmc_ccgt=(
                _ACCC_STATIC_CONTEXT["lng_netback_gj_2024_q1"]
                * _HEAT_RATE_GJ_MWH["CCGT"] + _GAS_VARIABLE_OM
            ),
            srmc_ocgt=(
                _ACCC_STATIC_CONTEXT["lng_netback_gj_2024_q1"]
                * _HEAT_RATE_GJ_MWH["OCGT"] + _GAS_VARIABLE_OM
            ),
            recent_prices=[],
            lng_netback_gj=_ACCC_STATIC_CONTEXT["lng_netback_gj_2024_q1"],
            price_trend="unknown",
            high_price_alert=False,
            crisis_alert=False,
            source="ACCC_GAS_INQUIRY_Q1_2024",
            raw_ref=_ACCC_STATIC_CONTEXT["accc_source"],
        )

    latest = regional[0]
    price = latest.price_gj

    # Trend: compare to 7 days ago
    week_ago = [p for p in regional if p.settlement_date <= latest.settlement_date - timedelta(days=6)]
    if week_ago:
        old_price = week_ago[0].price_gj
        if price > old_price * 1.05:
            trend = "rising"
        elif price < old_price * 0.95:
            trend = "falling"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    return GasMarketState(
        region=region,
        as_of=now,
        latest_hub_price_gj=price,
        hub_name=latest.hub,
        srmc_ccgt=latest.srmc_ccgt,
        srmc_ocgt=latest.srmc_ocgt,
        recent_prices=regional[:14],
        lng_netback_gj=_ACCC_STATIC_CONTEXT["lng_netback_gj_2024_q1"],
        price_trend=trend,
        high_price_alert=price > 15.0,
        crisis_alert=price > 25.0,
        source="AEMO_STTM",
        raw_ref=_STTM_BASE,
    )


def get_causal_chain_text() -> str:
    """Return the static LNG→gas→NEM causal chain explanation text."""
    return _GAS_ELECTRICITY_CAUSAL_CHAIN.strip()


def get_accc_static_context() -> dict[str, Any]:
    """Return ACCC quarterly static context (used when live data unavailable)."""
    return dict(_ACCC_STATIC_CONTEXT)
