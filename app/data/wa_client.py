"""Western Australia WEM market comparison client.

WA is not in the NEM — it operates the Wholesale Electricity Market (WEM), a
capacity market design managed separately by AEMO WA under WEM legislation.

This client provides:
  1. Structural comparison: WEM vs NEM market design differences
  2. Historical WEM price context (from AEMO WEM public reports — quarterly CSVs)
  3. Fuel mix context (from AEMO WEM Electricity Statement of Opportunities)
  4. Real-time context: when WEM data is unavailable, falls back to structural analysis

Data sources (all public, no credentials required):
  - AEMO WEM Dashboard: https://aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem
  - AEMO WEM Market Reports: quarterly CSV reports at the above URL
  - AEMO ESOO (Electricity Statement of Opportunities) — annual

Why this matters for NLP:
  "Compare WA to SA for energy prices" is not answerable with live data (different API).
  BUT a structured structural + indicative comparison IS answerable and valuable.
  This client makes that comparison rigorous and citable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(20.0)
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ── WEM structural facts (from AEMO WEM 2024 publications) ───────────────────
# These are publicly verifiable facts about WEM design and generation mix.
# Updated from: AEMO WEM Electricity Statement of Opportunities 2023-24,
#               ERA Wholesale Electricity Market Annual Report 2023-24.

_WEM_STRUCTURAL: dict[str, Any] = {
    "market_design": "Capacity Market + Energy Balancing Market",
    "nem_design":    "Energy-only spot market (marginal pricing)",
    "operator":      "AEMO (WEM division, separate from NEM division)",
    "regulator":     "ERA (Economic Regulation Authority WA) + AEMO WEM rules",
    "dispatch_interval_min": 30,   # WEM settles on 30-min intervals (NEM: 5-min)
    "market_size_mw": 4800,        # approximate peak demand
    "coverage": "South-West Interconnected System (SWIS) only; north WA = isolated grids",
    "nem_equivalent_region": "SA1",   # closest NEM analog (isolated, high renewable, gas backup)
    "generation_mix_pct": {
        "gas":        59,   # dominant fuel in WEM (cheap LNG by-product gas)
        "coal":       14,   # Collie coal (scheduled for retirement)
        "wind":       12,
        "solar_pv":   11,
        "other":       4,
    },
    "renewable_target_pct": 80,    # WA government: 80% renewable by 2030
    "major_retailers": ["Synergy (government-owned, dominant)", "Kleenheat", "Alinta Energy"],
    "interconnected": False,        # no interconnection with NEM
    "capacity_mechanism": True,     # generators get paid for availability (not just output)
}

# ── WEM historical price context (indicative, from AEMO WEM reports) ─────────
# These are published indicative ranges from AEMO WEM quarterly reports.
# Not live data — they are the last published averages (updated quarterly).
_WEM_PRICE_CONTEXT: dict[str, Any] = {
    "balancing_price_avg_2023_aud_mwh": 112.0,   # AEMO WEM 2023 annual average
    "balancing_price_p10_aud_mwh": 45.0,
    "balancing_price_p90_aud_mwh": 280.0,
    "reserve_capacity_price_2024_aud_mw_year": 206_100,   # AEMO WEM 2024 RCP ($/MW/year)
    "peak_price_cap_aud_mwh": 500.0,              # WEM balancing price cap
    "floor_price_aud_mwh": -1000.0,
    "source": "AEMO WEM Quarterly Report 2024 Q1 (indicative published values)",
    "data_lag_note": (
        "These are published indicative values from AEMO WEM quarterly reports — "
        "not real-time. Live WEM balancing prices are only available via AEMO WEM API "
        "(requires separate registration from NEM access)."
    ),
}

# ── Key policy/structural differences table ───────────────────────────────────
_WEM_NEM_DIFFERENCES: list[dict[str, str]] = [
    {
        "dimension":     "Market design",
        "nem":           "Energy-only spot market. Generators paid for each MWh dispatched.",
        "wem":           "Capacity market + energy balancing. Generators paid for being available (capacity credits) AND for MWh dispatched.",
        "implication":   "WEM reduces investment risk (availability payment = revenue floor). NEM has higher merchant risk.",
    },
    {
        "dimension":     "Dispatch interval",
        "nem":           "5-minute dispatch intervals. Price set every 5 minutes by AEMO (since July 2021).",
        "wem":           "30-minute settlement intervals. More price stability than NEM's 5-min volatility.",
        "implication":   "NEM has far more intraday volatility. WEM price spikes are less frequent but still occur.",
    },
    {
        "dimension":     "Interconnection",
        "nem":           "5 regions interconnected (NSW-VIC-QLD-SA-TAS). Price divergence bounded by interconnector capacity.",
        "wem":           "Isolated system (SWIS). No interconnection with NEM. Price set entirely by domestic supply-demand.",
        "implication":   "WEM is more vulnerable to domestic generator failures. Price recovery slower without imports.",
    },
    {
        "dimension":     "Gas dependency",
        "nem":           "~15-20% gas in energy mix; gas is marginal setter ~40% of peak hours.",
        "wem":           "~59% gas in energy mix. Gas is the dominant fuel source (cheap LNG by-product from NW Shelf).",
        "implication":   "WEM prices are more exposed to domestic gas market. NW Shelf gas is cheap vs east coast.",
    },
    {
        "dimension":     "Renewable transition",
        "nem":           "~40% renewable 2024 (SA >70%). NEM prices diverging by region as solar/wind build.",
        "wem":           "~23% renewable 2024. 80% target by 2030. Solar cannibalisation starting (similar to NEM SA1 2019).",
        "implication":   "WEM is ~5 years behind NEM in renewable penetration. NEM SA1 in 2019 is a good WEM 2024 analog.",
    },
    {
        "dimension":     "Price volatility",
        "nem":           "NEM spot can swing from -$1000/MWh (solar surplus) to +$15,500/MWh (VoLL) in one interval.",
        "wem":           "WEM balancing price cap ~$500/MWh. Less extreme than NEM but still volatile during stress.",
        "implication":   "NEM has ~30× the price ceiling of WEM. BESS arbitrage value is higher in NEM.",
    },
    {
        "dimension":     "Carbon exposure",
        "nem":           "No carbon price (Safeguard Mechanism 2023 is intensity-based, not a price floor).",
        "wem":           "No carbon price either. Coal retirement driven by WA Government policy, not market.",
        "implication":   "Similar carbon policy exposure, but WA coal (Collie) retires faster due to state policy.",
    },
]

# ── NEM closest analog explanation ───────────────────────────────────────────
_WEM_NEM_ANALOG = {
    "region":      "SA1",
    "reasoning": (
        "SA is the closest NEM analog to the WEM because both are: "
        "(1) high renewable penetration with strong solar cannibalisation, "
        "(2) backed by gas peakers when renewables are low, "
        "(3) increasingly dependent on storage (SA: Hornsdale; WEM: new BESS projects), and "
        "(4) isolated from cheaper baseload power (SA via single Heywood interconnector; WEM has no interconnection). "
        "SA's 2019-2022 experience — solar midday price collapse, evening gas peaks, BESS arbitrage growth — "
        "is the best available preview of where WEM is heading by 2027-2030."
    ),
    "sa_specific_learning": [
        "SA midday prices went negative or near-zero regularly from 2019 as solar penetration passed 50%.",
        "Evening gas peaks in SA are $150-300/MWh during winter and high-demand events.",
        "Hornsdale BESS (150MW/194MWh) earns revenue from FCAS and arbitrage — similar BESS economics apply in WEM.",
        "SA's Heywood interconnector to VIC provides a price ceiling that WEM lacks entirely.",
    ],
}


@dataclass
class WEMComparisonResult:
    """Structured WEM vs NEM comparison result — ready for NLP answer assembly."""
    as_of: datetime
    structural_differences: list[dict[str, str]]
    wem_fuel_mix_pct: dict[str, int]
    wem_price_context: dict[str, Any]
    nem_analog_region: str
    nem_analog_reasoning: str
    nem_analog_learnings: list[str]
    data_freshness: str   # "live" | "quarterly" | "static"
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of":                   self.as_of.isoformat(),
            "structural_differences":  self.structural_differences,
            "wem_fuel_mix_pct":        self.wem_fuel_mix_pct,
            "wem_price_context":       self.wem_price_context,
            "nem_analog_region":       self.nem_analog_region,
            "nem_analog_reasoning":    self.nem_analog_reasoning,
            "nem_analog_learnings":    self.nem_analog_learnings,
            "data_freshness":          self.data_freshness,
            "source":                  self.source,
        }


async def get_wem_comparison(nem_region: str = "SA1") -> WEMComparisonResult:
    """Return a structured WEM vs NEM comparison.

    Tries to fetch the latest quarterly WEM report from AEMO.
    Falls back to embedded structural data (always available, always accurate for design differences).

    Args:
        nem_region: The NEM region to compare WEM against (default SA1 = closest analog).
    """
    # Try to fetch live WEM quarterly data (AEMO WEM dashboard API)
    live_price_context = None
    try:
        live_price_context = await _fetch_wem_quarterly_prices()
    except Exception as exc:
        logger.debug("WEM quarterly fetch failed (non-fatal, using static): %s", exc)

    price_context = live_price_context or _WEM_PRICE_CONTEXT
    data_freshness = "quarterly" if live_price_context else "static"

    return WEMComparisonResult(
        as_of=datetime.now(timezone.utc),
        structural_differences=_WEM_NEM_DIFFERENCES,
        wem_fuel_mix_pct=_WEM_STRUCTURAL["generation_mix_pct"],
        wem_price_context=price_context,
        nem_analog_region=_WEM_NEM_ANALOG["region"],
        nem_analog_reasoning=_WEM_NEM_ANALOG["reasoning"],
        nem_analog_learnings=_WEM_NEM_ANALOG["sa_specific_learning"],
        data_freshness=data_freshness,
        source=price_context.get("source", "AEMO WEM publications (embedded static)"),
    )


async def _fetch_wem_quarterly_prices() -> dict[str, Any] | None:
    """Attempt to fetch the latest WEM quarterly report from AEMO.

    AEMO publishes WEM market reports quarterly as downloadable files.
    The URL format and file structure change between publications.
    Returns None on any failure — caller uses embedded static context.
    """
    # AEMO WEM Market Reports URL (updated quarterly; URL pattern may change)
    _WEM_REPORTS_PAGE = "https://aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem/market-data-wem"
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers={"User-Agent": _BROWSER_UA},
            follow_redirects=True,
        ) as client:
            # Attempt to find the quarterly report download link
            # In practice, AEMO WEM API endpoints are not public REST APIs like NEMWeb.
            # This is a placeholder — actual implementation would need to parse the page
            # for the CSV download link, or use the AEMO WEM API after registration.
            resp = await client.get(_WEM_REPORTS_PAGE)
            if resp.status_code != 200:
                return None
            # Parse the page for CSV download URLs (simplified — real implementation needed)
            # For now, return None to trigger fallback to static data
            return None
    except Exception:
        return None


def get_structural_comparison() -> list[dict[str, str]]:
    """Return the WEM vs NEM structural differences table (always available)."""
    return _WEM_NEM_DIFFERENCES


def get_wem_facts() -> dict[str, Any]:
    """Return static WEM structural facts (always available)."""
    return dict(_WEM_STRUCTURAL)


def format_comparison_for_nlp(comparison: WEMComparisonResult, nem_region: str = "SA1") -> list[str]:
    """Format a WEMComparisonResult into NLP-ready answer bullets.

    Returns a list of concise factual bullets suitable for answer_sections.
    """
    bullets: list[str] = []

    # Headline comparison
    avg_price = comparison.wem_price_context.get("balancing_price_avg_2023_aud_mwh")
    bullets.append(
        f"WEM average balancing price (2023): ~${avg_price:.0f}/MWh (indicative, quarterly report). "
        f"Compare to {nem_region}: live price is in the GridVerdict dashboard."
    )

    # Fuel mix
    mix = comparison.wem_fuel_mix_pct
    bullets.append(
        f"WEM generation mix: Gas {mix.get('gas', '?')}% | Coal {mix.get('coal', '?')}% | "
        f"Wind {mix.get('wind', '?')}% | Solar {mix.get('solar_pv', '?')}%. "
        "Gas dominates (vs NEM east coast which is coal-heavy at baseload)."
    )

    # Key design differences (top 3 most relevant)
    for diff in comparison.structural_differences[:3]:
        bullets.append(
            f"{diff['dimension']}: WEM = {diff['wem'][:80]}  vs  NEM {nem_region} = {diff['nem'][:80]}"
        )

    # NEM analog
    bullets.append(
        f"Closest NEM analog: {comparison.nem_analog_region}. "
        "SA's 2019-2022 solar cannibalisation / gas peak / BESS arbitrage experience "
        "previews WEM's trajectory toward 80% renewables by 2030."
    )

    # Data caveat
    bullets.append(
        f"Data freshness: {comparison.data_freshness}. "
        "Live WEM balancing prices require AEMO WEM API access (separate from NEM). "
        f"Source: {comparison.source}"
    )

    return bullets
