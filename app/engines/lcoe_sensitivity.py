"""LCOE sensitivity analysis engine.

Computes Levelised Cost of Electricity (LCOE) for NEM-relevant technologies
using CSIRO GenCost 2023-24 parameters. The key use case is showing how
interest rates (discount rates) affect the competitiveness of capital-intensive
renewables vs fuel-intensive gas, answering questions like:

  "If interest rates rise 2%, what happens to solar vs coal costs?"
  "Which technology is cheapest to build right now in NSW?"
  "Why is renewable investment slowing?"

Formula:
  LCOE = (Capex × CRF + Fixed O&M) / Annual Generation + Variable O&M + Fuel Cost

  CRF (Capital Recovery Factor) = r(1+r)^n / ((1+r)^n - 1)
    r = discount rate (weighted average cost of capital)
    n = project life in years

Data sources:
  - Technology parameters: CSIRO GenCost 2023-24 (free, public Excel)
  - Discount rates: AEMO IASR 2024 + RBA cash rate + sector risk premia
  - Gas price input: links to gbb_client for live east coast gas price

Key insight: Solar/wind are ~75-85% capex → most rate-sensitive.
  At 5% WACC: solar utility LCOE ≈ $55/MWh
  At 8% WACC: solar utility LCOE ≈ $72/MWh (+31%)
  Coal (fully amortised fleet) not affected by new rates.
  New coal at 8% WACC: LCOE ≈ $160-220/MWh (economically unviable).

All values are indicative. CSIRO GenCost is the authoritative reference
for Australian technology cost data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── CSIRO GenCost 2023-24 technology parameters ───────────────────────────────
# Source: CSIRO GenCost 2023-24, Table 3.1 (indicative 2024 values)
# All costs in real 2023-24 A$. Capacity factors are Australian-typical.
#
# Explanation of fields:
#   capex_per_kw: overnight capital cost ($/kW installed capacity)
#   fom_per_kw_yr: fixed O&M ($/kW/year)
#   vom_per_mwh: variable O&M ($/MWh generated)
#   capacity_factor: fraction of time at full output (annualised)
#   life_yr: project design life
#   heat_rate_gj_mwh: fuel consumption (GJ per MWh output) — None for zero-fuel tech
#   fuel_type: "gas" | "coal" | None
#   capex_sensitivity: fraction of LCOE that is capex-driven (higher = more rate-sensitive)

GENTECH: dict[str, dict[str, Any]] = {
    "solar_utility": {
        "label": "Utility-scale solar PV",
        "capex_per_kw": 1_150,        # CSIRO GenCost 2023-24 central estimate
        "fom_per_kw_yr": 17,
        "vom_per_mwh": 0,
        "capacity_factor": 0.26,      # NSW/QLD average (latitude-adjusted)
        "life_yr": 25,
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.85,       # ~85% of LCOE is capex-driven
        "tech_notes": "Single-axis tracking, utility scale. CF varies: QLD 0.28, NSW 0.26, VIC 0.21",
    },
    "wind_onshore": {
        "label": "Onshore wind",
        "capex_per_kw": 2_100,
        "fom_per_kw_yr": 25,
        "vom_per_mwh": 0,
        "capacity_factor": 0.35,      # Good wind sites (New England, Gippsland, SA)
        "life_yr": 25,
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.80,
        "tech_notes": "High-quality wind sites. CF varies by location: 0.25-0.45",
    },
    "wind_offshore": {
        "label": "Offshore wind",
        "capex_per_kw": 4_800,        # much higher than onshore (foundations, cabling)
        "fom_per_kw_yr": 120,
        "vom_per_mwh": 0,
        "capacity_factor": 0.42,      # offshore typically higher CF
        "life_yr": 25,
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.88,
        "tech_notes": "Fixed-foundation offshore. Star of the South (VIC), Hunter (NSW)",
    },
    "gas_ccgt": {
        "label": "Gas CCGT",
        "capex_per_kw": 1_400,
        "fom_per_kw_yr": 15,
        "vom_per_mwh": 4,
        "capacity_factor": 0.85,      # baseload/mid-merit operation assumption
        "life_yr": 25,
        "heat_rate_gj_mwh": 6.5,     # combined-cycle efficiency ~46%
        "fuel_type": "gas",
        "capex_fraction": 0.35,       # only ~35% capex-driven; fuel cost dominates
        "tech_notes": "Combined-cycle gas turbine. SRMC = gas_price × 6.5 + 4 $/MWh",
    },
    "gas_ocgt": {
        "label": "Gas OCGT (peaker)",
        "capex_per_kw": 750,
        "fom_per_kw_yr": 10,
        "vom_per_mwh": 6,
        "capacity_factor": 0.15,      # peaker — low utilisation
        "life_yr": 25,
        "heat_rate_gj_mwh": 10.0,
        "fuel_type": "gas",
        "capex_fraction": 0.20,       # low capex fraction due to low CF
        "tech_notes": "Open-cycle peaker. Very low CF; high SRMC sets peak prices",
    },
    "coal_black_new": {
        "label": "New black coal (hypothetical)",
        "capex_per_kw": 4_500,        # CSIRO GenCost: very high for new coal
        "fom_per_kw_yr": 65,
        "vom_per_mwh": 8,
        "capacity_factor": 0.75,
        "life_yr": 30,
        "heat_rate_gj_mwh": 10.5,
        "fuel_type": "coal",
        "capex_fraction": 0.55,
        "tech_notes": (
            "No new coal is planned in Australia. LCOE included for scenario comparison only. "
            "At 8% WACC, new coal LCOE ≈ $160-220/MWh — not commercially viable."
        ),
    },
    "bess_2h": {
        "label": "Battery storage (2-hour)",
        "capex_per_kw": 1_200,        # includes inverter and balance-of-plant
        "fom_per_kw_yr": 12,
        "vom_per_mwh": 2,
        "capacity_factor": 0.20,      # ~2 cycles/day × 365 days
        "life_yr": 15,               # battery pack life before replacement
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.90,
        "tech_notes": "Li-ion 2-hour BESS. Revenue from arbitrage + FCAS co-optimisation",
    },
    "bess_4h": {
        "label": "Battery storage (4-hour)",
        "capex_per_kw": 1_800,
        "fom_per_kw_yr": 15,
        "vom_per_mwh": 2,
        "capacity_factor": 0.15,
        "life_yr": 15,
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.92,
        "tech_notes": "Li-ion 4-hour BESS. Suitable for evening peak shifting",
    },
    "solar_rooftop": {
        "label": "Rooftop solar (residential)",
        "capex_per_kw": 1_050,        # installed cost residential (smaller scale premium removed)
        "fom_per_kw_yr": 10,
        "vom_per_mwh": 0,
        "capacity_factor": 0.18,      # residential: less optimal tilt/orientation
        "life_yr": 25,
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.90,
        "tech_notes": "Residential 6.6kW system. LCOE relevant to FiT rate assessment",
    },
    "pumped_hydro": {
        "label": "Pumped hydro (new)",
        "capex_per_kw": 2_800,        # Snowy 2.0 revealed much higher costs
        "fom_per_kw_yr": 20,
        "vom_per_mwh": 1,
        "capacity_factor": 0.20,
        "life_yr": 50,               # very long life
        "heat_rate_gj_mwh": None,
        "fuel_type": None,
        "capex_fraction": 0.92,
        "tech_notes": "New build pumped hydro (Snowy 2.0 class). Long life reduces CRF impact",
    },
}

# Standard discount rates for NEM project analysis (AEMO IASR 2024 + sector premia)
STANDARD_DISCOUNT_RATES: dict[str, float] = {
    "rba_cash_rate_2024": 0.043,        # RBA cash rate May 2024
    "regulated_network": 0.055,          # AEMC-allowed WACC for regulated networks
    "merchant_renewable_low": 0.065,     # merchant project with PPA/RESS support
    "merchant_renewable_central": 0.080, # central merchant assumption (no subsidy)
    "merchant_renewable_high": 0.095,    # high-risk merchant (no subsidy, merchant basis)
    "merchant_gas": 0.090,              # gas projects (fuel risk adds premium)
    "merchant_coal_new": 0.120,         # hypothetical new coal (stranded asset risk)
    "aemo_isp_assumed": 0.067,          # AEMO IASR 2024 assumed project WACC
}


@dataclass
class LCOEResult:
    tech_key: str
    tech_label: str
    discount_rate: float
    gas_price_gj: float | None
    lcoe_per_mwh: float
    capex_component: float   # $/MWh from annualised capex
    fom_component: float     # $/MWh from fixed O&M
    vom_component: float     # $/MWh from variable O&M
    fuel_component: float    # $/MWh from fuel cost (0 for renewables)
    capacity_factor: float
    tech_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tech": self.tech_key,
            "label": self.tech_label,
            "discount_rate_pct": round(self.discount_rate * 100, 1),
            "gas_price_gj": self.gas_price_gj,
            "lcoe_mwh": round(self.lcoe_per_mwh, 1),
            "breakdown": {
                "capex_mwh": round(self.capex_component, 1),
                "fom_mwh": round(self.fom_component, 1),
                "vom_mwh": round(self.vom_component, 1),
                "fuel_mwh": round(self.fuel_component, 1),
            },
            "capacity_factor": self.capacity_factor,
            "tech_notes": self.tech_notes,
        }


def _capital_recovery_factor(rate: float, life_yr: int) -> float:
    """CRF = r(1+r)^n / ((1+r)^n - 1). Annualises capital cost over project life."""
    if rate <= 0:
        return 1.0 / life_yr
    return rate * (1 + rate) ** life_yr / ((1 + rate) ** life_yr - 1)


def compute_lcoe(
    tech_key: str,
    discount_rate: float,
    gas_price_gj: float | None = None,
    capacity_factor_override: float | None = None,
) -> LCOEResult:
    """Compute LCOE for a technology at a given discount rate.

    Args:
        tech_key: Key in GENTECH dict (e.g. "solar_utility", "gas_ccgt")
        discount_rate: WACC as decimal (e.g. 0.08 = 8%)
        gas_price_gj: East coast gas price $/GJ (required for gas technologies)
        capacity_factor_override: Override the default CF for location adjustment

    Returns LCOEResult with component breakdown.
    """
    tech = GENTECH.get(tech_key)
    if tech is None:
        raise ValueError(f"Unknown technology: {tech_key!r}. Valid: {list(GENTECH)}")

    cf = capacity_factor_override if capacity_factor_override is not None else tech["capacity_factor"]
    life = tech["life_yr"]
    capex_per_kw = tech["capex_per_kw"]
    fom_per_kw_yr = tech["fom_per_kw_yr"]
    vom_per_mwh = tech["vom_per_mwh"]

    # Annual generation per kW installed (MWh/kW/yr)
    annual_gen_mwh_per_kw = cf * 8760 / 1000

    if annual_gen_mwh_per_kw <= 0:
        raise ValueError(f"Capacity factor must be > 0 for {tech_key}")

    crf = _capital_recovery_factor(discount_rate, life)

    # $/MWh components
    capex_component = (capex_per_kw * crf) / annual_gen_mwh_per_kw
    fom_component = fom_per_kw_yr / annual_gen_mwh_per_kw
    vom_component = vom_per_mwh

    # Fuel cost
    fuel_component = 0.0
    if tech["heat_rate_gj_mwh"] is not None:
        effective_gas = gas_price_gj if gas_price_gj is not None else 10.0  # fallback: $10/GJ
        fuel_component = tech["heat_rate_gj_mwh"] * effective_gas

    lcoe = capex_component + fom_component + vom_component + fuel_component

    return LCOEResult(
        tech_key=tech_key,
        tech_label=tech["label"],
        discount_rate=discount_rate,
        gas_price_gj=gas_price_gj,
        lcoe_per_mwh=lcoe,
        capex_component=capex_component,
        fom_component=fom_component,
        vom_component=vom_component,
        fuel_component=fuel_component,
        capacity_factor=cf,
        tech_notes=tech.get("tech_notes", ""),
    )


def lcoe_sensitivity(
    tech_key: str,
    discount_rates: list[float],
    gas_price_gj: float | None = None,
) -> dict[str, Any]:
    """Compute LCOE at multiple discount rates — shows interest rate sensitivity.

    Returns a table showing how LCOE changes as rates rise/fall.
    """
    tech = GENTECH.get(tech_key)
    if tech is None:
        raise ValueError(f"Unknown technology: {tech_key!r}")

    results = []
    for rate in sorted(discount_rates):
        r = compute_lcoe(tech_key, rate, gas_price_gj)
        results.append(r.to_dict())

    # Sensitivity: change from lowest to highest rate
    if len(results) >= 2:
        lcoe_low = results[0]["lcoe_mwh"]
        lcoe_high = results[-1]["lcoe_mwh"]
        rate_low = results[0]["discount_rate_pct"]
        rate_high = results[-1]["discount_rate_pct"]
        delta = lcoe_high - lcoe_low
        delta_pct = (delta / lcoe_low * 100) if lcoe_low > 0 else 0
        sensitivity_note = (
            f"A {rate_high - rate_low:.0f}pp rise in discount rate "
            f"({rate_low:.0f}% → {rate_high:.0f}%) increases "
            f"{tech['label']} LCOE by ${delta:.0f}/MWh ({delta_pct:.0f}%). "
            f"Capex accounts for {tech['capex_fraction']*100:.0f}% of LCOE at this CF, "
            "explaining why capital-intensive technologies are most rate-sensitive."
        )
    else:
        sensitivity_note = "Provide at least two discount rates for sensitivity analysis."

    return {
        "tech": tech_key,
        "label": tech["label"],
        "capex_fraction": tech["capex_fraction"],
        "life_yr": tech["life_yr"],
        "rates_tested": results,
        "sensitivity_note": sensitivity_note,
        "source": "CSIRO GenCost 2023-24 (indicative 2024 A$)",
    }


def compare_technologies(
    discount_rate: float = 0.08,
    gas_price_gj: float = 10.0,
    techs: list[str] | None = None,
    exclude_coal_new: bool = True,
) -> dict[str, Any]:
    """Compare LCOE across technologies at a single discount rate.

    This is the primary function for "which technology is cheapest?" and
    "how does a rate change affect the technology stack?" questions.
    """
    if techs is None:
        techs = [k for k in GENTECH if not (exclude_coal_new and k == "coal_black_new")]

    results: list[dict] = []
    for key in techs:
        try:
            r = compute_lcoe(key, discount_rate, gas_price_gj)
            results.append(r.to_dict())
        except Exception:
            continue

    results.sort(key=lambda x: x["lcoe_mwh"])
    cheapest = results[0] if results else None
    most_expensive = results[-1] if results else None

    rate_label = STANDARD_DISCOUNT_RATES.get(
        min(STANDARD_DISCOUNT_RATES, key=lambda k: abs(STANDARD_DISCOUNT_RATES[k] - discount_rate)),
        f"{discount_rate*100:.0f}% WACC",
    )

    return {
        "discount_rate_pct": round(discount_rate * 100, 1),
        "gas_price_gj": gas_price_gj,
        "technologies": results,
        "cheapest": cheapest["label"] if cheapest else None,
        "cheapest_lcoe_mwh": cheapest["lcoe_mwh"] if cheapest else None,
        "most_expensive": most_expensive["label"] if most_expensive else None,
        "interpretation": (
            f"At {discount_rate*100:.0f}% WACC and ${gas_price_gj}/GJ gas, "
            f"{cheapest['label']} is cheapest at ${cheapest['lcoe_mwh']:.0f}/MWh. "
            "Note: LCOE is the cost to build new capacity — existing amortised coal "
            "has near-zero capex cost and competes only on SRMC ($15-30/MWh)."
        ) if cheapest else "No results.",
        "important_caveat": (
            "LCOE is the cost to build NEW generation. It does not directly determine "
            "spot prices (which are set by SRMC of the marginal unit). "
            "LCOE affects investment decisions → future supply mix → long-run prices."
        ),
        "source": "CSIRO GenCost 2023-24",
    }


def rate_change_impact(
    rate_from: float,
    rate_to: float,
    gas_price_gj: float = 10.0,
    techs: list[str] | None = None,
) -> dict[str, Any]:
    """Show LCOE impact of a rate change across all technologies.

    Key use case: "If RBA raises rates by 2pp, what happens to solar vs coal?"
    """
    if techs is None:
        techs = ["solar_utility", "wind_onshore", "gas_ccgt", "gas_ocgt",
                 "bess_2h", "pumped_hydro", "coal_black_new"]

    rows: list[dict] = []
    for key in techs:
        try:
            before = compute_lcoe(key, rate_from, gas_price_gj)
            after = compute_lcoe(key, rate_to, gas_price_gj)
            delta = after.lcoe_per_mwh - before.lcoe_per_mwh
            delta_pct = (delta / before.lcoe_per_mwh * 100) if before.lcoe_per_mwh > 0 else 0
            tech_meta = GENTECH[key]
            rows.append({
                "tech": key,
                "label": tech_meta["label"],
                "capex_fraction": tech_meta["capex_fraction"],
                "lcoe_before": round(before.lcoe_per_mwh, 1),
                "lcoe_after": round(after.lcoe_per_mwh, 1),
                "delta_mwh": round(delta, 1),
                "delta_pct": round(delta_pct, 1),
            })
        except Exception:
            continue

    rows.sort(key=lambda x: x["delta_pct"], reverse=True)
    most_affected = rows[0] if rows else None
    least_affected = rows[-1] if rows else None

    return {
        "rate_from_pct": round(rate_from * 100, 1),
        "rate_to_pct": round(rate_to * 100, 1),
        "rate_delta_pp": round((rate_to - rate_from) * 100, 1),
        "gas_price_gj": gas_price_gj,
        "technologies": rows,
        "most_rate_sensitive": most_affected["label"] if most_affected else None,
        "least_rate_sensitive": least_affected["label"] if least_affected else None,
        "interpretation": (
            f"A {abs(rate_to - rate_from)*100:.0f}pp {'rise' if rate_to > rate_from else 'cut'} "
            f"in discount rate ({rate_from*100:.0f}% → {rate_to*100:.0f}%) most affects "
            f"{most_affected['label']} (+${most_affected['delta_mwh']}/MWh, "
            f"+{most_affected['delta_pct']:.0f}%). "
            f"Gas CCGT is least affected ({rows[-1]['delta_pct']:+.0f}%) because "
            "fuel cost — not capex — dominates its LCOE."
        ) if rows else "No results.",
        "mechanism_note": (
            "Interest rates affect LCOE (new investment cost), not SRMC (current spot price). "
            "Higher rates → renewables less competitive → slower buildout → "
            "higher long-run prices (5-15 year lag). Short-run spot prices are unaffected."
        ),
        "source": "CSIRO GenCost 2023-24",
    }
