"""Adjacent query handlers — structured partial-scope and evidence-bridge responses.

Each handler corresponds to a requested_output value produced by the decomposer
when it classifies a query as PARTIAL_SCOPE, EVIDENCE_BRIDGE, or GEOGRAPHIC_REDIRECT.

Design contract:
  - Every handler returns a dict matching the FactualVerdict answer_sections schema.
  - Handlers ALWAYS explain what was answered AND what wasn't, and why.
  - No handler claims SUPPORTED — all are capped at PARTIAL_SCOPE.
  - Every answer includes a redirect_resource pointing to the right external source.

Handler registry (matched by decomp.requested_output):
  "solar_household_context"        → handle_solar_household
  "renewable_investment_price_context" → handle_investment_price_context
  "geographic_redirect"            → handle_geographic_redirect
  "policy_evidence_bridge"         → handle_policy_bridge
  "macro_mechanism_bridge"         → handle_macro_bridge
  "gas_electricity_nexus"          → handle_gas_nexus
  "fiscal_budget_bridge"           → handle_fiscal_bridge
  "evidence_bridge"                → handle_generic_evidence_bridge
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Shared structures ─────────────────────────────────────────────────────────

_PARTIAL_SCOPE_DISCLAIMER = (
    "GridVerdict answered the NEM-relevant portion of your question using "
    "live AEMO data and public reference data. The sections marked "
    "'Outside GridVerdict scope' require external data sources."
)


def _make_section(title: str, items: list[str]) -> dict[str, Any]:
    return {"title": title, "items": items}


def _scope_boundary_section(answerable: str, unanswerable: str, redirect: str) -> dict[str, Any]:
    return _make_section("Scope boundary", [
        f"GridVerdict answered: {answerable}",
        f"Outside GridVerdict scope: {unanswerable}",
        f"For the rest: {redirect}",
    ])


# ── B2: Household / rooftop solar ────────────────────────────────────────────

async def handle_solar_household(
    decomp: Any,
    region: str,
    db: Any | None = None,
) -> dict[str, Any]:
    """Answer household solar revenue questions using NEM wholesale price signal.

    Answerable: solar-window spot price distribution, cannibalisation trend,
    seasonal patterns, evening peak (battery value case).
    Not answerable: actual FiT rate, STC rebate, equipment cost, payback period.
    """
    region = region.upper()
    adjacent = decomp.adjacent_context or {}

    # Fetch historical price distribution for solar-window hours (9am-3pm)
    # Try to pull from DB; fall back to indicative values if unavailable.
    solar_window_price: dict[str, Any] = {}
    try:
        if db is not None:
            from app.engines.historical_price import get_historical_price_distribution
            from datetime import datetime, timezone
            solar_window_price = await get_historical_price_distribution(
                db, region, datetime.now(timezone.utc), period="last_year"
            ) or {}
    except Exception as exc:
        logger.debug("Solar household: historical price fetch failed (non-fatal): %s", exc)

    median_price = solar_window_price.get("median", 70)
    p10 = solar_window_price.get("p10", 20)
    p90 = solar_window_price.get("p90", 180)

    # Cannibalisation context (solar noon suppression)
    cannibalisation_context = {
        "NSW1": "NSW midday prices have declined ~15-25% over 2021-2024 as solar penetration increased, compressing the value of solar exports around 11am-1pm.",
        "QLD1": "QLD experiences significant solar cannibalisation — midday spot prices regularly go negative in summer when solar output peaks. Export value is highest in mornings (8-10am) and evenings (5-7pm).",
        "SA1": "SA has the highest solar penetration (>70% of demand at midday in summer). Regular negative prices 10am-2pm mean solar export value is concentrated in shoulder periods.",
        "VIC1": "VIC is lower irradiance than QLD/NSW but solar cannibalisation is growing. Midday prices are typically lower than morning/evening peaks.",
        "TAS1": "TAS has minimal solar penetration and is hydro-dominated. Solar cannibalisation is not yet a significant factor.",
    }.get(region, "Solar cannibalisation varies by region — check current dispatch data.")

    sections = [
        _make_section("What GridVerdict can tell you about solar economics", [
            f"Wholesale spot price signal for {region} (the reference signal retailers use for FiT rates)",
            "Solar generation window (9am-3pm) price distribution — this is when your panels export",
            "Solar cannibalisation trend — how rising solar penetration is compressing midday prices",
            "Evening peak pricing (5-8pm) — relevant if you add battery storage",
        ]),
        _make_section(f"{region} wholesale price context (NEM spot, last 12 months)", [
            f"Median: ~${median_price:.0f}/MWh  |  P10: ~${p10:.0f}/MWh  |  P90: ~${p90:.0f}/MWh",
            "FiT rates paid by retailers are typically 5-15% of the wholesale price (retailer margin absorbed)",
            "Your roof solar exports during the solar window — this is the price signal that matters",
        ]),
        _make_section("Solar cannibalisation — the key risk for rooftop solar returns", [
            cannibalisation_context,
            "As more households add solar, midday prices fall. Early adopters earned higher effective FiT rates.",
            "Adding battery storage shifts export to the 5-8pm peak window where prices are 2-4× higher.",
        ]),
        _make_section("Evening peak opportunity (battery value case)", [
            f"Evening peak ({region} 5-8pm) spot prices are typically 2-4× midday prices in summer",
            "A 10kWh battery storing solar-noon energy and discharging at 6pm earns the spread",
            "This is the economic case for solar + battery (not solar alone) in high-penetration states",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "NEM wholesale price signal, cannibalisation trend"),
            unanswerable=adjacent.get("unanswerable_sub", "Actual FiT rates, STC rebates, installation costs"),
            redirect=adjacent.get("redirect_resource", "AER energy.gov.au, Clean Energy Regulator"),
        ),
        _make_section("What GridVerdict cannot answer", [
            "Actual feed-in tariff (FiT) rates — set by individual retailers, not AEMO",
            "Small-scale Technology Certificate (STC) rebate value — federal scheme (CER)",
            "Equipment + installation costs — contact solar installers for quotes",
            "Payback period and NPV — depends on your usage, retailer rate, and equipment costs",
        ]),
        _make_section("Next steps", [
            "Check current FiT rates: AER comparison tool at energy.gov.au/households/solar-panels",
            "STC calculator: cleanenergyregulator.gov.au (for upfront rebate estimate)",
            "For live NEM spot context: ask GridVerdict 'What is the NSW spot price right now?'",
            "For battery case: ask GridVerdict 'What is the evening peak price in NSW this week?'",
        ]),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            f"Your question about rooftop solar economics has a NEM-answerable part and an outside-scope part. "
            f"GridVerdict can show you the wholesale spot price signal in {region} — this is the market "
            "signal that retailers reference when setting feed-in tariff (FiT) rates. "
            "What we cannot provide are the actual FiT rates (set by retailers), STC rebate values "
            "(set by the federal government), or equipment costs. "
            f"The key insight: solar cannibalisation is real — midday prices in {region} have "
            "been declining as solar penetration grows, which reduces the value of solar-only exports. "
            "Battery storage shifts export to the higher-value evening peak window."
        ),
        "missing_data": [
            "Retailer-specific FiT rates (not AEMO data)",
            "STC certificate value (Clean Energy Regulator)",
            "Equipment and installation costs",
        ],
        "upgrade_path": [
            "Ask GridVerdict: 'What is the NSW spot price right now?' for live context",
            "Ask GridVerdict: 'How does the NSW midday price compare to the evening peak?' for battery case",
        ],
        "adjacent_context": adjacent,
    }


# ── B6: Commercial renewable investment price context ─────────────────────────

async def handle_investment_price_context(
    decomp: Any,
    region: str,
    db: Any | None = None,
) -> dict[str, Any]:
    """Answer commercial wind/solar investment questions using spot + ISP data.

    Answerable: spot distributions, ISP scenario price trajectories, capture rate analysis.
    Not answerable: LCOE, IRR/NPV, grid connection costs, PPA pricing.
    """
    region = region.upper()
    adjacent = decomp.adjacent_context or {}

    # Fetch ISP scenario for the region
    isp_result = None
    try:
        from app.mcp.isp_client import get_isp_scenario
        isp_result = get_isp_scenario(region, year_from=2025, year_to=2040, scenario="Step Change")
    except Exception as exc:
        logger.debug("ISP fetch failed (non-fatal): %s", exc)

    # Historical price distribution
    hist_price: dict[str, Any] = {}
    try:
        if db is not None:
            from app.engines.historical_price import get_historical_price_distribution
            from datetime import datetime, timezone
            hist_price = await get_historical_price_distribution(
                db, region, datetime.now(timezone.utc), period="last_year"
            ) or {}
    except Exception as exc:
        logger.debug("Investment price: historical fetch failed (non-fatal): %s", exc)

    # ISP price trajectory bullets
    isp_bullets = []
    if isp_result and isp_result.price_points:
        for pt in isp_result.price_points[:4]:
            isp_bullets.append(
                f"{pt['year']}: P10 ${pt['p10']}/MWh — P50 ${pt['p50']}/MWh — P90 ${pt['p90']}/MWh "
                f"(real 2023-24 A$, Step Change scenario)"
            )
    else:
        isp_bullets = [
            "ISP price data unavailable for this query.",
            "Source: aemo.com.au/isp — download IASR 2024 Excel appendices for live trajectory data.",
        ]

    # Coal retirements relevant to this region
    coal_bullets = []
    if isp_result:
        for ret in isp_result.coal_retirements:
            coal_bullets.append(
                f"{ret['unit']} ({ret['capacity_mw']}MW) — planned closure {ret['planned_closure']} "
                f"— {ret['status']}"
            )

    # REZ pipeline
    rez_bullets = []
    if isp_result:
        for rez in isp_result.rez_pipeline:
            rez_bullets.append(
                f"{rez['rez']}: {rez['capacity_gw']}GW {rez['tech']} — status: {rez['status']}"
            )

    sections = [
        _make_section(f"{region} spot price context (NEM wholesale, last 12 months)", [
            f"Median: ~${hist_price.get('median', '—')}/MWh",
            f"P10 (cheap periods): ~${hist_price.get('p10', '—')}/MWh",
            f"P90 (expensive periods): ~${hist_price.get('p90', '—')}/MWh",
            "These are the gross wholesale revenues your farm would receive before network/retailer costs",
        ]),
        _make_section("AEMO ISP 2024 Step Change — long-run price trajectory", isp_bullets + [
            "CAVEAT: These are real long-run equilibrium values, NOT short-run spot forecasts.",
            "Actual spot prices vary widely around these trajectories — P90/P10 spread shows the range.",
            f"Source: AEMO ISP 2024, real 2023-24 A$/MWh ({isp_result.data_source if isp_result else 'embedded static'})",
        ]),
        _make_section("Capture rate risk — solar and wind price suppression", [
            "As more solar/wind is built, prices fall during high-generation periods — reducing revenue per MWh.",
            "A solar farm built today captures today's prices. By 2030, solar-window prices are likely lower.",
            "This 'capture rate decline' is the key risk for new renewable investment.",
            "QLD and SA already show strong solar cannibalisation — NSW is following the same trajectory.",
        ]),
        _make_section(f"Planned coal retirements (capacity market context for {region})", coal_bullets or ["No major coal retirements in this region in the ISP window"]),
        _make_section(f"Renewable Energy Zone pipeline ({region})", rez_bullets or ["No major REZ in this region in the ISP pipeline"]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "Spot distributions, ISP trajectories, capture rates"),
            unanswerable=adjacent.get("unanswerable_sub", "LCOE, IRR, grid connection, PPA pricing"),
            redirect=adjacent.get("redirect_resource", "AEMO ISP 2024, CSIRO GenCost 2023-24, CEFC"),
        ),
        _make_section("What GridVerdict cannot answer", [
            "LCOE (levelised cost of electricity) — requires technology-specific capex quotes",
            "IRR / NPV calculation — requires your specific financing structure",
            "Grid connection cost — requires TNSP connection study (AusNet, TransGrid, ElectraNet, etc.)",
            "PPA (Power Purchase Agreement) pricing — commercial negotiation with an offtake partner",
            "RESS/LTESA eligibility — NSW state government program (dpie.nsw.gov.au)",
        ]),
        _make_section("Where to get the full investment analysis", [
            "AEMO ISP 2024: aemo.com.au/isp (Step Change scenario Excel appendices)",
            "CSIRO GenCost 2023-24: csiro.au (free download — technology cost benchmarks)",
            "Clean Energy Finance Corporation: cefc.com.au (financing for renewable projects)",
            "Ask GridVerdict: 'Why is the NSW spot price elevated right now?' for live market context",
        ]),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            f"Your wind/solar investment question has an answerable NEM part and an outside-scope part. "
            f"GridVerdict can show you: current {region} wholesale spot price distributions (your gross revenue signal), "
            "AEMO ISP 2024 Step Change long-run price trajectories (directional 10-year signal, NOT a forecast), "
            "and the capture rate risk from solar penetration. "
            "For LCOE, IRR, grid connection costs, and PPA modelling, you need AEMO ISP appendices, "
            "CSIRO GenCost, and commercial advisers."
        ),
        "missing_data": [
            "LCOE (requires CSIRO GenCost + your specific capex)",
            "Grid connection cost (requires TNSP connection study)",
            "PPA/offtake pricing (commercial negotiation)",
        ],
        "upgrade_path": [
            "For live ISP data: download AEMO IASR 2024 and parse with isp_client.parse_iasr_excel()",
            "Ask GridVerdict: 'What is the current price in NSW?' for real-time market context",
        ],
        "adjacent_context": adjacent,
    }


# ── B3: Geographic redirect (WA, NT, NZ) ─────────────────────────────────────

async def handle_geographic_redirect(
    decomp: Any,
    region: str = "NSW1",
    db: Any | None = None,
) -> dict[str, Any]:
    """Explain non-NEM market structure and offer the nearest NEM equivalent.

    For WA queries: uses wa_client.py to fetch structured WEM comparison data
    (fuel mix, indicative price context, structural differences table).
    Falls back to embedded static facts on any error.
    """
    adjacent = decomp.adjacent_context or {}
    geo = decomp.geographic_market or "UNKNOWN"

    # ── WA: use wa_client for richer structured comparison ────────────────────
    if geo == "WA_WEM":
        return await _handle_wa_comparison(decomp, region, adjacent)

    # Market structure explanation by geography (NT, NZ, unknown)
    market_explanations = {
        "WA_WEM": {
            "title": "Western Australia — Wholesale Electricity Market (WEM)",
            "overview": (
                "WA operates the WEM (Wholesale Electricity Market) — a fundamentally different "
                "design from the NEM. Key differences:"
            ),
            "differences": [
                "DESIGN: WEM is a Capacity Market + Energy Balancing Market. The NEM is an energy-only spot market.",
                "WEM CAPACITY MARKET: Generators get paid for being available (capacity credits), regardless of output. NEM generators only get paid when they dispatch.",
                "WEM BALANCING PRICE: The equivalent of the NEM spot price is the 'balancing price' — but it's set differently, with more bilateral contract markets.",
                "GENERATION MIX: WA is ~60% gas (cheap because of LNG by-product gas), ~15% coal (Collie), ~25% renewables. Very different from NEM east coast mix.",
                "REGULATOR: ERA (Economic Regulation Authority WA) not AER. Market rules by AEMO but separate WEM rules.",
                "MARKET OPERATOR: AEMO also operates the WEM but under separate WEM legislation.",
            ],
            "nem_equivalent": (
                "The closest NEM analog is SA — high renewable penetration, gas backup, "
                "isolated from the rest of the NEM by a single interconnector. "
                "SA spot price dynamics (solar cannibalisation, gas peaker dependency) "
                "give a preview of where WA is heading."
            ),
            "resource": "AEMO WEM: aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem",
        },
        "NT_GRID": {
            "title": "Northern Territory — Darwin-Katherine Interconnected System",
            "overview": "The NT operates an isolated grid with no connection to the NEM or WEM.",
            "differences": [
                "OPERATOR: Power and Water Corporation (government-owned utility), not AEMO.",
                "MARKET: Not a competitive wholesale market — essentially a regulated monopoly.",
                "GENERATION MIX: Predominantly gas (Territory Generation), growing solar.",
                "RETAIL: Jacana Energy (government-owned) is the dominant retailer.",
                "REGULATION: NT Utilities Commission, not AER.",
            ],
            "nem_equivalent": (
                "TAS (Tasmania) is the closest NEM analog — isolated grid (before Basslink), "
                "historically dominated by a single technology (hydro for TAS, gas for NT). "
                "TAS post-Basslink is now NEM-connected; NT has no interconnection."
            ),
            "resource": "Power and Water Corporation: powerwater.com.au",
        },
        "NZ_GRID": {
            "title": "New Zealand — Wholesale Electricity Market",
            "overview": "NZ operates a separate wholesale market under the Electricity Authority.",
            "differences": [
                "OPERATOR: Electricity Authority NZ + System Operator (Transpower).",
                "DESIGN: Similar to NEM (energy-only spot market, nodal pricing).",
                "GENERATION: ~80% renewable (hydro 60%, geothermal 15%, wind 7%).",
                "PRICING: Nodal (locational marginal pricing) vs NEM's regional pricing.",
                "CURRENCY: NZ$ not A$.",
            ],
            "nem_equivalent": (
                "TAS is the closest NEM analog — high hydro penetration, "
                "price sensitive to rainfall/storage levels."
            ),
            "resource": "Electricity Authority NZ: ea.govt.nz",
        },
    }

    market = market_explanations.get(geo, {
        "title": f"Non-NEM market: {geo}",
        "overview": "This market is not covered by GridVerdict.",
        "differences": ["No data available in GridVerdict for this market."],
        "nem_equivalent": "Ask GridVerdict about a NEM region instead.",
        "resource": "AEMO: aemo.com.au",
    })

    sections = [
        _make_section(market["title"], [
            market["overview"],
        ]),
        _make_section("Key differences from the NEM", market["differences"]),
        _make_section("Closest NEM equivalent", [
            market["nem_equivalent"],
            "Ask GridVerdict about this NEM region for comparable live data and analysis.",
        ]),
        _make_section("What GridVerdict CAN answer instead", [
            f"Live NEM spot prices and dispatch for NSW1, VIC1, QLD1, SA1, TAS1",
            "Causal explanations of NEM price events",
            "Historical analogs and scenario analysis for NEM regions",
            "Comparison: 'How does SA differ from NSW in terms of renewable penetration?'",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "NEM market structure and closest NEM equivalent"),
            unanswerable=adjacent.get("unanswerable_sub", f"Live data for {geo} (not an NEM market)"),
            redirect=adjacent.get("redirect_resource", market.get("resource", "See the relevant market operator")),
        ),
        _make_section("External resource for your market", [
            market.get("resource", "See AEMO: aemo.com.au"),
        ]),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            f"Your question is about a market outside the NEM (Australian National Electricity Market). "
            f"GridVerdict only covers the five NEM regions (NSW1, VIC1, QLD1, SA1, TAS1). "
            f"The {market['title']} operates under different rules, design, and data. "
            f"I've explained the key differences and the closest NEM analog above. "
            f"For live data on your market, see: {market.get('resource', 'the relevant market operator')}."
        ),
        "missing_data": [f"Live data for {geo} (not an NEM market)"],
        "upgrade_path": [
            "Ask: 'How does SA compare to NSW in terms of renewable penetration?'",
            "Ask: 'What is the current spot price in SA?' (closest to WA market dynamics)",
        ],
        "adjacent_context": adjacent,
    }


# ── B4: Policy evidence bridge ────────────────────────────────────────────────

def handle_policy_bridge(decomp: Any, region: str = "NSW1") -> dict[str, Any]:
    """Answer policy counterfactual questions with NEM evidence + mechanism explanation."""
    region = region.upper()
    adjacent = decomp.adjacent_context or {}

    # Load ISP scenario comparison (Step Change vs Slow Change) for policy framing
    scenario_comparison = None
    try:
        from app.mcp.isp_client import compare_scenarios, get_coal_retirement_schedule
        scenario_comparison = compare_scenarios(region, year=2030)
        retirements = get_coal_retirement_schedule(region)
    except Exception:
        retirements = []

    historical_closure_impacts = [
        "Hazelwood closure (2017, VIC, 1600MW): Victorian baseload prices rose ~$30-50/MWh within 12 months. NEM-wide baseload increased ~$15-25/MWh.",
        "Liddell units 3&4 closure (2023, NSW, 1000MW): NSW prices saw elevated spikes in winter 2023, partially attributed to reduced headroom.",
        "Northern Power Station closure (2016, SA, 520MW): SA already high-renewable; no acute price step-change but reduced inertia services.",
        "Anglesea closure (2015, VIC, 150MW): Minimal market impact due to small size and VIC's coal surplus at the time.",
    ]

    build_time_note = (
        "A new coal plant takes approximately 7-10 years from investment decision to commissioning. "
        "Under current market conditions (low coal economics, high construction costs, stranded asset risk), "
        "no financier is funding new coal in Australia. The LCOE for new black coal at 8% WACC is "
        "~$160-220/MWh — far above current NEM prices. Any 'coal investment policy' would require "
        "direct government subsidy (comparable to the capacity mechanisms in some US states)."
    )

    scenario_bullets = []
    if scenario_comparison:
        delta = scenario_comparison.get("slow_vs_step_delta_mwh", "—")
        scenario_bullets = [
            f"AEMO Step Change (planned decarbonisation): {region} 2030 price ~${scenario_comparison.get('step_change_p50', '—')}/MWh (real 2023-24 A$)",
            f"AEMO Slow Change (slower transition, more fossil): {region} 2030 price ~${scenario_comparison.get('slow_change_p50', '—')}/MWh",
            f"Slow Change premium vs Step Change: +${delta}/MWh — this is the cost of delayed decarbonisation",
            "Source: AEMO ISP 2024, real 2023-24 A$/MWh",
        ]
    else:
        scenario_bullets = [
            "Load AEMO ISP 2024 IASR Excel for region-specific scenario comparison.",
            "Step Change (central) vs Slow Change (more fossil) shows ~$15-25/MWh 2030 premium for slower transition.",
        ]

    sections = [
        _make_section("What GridVerdict can tell you from NEM evidence", [
            "How coal currently shapes NEM dispatch (regional % of dispatch intervals)",
            "Historical evidence from past coal capacity closures → price impacts",
            "AEMO ISP scenario comparison (Step Change vs Slow Change) for long-run price trajectory",
            "Current coal technology SRMC and its role as marginal price setter",
        ]),
        _make_section("Historical evidence: coal capacity → price relationship", historical_closure_impacts),
        _make_section("AEMO ISP 2024 policy scenario comparison", scenario_bullets),
        _make_section("Build time and economics — why new coal investment takes years to affect prices", [
            build_time_note,
            "Short-run price impact of a new coal investment announcement: near zero (7-10 year lag).",
            "Long-run impact (if actually built): increased baseload supply → lower overnight prices, unchanged peak prices.",
        ]),
        _make_section(f"Planned coal retirements ({region})", [
            f"{r['unit']} ({r['capacity_mw']}MW) — {r['planned_closure']} — {r['status']}"
            for r in retirements
        ] or ["No coal retirements scheduled in this region in ISP window."]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "Historical evidence, ISP scenarios, current dispatch"),
            unanswerable=adjacent.get("unanswerable_sub", "Long-run equilibrium modelling (requires ISP/Plexos)"),
            redirect=adjacent.get("redirect_resource", "AEMO ISP 2024 Progressive Change and Slow Change scenarios"),
        ),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            "Your policy question has an evidence-grounded NEM part and a modelling-dependent part. "
            "GridVerdict can show: how coal currently sets prices in the NEM, what history shows about "
            "capacity changes affecting prices (Hazelwood +$30-50/MWh in VIC), and how AEMO's "
            "Slow Change scenario (more fossil, slower transition) compares to Step Change. "
            "For quantified equilibrium price modelling under a new coal policy, AEMO ISP is the reference — "
            "Step Change vs Slow Change is exactly this comparison."
        ),
        "missing_data": [
            "Quantified equilibrium price impact of specific policy (requires ISP capacity expansion model)",
            "Policy implementation timeline and funding mechanism",
        ],
        "upgrade_path": [
            "Ask GridVerdict: 'What is the current coal dispatch share in NSW?'",
            "Ask GridVerdict: 'What happened to Victorian prices after Hazelwood closed?'",
        ],
        "adjacent_context": adjacent,
    }


# ── B5: Macro mechanism bridge (interest rates) ───────────────────────────────

def handle_macro_bridge(decomp: Any, region: str = "NSW1") -> dict[str, Any]:
    """Answer interest rate / cost of capital questions with LCOE evidence."""
    region = region.upper()
    adjacent = decomp.adjacent_context or {}

    # Compute LCOE at standard rate scenarios
    rate_impact = None
    tech_comparison = None
    try:
        from app.engines.lcoe_sensitivity import rate_change_impact, compare_technologies
        rate_impact = rate_change_impact(
            rate_from=0.065,  # merchant renewable low (with support)
            rate_to=0.095,    # merchant renewable high (no support)
            gas_price_gj=10.0,
        )
        tech_comparison = compare_technologies(discount_rate=0.08, gas_price_gj=10.0)
    except Exception as exc:
        logger.debug("LCOE computation failed (non-fatal): %s", exc)

    # LCOE sensitivity bullets
    lcoe_bullets = []
    if rate_impact and rate_impact.get("technologies"):
        for t in rate_impact["technologies"][:5]:
            lcoe_bullets.append(
                f"{t['label']}: ${t['lcoe_before']}/MWh → ${t['lcoe_after']}/MWh "
                f"({t['delta_pct']:+.0f}%) — capex fraction {t.get('capex_fraction', '?'):.0%}"
            )
        lcoe_bullets.append(rate_impact.get("interpretation", ""))
    else:
        lcoe_bullets = [
            "Solar utility: at 6.5% WACC ~$55/MWh → at 9.5% WACC ~$75/MWh (+36%)",
            "Wind onshore: at 6.5% ~$75/MWh → at 9.5% ~$100/MWh (+33%)",
            "Gas CCGT: at 6.5% ~$95/MWh → at 9.5% ~$102/MWh (+7%) — fuel cost dominates",
            "Coal (new): at 6.5% ~$145/MWh → at 9.5% ~$185/MWh (+28%) — already uneconomic",
        ]

    tech_comparison_bullets = []
    if tech_comparison and tech_comparison.get("technologies"):
        for t in tech_comparison["technologies"][:5]:
            tech_comparison_bullets.append(
                f"{t['label']}: ${t['lcoe_mwh']:.0f}/MWh (at 8% WACC, $10/GJ gas)"
            )
    else:
        tech_comparison_bullets = [
            "Utility solar: ~$65/MWh (most rate-sensitive due to 85% capex fraction)",
            "Onshore wind: ~$85/MWh",
            "Gas CCGT: ~$95/MWh (least rate-sensitive — fuel cost dominates)",
            "New coal: ~$165/MWh (not economically viable at any plausible rate)",
        ]

    sections = [
        _make_section("SRMC vs LCOE — why rates don't affect TODAY's spot prices", [
            "Spot prices are set by Short-Run Marginal Cost (SRMC) — fuel + variable O&M. No capex.",
            "SRMC is not affected by interest rates. Today's gas CCGT SRMC is ~$65-90/MWh regardless of rates.",
            "LCOE is the Levelised Cost to BUILD NEW generation. This IS affected by interest rates.",
            "The rate-price linkage is indirect and long-run: rates → LCOE → investment decision → supply mix → prices (5-15 year lag).",
        ]),
        _make_section("LCOE sensitivity to discount rate — from CSIRO GenCost 2023-24", lcoe_bullets),
        _make_section("Technology LCOE comparison (at 8% WACC, $10/GJ gas)", tech_comparison_bullets),
        _make_section("Which technologies are most rate-sensitive?", [
            "Solar PV (utility): ~85% capex-driven → most rate-sensitive (each 1pp adds ~$2-3/MWh LCOE)",
            "Wind: ~80% capex-driven → second most rate-sensitive",
            "Battery storage: ~90% capex-driven → very rate-sensitive (but revenue from FCAS partly offsets)",
            "Gas CCGT: ~35% capex-driven → least rate-sensitive (fuel cost dominates at $10/GJ gas)",
            "Existing coal fleet: fully amortised → interest rate changes don't affect their SRMC",
        ]),
        _make_section("Long-run mechanism: how rates affect NEM prices", [
            "Higher rates → renewables LCOE rises → fewer projects economically viable → slower buildout",
            "Slower renewable buildout → coal/gas stays in the mix longer → higher long-run prices",
            "This effect materialises over 5-15 years, not immediately in spot prices",
            "The 2022-2024 rate rises (RBA from 0.1% to 4.35%) will slow the 2027-2030 renewable pipeline",
            "AEMO's Slow Change scenario captures this: more gas, higher 2030-2035 prices",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "LCOE sensitivity, SRMC structure, mechanism explanation"),
            unanswerable=adjacent.get("unanswerable_sub", "Quantified equilibrium price impact (requires capacity expansion model)"),
            redirect=adjacent.get("redirect_resource", "AEMO ISP 2024, CSIRO GenCost 2023-24, RBA Statement on Monetary Policy"),
        ),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            "Interest rates don't directly affect today's NEM spot price — they affect the cost of "
            "building NEW generation (LCOE), which feeds into investment decisions, which affects "
            "the future supply mix, which then affects long-run prices (5-15 year lag). "
            "GridVerdict can show you the LCOE sensitivity using CSIRO GenCost data: "
            "solar/wind are ~80-85% capex-driven and most rate-sensitive. "
            "Gas is ~35% capex-driven and least rate-sensitive. "
            "For quantified long-run price impact, AEMO ISP Slow Change vs Step Change shows "
            "the policy scenario comparison that bounds this effect."
        ),
        "missing_data": [
            "Quantified equilibrium NEM price impact of sustained rate change (requires ISP model)",
            "Forward interest rate curve beyond RBA published expectations",
        ],
        "upgrade_path": [
            "Ask GridVerdict: 'Which technology is setting the NSW price right now?'",
            "Ask GridVerdict: 'What is the current AEMO ISP scenario comparison?'",
        ],
        "adjacent_context": adjacent,
    }


# ── C2 handler: Gas-electricity nexus ────────────────────────────────────────

async def handle_gas_nexus(
    decomp: Any,
    region: str = "NSW1",
    db: Any | None = None,
) -> dict[str, Any]:
    """Answer LNG/gas market questions with the causal chain to electricity prices."""
    region = region.upper()
    adjacent = decomp.adjacent_context or {}

    # Fetch current gas market state
    gas_state = None
    try:
        from app.mcp.gbb_client import get_gas_market_state, get_causal_chain_text
        gas_state = await get_gas_market_state(region)
        causal_text = get_causal_chain_text()
    except Exception as exc:
        logger.debug("Gas nexus: GBB fetch failed (non-fatal): %s", exc)
        causal_text = (
            "JKM (Asian LNG) → domestic gas price (Wallumbilla/STTM hub) → "
            "gas generator SRMC → NEM spot price"
        )

    gas_bullets = []
    if gas_state:
        gas_bullets = [
            f"Current hub price: ${gas_state.latest_hub_price_gj:.2f}/GJ ({gas_state.hub_name or 'hub'}, {gas_state.source})",
            f"Implied CCGT SRMC: ${gas_state.srmc_ccgt:.0f}/MWh",
            f"Implied OCGT peaker SRMC: ${gas_state.srmc_ocgt:.0f}/MWh",
            f"Price trend: {gas_state.price_trend}",
            f"Alert: {'⚠ HIGH PRICE (>$15/GJ)' if gas_state.high_price_alert else 'Normal range'}",
        ]
    else:
        gas_bullets = [
            "Live STTM data unavailable — ACCC Gas Inquiry Q1 2024 context used",
            "LNG netback (ACCC Q1 2024): ~$10.20/GJ",
            "Implied CCGT SRMC at $10.20/GJ: ~$70/MWh",
            "Normal domestic gas range (pre-2021): $5-8/GJ; crisis peak (2022): $30/GJ",
        ]

    sections = [
        _make_section("The LNG → gas → electricity causal chain", causal_text.split("\n") if causal_text else ["See adjacent_context"]),
        _make_section(f"Current east coast gas market ({region})", gas_bullets),
        _make_section("Why the 2022 energy crisis happened (evidence-grounded)", [
            "1. Russia-Ukraine war (Feb 2022) → European LNG demand spike → JKM hit $70/MMBtu",
            "2. Australian LNG netback rose to equivalent of $25-30/GJ domestic gas",
            "3. Gas generators' SRMC rose to $400-600/MWh",
            "4. NEM hit market price cap ($15,500/MWh) for multiple intervals",
            "5. AEMO invoked administered pricing (capped prices) — triggered by sustained SRMC > $300/MWh",
            "6. AEMO emergency directions: gas generators told to offer below cost (first time since 2008)",
        ]),
        _make_section("How much of NEM peak prices are set by gas generators?", [
            "~40% of NEM peak demand hours have gas generators as the marginal price setter",
            "Coal is marginal for baseload (overnight, shoulder periods)",
            "Gas (OCGT) is marginal for peak demand (5-8pm, hot days)",
            "Renewables set negative prices during midday solar surplus hours",
            "This is why gas price spikes translate directly into NEM peak price spikes",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "STTM gas hub prices, SRMC calculation, historical nexus"),
            unanswerable=adjacent.get("unanswerable_sub", "Real-time JKM (requires Platts/ICIS subscription)"),
            redirect=adjacent.get("redirect_resource", "ACCC Gas Inquiry (accc.gov.au), AEMO GBB (gbb.aemo.com.au)"),
        ),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            f"Your question is about the LNG/gas market's impact on electricity prices in {region}. "
            "GridVerdict can explain and evidence the causal chain: Asian LNG prices set the floor "
            "for Australian domestic gas, which feeds directly into gas generator SRMCs, "
            "which set NEM peak prices about 40% of the time. "
            "The 2022 crisis is the clearest recent example of this chain. "
            "For real-time JKM (Asian LNG spot), you'd need a Platts/ICIS subscription."
        ),
        "missing_data": [
            "Real-time JKM (Asian LNG spot price) — requires Platts or ICIS subscription",
            "Forward gas contract prices (AEMO gas market futures)",
        ],
        "adjacent_context": adjacent,
    }


# ── Fiscal budget bridge ──────────────────────────────────────────────────────

async def handle_fiscal_bridge(
    decomp: Any,
    region: str = "NSW1",
    db: Any | None = None,
) -> dict[str, Any]:
    """Answer government budget / energy spending questions using budget RAG."""
    adjacent = decomp.adjacent_context or {}

    # Try to get budget context from fiscal_budget_client
    budget_context = None
    try:
        from app.mcp.fiscal_budget_client import search_budget_measures
        query_text = decomp.raw_query or ""
        budget_context = await search_budget_measures(query_text)
    except Exception as exc:
        logger.debug("Fiscal bridge: budget search failed (non-fatal): %s", exc)

    budget_bullets = []
    if budget_context and budget_context.get("measures"):
        for m in budget_context["measures"][:5]:
            budget_bullets.append(
                f"{m.get('program', '?')}: {m.get('amount', '?')} — {m.get('description', '?')}"
            )
    else:
        budget_bullets = [
            "ARENA (Australian Renewable Energy Agency): $2.4B over 10 years (2023-24 budget)",
            "CEFC (Clean Energy Finance Corporation): $20B extended mandate (2023)",
            "Rewiring the Nation: $20B for transmission infrastructure",
            "Capacity Investment Scheme: 23GW of new dispatchable capacity backed by government",
            "Hydrogen Headstart: $2B for green hydrogen production",
        ]

    sections = [
        _make_section("Energy measures in the Australian Government Budget", budget_bullets),
        _make_section("How budget spending affects NEM prices", [
            "ARENA/CEFC funding reduces project risk → lower WACC → lower LCOE → faster renewable buildout",
            "CIS (Capacity Investment Scheme) backs 23GW new capacity — most significant demand-side support",
            "Transmission investment (Rewiring the Nation) unlocks constrained renewable zones",
            "Hydrogen Headstart doesn't directly affect electricity prices (different market)",
            "Budget spending works through LCOE → investment → supply mix → long-run prices (5-15 year lag)",
        ]),
        _make_section("What GridVerdict can tell you from NEM data", [
            "Current renewable investment pipeline visible in AEMO market notices",
            "Constraint relief as new transmission is built (binding constraint frequency over time)",
            "Live NEM spot price context for any region",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "Budget energy measures, mechanism explanation"),
            unanswerable=adjacent.get("unanswerable_sub", "Detailed fiscal modelling, program effectiveness"),
            redirect=adjacent.get("redirect_resource", "budget.gov.au, arena.gov.au, cefc.com.au"),
        ),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            "Your budget/fiscal question relates to how government energy spending affects the NEM. "
            "GridVerdict can identify the major energy programs and explain the mechanism by which "
            "they affect electricity prices (through LCOE reduction → investment → supply mix). "
            "For detailed fiscal analysis and program effectiveness, the official budget papers "
            "(budget.gov.au) and agency reports (ARENA, CEFC) are the authoritative sources."
        ),
        "missing_data": [
            "Detailed program effectiveness data (requires Treasury/agency evaluation reports)",
            "Forward budget estimates beyond 4-year forward estimates",
        ],
        "adjacent_context": adjacent,
    }


# ── Generic evidence bridge fallback ─────────────────────────────────────────

def handle_generic_evidence_bridge(decomp: Any, region: str = "NSW1") -> dict[str, Any]:
    """Generic handler for EVIDENCE_BRIDGE queries without a specific handler."""
    adjacent = decomp.adjacent_context or {}

    sections = [
        _make_section("What GridVerdict can tell you from NEM evidence", [
            adjacent.get("answerable_sub", "Live NEM dispatch, historical data, causal attribution"),
        ]),
        _make_section("NEM evidence available", [
            "Live spot prices for all 5 NEM regions",
            "Historical dispatch data and causal attribution",
            "AEMO market notices and constraint events",
            "Weather correlation with demand/renewables",
        ]),
        _scope_boundary_section(
            answerable=adjacent.get("answerable_sub", "NEM market data and causal analysis"),
            unanswerable=adjacent.get("unanswerable_sub", "External market data or modelling"),
            redirect=adjacent.get("redirect_resource", "See AEMO: aemo.com.au"),
        ),
    ]

    return {
        "sections": sections,
        "why_plain_english": (
            "Your question bridges the NEM and an external market or mechanism. "
            "GridVerdict can provide the NEM-side evidence. "
            f"{adjacent.get('bridge_mechanism', 'See the scope boundary above for what else is needed.')}"
        ),
        "missing_data": [adjacent.get("unanswerable_sub", "External data source")],
        "adjacent_context": adjacent,
    }


async def _handle_wa_comparison(
    decomp: Any,
    region: str,
    adjacent: dict[str, Any],
) -> dict[str, Any]:
    """WA NEM-vs-WEM comparison using wa_client structured data.

    Provides: fuel mix, indicative price context, structural differences table,
    NEM analog reasoning, and NLP-ready answer bullets.
    """
    comparison = None
    try:
        from app.data.wa_client import get_wem_comparison, format_comparison_for_nlp
        comparison = await get_wem_comparison(nem_region="SA1")
        bullets = format_comparison_for_nlp(comparison, nem_region="SA1")
    except Exception as exc:
        logger.debug("WA comparison client failed (using static fallback): %s", exc)
        bullets = [
            "WEM (WA): Capacity Market + Energy Balancing Market. NEM: energy-only spot market.",
            "WEM fuel mix: ~59% gas, ~14% coal, ~23% renewable. NEM: ~40% renewable (varies by region).",
            "WEM balancing price cap: ~$500/MWh vs NEM VoLL: $15,500/MWh.",
            "WEM dispatch interval: 30 min. NEM: 5 min (more volatile).",
            "WEM has no interconnection to NEM. SA1 is isolated by a single line — the closest NEM analog.",
        ]
        comparison = None

    # Structural differences table — always available
    diff_bullets = []
    if comparison:
        for diff in comparison.structural_differences[:5]:
            diff_bullets.append(
                f"{diff['dimension']}: "
                f"WEM = {diff['wem'][:100]}  |  "
                f"NEM ({region}) = {diff['nem'][:100]}"
            )
    else:
        diff_bullets = [
            "Market design: WEM = capacity market (availability payments). NEM = energy-only (dispatch payments).",
            "Dispatch: WEM = 30-min settlement. NEM = 5-min dispatch since 2021.",
            "Interconnection: WEM = isolated SWIS. NEM = 5-region interconnected grid.",
            "Gas role: WEM = ~59% gas (dominant). NEM = ~15% gas (peaker-only).",
            "Price cap: WEM = ~$500/MWh. NEM = $15,500/MWh (VoLL) — 30x more volatile.",
        ]

    analog_reasoning = comparison.nem_analog_reasoning if comparison else (
        "SA1 (South Australia) is the closest NEM analog: both are isolated systems with high renewable "
        "penetration backed by gas peakers. SA's 2019-2022 solar cannibalisation trajectory previews "
        "where WEM is heading toward its 80% renewable target by 2030."
    )
    analog_learnings = comparison.nem_analog_learnings if comparison else [
        "SA midday prices collapsed to near-zero once solar exceeded 50% of demand.",
        "Evening gas peaks in SA reach $150-300/MWh during winter demand peaks.",
        "Hornsdale BESS (SA) earns strong FCAS + arbitrage revenue — same opportunity exists in WEM.",
    ]

    sections = [
        _make_section("WEM vs NEM — structural comparison", bullets),
        _make_section("Key design differences by dimension", diff_bullets),
        _make_section("Closest NEM analog: SA1 (South Australia)", [
            analog_reasoning,
        ]),
        _make_section("What SA1's experience tells us about WEM's future", analog_learnings),
        _make_section("Queryable NEM questions that inform WA decisions", [
            "Ask: 'What is the current SA spot price?' — SA is the closest live analog",
            "Ask: 'Why is SA price elevated right now?' — gas peaker dependency (same as WEM)",
            "Ask: 'What is the evening peak price pattern in SA?' — WEM will follow similar curve",
            "Ask: 'How does SA renewable penetration compare to NSW?' — SA shows WEM's future trajectory",
        ]),
        _scope_boundary_section(
            answerable="WEM market structure, fuel mix, indicative price context, NEM analog analysis",
            unanswerable="Live WEM balancing prices, WEM market notices, WEM FCAS (requires AEMO WEM API)",
            redirect="AEMO WEM: aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem",
        ),
    ]

    data_note = (
        f"Source: {comparison.source} (freshness: {comparison.data_freshness})"
        if comparison else "Source: embedded structural data from AEMO WEM 2024 publications"
    )

    return {
        "sections": sections,
        "why_plain_english": (
            "Your question is about the WEM (Western Australia's Wholesale Electricity Market), "
            "which is not part of the NEM. GridVerdict provides a rigorous structural comparison: "
            "WEM is a capacity market (generators paid for availability) vs the NEM's energy-only spot market. "
            "WEM runs on 30-min dispatch intervals vs NEM's 5-min. WEM is ~59% gas vs NEM's ~15%. "
            "The closest NEM analog is SA1 — high renewables, isolated system, gas peaker backup. "
            "SA's experience since 2019 is a leading indicator for where WEM is heading. "
            f"{data_note}."
        ),
        "missing_data": [
            "Live WEM balancing price (requires AEMO WEM API — separate from NEM NEMWeb access)",
            "WEM FCAS prices and market notices (available through WEM API)",
        ],
        "upgrade_path": [
            "Ask: 'What is the current SA spot price?' — best live NEM proxy for WEM dynamics",
            "Ask: 'Why is SA price elevated right now?' — identical supply-stack logic to WEM",
            "For live WEM data: register at AEMO WEM portal (aemo.com.au)",
        ],
        "adjacent_context": adjacent,
    }


# ── Router ────────────────────────────────────────────────────────────────────

HANDLER_REGISTRY = {
    "solar_household_context":           handle_solar_household,
    "renewable_investment_price_context": handle_investment_price_context,
    "geographic_redirect":               handle_geographic_redirect,
    "policy_evidence_bridge":            handle_policy_bridge,
    "macro_mechanism_bridge":            handle_macro_bridge,
    "gas_electricity_nexus":             handle_gas_nexus,
    "fiscal_budget_bridge":              handle_fiscal_bridge,
    "evidence_bridge":                   handle_generic_evidence_bridge,
}


async def dispatch_adjacent_handler(
    decomp: Any,
    region: str,
    db: Any | None = None,
) -> dict[str, Any]:
    """Route a decomposition to the appropriate adjacent handler.

    Returns a structured dict with 'sections', 'why_plain_english',
    'missing_data', and 'upgrade_path' keys.
    """
    requested = decomp.requested_output or "evidence_bridge"
    handler = HANDLER_REGISTRY.get(requested)

    if handler is None:
        logger.warning("No adjacent handler for requested_output=%r — using generic", requested)
        handler = handle_generic_evidence_bridge

    try:
        # Async handlers need await; sync handlers don't
        import asyncio
        import inspect
        if inspect.iscoroutinefunction(handler):
            result = await handler(decomp, region, db)
        else:
            result = handler(decomp, region)
        return result
    except Exception as exc:
        logger.error("Adjacent handler %r failed: %s", requested, exc)
        return {
            "sections": [_make_section("Partial answer unavailable", [
                "The adjacent query handler encountered an error.",
                f"requested_output={requested!r}",
                str(exc),
            ])],
            "why_plain_english": f"Adjacent handler failed for {requested}: {exc}",
            "missing_data": ["Handler error — see logs"],
        }
