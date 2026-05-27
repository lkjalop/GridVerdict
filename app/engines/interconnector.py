"""Interconnector-aware causality engine (Sprint C).

Parses MarketDriverEvent rows of driver_type == "interconnector" and determines
whether interconnector flow constraints are a causal or contributing factor in
a regional price event.

NEM topology: five regions connected by six major interconnectors.
Positive flow convention follows AEMO's MMSDM: positive = flow in the
"from" → "to" direction (varies by interconnector naming).

Usage (called from assemble_why_sources):
    statuses = parse_interconnector_events(driver_events, region)
    ctx = classify_interconnector_causality(statuses, region, price_rrp, regime)

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── NEM interconnector topology ────────────────────────────────────────────────
# Maps lowercase element_id prefix → metadata.
# from_region → to_region when flow is positive.
_TOPOLOGY: dict[str, dict[str, str]] = {
    "nsw1-qld1":  {"from": "NSW1", "to": "QLD1", "name": "QNI"},
    "n-q-mnsp1":  {"from": "NSW1", "to": "QLD1", "name": "DirectLink"},
    "vic1-nsw1":  {"from": "VIC1", "to": "NSW1", "name": "VIC-NSW"},
    "v-sa":       {"from": "VIC1", "to": "SA1",  "name": "Heywood"},
    "v-s-mnsp1":  {"from": "SA1",  "to": "VIC1", "name": "Murraylink"},
    "t-v-mnsp1":  {"from": "TAS1", "to": "VIC1", "name": "Basslink"},
}


def _resolve_topology(element_id: str) -> dict[str, str] | None:
    key = (element_id or "").lower().strip()
    for prefix, meta in _TOPOLOGY.items():
        if key.startswith(prefix):
            return meta
    return None


@dataclass
class InterconnectorStatus:
    element_id: str
    name: str
    from_region: str
    to_region: str
    flow_mw: float                    # positive = from_region → to_region
    export_limit: float | None        # MW cap on positive flow
    import_limit: float | None        # MW cap on negative flow (stored as negative)
    utilisation_pct: float            # 0–1, how close to limit
    is_binding: bool                  # flow within 5% of the relevant limit
    binding_side: str                 # "export" | "import" | "none"
    affects_region: str               # which region's price this most directly affects
    causal_note: str                  # one-line human explanation
    valid_time: Any = None
    raw_ref: str = ""


@dataclass
class InterconnectorCausalityContext:
    statuses: list[InterconnectorStatus] = field(default_factory=list)
    causal_role: str = "unknown"          # "causal" | "contributing" | "not_relevant" | "unknown"
    binding_count: int = 0
    primary_note: str = ""                # top-level narrative sentence
    evidence_element: str = ""            # element_id for evidence_refs


def parse_interconnector_events(
    driver_events: list[dict[str, Any]],
    query_region: str,
) -> list[InterconnectorStatus]:
    """Parse driver_events for interconnector rows and classify each."""
    statuses: list[InterconnectorStatus] = []
    region_up = query_region.upper()

    for ev in driver_events:
        if ev.get("driver_type") != "interconnector":
            continue
        element_id = ev.get("element_id", "")
        values = ev.get("values", {}) or {}

        topo = _resolve_topology(element_id)
        if topo is None:
            continue

        flow_raw = values.get("mw_flow") or values.get("metered_mw_flow")
        if flow_raw is None:
            continue
        flow = float(flow_raw)

        export_limit = _float_or_none(values.get("export_limit"))
        import_limit = _float_or_none(values.get("import_limit"))

        # Compute utilisation and binding side
        binding_side, utilisation = _assess_binding(flow, export_limit, import_limit)
        is_binding = utilisation >= 0.95

        # Which region does this affect most?
        # If binding on export: supply to "to" region is constrained
        # If binding on import: "from" region is constrained on imports into "to"
        if binding_side == "export":
            affected = topo["to"]
        elif binding_side == "import":
            affected = topo["from"]
        else:
            # Determine based on flow direction
            affected = topo["to"] if flow > 0 else topo["from"]

        causal_note = _build_causal_note(
            topo["name"], topo["from"], topo["to"], flow, export_limit, import_limit,
            binding_side, utilisation, region_up,
        )

        statuses.append(InterconnectorStatus(
            element_id=element_id,
            name=topo["name"],
            from_region=topo["from"],
            to_region=topo["to"],
            flow_mw=round(flow, 1),
            export_limit=export_limit,
            import_limit=import_limit,
            utilisation_pct=round(utilisation, 3),
            is_binding=is_binding,
            binding_side=binding_side,
            affects_region=affected,
            causal_note=causal_note,
            valid_time=ev.get("valid_time"),
            raw_ref=ev.get("raw_ref", ""),
        ))

    # Sort: binding → high utilisation
    statuses.sort(key=lambda s: (-int(s.is_binding), -s.utilisation_pct))
    return statuses


def classify_interconnector_causality(
    statuses: list[InterconnectorStatus],
    query_region: str,
    price_rrp: float,
    regime: str,
) -> InterconnectorCausalityContext:
    """Determine whether interconnectors are a causal/contributing factor."""
    if not statuses:
        return InterconnectorCausalityContext(causal_role="unknown")

    region_up = query_region.upper()
    binding = [s for s in statuses if s.is_binding]
    region_relevant = [
        s for s in binding
        if s.from_region == region_up or s.to_region == region_up or s.affects_region == region_up
    ]

    if not region_relevant:
        return InterconnectorCausalityContext(
            statuses=statuses,
            causal_role="not_relevant",
            binding_count=len(binding),
        )

    # Causal = binding + elevated/spike/extreme regime + the binding side restricts supply into region
    supply_restricting = [
        s for s in region_relevant
        if s.affects_region == region_up and s.is_binding
    ]

    if supply_restricting and regime in ("elevated", "spike", "extreme"):
        role = "causal"
        top = supply_restricting[0]
        note = (
            f"{top.name} interconnector is at {top.binding_side} limit "
            f"({top.flow_mw:+.0f} MW, {top.utilisation_pct * 100:.0f}% utilisation) — "
            f"constraining supply flow to {region_up} and contributing to the price spike."
        )
    elif region_relevant:
        role = "contributing"
        top = region_relevant[0]
        note = (
            f"{top.name} is near its {top.binding_side} limit "
            f"({top.flow_mw:+.0f} MW, {top.utilisation_pct * 100:.0f}%), "
            f"which may be limiting imports into {region_up}."
        )
    else:
        role = "not_relevant"
        note = ""

    primary = supply_restricting[0] if supply_restricting else (region_relevant[0] if region_relevant else statuses[0])
    return InterconnectorCausalityContext(
        statuses=statuses,
        causal_role=role,
        binding_count=len(binding),
        primary_note=note,
        evidence_element=primary.element_id,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assess_binding(
    flow: float,
    export_limit: float | None,
    import_limit: float | None,
    tolerance: float = 0.05,
) -> tuple[str, float]:
    """Return (binding_side, utilisation_ratio)."""
    best_side = "none"
    best_util = 0.0

    if export_limit is not None and abs(export_limit) > 0 and flow > 0:
        util = flow / abs(export_limit)
        if util > best_util:
            best_util = util
            best_side = "export" if util >= (1.0 - tolerance) else "none"

    if import_limit is not None and abs(import_limit) > 0 and flow < 0:
        util = abs(flow) / abs(import_limit)
        if util > best_util:
            best_util = util
            best_side = "import" if util >= (1.0 - tolerance) else "none"

    if best_side == "none" and best_util == 0.0:
        # No limits stored — use flow magnitude as proxy
        best_util = min(abs(flow) / max(abs(flow), 1.0), 1.0)

    return best_side, min(best_util, 1.0)


def _build_causal_note(
    name: str,
    from_r: str,
    to_r: str,
    flow: float,
    export_limit: float | None,
    import_limit: float | None,
    binding_side: str,
    utilisation: float,
    query_region: str,
) -> str:
    direction = f"{from_r}→{to_r}" if flow >= 0 else f"{to_r}→{from_r}"
    util_pct = f"{utilisation * 100:.0f}%"
    if binding_side == "none":
        return f"{name}: {abs(flow):.0f} MW {direction} ({util_pct} utilisation, not binding)"
    side_word = "export" if binding_side == "export" else "import"
    return (
        f"{name}: {abs(flow):.0f} MW {direction}, at {side_word} limit "
        f"({util_pct} utilised) — supply to {query_region} constrained"
    )


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None
