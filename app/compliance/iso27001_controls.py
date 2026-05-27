"""ISO/IEC 27001:2022 Annex A control mapping for GridVerdict SecurityObserver signals.

Maps each SecurityObserver signal name to the Annex A control it evidences.
This mapping lets audit logs carry machine-readable compliance artefacts and
lets the compliance report endpoint generate a 27001 control coverage summary.

Controls covered are from ISO/IEC 27001:2022 Annex A (113 controls across
4 organisational themes: Organisational, People, Physical, Technological).
"""
from __future__ import annotations

# ── Signal → Annex A control ──────────────────────────────────────────────────
# Keyed by ObserverSignal.name; value is the Annex A clause reference.

_SIGNAL_TO_CONTROL: dict[str, str] = {
    # Pass 1 — Input sanitisation
    "prompt_injection":            "A.8.28",  # Secure coding
    "manipulation_signal":         "A.8.28",  # Secure coding (adversarial input)
    "pii_in_input":                "A.5.34",  # Privacy and protection of PII
    "oos_topic":                   "A.8.7",   # Protection against malware / input validation
    "excessive_length":            "A.8.20",  # Networks security (resource exhaustion)
    "unicode_anomaly":             "A.8.28",  # Secure coding (encoding/homoglyph attacks)
    # Pass 2 — Decomposition validation
    "unsafe_execution_intent":     "A.5.29",  # Information security during disruption
    "oos_intent":                  "A.8.7",   # Input validation
    "very_low_decomp_confidence":  "A.8.7",   # Input validation (ambiguous/garbled query)
    "low_decomp_confidence":       "A.8.7",   # Input validation
    "portfolio_data_requested":    "A.5.10",  # Acceptable use of information and assets
    # Pass 3 — Tool output validation
    "tool_output_injection":       "A.8.22",  # Controlling access to application system
    "price_anomaly":               "A.8.7",   # Input/output validation
    "demand_anomaly":              "A.8.7",
    "availability_anomaly":        "A.8.7",
    "weather_temperature_anomaly": "A.8.7",
    "weather_wind_anomaly":        "A.8.7",
    # Pass 4 — Answer validation
    "unsupported_claim":           "A.5.36",  # Compliance with policies, rules, standards
    "missing_disclaimer":          "A.5.36",  # Compliance
    "overconfidence":              "A.5.36",  # Compliance
    "missing_counterargument":     "A.5.36",  # Compliance
    "null_evidence_value":         "A.5.36",  # Compliance
}

# ── Annex A control descriptions ──────────────────────────────────────────────

_CONTROL_METADATA: dict[str, dict[str, str]] = {
    "A.5.10": {
        "title": "Acceptable use of information and other associated assets",
        "theme": "Organisational",
        "objective": "Ensure information and assets are used appropriately and not misused.",
    },
    "A.5.29": {
        "title": "Information security during disruption",
        "theme": "Organisational",
        "objective": "Protect information security when operations are disrupted.",
    },
    "A.5.34": {
        "title": "Privacy and protection of personally identifiable information",
        "theme": "Organisational",
        "objective": "Ensure privacy and protection of PII as required by applicable legislation.",
    },
    "A.5.36": {
        "title": "Compliance with policies, rules and standards for information security",
        "theme": "Organisational",
        "objective": "Ensure information security and its implementation comply with organisational policies and procedures.",
    },
    "A.8.7": {
        "title": "Protection against malware",
        "theme": "Technological",
        "objective": "Protect against malware; includes input/output validation controls.",
    },
    "A.8.20": {
        "title": "Networks security",
        "theme": "Technological",
        "objective": "Protect networks and network services against threats, including DoS/resource exhaustion.",
    },
    "A.8.22": {
        "title": "Segregation in networks",
        "theme": "Technological",
        "objective": "Separate groups of information services, users, and systems in networks.",
    },
    "A.8.28": {
        "title": "Secure coding",
        "theme": "Technological",
        "objective": "Ensure software is written securely; includes injection prevention and input encoding.",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_control_ref(signal_name: str) -> str | None:
    """Return the Annex A control reference for a given observer signal name."""
    return _SIGNAL_TO_CONTROL.get(signal_name)


def get_control_metadata(control_ref: str) -> dict | None:
    """Return title, theme, and objective for an Annex A control reference."""
    meta = _CONTROL_METADATA.get(control_ref)
    if meta is None:
        return None
    return {"control_ref": control_ref, **meta}


def get_all_controls() -> list[dict]:
    """Return all covered Annex A controls with coverage status."""
    covered = set(_SIGNAL_TO_CONTROL.values())
    return [
        {
            "control_ref": ref,
            **_CONTROL_METADATA[ref],
            "covered_by_observer": ref in covered,
            "covering_signals": [
                s for s, c in _SIGNAL_TO_CONTROL.items() if c == ref
            ],
        }
        for ref in sorted(_CONTROL_METADATA)
    ]


def get_coverage_summary() -> dict:
    """Return a summary of 27001 Annex A coverage from the SecurityObserver."""
    covered_refs = set(_SIGNAL_TO_CONTROL.values())
    return {
        "standard": "ISO/IEC 27001:2022",
        "annex_a_controls_covered": len(covered_refs),
        "signal_mappings": len(_SIGNAL_TO_CONTROL),
        "controls": sorted(covered_refs),
        "note": (
            "Coverage counts controls evidenced by the SecurityObserver signal mapping. "
            "Full ISO 27001 certification requires ISMS documentation, risk assessment, "
            "and independent audit — this mapping is an evidence artefact, not a certification claim."
        ),
    }
