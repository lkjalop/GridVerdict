"""ISO/IEC 42001:2023 AI Risk Register for GridVerdict.

Documents AI-specific risks, inherent risk ratings, implemented controls,
and residual risk classifications per:

  ISO/IEC 42001:2023 §6.1  — Actions to address risks and opportunities
  ISO/IEC 42001:2023 §8.4  — AI system documentation
  ISO/IEC 42001:2023 §8.5  — AI system operation

Residual risk levels: Critical | High | Medium | Low
Review frequency: Continuous | Monthly | Quarterly | Annually
"""
from __future__ import annotations

from datetime import date
from typing import Any

_REGISTER_DATE = "2026-05-25"

RISK_REGISTER: list[dict[str, Any]] = [
    {
        "risk_id": "AI-001",
        "category": "AI System Performance",
        "description": (
            "AI produces incorrect market dispatch recommendation, leading to "
            "suboptimal BESS operation or financial loss for the operator."
        ),
        "likelihood": "Medium",
        "impact": "High",
        "inherent_risk": "High",
        "controls": [
            "simulation_only=True is hardcoded in every DecisionAuditLog row — no real market action can be executed",
            "confidence field surfaces evidence quality per decision: supported | low_confidence | insufficient_data",
            "why_summary stores up to 5 traceable rationale bullets per recommendation",
            "risk_flags captures detected concerns at the time of decision",
            "missing_before_action checklist explicitly lists data gaps before any action is recommended",
            "All outputs carry mandatory disclaimer: 'Simulation and decision-support only. Not financial advice.'",
        ],
        "control_refs": ["ISO42001-§8.5", "ISO42001-§7.3", "NER-Chapter4"],
        "residual_risk": "Low",
        "owner": "GridVerdict Platform Team",
        "review_frequency": "Quarterly",
    },
    {
        "risk_id": "AI-002",
        "category": "Data Quality and Provenance",
        "description": (
            "Forecast models trained on stale, incomplete, or unrepresentative "
            "historical data, reducing recommendation accuracy without the operator "
            "being aware."
        ),
        "likelihood": "Medium",
        "impact": "Medium",
        "inherent_risk": "Medium",
        "controls": [
            "model_version in DecisionAuditLog records which model version produced each decision",
            "training_data_ref records the training data window (region, date range, n_rows, SHA-256 prefix)",
            "Conformal calibration (CQR split-conformal) provides finite-sample coverage guarantees on prediction intervals",
            "Regime-aware training: separate model weights for normal / elevated / spike price regimes",
            "MetaEnsemble degrades gracefully when individual models are absent (renormalises weights)",
            "evidence_quality field degrades to 'insufficient' when fewer than 3 data sources are available",
            "Model cards published in /docs/model_cards/ with known limitations documented",
        ],
        "control_refs": ["ISO42001-§8.4", "ISO42001-§9.1"],
        "residual_risk": "Low",
        "owner": "GridVerdict Data Science Team",
        "review_frequency": "Monthly",
    },
    {
        "risk_id": "AI-003",
        "category": "Security — Adversarial Input",
        "description": (
            "Adversarial user input (prompt injection, market manipulation language, "
            "encoded payloads) corrupts AI reasoning or causes the system to produce "
            "unsafe recommendations."
        ),
        "likelihood": "Low",
        "impact": "High",
        "inherent_risk": "High",
        "controls": [
            "SecurityObserver 4-pass pipeline inspects every query lifecycle stage before LLM call",
            "Pass 1 (Input): rejects prompt injection (score 90), market manipulation (85), PII (35), unicode anomalies (40)",
            "Pass 2 (Decomposition): blocks unsafe execution intents (score 95) and OOS classification (65)",
            "Pass 3 (Tool output): detects injection in retrieved data (85) and anomalous numeric values",
            "Pass 4 (Answer): requires evidence_refs for SUPPORTED verdicts; flags missing disclaimers",
            "ISO 27001 Annex A control refs tagged on every observer signal for audit traceability",
            "System prompt is static; raw user text only goes into role=user messages (injection prevention)",
        ],
        "control_refs": ["ISO27001-A.8.28", "ISO27001-A.5.36", "ISO42001-§8.5"],
        "residual_risk": "Low",
        "owner": "GridVerdict Security Team",
        "review_frequency": "Continuous",
    },
    {
        "risk_id": "AI-004",
        "category": "Privacy and Data Confidentiality",
        "description": (
            "Commercially sensitive operator position data (SOC %, capacity, strategy) "
            "or personally identifiable information is exposed to cloud AI services."
        ),
        "likelihood": "Low",
        "impact": "High",
        "inherent_risk": "High",
        "controls": [
            "Portfolio data never sent to cloud LLM — hard block enforced in SecurityObserver Pass 2 (portfolio_data_requested signal)",
            "All LLM calls use static system prompt; user text never contains injected portfolio fields",
            "Multi-tenant isolation: tenant_id on every DB table; all queries scoped by tenant",
            "PII detection (SSN, credit card numbers, email addresses, API keys) blocks input in Pass 1",
            "Audit export endpoint scoped to tenant_id — operators cannot access other tenants' records",
        ],
        "control_refs": ["ISO27001-A.5.34", "ISO27001-A.5.10", "ISO42001-§8.5"],
        "residual_risk": "Low",
        "owner": "GridVerdict Security Team",
        "review_frequency": "Quarterly",
    },
    {
        "risk_id": "AI-005",
        "category": "Explainability and Auditability",
        "description": (
            "An AI recommendation cannot be explained or audited after the fact, "
            "preventing regulatory review or post-incident investigation."
        ),
        "likelihood": "Low",
        "impact": "High",
        "inherent_risk": "Medium",
        "controls": [
            "DecisionAuditLog is write-once and immutable — rows are never updated or deleted",
            "why_summary (up to 5 ordered rationale bullets) stored per decision",
            "evidence_quality documents data completeness at decision time",
            "risk_flags captures detected concerns at decision time",
            "model_version and training_data_ref record which model and data produced the recommendation",
            "Regulatory export: GET /api/portfolio/audit/export returns JSON or CSV for external review",
            "Participant behavior profiler (behavioral_tier) traces rebid-driven decisions",
        ],
        "control_refs": ["ISO42001-§8.4", "ISO42001-§9.1"],
        "residual_risk": "Low",
        "owner": "GridVerdict Compliance Team",
        "review_frequency": "Quarterly",
    },
    {
        "risk_id": "AI-006",
        "category": "Regulatory Exposure",
        "description": (
            "System is mistaken for a licensed market participant or financial adviser, "
            "creating regulatory exposure under the Corporations Act or NER."
        ),
        "likelihood": "Low",
        "impact": "Critical",
        "inherent_risk": "High",
        "controls": [
            "simulation_only=True is hardcoded in every audit row — cannot be overridden via API",
            "Every recommendation carries the mandatory disclaimer: 'Simulation and decision-support only. Not financial advice. Not a market participant.'",
            "No AEMO/NEMDE write access in the codebase — system is read-only from NEM data sources",
            "NER Chapter 4 / AEMC dispatch rules reviewed at design time — no AFSL-regulated activity performed",
            "IEC 62443 SL-2 assessment required before any real-execution integration",
        ],
        "control_refs": ["ISO42001-§4.2", "NER-Chapter4", "AFSL-Carve-out"],
        "residual_risk": "Low",
        "owner": "GridVerdict Legal / Compliance Team",
        "review_frequency": "Annually",
    },
    {
        "risk_id": "AI-007",
        "category": "AI System Performance — Distribution Shift",
        "description": (
            "Models overfit to historical price regime and fail silently during novel "
            "market conditions (e.g. VoLL event, interconnector failure, mass withdrawal)."
        ),
        "likelihood": "Medium",
        "impact": "Medium",
        "inherent_risk": "Medium",
        "controls": [
            "Regime-aware forecasting: separate combiner weights for normal / elevated / spike / extreme regimes",
            "MetaEnsemble shifts weight toward LNN (temporal) when spike regime is detected",
            "Conformal calibration widens prediction intervals under distributional uncertainty",
            "Participant behavior profiler detects habitual rebidders and elevates evidence tier",
            "evidence_quality degrades to 'insufficient' when fewer than 3 data sources available",
            "LEAR and QRA models retrain from scratch each forecast call — no stale weights",
        ],
        "control_refs": ["ISO42001-§8.4", "ISO42001-§9.1"],
        "residual_risk": "Medium",
        "owner": "GridVerdict Data Science Team",
        "review_frequency": "Monthly",
    },
]


def get_risk_register() -> list[dict[str, Any]]:
    """Return the full AI risk register."""
    return RISK_REGISTER


def get_risk_summary() -> dict[str, Any]:
    """Return a summary suitable for the compliance dashboard."""
    by_residual: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for r in RISK_REGISTER:
        lvl = r["residual_risk"]
        cat = r["category"]
        by_residual[lvl] = by_residual.get(lvl, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
    return {
        "standard": "ISO/IEC 42001:2023",
        "total_risks": len(RISK_REGISTER),
        "by_residual_risk": by_residual,
        "by_category": by_category,
        "register_date": _REGISTER_DATE,
        "caveat": (
            "This risk register documents known AI risks and their controls as of the register date. "
            "It is an evidence artefact for ISO/IEC 42001 alignment — not a certification claim. "
            "Independent audit and ISMS documentation are required for formal certification."
        ),
    }
