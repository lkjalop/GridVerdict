"""Australian Energy Sector Cybersecurity Framework (AESCSF) self-assessment.

The AESCSF is the primary cybersecurity framework for Australian energy sector
participants (AEMO, networks, generators, retailers). It maps to ISO/IEC 27001
but adds energy-sector-specific requirements around OT/IT boundary security,
incident notification obligations, and supply chain risk.

This module provides a structured self-assessment template with GridVerdict's
current implementation status per AESCSF domain.

Reference: AESCSF v2.0 (2023), Australian Energy Market Operator.
"""
from __future__ import annotations

from typing import Any

# Maturity levels: Initial(1) → Developing(2) → Defined(3) → Managed(4) → Optimising(5)
# For a software decision-support platform, Level 3 (Defined) is the target before
# enterprise sales. Level 4+ is for critical infrastructure operators.

SELF_ASSESSMENT: dict[str, Any] = {
    "framework": "Australian Energy Sector Cybersecurity Framework (AESCSF) v2.0",
    "assessment_date": "2026-05-25",
    "system_description": (
        "GridVerdict — AI-assisted BESS dispatch decision-support platform. "
        "Simulation-only; does not execute real market actions. "
        "Consumes public AEMO/NEMWeb data via read-only APIs."
    ),
    "target_maturity": 3,
    "domains": [
        {
            "domain_id": "ID",
            "domain_name": "Identify",
            "description": "Asset management, governance, risk assessment, supply chain.",
            "controls": [
                {
                    "control_id": "ID.AM-1",
                    "description": "Software assets and data flows are inventoried",
                    "status": "Implemented",
                    "evidence": [
                        "All DB tables have tenant_id and ingested_at timestamps",
                        "Model registry (model_registry.py) tracks model versions and training data provenance",
                        "OpenAPI schema at /api/openapi.json documents all endpoints",
                    ],
                    "maturity": 3,
                },
                {
                    "control_id": "ID.RA-1",
                    "description": "Asset vulnerabilities are identified and documented",
                    "status": "Partial",
                    "evidence": [
                        "ISO 42001 AI risk register documents 7 AI-specific risks with controls",
                        "Dependency scanning not yet automated (gap: add Dependabot or pip-audit to CI)",
                    ],
                    "maturity": 2,
                    "gap": "Automated dependency vulnerability scanning not yet implemented",
                },
                {
                    "control_id": "ID.GV-1",
                    "description": "Cybersecurity policy is established and communicated",
                    "status": "Partial",
                    "evidence": [
                        "Security constraints documented in module docstrings (simulation_only, no real execution)",
                        "ISO 27001 Annex A control mapping documented in iso27001_controls.py",
                    ],
                    "maturity": 2,
                    "gap": "Formal ISMS policy document not yet drafted",
                },
            ],
        },
        {
            "domain_id": "PR",
            "domain_name": "Protect",
            "description": "Access control, data security, training, maintenance, protective technology.",
            "controls": [
                {
                    "control_id": "PR.AC-1",
                    "description": "Identities and credentials are managed for authorised users",
                    "status": "Implemented",
                    "evidence": [
                        "JWT-based authentication on all API endpoints",
                        "Multi-tenant isolation with tenant_id on all DB tables",
                        "GRIDVERDICT_DEV_NO_AUTH flag is False in production",
                    ],
                    "maturity": 3,
                },
                {
                    "control_id": "PR.DS-1",
                    "description": "Data-at-rest is protected",
                    "status": "Partial",
                    "evidence": [
                        "PostgreSQL at-rest encryption delegated to infrastructure (RDS/CloudSQL)",
                        "No sensitive position data stored beyond what operator explicitly submits",
                    ],
                    "maturity": 2,
                    "gap": "At-rest encryption depends on infrastructure configuration; not enforced at application layer",
                },
                {
                    "control_id": "PR.PT-1",
                    "description": "Audit/log records are determined, documented, implemented, and reviewed",
                    "status": "Implemented",
                    "evidence": [
                        "DecisionAuditLog: write-once, immutable records with model_version and training_data_ref",
                        "ObserverEvent: persisted security signals with ISO 27001 Annex A control_ref",
                        "Regulatory export endpoint: GET /api/portfolio/audit/export (JSON + CSV)",
                        "Rate-limit middleware logs all requests",
                    ],
                    "maturity": 4,
                },
                {
                    "control_id": "PR.IP-1",
                    "description": "A baseline configuration is created and maintained",
                    "status": "Partial",
                    "evidence": [
                        "config/settings.py manages all configuration via environment variables",
                        "Pydantic settings validation at startup",
                    ],
                    "maturity": 2,
                    "gap": "Infrastructure-as-code baseline (Terraform/CDK) not yet formalised",
                },
            ],
        },
        {
            "domain_id": "DE",
            "domain_name": "Detect",
            "description": "Anomaly detection, continuous monitoring, detection processes.",
            "controls": [
                {
                    "control_id": "DE.AE-1",
                    "description": "A baseline of normal operations is established and managed",
                    "status": "Implemented",
                    "evidence": [
                        "SecurityObserver 4-pass pipeline detects anomalous inputs and outputs on every query",
                        "Rate limiting middleware detects abnormal request volumes (A.8.20)",
                        "Price/demand/availability anomaly detection in Pass 3 tool output validation",
                    ],
                    "maturity": 3,
                },
                {
                    "control_id": "DE.CM-1",
                    "description": "The network is monitored to detect potential events",
                    "status": "Partial",
                    "evidence": [
                        "Rate-limit middleware monitors request cadence",
                        "Security status endpoint: GET /api/security/status",
                    ],
                    "maturity": 2,
                    "gap": "Network-layer monitoring (WAF, IDS) delegated to infrastructure; not application-layer",
                },
            ],
        },
        {
            "domain_id": "RS",
            "domain_name": "Respond",
            "description": "Response planning, communications, analysis, mitigation, improvements.",
            "controls": [
                {
                    "control_id": "RS.RP-1",
                    "description": "Response plan is executed during or after an event",
                    "status": "Partial",
                    "evidence": [
                        "SecurityObserver halt verdict (score ≥80) stops query pipeline immediately",
                        "Observer events logged with ISO 27001 control refs for incident reconstruction",
                    ],
                    "maturity": 2,
                    "gap": "Formal incident response runbook not yet documented",
                },
                {
                    "control_id": "RS.AN-1",
                    "description": "Notifications from detection systems are investigated",
                    "status": "Partial",
                    "evidence": [
                        "GET /api/security/signals provides paginated observer signal log",
                        "GET /api/compliance/27001/observer-events provides DB-persisted event history",
                    ],
                    "maturity": 2,
                    "gap": "No automated alerting (PagerDuty/OpsGenie integration) on halt verdicts",
                },
            ],
        },
        {
            "domain_id": "RC",
            "domain_name": "Recover",
            "description": "Recovery planning, improvements, communications.",
            "controls": [
                {
                    "control_id": "RC.RP-1",
                    "description": "Recovery plan is executed during or after a cybersecurity event",
                    "status": "Not Implemented",
                    "evidence": [],
                    "maturity": 1,
                    "gap": "Formal business continuity and disaster recovery plan not yet drafted (ISO 22301 gap)",
                },
            ],
        },
    ],
    "overall_maturity_estimate": 2,
    "next_steps": [
        "Draft formal ISMS policy document (closes ID.GV-1 gap)",
        "Add automated dependency vulnerability scanning to CI pipeline (closes ID.RA-1 gap)",
        "Document infrastructure-as-code baseline (closes PR.IP-1 gap)",
        "Draft incident response runbook (closes RS.RP-1 gap)",
        "Add automated alerting on SecurityObserver halt verdicts (closes RS.AN-1 gap)",
        "Draft business continuity plan (closes RC.RP-1 gap, addresses ISO 22301)",
    ],
    "caveat": (
        "This self-assessment is a point-in-time snapshot for internal gap analysis. "
        "It is not a compliance certification. AESCSF assessment by an accredited assessor "
        "is required for formal compliance claims with AEMO or the AER."
    ),
}


def get_self_assessment() -> dict[str, Any]:
    return SELF_ASSESSMENT


def get_maturity_summary() -> dict[str, Any]:
    domains = SELF_ASSESSMENT["domains"]
    scores = []
    gaps = []
    for d in domains:
        for c in d["controls"]:
            scores.append(c["maturity"])
            if c.get("gap"):
                gaps.append({"control_id": c["control_id"], "gap": c["gap"]})
    avg = round(sum(scores) / len(scores), 1) if scores else 0
    return {
        "framework": "AESCSF v2.0",
        "assessment_date": SELF_ASSESSMENT["assessment_date"],
        "overall_maturity_estimate": avg,
        "target_maturity": SELF_ASSESSMENT["target_maturity"],
        "gap_count": len(gaps),
        "gaps": gaps,
    }
