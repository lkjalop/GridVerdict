# Runbook: ClaimDowngradeSpike / ClaimDowngradeCritical

**Alert:** `ClaimDowngradeSpike` (>0.5/s for 5 min) | `ClaimDowngradeCritical` (>2/s for 2 min)
**Metric:** `gv_claim_verifier_downgrades_total`
**Team:** ml

---

## What is a claim downgrade?

The `ClaimVerifier` applies rule-based checks to every answer before it is returned.
A downgrade occurs when the verifier detects a violation of the evidence contract:
- An LLM asserted a specific price/demand number not present in the deterministic layer
- A narrative cited a source that wasn't in the tool outputs
- A probability claim was made without supporting evidence
- An action recommendation lacks a hedging counterargument

On downgrade, the answer's confidence is lowered and/or the verdict is degraded
to `LOW_CONFIDENCE` or `INSUFFICIENT_DATA`.

---

## Diagnosis

1. **Check recent claim verifier findings in logs**
   ```
   grep "Claim verifier" app.log | tail -50
   ```
   Look for which `rule` is triggering: `number_not_in_evidence`, `ungrounded_citation`,
   `no_counterargument`, `probability_without_support`, etc.

2. **Check if the LLM output changed** — downgrades spike after:
   - Ollama model update
   - Claude API prompt version change
   - Prompt template changes in `why_builder.py`

3. **Check if forecast model changed** — if LEAR/QRA/LNN started emitting very high
   confidence intervals, the why_builder may include those in the narrative.

---

## Remediation

| Scenario | Action |
|---|---|
| Rule-based why_builder glitch | Check `why_builder.py` recent changes |
| LLM model update broke grounding | Roll back Ollama model or prompt template |
| Forecast model confidence inflation | Check calibration error in `/api/market/trust` |
| False positives from verifier | Review `claim_verifier.py` rules; adjust thresholds if needed |

**`ClaimDowngradeCritical` (>2/s):** Consider temporarily routing all answers through the
rule-based why_builder only (bypass LLM narration) until the cause is identified.

---

## Impact

Downgrades are a safety feature. High rates mean:
- Users receive cautious `LOW_CONFIDENCE` answers (appropriate degradation)
- Audit log records show `model_version` + `training_data_ref` for traceability
- No incorrect market data is asserted (the evidence contract is enforced)

A sustained critical rate may indicate a systematic model quality issue
requiring ML team review.
