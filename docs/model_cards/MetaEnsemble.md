# Model Card: MetaEnsemble — Regime-Conditional Forecast Blending

**ISO/IEC 42001:2023 §8.4 — AI System Documentation**

---

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | MetaEnsemble |
| Version | 1.0.0 |
| Architecture | Regime-conditional weighted average of LEAR, QRA, and LNN |
| Training approach | No training — expert-defined weight tables |
| Primary use | Primary forecast output for all GridVerdict BESS recommendations |
| Deployment context | GridVerdict — simulation-only decision support |
| Owner | GridVerdict Data Science Team |
| Review frequency | Quarterly (weight table review) |

---

## 2. Intended Use

MetaEnsemble is the **primary forecast model** seen by the BESS dispatch policy.
It blends LEAR, QRA, and LNN outputs using regime-conditional weights, with the
intuition that the LNN's temporal dynamics are more valuable during price spikes.

---

## 3. Architecture

```python
# Regime weight tables (expert-defined, not learned):
_REGIME_WEIGHTS = {
    "normal":   {"qra": 0.50, "lear": 0.30, "experimental_lnn": 0.20},
    "elevated": {"qra": 0.40, "lear": 0.30, "experimental_lnn": 0.30},
    "spike":    {"qra": 0.30, "lear": 0.30, "experimental_lnn": 0.40},
    "extreme":  {"qra": 0.25, "lear": 0.25, "experimental_lnn": 0.50},
}
```

**Blending procedure:**
1. Index available forecasts by model name.
2. Select the regime weight row matching the current price regime.
3. Remove absent models and renormalise weights to sum to 1.0.
4. Compute weighted average of P10, P50, P90 per horizon step.
5. Tag output as `calibrated: True` only when all component models are calibrated.

`target_times` inheritance priority: QRA > LEAR > LNN (QRA has most reliable horizon alignment).

---

## 4. Training Data

MetaEnsemble has **no training data and no learned parameters**. The weight tables are
expert-defined based on:

- Academic benchmarks (Nowotarski & Weron 2015; Uniejewski et al. 2019)
- NEM regime frequency analysis (normal >> elevated >> spike)
- LNN experimental status (lower weight until backtesting matures)

The weight tables are reviewed quarterly. Changes are tracked via version control.

---

## 5. Performance Characteristics

| Metric | Notes |
|---|---|
| Regime adaptation | Shifts to LNN-heavy weighting during spike (0.40) to capture temporal dynamics |
| Missing model handling | Renormalises weights; never errors on missing component |
| Conformal coverage | Inherited from component models; `calibrated: True` only when all calibrated |
| Primary model | Always becomes the `primary` forecast when produced |

---

## 6. Known Limitations

1. **Expert-defined weights**: Not learned from data. Weight optimality for specific
   regions or time periods is not validated. Quarterly review is manual.
2. **No uncertainty propagation**: P10/P90 interval widths are a linear blend; they do
   not account for inter-model correlation. Intervals may be over- or under-confident.
3. **Regime classification dependency**: Weight selection uses `price_regime` from the
   MarketSnapshot, which is classified by a threshold rule
   (≤300 = normal, 300–1000 = elevated, >1000 = spike). Misclassification propagates.
4. **LNN experimental status**: LNN is included but explicitly downweighted in normal
   regime. As LNN backtesting evidence accumulates, weights should be revisited.

---

## 7. Human Oversight

- MetaEnsemble is the default primary forecast; the operator sees blended output, not component outputs.
- `evidence_quality` in MarketSnapshot is assessed separately from forecast quality.
- `simulation_only=True` hardcoded. Operator validation required before any action.
- Quarterly weight table review by GridVerdict Data Science Team is a registered control
  in the ISO 42001 AI risk register (AI-002).

---

## 8. Regulatory References

- ISO/IEC 42001:2023 §8.4 (AI system documentation)
- ISO/IEC 42001:2023 §9.1 (Monitoring and measurement — quarterly weight review)
- Nowotarski & Weron (2015); Uniejewski et al. (2019) — peer-reviewed methodology basis
