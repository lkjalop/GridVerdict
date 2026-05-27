# Model Card: QRA — Quantile Regression Averaging

**ISO/IEC 42001:2023 §8.4 — AI System Documentation**

---

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | QRA |
| Version | 1.0.0 |
| Architecture | Quantile meta-combiner over LEAR and LNN base models |
| Training approach | Supervised, calibration window, refitted each forecast call |
| Primary use | Probabilistic NEM price ensemble combining |
| Deployment context | GridVerdict — simulation-only decision support |
| Owner | GridVerdict Data Science Team |
| Review frequency | Monthly |

---

## 2. Intended Use

QRA (Nowotarski & Weron 2015) combines outputs from LEAR and LNN using a learned
linear quantile regression combiner. It is the primary ensemble model in normal and
elevated price regimes.

**In scope:** NEM price forecast ensemble combining.

**Out of scope:** Standalone forecasting (requires base model inputs), market execution.

---

## 3. Architecture

For each target quantile, QRA fits a `QuantileRegressor` whose inputs are the
corresponding quantile forecasts from all available base models:

```
inputs: [LEAR_p10, LNN_p10, LEAR_p50, LNN_p50, LEAR_p90, LNN_p90]
targets: actual observed price

One QuantileRegressor per quantile level.
LASSO α = 0.01 — can shrink useless models to zero weight.
```

**Regime-aware combiners**: Separate combiners trained for each of
`normal`, `elevated`, and `spike` price regimes. Falls back to global combiner
when fewer than 10 regime-specific calibration rows are available.

---

## 4. Training Data

| Property | Value |
|---|---|
| Source | Base model forecast outputs + actual NEM prices (calibration window) |
| Window | 20% of the 30-day rolling lookback (most recent rows) |
| Regime classification | Price ≤ 300 $/MWh = normal; 300–1000 = elevated; > 1000 = spike |
| Provenance tracking | `training_data_ref` field in `DecisionAuditLog` |

---

## 5. Performance Characteristics

| Metric | Notes |
|---|---|
| Coverage calibration | Inherits conformal calibration from base models |
| Regime combiner | Regime-specific weights improve coverage during elevated/spike periods |
| Strict causality | Combiner trained on historical forecasts vs actuals — no lookahead |
| Non-negative combining weights | LASSO regularisation keeps weights interpretable |

---

## 6. Known Limitations

1. **Inherits base model biases**: QRA cannot correct for systematic bias in both
   LEAR and LNN simultaneously.
2. **Regime combiners need sufficient history**: `_MIN_REGIME_CAL_ROWS = 10`. Rare
   spike regimes may always use the global combiner in practice.
3. **Requires both base models**: If LEAR or LNN is unavailable, QRA degrades to
   a single-model combiner. MetaEnsemble handles this gracefully.

---

## 7. Human Oversight

Same as LEAR. See LEAR model card §7.

---

## 8. Regulatory References

- ISO/IEC 42001:2023 §8.4 (AI system documentation)
- Nowotarski & Weron (2015) "Computing electricity spot price prediction intervals using
  quantile regression and forecast averaging" — peer-reviewed methodology basis
