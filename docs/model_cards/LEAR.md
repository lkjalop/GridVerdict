# Model Card: LEAR — Lasso-Estimated AutoRegressive Model

**ISO/IEC 42001:2023 §8.4 — AI System Documentation**

---

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | LEAR |
| Version | 1.0.0 |
| Architecture | Quantile linear autoregression with LASSO regularisation |
| Training approach | Supervised, rolling window, refitted each forecast call |
| Primary use | NEM spot price probabilistic forecasting (P10/P50/P90) |
| Deployment context | GridVerdict — simulation-only decision support |
| Owner | GridVerdict Data Science Team |
| Review frequency | Monthly |

---

## 2. Intended Use

LEAR produces short-horizon NEM price forecasts used as one component in the MetaEnsemble.
It is **not** used to execute market actions. All outputs are clearly labelled
`simulation_only=True` in the DecisionAuditLog.

**In scope:** 5-minute ahead NEM price forecasting for regions NSW1, VIC1, QLD1, SA1, TAS1.

**Out of scope:** Intraday bidding, real-time SCADA control, financial derivatives pricing.

---

## 3. Architecture

LEAR (Uniejewski, Nowotarski & Weron 2019) fits one `QuantileRegressor` per target quantile
using autoregressive price lags as features:

```
Lags: [1, 2, 3, 12, 288, 576] × 5-min intervals
      (5 min, 10 min, 15 min, 1 hr, 24 hr, 48 hr ago)
```

Features are standardised with `StandardScaler`. Regularisation parameter α = 0.01 (LASSO).
Non-quantile features (demand, availability, headroom) are appended after the AR lags.

Output: `QuantileForecast` with `p10`, `p50`, `p90` arrays for each forecast horizon step.

---

## 4. Training Data

| Property | Value |
|---|---|
| Source | NEM dispatch prices (AEMO public data via NEMWeb MMSDM) |
| Window | Rolling 30-day lookback at fit time |
| Frequency | 5-minute dispatch intervals |
| Features | Price lags, demand, availability, price regime flags |
| Provenance tracking | `training_data_ref` field in `DecisionAuditLog` |

Training data is **public AEMO dispatch data only**. No operator position data or
commercially sensitive information is used.

---

## 5. Performance Characteristics

| Metric | Notes |
|---|---|
| Quantile coverage | Calibrated via CQR split-conformal calibration (80/20 split) |
| Conformal coverage target | 90% (α = 0.10) |
| Regime behaviour | Uses global model; regime-specific behaviour via MetaEnsemble weights |
| Fallback | Persistence forecast when insufficient training data |

---

## 6. Known Limitations

1. **Linear model**: Cannot capture non-linear spike dynamics. MetaEnsemble compensates by
   increasing LNN weight during spike regimes.
2. **Assumes approximate stationarity**: Model quality degrades during structural market changes
   (e.g. mass renewable entry, new interconnector). Window retraining mitigates this.
3. **No FCAS features**: FCAS prices are not included in LEAR features; FCAS-driven spikes
   are partially captured only via AR lags.
4. **Cold start**: Requires ≥ 48 intervals (4 hours) to produce meaningful forecasts;
   falls back to persistence before that.

---

## 7. Human Oversight

- All LEAR outputs are fed into MetaEnsemble; no single model output is shown to the operator without blending.
- `evidence_quality` field in MarketSnapshot degrades to `insufficient` when LEAR is not available.
- Operator validation is required before any market action; `simulation_only=True` is hardcoded.

---

## 8. Regulatory References

- ISO/IEC 42001:2023 §8.4 (AI system documentation)
- ISO/IEC 42001:2023 §9.1 (Monitoring and measurement)
- AEMO IT Security Requirements (read-only data consumption)
