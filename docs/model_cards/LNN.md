# Model Card: LNN — Liquid Neural Network (Temporal Forecaster)

**ISO/IEC 42001:2023 §8.4 — AI System Documentation**

---

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | LNN (experimental_lnn in MetaEnsemble) |
| Version | 1.0.0 |
| Architecture | Liquid Time-Constant (LTC) recurrent network with regime-aware quantile head |
| Training approach | Supervised, rolling window, GPU/CPU, pinball loss |
| Primary use | Temporal price pattern recognition for NEM spike forecasting |
| Deployment context | GridVerdict — simulation-only, labelled "experimental" in UI |
| Owner | GridVerdict Data Science Team |
| Review frequency | Monthly |

---

## 2. Intended Use

The LNN is a neural ODE-inspired recurrent architecture that excels at learning
irregular temporal dynamics — critical for NEM price spikes that exhibit non-linear
autocorrelation not captured by linear AR models.

**In scope:** 5-minute NEM price probabilistic forecasting, especially during
elevated and spike regimes where linear models underperform.

**Out of scope:** Market execution, real-time control, financial advice.

> **UI label**: This model is labelled `experimental_lnn` in the MetaEnsemble
> and in any user-facing outputs. Operators should treat LNN outputs with
> additional scepticism until the model accumulates sufficient backtesting evidence.

---

## 3. Architecture

```
Input: feature matrix X ∈ ℝ^(T × 17)
  - T = sequence length (12 intervals = 1 hour)
  - 17 features: price, demand, availability, headroom, price regime flags,
                 temporal markers, temp_c (weather), wind_kmh (weather)

Encoder: LTC (Liquid Time-Constant) cells
  - hidden_size = 32
  - Non-linear ODE dynamics: dh/dt = f(h, x, t)

Head: RegimeAwareQuantileHead
  - regime_ctx = [log1p(|last_price|), is_elevated, is_spike]
  - Concatenated with hidden state h
  - fc1 → ReLU → {fc_p50, fc_lo, fc_hi}
  - P10 = P50 - softplus(fc_lo)  ← non-crossing guaranteed by construction
  - P90 = P50 + softplus(fc_hi)  ← non-crossing guaranteed by construction
```

Non-crossing quantile guarantee: P10 ≤ P50 ≤ P90 is enforced structurally
via softplus offsets. No post-hoc sorting required.

---

## 4. Training Data

| Property | Value |
|---|---|
| Source | NEM dispatch prices + derived features (public AEMO data) |
| Minimum required | 288 intervals (24 hours) |
| Window | Rolling 30-day lookback |
| Weather features | temp_c, wind_kmh (from weather cache, 1 dispatch cycle stale) |
| Weights persistence | `data/lnn_weights/ltc_{region}.pt` (PyTorch checkpoint) |
| Metadata persistence | `data/lnn_weights/ltc_{region}_meta.json` (hyperparams + norm state) |
| Provenance tracking | `training_data_ref` field in `DecisionAuditLog` |

---

## 5. Performance Characteristics

| Metric | Notes |
|---|---|
| Regime focus | Higher weight in MetaEnsemble during spike regime (0.40 vs 0.20 in normal) |
| Conformal calibration | CQR calibration applied post-training on held-out 20% window |
| Non-crossing | Guaranteed by softplus head construction |
| Training speed | ~60 epochs on CPU; GPU-accelerated when available |

---

## 6. Known Limitations

1. **Experimental**: LNN has less production backtesting evidence than LEAR/QRA.
   It is weighted lower in normal regime (0.20) for this reason.
2. **Requires substantial data**: Minimum 288 intervals. Falls back to zeroed forecast
   when insufficient training data.
3. **Weather features**: `temp_c` and `wind_kmh` are one dispatch cycle (5 min) stale
   because weather is fetched in a parallel pipeline task. Acceptable for 30-min horizon;
   not suitable for weather-sensitive forecasting at sub-5-min timescales.
4. **Non-stationary spike distribution**: Spike events are rare; LTC may not generalise
   to novel spike patterns (e.g. first VoLL event of a season). MetaEnsemble's
   conformal calibration widens intervals to compensate.
5. **No gradient explainability**: LTC gradients are not directly human-interpretable.
   The `why_summary` in DecisionAuditLog does not derive from LNN internals.

---

## 7. Human Oversight

- Labelled `experimental_lnn` in all API responses.
- Weighted lower than QRA in normal regime to reduce influence of less-validated model.
- All outputs pass through MetaEnsemble before reaching the operator.
- `simulation_only=True` hardcoded. Operator validation required.

---

## 8. Regulatory References

- ISO/IEC 42001:2023 §8.4 (AI system documentation)
- ISO/IEC 42001:2023 §8.5 (AI system operation — monitoring of experimental components)
- Hasani et al. (2021) "Liquid Time-constant Networks" — NeurIPS peer-reviewed basis
