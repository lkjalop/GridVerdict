# GridVerdict Model Explainer

How each model works, why it was chosen, and what its outputs mean.

---

## LEAR (Lasso Estimated AutoRegressive)

### What it is
LEAR fits one Ridge/Lasso regression model **per quantile** (P10, P25, P50, P75, P90). Each model learns a linear mapping from the 19-dimensional market feature vector to the price quantile at the target horizon.

### Why it's here
- Fastest inference (microseconds per prediction)
- Fully interpretable — `feature_importances(top_k)` returns mean absolute coefficient ranks
- Regime-specific conformal calibration means the P90 is empirically valid within each market regime

### Conformal calibration
After training, call `fit_regime_conformal(X_cal, y_cal, regimes)`. This computes per-regime calibration residuals (the `q_hat` conformal quantile). At inference, `get_conformal_q_hat(regime)` adjusts the raw quantile width so the empirical coverage matches the target (90% by default).

### Feature importances
```python
lear.feature_importances(top_k=5)
# Returns: [{"feature": "price_rrp", "importance": 0.42, "rank": 1}, ...]
```

### Failure modes
- Assumes linear relationships — misses highly non-linear spike dynamics
- Quantile crossing possible on extreme inputs (not post-processed)
- Regime labels must be consistent between train and inference

---

## LNN (Liquid Neural Network)

### What it is
An LNN uses **Neural Circuit Policy (NCP)** wiring — a sparse, biologically-inspired connectivity pattern where neurons are divided into sensory, inter, command, and motor cells. Unlike a dense MLP, NCP wires are fixed and sparse, which reduces overfitting on small datasets.

The LNN is implemented **from scratch** using `ncps` + PyTorch. It replaces recurrence with a continuous-time ODE cell (`LTCCell`) that integrates inputs over time — better for irregularly-sampled or non-stationary time series.

### Why it's here
- Handles non-stationarity better than standard RNNs (the ODE cell adapts its time constants)
- NCP sparsity provides implicit regularisation on the 19-feature input
- Spike-risk head trained jointly: shares the LTC backbone with the quantile head

### Spike-risk head
A binary classification head (`nn.Linear(hidden_dim, 3)`) is trained alongside the quantile head using Binary Cross Entropy loss. Outputs sigmoid-activated probabilities for:
- `gt_300` — P(price > $300/MWh) in the next interval
- `gt_1000` — P(price > $1000/MWh)
- `lt_0` — P(price < $0/MWh)

**Sigmoid overflow protection:** logits are clipped to [-60, 60] before sigmoid to prevent `RuntimeWarning: overflow in exp`.

### Training
```python
loss = pinball_loss(quantile_pred, target) + 0.5 * bce_loss(spike_logits, spike_labels)
```

The 0.5 weight (`spike_loss_weight`) balances the two objectives. Tune upward if spike-risk accuracy is more important than median accuracy.

### Failure modes
- Requires `torch` and `ncps` — falls back gracefully if unavailable
- NCP wiring is random-seeded; different seeds give different connectivity (not reproducible without fixing the seed)
- The ODE cell has no `.eval()` effect (no dropout) — but weight consistency requires `.eval()` before inference

---

## TCN (Temporal Convolutional Network)

### What it is
A TCN uses **dilated causal convolutions** stacked in residual blocks. Each block has two causal convolutions with exponentially increasing dilation (1, 2, 4, 8...) which gives a large effective receptive field with few parameters.

Architecture:
```
Input (B, T, 19) → InputProjection (Conv1d 1×1) → ResidualBlock(dilation=1)
→ ResidualBlock(dilation=2) → ResidualBlock(dilation=4) → ResidualBlock(dilation=8)
→ Final timestep (B, channels) → QuantileHead + SpikeHead
```

### Why it's here
- Fully parallelisable (unlike RNNs) — faster training
- Explicit receptive field = `(kernel_size - 1) * 2^n_layers`
- Dropout during training adds regularisation; `.eval()` before inference ensures deterministic predictions

### Causal padding
`_CausalConv1d` left-pads inputs so the output length equals the input length and no future information leaks in. The right-side overhang from `padding=(kernel_size-1)*dilation` is trimmed: `output[:, :, :x.shape[2]]`.

### Weight norm
`nn.utils.weight_norm()` is applied to the **inner** `nn.Conv1d` object (`self.causal1.conv`), not to the `_CausalConv1d` wrapper. The wrapper has no `weight` parameter directly.

### Eval/train toggle
```python
def predict_quantiles(self, X, target_times):
    self._backbone.eval(); self._head.eval()
    try:
        with torch.no_grad():
            pred = self._head(self._backbone(seqs)).cpu().numpy()
    finally:
        self._backbone.train(); self._head.train()
```
This pattern ensures dropout is off during prediction, making save/load round-trips deterministic.

### Failure modes
- Requires `torch` — falls back gracefully if unavailable
- Fixed sequence length (`seq_len=12` by default) — queries shorter than 12 intervals are left-padded
- Larger dilation stacks can overfit on sparse spike data — monitor CRPS per regime

---

## Ensemble and Forecast Trust

### Combining models
`app/engines/forecasting/live_forecast.py` runs all available models and aggregates into a consensus forecast. The primary output is:
- **P50 (median)** from the ensemble weighted by recent CRPS
- **P10/P90 band** for the uncertainty ribbon
- **Spike probabilities** averaged across models that have spike heads

### Backtest scoring
`app/engines/forecast_trust.py` evaluates each model against held-out NEMWeb data using:
- **CRPS** (Continuous Ranked Probability Score) — proper scoring rule for probabilistic forecasts
- **Pinball loss** — quantile-specific loss at each percentile
- **Calibration** — empirical coverage at P10, P50, P90

The `/market/trust` endpoint exposes these scores. Lower CRPS = better.

### No-lookahead guarantee
In backtest mode, the predispatch proxy feature uses `price[t-6]` (6 intervals = 30 minutes ago), not `price[t]`. This prevents leakage of the target variable into the feature set. Tested in `tests/test_sprint_e_forecast.py`.

---

## Feature Vector Reference

19 features, in column order:

| Col | Name | Description |
|---|---|---|
| 0 | `price_rrp` | Current dispatch price ($/MWh) |
| 1 | `demand_mw` | Scheduled demand (MW) |
| 2 | `dispatch_mw` | Total dispatched generation (MW) |
| 3 | `available_gen_mw` | Available registered capacity (MW) |
| 4 | `solar_mw` | Semi-scheduled solar (MW) |
| 5 | `wind_mw` | Semi-scheduled wind (MW) |
| 6 | `hydro_mw` | Hydro generation (MW) |
| 7 | `gas_mw` | Gas/OCGT/CCGT generation (MW) |
| 8 | `coal_mw` | Coal generation (MW) |
| 9 | `battery_mw` | Battery storage net dispatch (MW) |
| 10 | `price_velocity` | Price change over last 2 intervals |
| 11 | `demand_velocity` | Demand change over last 2 intervals |
| 12 | `vic_nsw_flow` | VIC→NSW interconnector MW |
| 13 | `sa_vic_flow` | SA→VIC interconnector MW |
| 14 | `qld_nsw_flow` | QLD→NSW interconnector MW |
| 15 | `tas_vic_flow` | TAS→VIC (Basslink) interconnector MW |
| 16 | `constraint_count_norm` | Normalised active constraint count |
| 17 | `headroom_mw` | Available gen minus demand (MW) — Sprint O |
| 18 | `constraint_count` | Raw active constraint count — Sprint O |
