# GridVerdict — Known Limitations

An honest account of what the system cannot do, what evidence is missing, and where the analysis may be wrong. Transparency about these limitations is a design goal.

---

## Data Limitations

### 1. No real-time bid stack
GridVerdict ingests dispatch *outcomes* (what was dispatched) but not generator *offer stacks* (what prices generators bid at). This means:
- It cannot explain price events caused purely by rebidding strategies.
- `coal_outage_commitment` and `fuel_costs` are always in `missing_data`.

### 2. No hydro water values
Hydro reservoir levels and water values (which drive hydro bidding) are not ingested. Hydro dispatch events can be explained by interconnector flows but not by storage depletion.

### 3. Constraint equations are opaque
AEMO publishes constraint IDs but not always the full marginal value or binding equation. GridVerdict can identify *which* constraint is binding but cannot always explain *why* it binds.

### 4. NEMWeb archive is finite
The accessible archive covers approximately 2022-08 to 2024-07 (see NEMWeb access notes). Historical analog base rates are limited to this window. HippoGraph will show sparse results for query types not represented in that period.

### 5. Weather is contextual, not causal
The weather consensus layer provides temperature, wind speed, and solar irradiance estimates. These are **contextual demand and renewable drivers**, not direct causal evidence of price events. A correlation note is always appended in the `WeatherCorrelation` schema.

### 6. AEMO notices are retrospective
Market notices are published by AEMO after the event (minutes to hours). GridVerdict checks the notice cache but cannot guarantee a relevant notice will be available at query time. The `notices_cache_stale` flag indicates when the notice cache is older than 30 minutes.

---

## Model Limitations

### 7. LEAR assumes linearity
The LEAR model uses linear regression per quantile. It cannot capture highly non-linear spike dynamics. In spike regimes, CRPS degrades noticeably vs LNN/TCN.

### 8. LNN seed sensitivity
NCP wiring is randomly initialised. Different seeds produce different connectivity patterns. Without fixing the random seed, two LNN instances trained on the same data may produce different forecasts. Not yet addressed.

### 9. TCN fixed sequence length
The TCN uses a fixed `seq_len=12` (60 minutes of 5-min intervals). Inputs shorter than 12 intervals are left-padded, which may produce artefacts for the first intervals after a data gap.

### 10. Spike labels are imbalanced
Price events above $300/MWh are rare (~2–5% of intervals in the archive). Spike-risk heads are trained on imbalanced data without class weighting or oversampling. Expect high precision but lower recall for spike events.

### 11. Conformal calibration is regime-specific, not region-specific
LEAR's conformal calibrator is fitted per regime (spike/elevated/normal) but not per NEM region. A calibration trained on NSW data is applied across all regions for the same regime. Region-specific calibration is a known improvement path.

---

## Architecture Limitations

### 12. No portfolio optimisation
GridVerdict recommends dispatch actions for individual assets but has no portfolio optimisation layer. It cannot coordinate multiple batteries or optimise across a portfolio.

### 13. Simulation only — forever
The `simulation_only=True` flag is hard-coded. The system is designed as decision-support only and cannot be configured to execute real market actions. This is intentional.

### 14. Single-tenant in MVP
The data model does not yet enforce multi-tenant data isolation at the row level. All users see the same market data. User-specific portfolio data is not ingested.

### 15. No FCAS market depth
GridVerdict tracks the FCAS price (from dispatch prices) but does not ingest FCAS enablement volumes, bid-stack depth, or trapezoid parameters. FCAS recommendations are limited to "participate or not" with low confidence.

### 16. LLM quality varies by backend
The rule-based fallback decomposer achieves ~78% intent accuracy. Ollama's accuracy depends heavily on the model version (`mistral`, `qwen3`, etc.). Claude (Anthropic API) is the highest-quality decomposer but incurs API cost.

---

## What "SUPPORTED" Verdict Means (and Doesn't Mean)

A `SUPPORTED` verdict means:
- At least one piece of direct AEMO evidence supports the explanation.
- The evidence is fresh (staleness < threshold).

It does **not** mean:
- The explanation is complete.
- Alternative causes have been ruled out.
- The forecast is accurate.

The `counterargument` field always states what the evidence does not prove. The `missing_data` list always states what was searched but not available. These fields are populated by the deterministic engine, not by the LLM.

---

## What the LLM Cannot Do

By design, the LLM narrator in GridVerdict:
- Cannot assert prices, quantities, or timestamps.
- Cannot modify the confidence score, verdict, or action label.
- Cannot add evidence refs that were not found by the Why Engine.
- Cannot access the database directly.

All facts in `why_plain_english` must already appear in the structured `FactualVerdict`. If the LLM adds a fact not in the verdict, that is a system bug, not a feature.
