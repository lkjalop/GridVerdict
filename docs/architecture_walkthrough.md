# GridVerdict Architecture Walkthrough

A layer-by-layer description of the system. Intended for technical interviews, onboarding, and portfolio review.

---

## Overview

GridVerdict is a **temporal reasoning framework** applied to the Australian National Electricity Market. It answers NEM questions with evidence-grounded prose, forecasts using an ensemble of ML models, and never executes market actions.

```
User query
    │
    ▼
Query Decomposer ────────────── LLM (Ollama / Claude) + rule-based fallback
    │                            Outputs: intent, regions, flags, causal_targets
    ▼
Why Engine ─────────────────── Deterministic driver scoring (no LLM)
    │                            Sources: DB + cache, HippoGraph, AEMO notices
    ▼
Why Formatter ──────────────── Assembles FactualVerdict with evidence_refs,
    │                            claim_map, next_watch, confidence
    ▼
LLM Narrator ──────────────── Writes prose. Cannot assert facts or numbers.
    │                            Input: structured verdict. Output: why_plain_english
    ▼
API Response (FactualVerdict) ── JSON with verdict, action, evidence manifest,
                                  claim_map, next_watch, counterargument
```

---

## Layer 1: Ingest Pipeline

**Files:** `app/scheduler/`, `app/engines/ingest/`

### What it does
- Fetches NEMWeb MMSDM 5-minute dispatch files (zip → CSV → PostgreSQL).
- Fetches AEMO market notices via RSS and HTML scraping.
- Weather consensus from multi-source API blend.
- Runs on a 5-minute cron; Redis leader election (`SET gv:scheduler:leader <id> NX EX 30`) ensures exactly one worker runs the scheduler.

### Key design decisions
- **Browser User-Agent required** for NEMWeb: AEMO rate-limits bots; impersonating a browser UA is necessary (documented in NEMWeb access notes).
- **Archive range:** 2022-08 to 2024-07 accessible via direct MMSDM zip download.
- **Staleness tracking:** every DB row has an `ingested_at` timestamp. The Why Engine checks staleness before trusting any source.

---

## Layer 2: Storage

### PostgreSQL
Primary store for all market data:
- `dispatch_price` — 5-min RRP per region
- `dispatch_constraint` — binding constraints per interval
- `dispatch_interconnector` — interconnector MW flows and limits
- `dispatch_unit_solution` — per-DUID generation (used for fuel attribution)
- `aemo_market_notice` — parsed notices with region, type, and full text
- `weather_consensus` — multi-source temperature/wind/solar consensus

Alembic migrations (3 migrations, head = `0003`). AST-based smoke tests verify chain integrity without requiring a live DB.

### Redis
- Leader election: scheduler leader key with 30s TTL, 10s heartbeat.
- Short-lived cache: weather consensus (300s TTL), notice cache (600s TTL).
- Serves as the canary for cache degradation detection.

---

## Layer 3: Query Decomposer

**File:** `app/engines/decomposition.py`

### Backend waterfall
1. **Ollama** (local, default) — `mistral` or any model on `localhost:11434`
2. **Claude** (Anthropic API) — fallback when Ollama unreachable
3. **Rule-based** — deterministic keyword classifier, no LLM dependency

### Output: `QueryDecomposition`
```python
intent: IntentLabel           # 8 intents
entities: dict                # regions, generators, technologies
requires_why: bool            # needs causal explanation
requires_history: bool        # needs archive/analogs
requires_forecast: bool       # needs prediction
requires_live_market: bool    # needs real-time dispatch data
requires_bess_context: bool   # needs battery state info
requires_incident_timeline: bool  # needs AEMO notice timeline
causal_targets: list[str]    # e.g. ["demand", "constraint"]
spike_thresholds: list[float] # e.g. [300.0, 1000.0]
confidence: float             # LLM's self-assessed confidence
ambiguities: list[str]        # unclear aspects
clarifying_question: str | None
```

### Security
- System prompt is **static** — never contains user text.
- Raw user query goes only into `role=user` message.
- This prevents prompt injection into the system prompt.

### Eval
`tests/fixtures/decomposer_eval_set.json` — 232 cases covering all 8 intent types, all 5 NEM regions, and new Sprint O fields. `scripts/eval_decomposer.py` produces a confusion matrix and per-intent F1. Rule-based baseline: **78.4% intent accuracy, macro F1 = 0.808**.

---

## Layer 4: Why Engine

**Files:** `app/agents/why_sources.py`, `app/agents/why_builder.py`, `app/agents/why_formatter.py`

Three-part pipeline:

### 4a. WhySources — data assembly
Pulls from DB and cache:
- `CurrentState` — live dispatch price, demand, headroom_mw, constraint_count, interconnector flows, regime
- `NewsContext` — AEMO notices matching region + time window
- `WeatherContext` — consensus temperature, wind, solar radiation
- `AnalogResult` — HippoGraph K-nearest historical states
- `ForecastResult` — ensemble quantile forecast (P10/P50/P90), spike probs

### 4b. WhyBuilder — deterministic scoring
- `_build_driver_tiers()` — scores each evidence category (confirmed / supported / plausible / unconfirmed) based on data freshness, magnitude thresholds, and notice presence.
- `_build_claim_tiers()` — converts driver tiers to structured evidence refs.
- `_build_claim_map()` — converts to `ClaimMapItem` list with `ClaimType` enum.
- `_build_next_watch()` — deterministic threshold-based monitoring list (price, headroom, forecast, constraints, analogs, weather).
- No LLM in this layer. Scores are pure logic.

### 4c. WhyFormatter — final assembly
- Calls `derive_verdict()`, `derive_action()`, `compute_confidence()` from `app/core/verdict.py`.
- Assembles `FactualVerdict` with all evidence.
- The LLM narrator (separate path) receives the verdict and writes `why_plain_english`.

**Key invariant:** the LLM receives the `FactualVerdict` as input. It cannot modify confidence, action, verdict, or evidence refs. It only produces the prose description.

---

## Layer 5: HippoGraph (Historical Analogs)

**File:** `app/engines/hippograph/`

### What it does
- Stores compressed 19-dimensional market state vectors (feature columns from `market_features.py`).
- On query: finds K nearest states using cosine similarity.
- Computes analog base rate: what fraction of similar past states resolved within the price window.

### Feature vector (19 columns)
```
price_rrp, demand_mw, dispatch_mw, available_gen_mw,
solar_mw, wind_mw, hydro_mw, gas_mw, coal_mw, battery_mw,
price_velocity, demand_velocity,
vic_nsw_flow, sa_vic_flow, qld_nsw_flow, tas_vic_flow,
constraint_count_norm,
headroom_mw,        ← Sprint O addition
constraint_count    ← Sprint O addition
```

---

## Layer 6: Forecasting Engine

**Files:** `app/engines/forecasting/`

### Model ensemble

| Model | Architecture | Strength |
|---|---|---|
| LEAR | Ridge regression per quantile | Fast, interpretable, regime-specific conformal |
| LNN | Liquid Neural Network (NCP wiring) | Temporal adaptation, handles non-stationarity |
| TCN | Dilated causal conv (WaveNet-style) | Explicit receptive field, parallelisable |

All three models:
- Trained with **pinball loss** for quantile regression (P10/P25/P50/P75/P90)
- Have a **spike-risk classification head** trained jointly with BCE loss
- Spike thresholds: P(>$300), P(>$1000), P(<$0)

### Conformal calibration
LEAR supports **regime-specific conformal calibration** (`fit_regime_conformal`). Per-regime residuals are stored; `get_conformal_q_hat(regime)` returns a calibrated width adjustment so P90 is empirically valid within each regime.

### No-lookahead guarantee
The backtest proxy uses `price[t-6]` (not `price[t]`) for the predispatch proxy feature. Earlier versions had a data leakage bug (`price[t] = y[t]`) that caused CRPS ≈ 0. This is fixed and tested.

---

## Layer 7: API

**Files:** `app/api/`

### FastAPI routes
- `POST /query` — main entry point, runs the full Why Engine pipeline
- `GET /market/current?region=NSW1` — live market state
- `GET /market/forecast?region=NSW1` — ensemble forecast
- `GET /market/trust?region=NSW1` — per-model backtest scores
- `GET /incidents/brief/{region}` — Incident Brief (full evidence snapshot)
- `GET /history/analogs?region=NSW1` — HippoGraph analog search

### Security
- JWT authentication (RS256 or HS256 depending on config).
- `simulation_only=True` enforced at the application layer — routes that would execute market actions reject with 403.
- Rate limiting via middleware.
- System prompt injection prevention: user text never enters system prompt.

### Observability
Prometheus metrics exposed at `/metrics`:
- `gv_query_latency_ms` — end-to-end query latency
- `gv_llm_decompose_latency_ms` — decomposer call latency
- `gv_forecast_latency_ms` — ensemble forecast latency
- `gv_scheduler_jobs_running` — active ingest jobs

---

## Layer 8: Frontend

**File:** `frontend/app.html`, `frontend/static/js/`

Single-page application built with **Alpine.js** (no build step) and **ECharts** for charts.

### 8 tabs
1. **Answer** — verdict, action, why text, confidence, claim map, next watch
2. **Evidence** — evidence refs table with source, interval, staleness, caveat
3. **Forecast** — quantile ribbon + spike probability bars
4. **Analogs** — HippoGraph results table
5. **Backtest** — per-model CRPS / pinball / calibration scores
6. **History** — historical price chart
7. **Debug** — raw JSON verdict, decomposition output
8. **Incident Brief** — full snapshot (market state, BESS, notices, freshness, claims)

### State management
`frontend/static/js/state.js` — Alpine.js reactive store. All API calls go through `frontend/static/js/api.js`. State includes `answerExpanded` flags for collapsible sections.

---

## Key Design Principles

### 1. LLM cannot assert facts
The LLM receives a fully-formed `FactualVerdict` as input. It produces only `why_plain_english`. It cannot change numbers, confidence, or verdict. This is enforced structurally, not by instruction.

### 2. Evidence refs trace every claim
Every fact in the output has an `EvidenceRefSchema` with source table, interval, region, field, raw hash, and staleness. The `evidence_manifest` in the API response exposes this.

### 3. Simulation only
`simulation_only=True` is a hard-coded default in `BESSImplication` and all action-taking routes. No path through the system can execute a real market action.

### 4. Temporal bitemporality
Every piece of evidence has two timestamps:
- `valid_time` — when the market event occurred
- `system_time` — when GridVerdict ingested it

HippoGraph and TemporalRAG filter by `system_time <= query_time` to prevent lookahead leakage.

### 5. Deterministic fallbacks
The rule-based decomposer, the fallback forecast (last-known price), and the OOS route all produce valid outputs without any LLM. The system degrades gracefully.

---

## What This Demonstrates (Interview Talking Points)

- **Temporal reasoning at scale:** bitemporality, rolling-window conformal calibration, seasonal backtesting
- **LLM safety patterns:** fact prohibition, static system prompts, evidence-grounded generation
- **Multi-model ML pipeline:** quantile regression + classification heads trained jointly
- **Production reliability:** leader election, staleness tracking, Alembic migrations, Prometheus metrics
- **Clean architecture:** strict layering — ingest never touches LLM, LLM never touches DB
