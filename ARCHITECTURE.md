# GridVerdict Architecture

Evidence-grounded decision-support cockpit for the Australian NEM.

---

## Core design principles

1. **LLM decomposes only** — the language model produces `intent` and `entities`. It never asserts market facts or produces numbers. All numeric reasoning is deterministic.
2. **Evidence contract** — every `SUPPORTED` verdict requires at least one `evidence_ref`. Confidence bands are auto-derived from evidence coverage, never from model self-assessment.
3. **Security by construction** — four observer passes run synchronously on every query. Tool outputs are untrusted evidence, never instructions.
4. **No real market action** — all dispatch recommendations are advisory/simulation only. The system has no write path to AEMO, MSATS, or any broker.
5. **Framework boundary** — `core/` and `engines/` have zero imports from `data/`, `mcp/`, or `domain/nem/`. Framework components are reusable across domains.

---

## System layers

```
┌─────────────────────────────────────────────────────┐
│  Frontend (Alpine.js + ECharts)                     │
│  app.html · state.js · api.js                       │
│  Charts: swimlane · analog · backtest_pnl           │
│  Panels: evidence · trace · backtest · security     │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP (FastAPI)
┌───────────────────▼─────────────────────────────────┐
│  API Layer                                          │
│  routes_query · routes_market · routes_trace        │
│  routes_backtest · routes_security · routes_sessions│
│  auth (JWT) · deps · security_observer (4-pass)     │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Agents                                             │
│  scatter_gather (T1 dispatch · T2 notices ·         │
│                  T3 analogs · T4 forecast)          │
│  why_sources · why_builder · why_formatter          │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Engines (framework boundary — no NEM imports)      │
│  decomposition (Ollama → Claude → rule-based)       │
│  chronograph/ (ADWIN · t-digest · RegimeClassifier) │
│  hippograph/  (graph · embedder · PPR)              │
│  lnn/         (ltc_cell · ltc_model · distribution ·│
│                trainer)                             │
│  forecasting/ (inference · harness · baselines ·   │
│                LNNQuantileModel)                    │
│  backtest     (run_region_backtest · _LTCAdapter)   │
│  analog_retriever                                   │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│  Data / MCP / Domain                                │
│  aemo_live_client · cache · scheduler               │
│  mcp/: registry · router · aemo_archive ·           │
│        aemo_notices_client                          │
│  domain/nem/: adapter (classify_regime)             │
│  db/: models · session · alembic migrations         │
└─────────────────────────────────────────────────────┘
```

---

## Key data flows

### Query pipeline (~1s standard path)

```
POST /sessions/{id}/query
  │
  ├─ SecurityObserver.pass_input()         [pass 1]
  ├─ decompose() → QueryDecomposition
  ├─ SecurityObserver.pass_decomposition() [pass 2]
  │
  ├─ scatter_gather()                      [parallel]
  │    T1: dispatch price (AEMO NEMWeb)
  │    T2: market notices (cache)
  │    T3: analogs (HippoGraph PPR)
  │    T4: forecast (LNN trainer buffer)
  │
  ├─ SecurityObserver.pass_tool_output()   [pass 3]
  ├─ assemble_why_sources()
  ├─ build_why()  → WhyOutput (deterministic)
  ├─ format_verdict() → FactualVerdict
  ├─ SecurityObserver.pass_answer()        [pass 4]
  ├─ write_trace() → Trace (bitemporal)
  └─ return QueryResponse {verdict, decomposition, viewport_type}
```

### ChronoGraph streaming update (every 5 min via scheduler)

```
dispatch_refresh job
  → AEMOLiveClient.fetch_latest_snapshot()
  → cache.set("dispatch_snapshot", ...)
  → feed_interval(region, {...})     [LNN trainer buffer]
  → RegimeClassifier.observe(price)  [ADWIN + t-digest]
```

### LNN training lifecycle

```
Day 0  : 288 intervals accumulated (min_train threshold)
Day 0+ : lnn_retrain job (hourly) calls maybe_train()
         → LTCTrainer.train() → pinball loss on rolling window
         → weights saved to data/lnn_weights/ltc_{region}.pt
Day 1+ : scatter_gather T4 returns P10/P50/P90
         → ForecastDrivers populated with quantiles
         → why_builder narrates the forecast
```

---

## Bitemporal trace

Every query writes two timestamps:

| Field | Meaning |
|---|---|
| `valid_time` | When the market data was true (AEMO dispatch interval) |
| `system_time` | When GridVerdict processed the query |

This enables retrospective queries: "What did GridVerdict know at T1 about market state at T2?"

---

## Security constraints (hard-coded, not configurable)

- No real market action execution — ever
- Portfolio-sensitive data never sent to cloud models without approval
- LLM never produces numbers or asserts facts
- System prompt is static; user text only in `role=user`
- Tool outputs are untrusted evidence (observer screens all)
- Risk score ≥ 80 halts the pipeline and returns 400

---

## Framework boundary rule

`app/engines/` and `app/core/` must have **zero imports** from:
- `app/data/`
- `app/mcp/`
- `domain/nem/`

The NEM-specific `NEM_REGIME_THRESHOLDS` and `classify_regime` are injected
as parameters at the boundary (in `why_sources.py` and `analog_retriever.py`).

---

## Configuration

| File | Purpose |
|---|---|
| `config/settings.py` | Pydantic settings (env vars / .env file) |
| `config/models.yaml` | LLM tier config (Ollama → Claude → rule-based) |
| `config/tools.yaml` | MCP tool registry (poll intervals, staleness thresholds) |

---

## Test coverage

```
tests/test_security_observer.py   43 tests  — 4-pass security observer
tests/test_verdict_contract.py    34 tests  — Factual Verdict Contract
tests/test_acceptance_matrix.py   31 tests  — 11 canonical queries end-to-end
                                 108 total
```

All tests use SQLite in-memory + AsyncMock for scatter_gather.
No external network calls in test suite.
