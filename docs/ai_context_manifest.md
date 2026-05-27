# GridVerdict — AI Onboarding Manifest

**Purpose:** A curated file index and reading guide for any AI (or human) that needs to
understand how GridVerdict works from first principles. Structured to match the pipeline
columns in `docs/query_trace_explainer.md`.

> **Read that document first.** This manifest assumes you have read `query_trace_explainer.md`
> and understand the four-stage pipeline: Decompose → Gather → Reason → Answer.

---

## Recommended Reading Order for Opus 4.7

Do not load all files. Load in this order, stopping when you have enough context for the task:

| Pass | Files | What you learn |
|------|-------|----------------|
| **1 — Orientation** | `docs/query_trace_explainer.md` | The full pipeline traced through 3 real queries |
| **2 — Contracts** | `app/core/schema.py` | Every data type that flows between stages |
| **3 — Conductor** | `app/api/routes_query.py` | The full pipeline in one function (~500 lines) |
| **4 — Stage brains** | `app/engines/decomposition.py`, `app/agents/why_builder.py`, `app/agents/answer_planner.py` | The three files that contain most of the intelligence |
| **5 — Evidence bus** | `app/agents/scatter_gather.py`, `app/agents/why_sources.py` | How evidence is collected and assembled |
| **6 — Models** | `docs/model_cards/` (4 files) | LEAR, QRA, LNN, MetaEnsemble capabilities and limitations |
| **7 — Security** | `app/security/observer.py` | The 4-pass safety layer |

Everything else is support infrastructure. Ask for specific files if needed.

---

## The Data Contracts (What Flows Between Stages)

Understanding what data flows between stages is more important than reading every file.

### Raw query → `QueryDecomposition`
```
QueryDecomposition:
  raw_query:         str            — original user text, never modified
  intent:            IntentLabel    — EXPLANATION | RETROSPECTIVE | ACTION_RECOMMENDATION | ...
  entities:          dict           — {regions: ["NSW1"], technologies: ["coal", "hydro"]}
  requires_why:      bool           — triggers WhyBuilder causal chain
  requires_history:  bool           — triggers recent_dispatch DB query
  requires_forecast: bool           — triggers forecasting model suite
  requested_output:  str            — routing key for AnswerPlanner (see below)
  causal_targets:    list[str]      — what to investigate: ["price", "demand", "weather", ...]
  confidence:        float          — decomposer confidence in its own classification
```

`requested_output` is the most important field. It is a string like
`"price_fluctuation_attribution"` or `"fuel_source_recommendation"` that acts as the
routing key in `plan_answer()`. Every planner path has a matching function.

### `QueryDecomposition` + `GatherResult` → `WhySources`
```
WhySources:
  decomp:           QueryDecomposition  — the original classification
  current:          CurrentDrivers      — live dispatch: price, demand, headroom, regime, staleness
  forecast:         ForecastDrivers     — model outputs: direction, P10/P50/P90, model_detail list
  analogs:          AnalogSummary       — HippoGraph: count, outcome_summary, top_items
  news:             NewsContext         — AEMO notices, RSS items, explained flag
  weather:          WeatherContext      — consensus, temperature, wind, cloud, source_count
  drivers:          DriverContext       — binding_constraints, tight_interconnectors
  technology:       TechnologyContext   — has_unit_evidence, unit_events
  recent_dispatch:  list[dict]          — last ~70 min of dispatch rows from DB
```

This is the single bundle that all of Stage ③ reads from. Nothing downstream re-fetches data.

### `WhySources` → `WhyOutput`
```
WhyOutput:
  contributing_factors:  list[dict]        — [{label, tier, present, confidence}]
  claim_map:             list[ClaimMapItem] — typed claims with evidence_ref_ids
  missing_data:          list[str]         — what was sought but not found
  evidence_refs:         list[EvidenceRef]  — every fact with source + interval + field
  confidence:            float             — weighted tier score
  next_watch:            list[str]         — recommended monitoring items
  counterargument:       str               — adversarial critic output
  answer_sections:       list[dict]        — formatted section titles + items (set in Stage ④)
```

### `WhyOutput` → `FactualVerdict` → final response
```
FactualVerdict:
  verdict:              VerdictLabel      — SUPPORTED | LOW_CONFIDENCE | INSUFFICIENT_DATA | ...
  action:               ActionLabel       — DISPATCH | HOLD | MONITOR | STAND_DOWN
  confidence:           float
  confidence_band:      ConfidenceBand    — VERY_LOW | LOW | MEDIUM | HIGH
  why_plain_english:    str               — long audit narrative (NOT shown in chat)
  answer_sections:      list[dict]        — SHORT visible sections (shown in chat)
  evidence_refs:        list[EvidenceRef]
  claim_map:            list[ClaimMapItem]
  missing_data:         list[str]
  counterargument:      str
  disclaimer:           str               — always present, never skippable
  trace_id:             str
```

The critical distinction: `why_plain_english` is the full audit narrative written by
`WhyBuilder`. `answer_sections` is the short structured answer written by `AnswerPlanner`.
The chat UI shows `answer_sections`. `why_plain_english` is available in the Sources panel.

---

## File Manifest by Pipeline Stage

---

### DOCUMENTATION (read first)

| File | Purpose |
|------|---------|
| `docs/query_trace_explainer.md` | **Start here.** Three live queries traced through the full pipeline with NEM market context explaining why each answer looks the way it does |
| `docs/architecture_walkthrough.md` | High-level architecture overview — less detailed than the trace explainer but broader in scope |
| `docs/model_explainer.md` | Plain-language explanations of LEAR, QRA, LNN, Meta-ensemble for non-ML audiences |
| `docs/known_limitations.md` | What the system explicitly cannot do and why — important for honest capability scoping |
| `docs/model_cards/LEAR.md` | LEAR model: inputs, outputs, calibration status, caveat conditions |
| `docs/model_cards/QRA.md` | QRA model: quantile regression averaging, ensemble method |
| `docs/model_cards/LNN.md` | LNN: Liquid Neural Network (continuous-time RNN), training requirements |
| `docs/model_cards/MetaEnsemble.md` | Meta-ensemble: how LEAR + QRA + LNN outputs are blended |

---

### CONTRACT / SCHEMA (the data definitions)

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/core/schema.py` | **The single most important file.** Every data type that flows between pipeline stages. Contains: `QueryDecomposition`, `IntentLabel`, `FactualVerdict`, `VerdictLabel`, `ActionLabel`, `EvidenceRef`, `ClaimMapItem`, `ClaimType`, `ConfidenceBand` | `ClaimType` enum (13 values), `EvidenceRef` (source + interval + tier), `QueryDecomposition.requested_output` field |
| `app/agents/why_sources.py` | Defines `WhySources` — the evidence bundle that Stage ③ reads from. Also defines `CurrentDrivers`, `ForecastDrivers`, `AnalogSummary`, `NewsContext`, `WeatherContext`, `DriverContext`, `TechnologyContext` | How `WhySources` is assembled from a `GatherResult` in `assemble_why_sources()` |
| `app/engines/forecasting/types.py` | Forecasting model output types: `ForecastResult`, `QuantileForecast` | P10/P50/P90 structure, `calibrated` flag |

---

### ENTRY POINT — The Conductor

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/api/routes_query.py` | **The full pipeline in one place.** The `submit_query` endpoint orchestrates all four stages in sequence. Reading this top-to-bottom gives the complete picture of how a query travels from HTTP request to HTTP response. ~600 lines. | Steps 0–7 comments in the function body, the `plan_answer()` call site, `_load_recent_dispatch_context()` |
| `app/api/main.py` | App factory: mounts all routes, runs startup (AEMO warm, HippoGraph rebuild, scheduler start) | `lifespan()` function — what happens at startup |
| `config/settings.py` | All configurable parameters: DB URL, AEMO endpoints, model paths, feature flags, decomposer backend | `decomposer_backend` (ollama/claude/rule_based), `gridverdict_dev_no_auth` |

---

### STAGE ① — NLP Decompose + Security

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/engines/decomposition.py` | **The decomposer.** Tries Ollama → Claude → rule-based in sequence. Rule-based is the deterministic fallback and is what tests use. Contains `_decompose_rules()`, `_requested_output_for()`, `_extract_causal_targets()` | `_requested_output_for()` — the routing key assignment logic. `_decompose_rules()` — the keyword matching that determines intent |
| `app/security/observer.py` | **4-pass SecurityObserver.** Runs at input, decomposition, tool output, and answer stages. Blocks: prompt injection, unsafe action intent, anomalous market data, hallucinated price assertions. | `pass_input()`, `pass_decomposition()`, `pass_tool_output()`, `pass_answer()`. The signal types and halt conditions |
| `app/engines/geo_aliases.py` | NEM region alias table: maps place names ("Sydney", "Penrith", "Griffith") to NEM region codes. Prevents the LLM from hallucinating non-existent regions | `detect_regions()`, `correct_region_hallucinations()` |
| `app/engines/temporal_utils.py` | Season bucket extraction: maps date references ("last summer", "winter peak") to calendar ranges for historical queries | `extract_season_buckets()` |

---

### STAGE ② — Gather (MCP Evidence Collection)

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/agents/scatter_gather.py` | **The evidence orchestrator.** Fires parallel async fetches for all evidence sources. Returns a `GatherResult`. Each fetch is guarded by the decomposition flags — weather only if relevant, analogs only if `requires_why`, etc. | `scatter_gather()` — the main async function. `GatherResult` dataclass. How `source_statuses` is populated |
| `app/data/aemo_live_client.py` | Fetches live dispatch price, demand, availability from the AEMO NEMWeb API. The only CONF-tier data source for current market state. | `fetch_latest_snapshot()`, `DispatchSnapshot`, rate limiting and fallback logic |
| `app/mcp/aemo_notices_client.py` | Fetches AEMO market notices (outage declarations, constraint activations, RERT notices). An AEMO notice is the strongest confirmation of a known market event. | `fetch_notices()`, notice classification by type |
| `app/mcp/weather_client.py` | Aggregates weather from multiple sources into a consensus reading. Returns temperature, wind speed, cloud cover, and confidence. | `weather_query_relevant()` — the gate function that decides whether to fetch. `WeatherConsensus` structure |
| `app/mcp/nem_news_client.py` | Fetches NEM-relevant RSS news. WattClarity is the primary source. Contextual only — not used for forecasting features, only for narrative context | `fetch_news_items()`, relevance filtering |
| `app/engines/hippograph/graph.py` | **HippoGraph** — the analog retrieval engine. Searches the graph of historical dispatch states for moments similar to the current state. Returns ranked analogs with outcome labels. | `retrieve_analogs()`, the similarity scoring function, `rebuild_from_db()` for startup warm |
| `app/engines/hippograph/ppr.py` | Personalized PageRank re-ranker for analogs. Re-ranks HippoGraph results using driver context (constraints, weather, notices) as edge weights | `rerank_analogs_with_drivers()` |
| `app/engines/temporalrag/retriever.py` | TemporalRAG bitemporal document retrieval. Queries the document store by valid_time range and system_time-at-query. Returns only documents that were "known" at query time. | `retrieve()`, `TemporalQuery`, the `system_time <= query_time` constraint |
| `app/engines/temporalrag/schema.py` | TemporalDocument schema: `doc_id`, `source_type`, `valid_time`, `system_time`, `content`, `relevance_score` | The bitemporal index design |
| `app/engines/fuel_mix.py` | Fuel mix model: ranks fuel types by marginal cost against current spot price. Returns merit order with data tier annotation (`dispatch`, `prior`, or `estimated`). | `get_fuel_mix()`, merit order computation, data tier assignment |
| `app/engines/driver_attribution.py` | Queries the DB for logged market driver events (constraint activations, interconnector flows) in the vicinity of the query time. | `retrieve_market_drivers()`, driver event schema |
| `app/engines/unit_attribution.py` | Queries the DB for unit-level dispatch events. If available, shows which generator (and fuel type) was marginal at each interval. Usually unavailable unless NEMWeb unit-dispatch data has been ingested. | `retrieve_unit_dispatch()`, DUID-to-fuel-type mapping |
| `app/mcp/source_status.py` | `SourceStatus` — records whether each evidence source was fresh, stale, or unavailable. These statuses flow through to evidence tier assignment. | `SourceStatus.to_dict()`, status categories |
| `app/data/cache.py` | Redis-backed in-memory cache for dispatch snapshots, forecast results, weather consensus. Avoids re-fetching for concurrent queries. | `MarketCache.get()`, `MarketCache.set()`, TTL configuration |

---

### STAGE ③ — Reason (WhyBuilder + Models + Adversarial Critic)

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/engines/why_builder.py` | **The causal reasoning engine.** Takes a `WhySources` bundle and produces a `WhyOutput`. Assigns evidence tiers, computes confidence, builds the claim map, runs next-watch logic, constructs the adversarial counterargument. | `build_why()` — the main function. `_build_claim_tiers()` — tier assignment. `_build_claim_map()` — typed claim construction. `_build_next_watch()` — monitoring signal generation. Staleness check logic |
| `app/agents/claim_verifier.py` | **The adversarial critic.** Verifies that each claim in the verdict has supporting evidence. Downgrades claims that lack evidence_ref IDs. Produces findings that are appended to the verdict. | `verify_answer()`, `apply_verification()`, downgrade conditions |
| `app/agents/why_formatter.py` | Formats the `WhyOutput` into a `FactualVerdict`. Maps confidence scores to `ConfidenceBand`, selects `VerdictLabel` and `ActionLabel`, writes `why_plain_english`. | `format_verdict()`, the confidence → verdict mapping table |
| `app/engines/forecasting/live_forecast.py` | Orchestrates the live forecast: runs all available models (LEAR, QRA, LNN, TCN, Meta-ensemble) for the current interval, aggregates results, returns `ForecastDrivers`. | `run_live_forecast()`, model availability checks, P10/P50/P90 aggregation |
| `app/engines/forecasting/models/lear_model.py` | **LEAR** — Least Absolute shrinkage Estimation with Recursive updates. The primary short-term price forecaster. Wide P90 range reflects fat-tail spike risk. | `predict()`, feature vector, recursive update logic |
| `app/engines/forecasting/models/qra_model.py` | **QRA** — Quantile Regression Averaging. Blends multiple quantile regression models. Tighter intervals than LEAR but potentially less robust to regime shifts. | `predict()`, quantile blend weights |
| `app/engines/lnn/ltc_model.py` | **LNN** — Liquid Neural Network (continuous-time RNN). The most computationally expensive model. Requires a trained checkpoint to produce outputs; returns "untrained" caveat if checkpoint absent. | `LTCModel.forward()`, the continuous-time ODE cell, checkpoint loading |
| `app/engines/lnn/ltc_cell.py` | The LTC (Liquid Time Constant) cell implementation. Each cell has a continuous-time state equation governed by learnable time constants. | The ODE formulation, `tau_sys` (time constant) parameter |
| `app/engines/forecasting/models/tcn_model.py` | **TCN** — Temporal Convolutional Network. The deep neural network forecaster. Causal dilated convolutions over the recent price-demand time series. | `TCNModel`, dilated causal convolution architecture |
| `app/engines/forecasting/models/meta_ensemble.py` | **Meta-ensemble.** Blends LEAR + QRA + LNN outputs with dynamically computed weights. The primary model shown to users when all constituent models are available. | `MetaEnsembleModel.predict()`, weight computation, fallback when constituent models fail |
| `app/engines/forecasting/models/gbm_model.py` | GBM (Gradient Boosting Machine) feature-importance model. Not a primary forecaster — used for feature attribution and driver identification. | Feature importance output, which features dominate at each regime |
| `app/engines/forecasting/evaluation/calibration.py` | Calibration evaluation: checks whether P10/P50/P90 intervals contain the right fraction of actual outcomes. A model with miscalibrated intervals gets a confidence penalty. | CRPS computation, coverage rate checks |
| `app/engines/forecasting/features/market_features.py` | Feature engineering: constructs the input vector for all forecasting models from raw dispatch data. Price lags, demand lags, time-of-day encoding, seasonal flags, weather features. | `build_feature_vector()`, time-of-day and day/night cycle features |
| `app/engines/spike_detector.py` | Detects price spikes (crossing $300, $1000, $15000 thresholds) from live dispatch ticks. Fires events to the commentary engine when spikes are detected. | `SpikeDetector.process_tick()`, threshold configuration |
| `app/engines/chronograph/regime.py` | Classifies the current market regime: NORMAL, ELEVATED, HIGH, SPIKE, MANAGED_RECOVERY. Used as a feature in analogs, forecasts, and answer routing. | `classify_regime()`, the price-headroom threshold matrix |
| `app/engines/chronograph/adwin.py` | ADWIN (Adaptive Windowing) change detection. Detects when the dispatch price distribution has shifted — used to flag regime changes in real time. | `ADWIN.update()`, change detection trigger |
| `app/engines/chronograph/tdigest.py` | T-Digest approximate quantile computation. Maintains a streaming quantile estimator over recent prices without storing all historical values. | `TDigest.update()`, `TDigest.quantile()` |
| `app/engines/rebid_engine.py` | Analyses rebid events: identifies generators that moved their bid prices significantly between dispatch intervals. If a rebid is detected at the query time, it is the most likely driver of a short-duration price spike. | `analyse_rebids()`, the bid-change threshold for "material rebid" classification |
| `app/engines/interconnector.py` | Monitors NSW1 interconnector flows (QNI, VIC1-NSW1). Detects when interconnectors are constrained. Constrained interconnectors remove cheap interstate supply, forcing local high-cost dispatch. | `check_interconnector_flows()`, constraint detection logic |
| `app/engines/incident_timeline.py` | Reconstructs the timeline of a market incident from stored events, notices, and dispatch history. Used for the Incident Brief panel. | `build_incident_timeline()`, event ordering and deduplication |

---

### STAGE ④ — Answer Planner + Formatter

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/agents/answer_planner.py` | **The answer formatter.** Routes to the correct planner function based on `requested_output`. Never calls a language model — reads from the evidence bundle and formats deterministically. Contains `_extract_query_price_path()` for parsing user-mentioned prices. | `plan_answer()` — the routing switch. Each `_plan_*()` function. `_trend_line()`, `_price_movement_line()`, `_extract_query_price_path()`. The `_cap()` function that limits section items to 3 |
| `app/agents/why_formatter.py` | Converts `WhyOutput` into `FactualVerdict`. Assigns verdict label, action label, confidence band. Writes `why_plain_english` as the full audit narrative. | `format_verdict()`, the confidence-to-verdict mapping |

---

### LIVE FEED — Commentary Engine (Separate Pipeline)

The Live Feed is a parallel pipeline that runs automatically on every dispatch tick,
independent of user queries. It monitors for material market changes and fires Why Engine
analysis without a user asking.

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/engines/commentary/engine.py` | **CommentaryEngine** — the Live Feed brain. Runs on every scheduler tick: loads the previous snapshot, detects changes, runs the Why Engine, stores the event, publishes to the event bus. | `process_tick()`, `_build_event()`, `_build_baseline_event()`, cooldown management |
| `app/engines/commentary/detector.py` | Detects material changes between consecutive dispatch snapshots: price spikes, headroom tightening, regime changes, constraint activations, weather pressure, data staleness. | `detect()`, `ChangeType` enum, `_CHANGE_DECOMPOSITION` mapping, `_COOLDOWN_SECONDS` |
| `app/engines/commentary/snapshot.py` | `RegionSnapshot` — the state object stored in Redis between ticks. Fields: price, demand, headroom, regime, spike probabilities, staleness, binding constraints, weather pressure. | `save_snapshot()`, `load_snapshot()`, the new Sprint R fields |
| `app/engines/commentary/prose.py` | Formats commentary event headlines for each `ChangeType`. Deterministic — no LLM. | `format_headline()`, `format_factors()` |
| `app/engines/commentary/store.py` | Persists `CommentaryEvent` objects to the database and provides recent-event retrieval for the API. | `write_event()`, `has_recent_baseline()` |
| `app/api/routes_commentary.py` | API endpoints for the Live Feed: `/api/commentary/recent`, `/api/commentary/{id}` | Event serialization, region filtering |
| `app/api/routes_events.py` | SSE (Server-Sent Events) stream endpoint `/api/events/stream`. Pushes commentary events, dispatch updates, and forecast updates to the browser in real time. | The `EventSource` handler, event type routing |

---

### DATA / INFRASTRUCTURE

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/db/models.py` | SQLAlchemy ORM models: `MarketEvent`, `Query`, `Session`, `AuditLog`, `SecurityEvent`, `CommentaryEvent`, `TemporalDocument`. The database schema. | `MarketEvent` (the dispatch price store), `Query` (every user query with its decomposition and answer), `CommentaryEvent` (Live Feed events) |
| `app/db/session.py` | Async SQLAlchemy session factory. Handles SQLite (dev/test) and PostgreSQL (production). `init_db()` creates tables from models. | `get_db()`, `init_db()` |
| `app/data/scheduler.py` | APScheduler background job: fires `dispatch_refresh` every 5 minutes and `archive_backfill` hourly. The `dispatch_refresh` job is what drives the Live Feed commentary engine on every tick. | `start_scheduler()`, `dispatch_refresh()`, `archive_backfill()` |
| `app/data/event_bus.py` | In-process event bus. Commentary events and market updates are published here; the SSE endpoint subscribes to broadcast to the browser. | `publish()`, `subscribe()`, the async queue |
| `app/data/redis_client.py` | Redis client factory. Used for: market cache, HippoGraph persistence, commentary cooldowns, snapshot storage. Falls back gracefully if Redis is unavailable. | `get_redis()`, connection pooling, the `None` fallback path |
| `app/engines/analog_retriever.py` | Loads and re-ranks analog results from HippoGraph for the query handler. Wraps `hippograph.graph` with driver-context re-ranking. | `rerank_analogs_with_drivers()` |
| `app/engines/participant_profiler.py` | Builds profiles of market participants (generators/retailers) from historical bidding behaviour. Used for detecting unusual participant behaviour as a price driver. | `profile_participant()`, bid pattern analysis |

---

### SECURITY LAYER

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/security/observer.py` | **The 4-pass SecurityObserver.** The most important security file. Each pass has a different concern and can halt the pipeline independently. | `pass_input()` — injection and OOS detection. `pass_decomposition()` — unsafe intent detection. `pass_tool_output()` — anomalous data detection. `pass_answer()` — hallucination and financial advice detection. `ObserverSignal.severity`, `ObserverResult.should_halt()` |
| `app/audit/audit_logger.py` | Structured audit log for every observer event, query, and verdict. Append-only. | `log_event()`, event schema |
| `app/compliance/ai_risk_register.py` | AI risk register: documents known risks, mitigations, and residual risks for the platform. | Risk IDs, mitigation mappings |
| `app/compliance/aescsf.py` | AESCSF (Australian Energy Sector Cyber Security Framework) compliance checks | Security control validations |

---

### PORTFOLIO / BESS (Battery Energy Storage)

| File | Purpose | Key things to read |
|------|---------|-------------------|
| `app/portfolio/bess_engine.py` | BESS dispatch optimiser: given current price, forecast, and storage level, computes the optimal charge/discharge decision. `simulation_only=True` always — never executes real dispatches. | `compute_dispatch()`, the simulation-only constraint, opportunity cost model |
| `app/portfolio/dispatch_policy.py` | Defines dispatch policy rules for BESS: when to charge, when to discharge, threshold-based and forecast-based modes. | `DispatchPolicy`, threshold configuration |
| `app/portfolio/fleet_coordinator.py` | Multi-unit fleet coordinator: aggregates decisions across multiple BESS units or portfolio positions. | `coordinate_fleet()` |

---

### TESTING (behaviour documentation)

Tests are the most accurate documentation of system behaviour. These are the most useful
for understanding what the system guarantees:

| File | What it documents |
|------|-------------------|
| `tests/test_answer_planner.py` | Every planner routing path with assertions on output content. Shows exactly what each query type produces. Includes price-path extraction tests. |
| `tests/test_why_engine.py` | WhyBuilder invariants: confidence computation, claim tier assignment, staleness downgrade logic |
| `tests/test_sprint_r.py` | Sprint R: evidence claim map, staleness cap, new change types. The most recent complete feature test suite |
| `tests/test_sprint_q.py` | Sprint Q: Live Feed commentary quality, baseline events, cooldowns |
| `tests/test_decomposer_quality.py` | Decomposer routing accuracy: 40+ query fixtures with expected intent labels |
| `tests/test_causality_tiers.py` | Evidence tier assignment invariants |
| `tests/test_claim_verifier.py` | Adversarial critic / claim verifier rules |
| `tests/test_security_observer.py` | All 4 observer passes with injection, hallucination, and OOS fixtures |
| `tests/e2e/test_price_fluctuation_gap.py` | End-to-end gap test for the fluctuation query: API routing + UI rendering + price path extraction |
| `tests/e2e/test_frontend.py` | Browser smoke tests: JS bootstrap, Alpine.js init, vendor assets |
| `tests/e2e/test_live_feed.py` | Live Feed UI tests: commentary cards, SSE push, dismiss, ask-about |

---

## Key Invariants (What the System Guarantees)

These are not aspirational — they are enforced by the architecture:

1. **The LLM never produces numbers.** The decomposer LLM classifies intent only. The answer planner is a deterministic function. No number in any answer was generated by a language model.

2. **Every claim has an evidence_ref.** `ClaimMapItem` objects without `evidence_ref_ids` are downgraded from CONFIRMED to PLAUSIBLE by the `ClaimVerifier`.

3. **Missing data is always surfaced.** The `WhyBuilder` builds the `missing_data` list from sources that were sought but unavailable. The `AnswerPlanner` includes a Missing section in every answer. There is no path to a clean answer without showing what is absent.

4. **Confidence is computable, not asserted.** Confidence is a weighted function of the tier mix in the evidence bundle. It cannot be manually set or overridden by a language model.

5. **The adversarial critic runs on every verdict.** The `ClaimVerifier` produces a counterargument for every answer. An answer without a counterargument is a system error, not a feature.

6. **`simulation_only = True` always.** The BESS engine and all portfolio components carry this flag. No code path exists that would execute a real market action.

7. **The 4-pass observer cannot be bypassed.** The observer runs before data is fetched, before reasoning begins, on raw tool outputs, and on the final answer. Each pass can halt the pipeline independently.

---

## My Disagreement on the "Upload All Files" Approach

Raw file dumps are a poor way to brief an AI on a codebase. Here is why, and what to do instead.

**The problem with uploading 50+ Python files:**

- An AI reading 50 files cold spends most of its context budget on boilerplate: imports,
  `__init__.py` files, migration scripts, `model_config = {}` lines. The signal-to-noise
  ratio is low.
- Files don't explain intent. `why_builder.py` is 600 lines of Python. Reading it tells
  you what the code does. It does not tell you *why* the evidence tier system was designed
  the way it was, what bug the staleness cap was introduced to fix, or why the
  adversarial critic runs before the answer reaches the user rather than after.
- Context is consumed in proportion to file size, not importance. `app/compliance/aescsf.py`
  is large but rarely relevant. `app/core/schema.py` is smaller but critical.

**What to do instead:**

Upload in this priority order and stop when the AI has enough to answer your question:

| Priority | Upload | Reason |
|----------|--------|--------|
| 1 | `docs/query_trace_explainer.md` | Architecture + live examples + NEM market context |
| 2 | `app/core/schema.py` | All data contracts in ~400 lines |
| 3 | `app/api/routes_query.py` | The full pipeline orchestration |
| 4 | `app/engines/decomposition.py` | Stage ① brain |
| 5 | `app/agents/why_builder.py` | Stage ③ brain |
| 6 | `app/agents/answer_planner.py` | Stage ④ brain |
| 7 | Ask for specific files | Let the AI request what it needs for the specific task |

This 6-file set (plus this manifest) gives Opus 4.7 enough to understand the full
pipeline, answer architecture questions, suggest improvements, and write new tests or
features — without wasting context on boilerplate.

**The files NOT worth uploading unless specifically needed:**

- `__init__.py` files (empty or near-empty)
- Migration scripts (`app/db/migrations/`)
- Compliance boilerplate (`app/compliance/`)
- Vendor-specific adapters (`app/data/wa_client.py` — Western Australia grid, not NEM)
- Evaluation harnesses (`app/engines/forecasting/evaluation/`)
- Export utilities (`app/audit/export.py`)

---

## File Count Summary

| Category | File count | Upload priority |
|----------|-----------|-----------------|
| Documentation | 9 | High — read first |
| Schema / contracts | 3 | Critical |
| Entry point | 3 | Critical |
| Stage ① Decompose + Security | 4 | High |
| Stage ② Gather (MCP) | 14 | Medium — fetch relevant ones |
| Stage ③ Reason (models) | 13 | Medium — fetch relevant ones |
| Stage ④ Answer | 2 | High |
| Live Feed | 6 | Medium |
| Data / Infrastructure | 8 | Low |
| Security / Audit | 4 | Medium |
| Portfolio / BESS | 3 | Low |
| Tests | 11 | High for behaviour questions |
| **Total relevant** | **~80** | **Upload 6–8 for most tasks** |
