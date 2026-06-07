# GridVerdict

**Evidence-grounded decision-support cockpit for the Australian National Electricity Market (NEM).**

Ask natural-language questions about live market conditions. Get short, cited, defensible answers — every supported claim backed by a traceable evidence reference.

> The LLM decomposes the question. Deterministic code answers it with evidence.

```
docker compose up
```
→ Open `http://localhost:8000`

---

## Contents

- [What it is](#what-it-is)
- [ASCII Architecture](#ascii-architecture)
- [NLP Pipeline — how a question becomes an answer](#nlp-pipeline--how-a-question-becomes-an-answer)
- [Technical Components](#technical-components)
- [What Works vs Stub/Demo](#what-works-vs-stubdemo)
- [USP: What Delta Gap Does This Bridge?](#usp-what-delta-gap-does-this-bridge)
- [Skillsets Demonstrated](#skillsets-demonstrated)
- [Quick Start](#quick-start)
- [Test Suite](#test-suite)
- [Project Layout](#project-layout)
- [Design Constraints](#design-constraints)

---

## What it is

GridVerdict is a natural-language interface to the NEM — the interconnected electricity grid spanning Queensland, New South Wales, Victoria, South Australia, and Tasmania. It ingests live AEMO dispatch prices every 5 minutes, backfills 2+ years of historical data from the MMSDM archive, and answers questions like:

- *"Why is NSW price elevated right now?"* → tiered driver evidence (constraints, interconnectors, unit dispatch, rebids, weather, AEMO notices) with explicit not-confirmed flags for missing data
- *"Is this cheap compared to last year?"* → P25/median/P75/P90 for this hour/season window from the archive
- *"Earlier today wind was $15/MWh, why am I paying double now?"* → real intraday fuel timeline showing the solar cliff and dispatch transition
- *"What would my BESS have earned last quarter?"* → bitemporal backtest with no-lookahead replay
- *"What's the price forecast for the next 30 minutes?"* → LEAR + QRA + GBM + TCN + LNN meta-ensemble, P10/P50/P90
- *"Compare all NEM states right now"* → parallel scatter-gather across all 5 regions, cross-region spread analysis

It is **not** a chatbot that synthesises an answer from training data. The LLM does one job — decompose the question into a typed schema. Every number in the answer comes from a live data fetch or an archived database row, and every `SUPPORTED` verdict requires a cited `evidence_ref` before it can be returned.

---

## ASCII Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GRIDVERDICT PLATFORM                               │
│                                                                                 │
│  ┌──────────────┐   HTTP/SSE    ┌──────────────────────────────────────────┐   │
│  │  Alpine.js   │◄─────────────►│           FastAPI  /api/v1               │   │
│  │  SPA + SSE   │               │  auth · sessions · query · market ·      │   │
│  │  ECharts     │               │  events · incidents · models · trace ·   │   │
│  └──────────────┘               │  backtest · compliance · portfolio        │   │
│                                 └──────────────┬───────────────────────────┘   │
│                                                │                               │
│                        ┌───────────────────────▼──────────────────────────┐    │
│                        │         SECURITY OBSERVER  (4-pass pipeline)     │    │
│                        │  Pass 1: Input hygiene (injection, PII, manip)   │    │
│                        │  Pass 2: Decomposition validation                 │    │
│                        │  Pass 3: Tool output hygiene (untrusted data)     │    │
│                        │  Pass 4: Answer guard (evidence_ref required)     │    │
│                        └───────────────────────┬──────────────────────────┘    │
│                                                │                               │
│                ┌───────────────────────────────▼──────────────────────────┐    │
│                │                  DECOMPOSER  (hybrid)                    │    │
│                │                                                          │    │
│                │  Rule-based → always runs (safety gate, deterministic)   │    │
│                │       ↓                                                  │    │
│                │  LLM enrichment → Ollama (local) | Claude | skip         │    │
│                │       ↓                                                  │    │
│                │  Merge → rules win on routing; LLM wins on entities      │    │
│                │                                                          │    │
│                │  Output: QueryDecomposition                               │    │
│                │    intent: explanation | lookup | comparison | …(11)     │    │
│                │    requested_output: causal_explanation | fuel_source_…  │    │
│                │    sub_questions: [{type: historical_price_dist}, …]     │    │
│                │    entities: {regions, technologies, generators}          │    │
│                │    confidence: 0.0–1.0                                    │    │
│                └───────────────────────────────┬──────────────────────────┘    │
│                                                │                               │
│                ┌───────────────────────────────▼──────────────────────────┐    │
│                │              SCATTER-GATHER  (~330ms, async)             │    │
│                │                                                          │    │
│                │  T1 ─ AEMO dispatch price       (NEMWeb live)            │    │
│                │  T2 ─ AEMO market notices       (cache + live)           │    │
│                │  T3 ─ HippoGraph PPR analogs    (in-process graph)       │    │
│                │  T4 ─ LNN quantile forecast     (in-process PyTorch)     │    │
│                │  T5 ─ AEMO pre-dispatch 30min   (NEMWeb)                 │    │
│                │  T6 ─ NEM news RSS              (WattClarity RSS)        │    │
│                │  T7 ─ Weather consensus         (BOM, conditional)       │    │
│                │                                                          │    │
│                │  All tasks timeout-gated (8s). Failure = is_available    │    │
│                │  false in SourceStatus — never blocks the pipeline.      │    │
│                └───────────────────────────────┬──────────────────────────┘    │
│                                                │                               │
│                ┌───────────────────────────────▼──────────────────────────┐    │
│                │                    ENRICHMENT                            │    │
│                │                                                          │    │
│                │  TemporalRAG    — bitemporal evidence retrieval           │    │
│                │  FuelMix        — per-fuel-type dispatch breakdown        │    │
│                │  HistoricalDist — P10/P25/P50/P75/P90 from archive       │    │
│                │  IntradayFuel   — solar cliff + dispatch transition       │    │
│                │  OpenNEM        — real hourly/monthly trend data          │    │
│                │  BOM 7-day      — weather forecast for future queries     │    │
│                └───────────────────────────────┬──────────────────────────┘    │
│                                                │                               │
│                ┌───────────────────────────────▼──────────────────────────┐    │
│                │              WHY ENGINE  (deterministic)                 │    │
│                │                                                          │    │
│                │  WhySources  → normalise raw evidence into dataclasses   │    │
│                │  WhyBuilder  → tiered driver attribution + claim map     │    │
│                │  ClaimVerifier → downgrade unsupported claims            │    │
│                │  AnswerPlanner → build 3–5 line answer sections          │    │
│                │  CoverageAudit → re-route if sub-questions unmet         │    │
│                └───────────────────────────────┬──────────────────────────┘    │
│                                                │                               │
│                                      FactualVerdict                            │
│                                      + pipeline_events                          │
│                                      + evidence_refs                           │
│                                      + suggested_questions                     │
│                                                │                               │
└────────────────────────────────────────────────┼────────────────────────────────┘
                                                 │
                     ┌───────────────────────────▼──────────────────────────┐
                     │               DATA LAYER  (MCP registry)             │
                     │                                                       │
                     │  AEMO NEMWeb   → live dispatch, notices, pre-dispatch │
                     │  AEMO MMSDM    → 2yr historical archive (backfilled)  │
                     │  OpenElectricity API → real hourly/monthly trends     │
                     │  BOM forecast  → 7-day temperature + wind             │
                     │  WattClarity RSS → NEM news (retrospective only)      │
                     │  ISP / GBB     → AEMO scenarios, gas hub prices       │
                     │                                                       │
                     │  PostgreSQL (prod) / SQLite (dev)                    │
                     │  Redis event bus (optional SSE broadcast)            │
                     └──────────────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────────────┐
                     │              ML ENGINE  (engines/)                   │
                     │                                                       │
                     │  LNN/LTC  → Liquid Time-Constant ODE (from scratch)  │
                     │  ChronoGraph → ADWIN + t-digest streaming analytics  │
                     │  HippoGraph → market state graph + PPR analogs       │
                     │  TemporalRAG → bitemporal evidence retrieval          │
                     │  Forecast ensemble → LEAR | QRA | GBM | TCN | LNN    │
                     │  DriftMonitor → ADWIN on live forecast residuals      │
                     └──────────────────────────────────────────────────────┘
```

---

## NLP Pipeline — how a question becomes an answer

GridVerdict uses a **hybrid decomposer** where deterministic rules run first for safety and routing, and an LLM (optionally) enriches entity extraction afterward. The pipeline never lets the LLM generate market facts — it only classifies intent.

### Step 0: Security pass 1 — input hygiene

Before any processing, `SecurityObserver.pass_input()` scans for prompt injection patterns, PII, market manipulation language, and excessive length. If the risk score ≥ 80, the query is rejected with a 400 before the decomposer ever sees it.

### Step 1: Decomposition (hybrid rules + LLM)

**Rule-based classifier runs first, always.** It is deterministic and never fails. It handles:

| Query signal | Intent assigned |
|---|---|
| "should I dispatch / bid / act" | `action_recommendation` |
| "why is / what caused / reason for" | `explanation` |
| "last time / what happened / yesterday" | `retrospective` |
| "what if / simulate / counterfactual" | `counterfactual` |
| "compare / versus / difference" | `comparison` |
| "western australia / perth" | `geographic_redirect` |
| "solar panels on my house / feed-in tariff" | `partial_scope` |
| "interest rates → renewables" | `evidence_bridge` |
| "ignore your instructions" | `out_of_scope` (halt, score 99) |

The rules also extract: NEM regions (with 200+ Australian city aliases), technologies, price thresholds (`$150/MWh`), seasonal buckets (summer/winter), specific month+year, future date references, and intraday comparison patterns.

**Safety gates are hard stops.** `out_of_scope`, `partial_scope`, `evidence_bridge`, and `geographic_redirect` intents return immediately — the LLM never sees these queries.

**LLM enrichment** (Ollama/Claude/skip) runs for the remaining intents to improve entity extraction — particularly generator names, fuel types, and ambiguous geography. The LLM output is validated against the `QueryDecomposition` schema; any parse failure silently falls back to the rule-based result.

**Merge logic:** Rules win on routing, intent, and requires_* flags. LLM wins on entity extraction. The result is a fully typed `QueryDecomposition`:

```python
QueryDecomposition(
    intent=IntentLabel.EXPLANATION,
    requested_output="causal_explanation_with_forecast",
    entities={"regions": ["NSW1"], "technologies": ["coal", "gas"]},
    sub_questions=[
        {"type": "current_price_reason"},
        {"type": "historical_price_distribution", "period": "last_year"},
        {"type": "forecast_outlook"},
    ],
    confidence=0.87,
    requires_why=True,
    requires_history=True,
    requires_forecast=True,
)
```

### Step 1a: Ambiguity gate

If decomposer confidence < 0.62 AND a clarifying question is available AND the query is non-trivial (>8 words), the system returns a `NEEDS_CLARIFICATION` verdict asking the user to specify region, time period, or intent — rather than guessing.

### Step 1b: Adjacent query routing

Three intent types bypass scatter-gather entirely and are handled by `adjacent_handlers`:
- `PARTIAL_SCOPE` — e.g., household solar (answers the NEM portion: wholesale price signal; redirects for feed-in tariff rates)
- `EVIDENCE_BRIDGE` — e.g., "how do interest rates affect renewables?" (explains the LCOE mechanism from SRMC structure)
- `GEOGRAPHIC_REDIRECT` — e.g., WA/NT questions (explains the separate grid, redirects to correct authority)

### Step 2: ScatterGather (parallel, ~330ms)

Seven async tasks run simultaneously with `asyncio.gather`. Each is wrapped with a timeout (8s) and a `SourceStatus` provenance record. A failing task contributes `is_available=False` to confidence scoring — it never blocks the pipeline.

```
T1  AEMO live dispatch price           → price_rrp, demand_mw, availability_mw
T2  AEMO market notices (cache)        → LOR, RECLASSIFY, DIRECTIONS notices
T3  HippoGraph PPR analogs             → top-10 similar historical states
T4  LNN + LEAR + QRA ensemble          → P10/P50/P90 quantile forecast
T5  AEMO pre-dispatch 30min ahead      → directional price signal
T6  NEM news RSS (WattClarity)         → retrospective context only
T7  Weather consensus (conditional)    → BOM consensus when query is weather-relevant
```

Historical and comparison queries use alternate gather paths:
- **RETROSPECTIVE intent** → `scatter_gather_historical()` fetches evidence from the DB at the anchor time, not live
- **COMPARISON intent** → runs T1 in parallel for all detected regions (up to 5 simultaneous fetches)
- **Routing alignment check** → if historical anchor is >2h old for an EXPLANATION query, self-corrects to live scatter

### Step 3: Context enrichment

Runs sequentially after scatter-gather, using `db.begin_nested()` savepoints to isolate each query:

1. **TemporalRAG** — retrieves up to 12 bitemporal evidence documents from the archive (valid_time window ±4h, relevance-ranked by temporal proximity + source credibility)
2. **FuelMix** — per-fuel-type generation breakdown (coal/gas/hydro/solar/wind/battery) to support fuel source recommendation
3. **BOM 7-day forecast** — injected when `requires_forecast=True` and no weather gathered in T7
4. **OpenNEM/OpenElectricity API** — real monthly/hourly price data for trend and diurnal queries
5. **Historical price distribution** — `get_historical_price_distribution()` queries the archive for the same hour (±2h window), same calendar quarter, returning P10/P25/P50/P75/P90 and a classification (`cheap`, `normal`, `elevated`, `high`, `spike`)
6. **Specific period stats** — for "what was the average price in July 2023?" queries, exact DB aggregate
7. **Intraday prices + fuel timeline** — today's 5-minute prices and fuel dispatch transitions (solar cliff detection, dominant fuel by hour)
8. **Driver attribution** — binding constraints, marginal setter identification from the archive
9. **Unit dispatch** — per-unit cleared MW for fuel attribution

### Step 4: Why Engine (deterministic)

```
WhySources    → typed dataclasses: CurrentState, ForecastContext, DriversContext,
                 WeatherContext, NewsContext, AnalogsContext, FcasContext
WhyBuilder    → tiered claim map: T1=CONFIRMED | T2=SUPPORTED | T3=INFERRED | T4=SPECULATIVE
ClaimVerifier → downgrades SUPPORTED verdicts with missing evidence_refs
AnswerPlanner → selects one of 15+ named planners based on requested_output
```

The `AnswerPlanner` dispatches to a named planner based on `decomp.requested_output`. Each planner builds `PlannedAnswer(headline, direct_answer, key_evidence, drivers, continuation, missing)` deterministically from evidence — no LLM generation. Planners include:

| `requested_output` | Planner behaviour |
|---|---|
| `causal_explanation` | Driver tiering, constraint attribution, analog outcome |
| `fuel_source_recommendation` | Per-fuel SRMC ranking, diurnal cycle context, data tier label |
| `historical_price_distribution` | P10/P25/P50/P75/P90, classification, archive citation |
| `regional_comparison` | Multi-region price/demand/headroom table |
| `diurnal_analysis` | Real OpenNEM hourly data (Path A) or seasonal NEM norms (Path B) |
| `trend_analysis` | Real OpenNEM monthly data (Path A) or DB archive summary (Path B) |
| `future_date_price_forecast` | Time-of-day table, BOM weather, seasonal P10/P50/P90 |
| `specific_period_stats` | DB aggregate for exact month/year, vs 4yr historical median |
| `price_fluctuation_attribution` | Price path reconstruction, swing label, fuel driver |
| `portfolio_action` | BESS economics, FCAS opportunity summary |

**Sub-question handling:** When the decomposer detects ≥2 distinct sub-question types in a query, `_plan_multi_part()` builds one section per type rather than forcing a single narrative. This prevents compound questions from receiving hollow generic answers.

### Step 5: Coverage audit (adversarial critic)

If confidence < 0.6 AND the query restatement detected gap risk, `audit_coverage()` compares the detected sub-questions against the answer sections and may re-route to a more specific planner. This runs synchronously using Ollama with thinking mode (~4–8s) and is skipped when Ollama is offline.

### Step 6: Security pass 4 — answer guard

Every numeric claim in the answer must have an `evidence_ref` pointing to a specific DB row, API response, or model output. Any `SUPPORTED` verdict with no evidence_refs triggers a downgrade to `LOW_CONFIDENCE`. If the security observer scores the answer ≥ 80, it is replaced with an `INSUFFICIENT_DATA` verdict rather than returning potentially dangerous output.

### Step 7: Persist and return

Every query writes:
- `Query` row (raw text, decomposition JSON, answer JSON, intent, verdict, region)
- `Trace` row (valid_time + system_time pair for bitemporal replay, full tool call log, observer result)

The response carries `pipeline_events` — a timestamped decision trace showing every step, source, and timing milestone from `QUERY_RECEIVED` to `COMPLETE`.

---

## Technical Components

| Component | Implementation | What it actually does |
|---|---|---|
| **LTC cell** | `app/engines/lnn/ltc_cell.py` — PyTorch ODE from scratch | Liquid Time-Constant recurrent cell: `dh/dt = (-h + gate·A) / τ`, semi-implicit Euler. Time constant τ adapts to input volatility — SA1 spikes handled differently from TAS1 baseload |
| **LNN distribution** | `app/engines/lnn/distribution.py` | Quantile head (P10/P50/P90) with pinball loss. Trained on 11 features: price, demand, availability, headroom, TOD sin/cos, DOW, log-price, FCAS R6S, renewable %, constraint count |
| **ChronoGraph** | `app/engines/chronograph/adwin.py` + `tdigest.py` | ADWIN (Bifet & Gavalda 2007) for regime change detection + streaming t-digest for real-time percentile estimation. Zero external ML dependencies — pure Python |
| **HippoGraph** | `app/engines/hippograph/graph.py` + `ppr.py` | Market state graph with temporal edges (t→t+1) and similarity edges (cosine ≥ 0.85). Personalised PageRank finds top-10 analog states. Max 8,640 nodes (30 days × 288 intervals) |
| **TemporalRAG** | `app/engines/temporalrag/retriever.py` | Bitemporal evidence retrieval: valid_time window + system_time fence. Relevance = 0.60 × temporal proximity (exp decay) + 0.40 × source credibility. Leakage-filtered (removes docs with system_time > query time) |
| **Forecast ensemble** | `app/engines/forecasting/models/` | LEAR (Linear ARX), QRA (Quantile Regression Averaging), GBM (sklearn), TCN (weight-normed CNN), LNN/LTC. Meta-ensemble with conformal calibration. Walk-forward backtest, CRPS evaluation, spike recall metrics |
| **DriftMonitor** | `app/engines/drift_monitor.py` | ADWIN on live forecast residuals per region. Triggers early retraining when error distribution shifts beyond expected NEM volatility |
| **SecurityObserver** | `app/security/observer.py` | 4-pass hygiene: regex injection detection, PII patterns, numeric anomaly bounds, evidence_ref completeness check. Halt threshold 80/100 |
| **BESS engine** | `app/portfolio/bess_engine.py` | Deterministic 5-min interval economics: SOC management, energy revenue, FCAS opportunity value, degradation cost, duration estimation |
| **ScatterGather** | `app/agents/scatter_gather.py` | 7 async tasks via `asyncio.gather`, each `_wrap_task` timeout-gated with `SourceStatus` provenance. Historical and comparison variants |
| **Decomposer** | `app/engines/decomposition.py` | Hybrid: rule-based (always) → Ollama/Claude (optional) → merge. 200+ city aliases, 11 intent labels, sub-question classification |

---

## What Works vs Stub/Demo

### Production-quality (tested, live data)

| Feature | Status | Notes |
|---|---|---|
| Live AEMO dispatch ingestion | ✓ Working | 5-min polling, DB persistence, DB fallback on NEMWeb outage |
| NLP decomposer (rule-based) | ✓ Working | Zero LLM dependency, full offline operation |
| NLP decomposer (Ollama/Claude) | ✓ Working | Graceful fallback when offline |
| 11 intent types + routing | ✓ Working | Validated across 1,995-test suite |
| Sub-question decomposition | ✓ Working | 14 sub-question types |
| Adjacent taxonomy (3 types) | ✓ Working | PARTIAL_SCOPE, EVIDENCE_BRIDGE, GEOGRAPHIC_REDIRECT |
| Ambiguity gate | ✓ Working | Confidence threshold 0.62, clarifying question |
| Session context (multi-turn) | ✓ Working | Last 3 Q&A pairs injected for short contextual follow-ups |
| ScatterGather parallel fetch | ✓ Working | ~330ms, 7 tasks, all timeout-gated |
| HippoGraph analog retrieval | ✓ Working | Rebuilds from DB at startup, PPR traversal |
| LNN/LTC forecast (per region) | ✓ Working | Trained on live dispatch data, weights persisted to disk |
| LEAR + QRA + GBM ensemble | ✓ Working | Meta-ensemble with conformal calibration |
| TemporalRAG retrieval | ✓ Working | Bitemporal, leakage-filtered |
| Historical price distribution | ✓ Working | P10–P90, hour/season/quarter windowing |
| Intraday fuel timeline | ✓ Working | Solar cliff detection, dispatch transitions |
| OpenNEM trend + diurnal data | ✓ Working | Real hourly/monthly data, graceful fallback |
| BOM 7-day weather forecast | ✓ Working | Injected for forecast queries |
| MMSDM archive backfill | ✓ Working | 2022-08 to 2024-07 accessible |
| Historical routing (anchored) | ✓ Working | RETROSPECTIVE queries route to archive |
| Routing alignment self-correction | ✓ Working | Detects anchor mismatch, re-runs live scatter |
| Multi-region comparison | ✓ Working | Parallel gather, cross-region spread narrative |
| FCAS attribution | ✓ Working | 8 NEM FCAS markets, tight-market detection |
| ChronoGraph regime detection | ✓ Working | ADWIN change-point, t-digest quantile rank |
| Answer planner (15+ planners) | ✓ Working | Deterministic, evidence-only |
| ClaimVerifier | ✓ Working | Downgrades unsupported causal/forecast claims |
| SecurityObserver (4 passes) | ✓ Working | Halt score 80, injection blocked pass 3 |
| Bitemporal trace | ✓ Working | valid_time + system_time on every query |
| BESS economics engine | ✓ Working | Pure deterministic, FCAS-aware |
| Markdown export | ✓ Working | Report with evidence table + claim map |
| Prometheus metrics | ✓ Working | query_latency_ms, llm_decompose_latency_ms |
| Grafana dashboard | ✓ Working | JSON in monitoring/ |
| ISO 27001 / AESCSF compliance surfaces | ✓ Working | Control mappings, AI risk register |
| Multi-tenant auth (JWT) | ✓ Working | Every row has tenant_id |
| Alembic migrations (6 versions) | ✓ Working | PostgreSQL + SQLite |
| Docker Compose stack | ✓ Working | App + Postgres + Redis |
| SSE live feed | ✓ Working | Commentary events broadcast via Redis |
| Follow-up question chips | ✓ Working | Deterministic, context-aware |
| Confidence gap explainer | ✓ Working | Explains WHY confidence is low |

### Implemented (Sprint AA upgrades)

These were previously listed as stubs and are now production-grade:

| Feature | Status | Notes |
|---|---|---|
| Coverage auditor | ✓ Deterministic-first | Rule-based gap detection always runs; Ollama is optional enhancement (5s timeout, fast-fail) |
| WA structural comparison | ✓ Working | `wa_client.py` returns full WEM vs NEM comparison: fuel mix, indicative price context, 6-dimension differences table, SA1 analog reasoning |
| OData export — Query + Trace entities | ✓ Working | `/odata/queries` and `/odata/traces` entity sets; full EDMX metadata with 4 entity types (PowerBI-compatible) |
| DISPATCHOFFERTRK parser | ✓ Working | Real-time rebid detection from 5-min DispatchIS reports — no 30-day confidentiality delay; `offer_track` table + parser in `aemo_parsers.py` |
| AER generator registry | ✓ Working | `scripts/seed_generator_registry.py` — embedded 50+ DUID registry with live AEMO NREL fetch option |
| Notice price signal classifier | ✓ Working | `notice_price_signal.py` — 18 AEMO notice types → spike probability (0–1), severity, action signal, regime probability shift; LNN forecast feature |
| ISP scenario data | ✓ Working | `isp_client.py` — Step Change/Slow Change trajectories, coal retirement schedule, REZ pipeline; wired into policy bridge handler |
| GBB gas hub prices | ✓ Working | `gbb_client.py` — STTM hub price fetch + SRMC calculation; wired into gas_electricity_nexus handler; ACCC static fallback always available |
| ST PASA client (scaffold) | Scaffold | `st_pasa_client.py` — 7-day supply/demand forecast parser; LOR risk detection; awaits wiring into forecast pipeline |

### Remaining Stubs

| Feature | Status | Notes |
|---|---|---|
| Rebid engine (live bid data) | Partial | `offer_track` table + DISPATCHOFFERTRK ingestion added (real-time). Full bid analysis (BIDDAYOFFER_D/BIDPEROFFER_D) needs 30-day archive ingestion pipeline wired to scheduler |
| Long-horizon forecast (>4h) | Scaffold | ST PASA client exists; needs wiring into LNN extended horizon + BOM 7-day integration. Structural limit: requires ST PASA + extended LEAR for 4h–7day accuracy |
| Participant intent profiling | Partial | Framework + `bid_offers` table schema complete; `seed_generator_registry.py` seeds DUID registry; needs BIDDAYOFFER_D scheduler job |
| WA live price feed | Not built | WEM API requires separate AEMO WEM registration (different from NEMWeb). Structural comparison is working; live balancing price is not |
| News RSS as ML forecast feature | Partial | `notice_price_signal.py` converts AEMO notices → price spike probability (immediately useful). RSS-as-feature requires news NLP extraction pipeline (not started) |

---

## USP: What Delta Gap Does This Bridge?

### Gap 1: "Why?" vs "What?"
Every existing NEM analytics platform (OpenNEM, WattClarity, AEMO's own tools) answers *what* — it shows you a chart of price, demand, and fuel mix. GridVerdict answers *why* — it tiered-attributes the cause, distinguishes confirmed from inferred drivers, and tells you exactly what evidence was missing. **No public NEM tool does causal attribution with explicit uncertainty quantification.**

### Gap 2: Evidence-grounded answers vs. LLM hallucination
Other AI assistants connected to energy data can confidently state wrong numbers. GridVerdict's design constraint: **every `SUPPORTED` verdict requires a cited `evidence_ref`** pointing to an actual DB row or API response. An answer with no evidence is automatically downgraded to `LOW_CONFIDENCE`. The LLM generates zero market facts.

### Gap 3: Temporal reasoning vs. snapshot tools
The bitemporal architecture (valid_time vs. system_time) enables questions no other tool can answer: *"What did the market know at 2pm last Tuesday — before the generator trip notice was published?"* The `Trace` table makes every past decision replayable with exactly the data available at that moment. This matters for post-event audit and regulatory compliance.

### Gap 4: Graceful degradation at scope boundaries
When a question crosses into territory GridVerdict cannot fully answer (household solar feed-in tariffs, WA electricity, interest rates → LCOE), it doesn't guess. It identifies the **answerable NEM sub-question**, answers it with evidence, and redirects for the rest with specific external resources. This is encoded in the adjacent taxonomy (8 handlers across 3 intent types). Most AI tools either refuse entirely or answer incorrectly.

### Gap 5: Hybrid NLP that survives offline
The rule-based decomposer runs with zero LLM dependency. The system handles all 11 intent types, 200+ Australian city aliases, seasonal bucket extraction, and sub-question classification without any external API. Most AI-first tools collapse when the LLM is unavailable. GridVerdict degrades gracefully — the answer may be less nuanced, but the pipeline never fails.

### Gap 6: Multi-tenant, audit-ready, deployable
Built from day one with multi-tenancy, JWT auth, ISO 27001 control mappings, AI risk register, model provenance, and a complete Alembic migration history. It is not a demo or a Jupyter notebook — it can be deployed and operated.

---

## Skillsets Demonstrated

This project demonstrates competency across the following technical domains:

**ML / AI Research**
- Liquid Time-Constant ODE (Hasani et al. 2021) implemented from scratch — no NCPS library
- Semi-implicit Euler solver for stiff ODEs; SA1 region-specific config for spike volatility
- Conformal prediction calibration for quantile intervals
- CRPS (Continuous Ranked Probability Score) evaluation, spike recall, walk-forward backtest
- ADWIN change-point detection (Bifet & Gavalda 2007) — pure Python, no external dependencies
- t-digest streaming quantile estimation
- Personalised PageRank for similarity-based analog retrieval

**Time Series Forecasting**
- Multi-model ensemble (LEAR, QRA, GBM, TCN, LNN) with conformal calibration
- P10/P50/P90 quantile outputs with pinball loss training
- Feature engineering: time-of-day cyclical encoding, log-price spike compression, FCAS proxy, renewable fraction
- Walk-forward backtesting with no-lookahead bitemporal replay
- Drift monitoring via ADWIN on live residuals

**NLP / LLM Engineering**
- Hybrid decomposer: rule-based safety gate → LLM entity enrichment → merge
- Structured output validation against Pydantic schema with fallback
- Query restatement, sub-question classification, session context management
- Adversarial coverage auditor with LLM re-routing
- Prompt injection defense at decomposition and tool output layers
- Multi-turn conversation coherence via rolling context injection

**Distributed Systems / Backend**
- FastAPI async application with 20+ route modules
- Redis event bus for SSE broadcast (optional)
- Background APScheduler for dispatch polling, notice fetching, archive backfill
- asyncio ScatterGather with timeout gating and provenance tracking
- SQLAlchemy async ORM, Alembic migrations, PostgreSQL + SQLite dual-mode
- Multi-tenant data model with row-level tenant isolation from day one

**Domain Expertise: Energy Markets**
- NEM dispatch mechanics (5-minute auction, marginal setter, merit order)
- FCAS market structure (8 services, raise/lower, tight-market detection)
- Interconnector causality (QNI, Heywood, Basslink)
- MMSDM data schema (DISPATCHPRICE, DISPATCHLOAD, PREDISPATCH, TRADING_REGIONSUM)
- Generator outage detection from AEMO market notices
- Rooftop solar signal extraction from TRADING_REGIONSUM
- Bidding strategy context (what can and cannot be inferred from public data)

**Security Engineering**
- OWASP-aligned 4-pass hygiene pipeline on every query lifecycle
- Untrusted data treatment for all tool outputs (prompt injection in API responses)
- PII detection (SSN, credit card, email, API key patterns)
- Market manipulation language detection
- ISO 27001 control mappings, AESCSF maturity profile
- AI risk register with impact/likelihood ratings

**API & Integration Design**
- REST + SSE hybrid for real-time streaming
- Rate limiting middleware
- OData-compatible export
- ngrok-compatible CORS (multi-tenant demo routing)
- Source provenance tracking (SourceStatus per data fetch)
- Markdown + JSON report export

**Frontend Engineering**
- Alpine.js SPA with reactive state management
- ECharts: swimlane timeline, fuel mix donut, analog scatter, backtest P&L, FCAS bar
- SSE live event feed with auto-reconnect
- Query progress polling (500ms interval, 5s delay before UI shows steps)
- Multi-region comparison table, causal chain panel

**DevOps / Observability**
- Docker Compose with Postgres, Redis, and app service
- Prometheus metrics (histogram and gauge)
- Grafana dashboard JSON with NEM-specific panels
- Prometheus alert rules (latency, stale dispatch)
- Operational runbooks for 5 failure scenarios

---

## Quick Start

**Requirements:** Docker + Docker Compose, 4 GB RAM (PyTorch CPU image)

```bash
cp .env.example .env      # edit DB_PASSWORD and JWT_SECRET
docker compose up
```

Open `http://localhost:8000`

On first startup the LNN bootstraps from the dispatch archive and trains all 5 NEM regions (~30 seconds). You'll see `LTC[NSW1] trained on 599 intervals, loss=98` in the logs.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `DB_PASSWORD` | yes | PostgreSQL password |
| `JWT_SECRET` | yes | 32+ char string for JWT signing |
| `GRIDVERDICT_DEV_NO_AUTH` | dev only | `true` to skip auth — never in prod |
| `DECOMPOSER_BACKEND` | no | `ollama` \| `claude` \| `rule_based` (default: `rule_based`) |
| `OLLAMA_BASE_URL` | no | Ollama endpoint (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | no | Model name (e.g. `qwen2.5:7b`) |
| `ANTHROPIC_API_KEY` | no | Claude API key for LLM fallback |
| `ENABLE_EXPERIMENTAL_SEQUENCE_FORECASTERS` | no | `true` to enable TCN + GBM in ensemble |

The rule-based decomposer runs fully offline — useful for CI and air-gapped demos. For best intent accuracy, use Ollama with `qwen2.5:7b` or `mistral`:

```bash
ollama pull qwen2.5:7b
# .env: DECOMPOSER_BACKEND=ollama, OLLAMA_MODEL=qwen2.5:7b
```

---

## Test Suite

```bash
# Core unit tests — no DB or network needed (~5s)
python -m pytest tests/test_decomposer_quality.py tests/test_answer_planner.py \
       tests/test_why_engine.py tests/test_verdict_contract.py -q

# Full offline suite — 1,995 tests, ~86s
python -m pytest tests/ \
       --ignore=tests/test_live_mcp_smoke.py \
       --ignore=tests/test_live_qa_regressions.py \
       --ignore=tests/integration -q

# Live MCP smoke (requires network + NEMWeb)
GRIDVERDICT_LIVE_TESTS=1 python -m pytest tests/test_live_mcp_smoke.py -q
```

**Test coverage:** unit (decomposer, planner, verdict contract, claim verifier), integration (scatter-gather, forecast, backtest, hippograph, temporal RAG), sprint regression (A through Z), acceptance matrix (NLP question routing), E2E (frontend via Playwright), QA (advanced scenarios and professional question playbook).

---

## Project Layout

```
app/
  agents/       ScatterGather · WhySources · WhyBuilder · ClaimVerifier · AnswerPlanner
  api/          FastAPI routes — query · market · incidents · models · events · trace ·
                  backtest · constraints · compliance · commentary · export · metrics
  audit/        Audit logger + structured JSON export
  compliance/   ISO 27001 controls · AESCSF profile · AI risk register
  core/         Schema (QueryDecomposition, FactualVerdict, EvidenceRefSchema) ·
                  bitemporal trace writer · verdict derivation · interfaces
  data/         Scheduler · cache · Redis event bus · AEMO live client
  db/           SQLAlchemy models · async session · Alembic migrations (6 versions)
  engines/
    chronograph/  ADWIN change-point + t-digest streaming quantiles
    forecasting/  LEAR · QRA · GBM · TCN · LNN meta-ensemble · walk-forward · calibration
    hippograph/   Market state graph · Personalised PageRank · embedder
    lnn/          LTC cell (ODE) · distribution head · trainer · feature extraction
    temporalrag/  Bitemporal retriever · schema · source credibility weights
    adjacent_handlers.py   8-handler adjacent taxonomy dispatcher
    decomposition.py       Hybrid NLP decomposer (rules + LLM + merge)
    fuel_mix.py            Per-fuel dispatch breakdown · intraday timeline
    historical_price.py    Archive distribution · period stats · classification
    marginal_setter.py     Marginal generator identification
    driver_attribution.py  Binding constraints · interconnector flows
    incident_timeline.py   Generator trip detection · incident brief
    why_builder.py         Tiered claim construction · evidence references
    [+ 10 more engines]
  mcp/          Read-only MCP registry — AEMO live/archive/notices · weather · news ·
                  OpenNEM · ISP · GBB · fiscal budget
  portfolio/    BESS economics · fleet coordinator · dispatch policy
  security/     SecurityObserver — 4-pass hygiene pipeline

frontend/
  app.html                 Alpine.js SPA entry point
  static/js/               main · api · state · charts/ · evidence · trace · history
  static/css/              theme (dark/light) · components
  static/vendor/           Alpine.js · ECharts (vendored, no CDN dependency)

tests/
  test_decomposer_quality.py    NLP intent accuracy + routing regression
  test_answer_planner.py        Per-planner output validation
  test_why_engine.py            Driver tier + evidence contract
  test_acceptance_matrix.py     End-to-end NLP routing matrix (40+ questions)
  test_e2e_integration.py       Full API pipeline integration
  e2e/test_frontend.py          Playwright browser tests
  qa_20_professional_questions.py  Energy-professional question playbook
  [+ 40 more test modules]

docs/
  architecture_walkthrough.md   Component-by-component walkthrough
  model_cards/                  LEAR · LNN · QRA · MetaEnsemble cards
  runbooks/                     5 operational runbooks (stale dispatch, Redis down, …)
  query_trace_explainer.md      How to read a pipeline_events trace
  known_limitations.md          Honest capability boundary documentation

config/
  settings.py         Pydantic settings with env var validation
  models.yaml         Forecast model profiles + feature sets
  tools.yaml          MCP tool registry
  geo_aliases.yaml    200+ Australian city → NEM region mappings

monitoring/
  grafana_dashboard.json        NEM-specific Grafana panels
  prometheus_alert_rules.yml    Latency + stale dispatch alerts
```

---

## Design Constraints

- **No bid execution** — simulation and advisory only; no write path to AEMO or any broker
- **Read-only MCP tools** — external fetches never write to external systems  
- **No unsupported causality** — drivers are tiered; unconfirmed causes are labelled, not hidden
- **No forecast authority without model status** — unavailable models surface as unavailable, not silently excluded
- **Auditability** — every query writes a bitemporal trace; any past decision is replayable
- **NEM-only scope** — WA (SWIS/NWIS) and NT (Darwin-Katherine system) are separate grids with no AEMO data

---

## Extending to a New Domain

GridVerdict is vertical one of a reusable temporal reasoning framework. To add shipping logistics, ICU monitoring, or bushfire tracking:

1. **Define your event schema** — extend `core.interfaces.IngestEvent` with domain fields
2. **Implement `DomainAdapter`** — `to_feature_vector()`, `to_evidence_ref()`, `decomposition_hints()`, `why_template()`
3. **Write a data MCP** — read-only Zone 1 tool
4. **Register in `config/verticals.yaml`**
5. **Write 10 acceptance questions** — same format as `tests/test_acceptance_matrix.py`

The entire `core/`, `engines/`, `agents/`, and `security/` stack runs unchanged. The framework boundary is enforced, not aspirational: `HippoGraph`, `ChronoGraph`, `TemporalRAG`, and `LNN` have zero imports from `data/`, `mcp/`, or `domain/nem/`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full framework design and reskin guide.

---

*Research and portfolio use only. GridVerdict does not execute bids or trades and makes no representation of fitness for trading or operational decisions. All data is sourced from publicly available AEMO NEMWeb resources.*
