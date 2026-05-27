# GridVerdict — Complete Build Roadmap
# Generated: 2026-05-23. Supersedes all prior build sequences.

---

## 0. All Architectural Decisions — Locked

Do not revisit these mid-build. If a decision needs changing, update this file.

| Decision | Choice | Reason |
|---|---|---|
| Deployment | Local + ngrok / Cloudflare Tunnel | Shareable for demo without cloud infra cost |
| Multi-tenancy | Multi-tenant from day one (`tenant_id` everywhere) | Cannot retrofit cleanly; `"local"` is the default tenant |
| Database | PostgreSQL via Docker | Concurrent writes from multiple tenants + background AEMO ingest |
| Auth | JWT per tenant, issued at login | Publicly reachable endpoint requires auth from day one |
| Frontend | Alpine.js + vanilla JS modules | Reactive state without build step; plays with modular JS spec |
| Charts | Apache ECharts (swimlane/price/forecast) + Chart.js (backtest P&L) | ECharts handles real-time financial-style charts natively |
| First screen | Live market dashboard (Monitor mode) | Real professional workflow: data first, query when needed |
| Data source | Public NEMWeb only (free) | No credentials; 5-min dispatch, pre-dispatch, archive zips |
| LNN approach | Two-track: ncps CfC (MVP battery Week 2) + from-scratch LTC (Week 4) | ncps gets harness running fast; LTC is the research showcase — both implement ForecastModel |
| Social media (X) | Not in MVP | Cost, noise, prompt injection risk; AEMO Market Notices is better signal |
| Framework boundary | Establish `core/` boundary in first commit | Zero cost now; high cost to extract later |
| AEMO notices | Add `mcp/aemo_notices.py` in MVP | Free, structured, authoritative — 80% of Twitter signal value |

---

## 1. What GridVerdict Actually Is (Two Layers)

GridVerdict is one vertical application of a reusable temporal reasoning framework.
Never couple the framework to AEMO code. The import boundary enforces this.

```
FRAMEWORK LAYER (domain-agnostic — lives in core/, engines/, agents/)
  ChronoGraph      ADWIN change-point detection + t-digest quantile tracking
  LNN / LTC        Liquid Time-Constant cells from scratch in PyTorch
  HippoGraph       Graph-based state representation + PPR analog retrieval
  TemporalRAG      Bitemporal archive retrieval (valid_time vs system_time)
  ScatterGather    Parallel agent orchestration + deepening + critic
  SecurityObserver 4-pass input/decomp/tool-output/answer hygiene
  BiTemporalTrace  Full replay of any past decision

NEM DOMAIN LAYER (AEMO-specific — lives in domain/nem/, data/, mcp/)
  Energy schema    Extends abstract IngestEvent with NEM fields
  NEM features     Extracts ChronoGraph/HippoGraph inputs from AEMO data
  AEMO clients     Live, archive, notices, weather MCPs
  Why templates    Plain-English NEM explanation patterns
```

**The rule that enforces this boundary:**
- `core/` and `engines/` contain ZERO imports from `data/`, `mcp/`, or `domain/nem/`
- They receive abstract `IngestEvent`, `FeatureVector`, `EvidenceRef` objects only
- Domain-specific code adapts NEM data INTO those abstract types before passing to engines

---

## 2. Final Repository Structure

```
gridverdict/
│
├── app/
│   │
│   ├── api/                            ── HTTP surface
│   │   ├── main.py                     FastAPI app init, CORS, router registration ONLY
│   │   ├── auth.py                     JWT middleware, tenant extraction, token issuance
│   │   ├── deps.py                     FastAPI dependency injection (db session, tenant)
│   │   ├── routes_query.py             POST /query/decompose, POST /query/answer
│   │   ├── routes_market.py            GET /market/state, GET /market/archive/status
│   │   ├── routes_backtest.py          POST /backtest/run
│   │   ├── routes_trace.py             GET /trace/{trace_id}
│   │   ├── routes_security.py          POST /security/observer/*, GET /security/observer/events
│   │   └── routes_health.py            GET /health/dependencies
│   │
│   ├── core/                           ── FRAMEWORK — no domain imports ever
│   │   ├── __init__.py
│   │   ├── interfaces.py               Abstract base classes: IngestEvent, FeatureVector,
│   │   │                               EvidenceRef, DecompositionHints, DomainAdapter
│   │   ├── schema.py                   Abstract answer, verdict, trace schemas (Pydantic)
│   │   ├── evidence.py                 Evidence ref builder, validator, coverage scorer
│   │   ├── verdict.py                  Deterministic verdict derivation (no LLM)
│   │   ├── trace.py                    Bitemporal trace persistence (valid_time/system_time)
│   │   └── uncertainty.py              Confidence banding, missing-data weighting
│   │
│   ├── engines/                        ── FRAMEWORK — receives abstract types from core/
│   │   ├── chronograph/
│   │   │   ├── __init__.py
│   │   │   ├── adwin.py                ADWIN change-point detection
│   │   │   ├── tdigest.py              t-digest quantile streaming
│   │   │   └── regime.py               Composes ADWIN + t-digest into regime verdict
│   │   │
│   │   ├── hippograph/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py                Market state graph builder
│   │   │   ├── ppr.py                  Personalised PageRank for analog retrieval
│   │   │   └── embedder.py             Local BGE embedder for state vectors
│   │   │
│   │   ├── lnn/                        ── THE RESEARCH CENTREPIECE
│   │   │   ├── __init__.py
│   │   │   ├── ltc_cell.py             Liquid Time-Constant cell — from-scratch PyTorch
│   │   │   │                           Implements: dx/dt = -x/τ(x,I,θ) + f(x,I,θ)
│   │   │   ├── ltc_model.py            Sequence model wrapping LTC cells
│   │   │   ├── distribution.py         Quantile output head (returns distribution not point)
│   │   │   └── trainer.py              Training loop, checkpoint save/load
│   │   │
│   │   ├── temporal_rag.py             Bitemporal archive retrieval — abstract over any source
│   │   ├── decomposition.py            Query → structured intent (Tier 1 LLM call)
│   │   ├── prefill.py                  Assemble evidence envelope from scatter results
│   │   ├── why_engine.py               Four grounded sources → plain-English explanation
│   │   ├── analog_retriever.py         PPR over archive via HippoGraph
│   │   ├── backtest.py                 Historical window simulation engine
│   │   └── forecast.py                 LNN inference → distribution output
│   │
│   ├── agents/
│   │   ├── orchestrator.py             Full pipeline: scatter → deepen → validate
│   │   │                               Timing: ~1s standard, ~5s Tier 2, ~30s backtest
│   │   ├── scatter_agents.py           Parallel: live_snapshot, regime, analog, forecast, portfolio
│   │   ├── deepening_agent.py          Tier 2 synthesis (fires on hard queries only)
│   │   └── critic_agent.py             Adversarial counterargument generation
│   │
│   ├── security/
│   │   ├── mcp_policy.py               Action zones 0-3, approval gate definitions
│   │   ├── security_observer.py        4-pass hygiene pipeline (main file)
│   │   ├── observer_passes.py          Isolated pass implementations (input/decomp/tool/answer)
│   │   ├── tool_registry.py            Signed allowlist, pinned versions, audit log
│   │   └── prompt_injection.py         Detector for user text + retrieved content
│   │
│   ├── mcp/                            ── DATA ACCESS LAYER
│   │   ├── registry.py                 All MCPs registered here; audit logging; version pins
│   │   ├── router.py                   Agents request capability name, not MCP name
│   │   │
│   │   ├── aemo_live.py                Zone 1: live dispatch, pre-dispatch, constraints
│   │   │
│   │   ├── aemo_archive.py             Zone 1: NEMWEB zip fetcher, local cache, gap detection
│   │   │
│   │   ├── aemo_notices.py             Zone 1: Market Notice fetcher/parser/router
│   │   │   ── Split into 3 files when > 350 lines:
│   │   │      aemo_notices_client.py   NEMWeb fetching, caching, polling interval
│   │   │      aemo_notices_parser.py   XML/HTML parsing → structured MarketNotice objects
│   │   │      aemo_notices_classifier.py  Notice type → severity → ChronoGraph signal
│   │   │
│   │   ├── weather.py                  Zone 1: BOM weather alerts, temperature forecast
│   │   └── portfolio.py                Zone 2: user asset assumptions, local-only, never cloud
│   │
│   ├── domain/                         ── NEM DOMAIN ADAPTER
│   │   └── nem/
│   │       ├── __init__.py
│   │       ├── schema.py               EnergyEvent — extends core.interfaces.IngestEvent
│   │       ├── features.py             NEM → FeatureVector for ChronoGraph/HippoGraph
│   │       ├── decomposition_hints.py  NEM intent patterns for decomposition engine
│   │       └── why_templates.py        NEM plain-English explanation templates
│   │
│   └── db/
│       ├── models.py                   SQLAlchemy models (ALL with tenant_id)
│       ├── session.py                  Async session factory
│       └── migrations/                 Alembic migration files
│
├── frontend/
│   └── static/
│       ├── app.html                    Shell + stable IDs + Alpine init (<200 lines)
│       ├── css/
│       │   ├── theme.css               Colours, typography, dark mode, NEM region palette
│       │   └── components.css          Cards, drawers, swimlane, badges, panels
│       └── js/
│           ├── api.js                  Fetch wrapper, JWT header, error normalisation
│           ├── state.js                Alpine store: region, tenant, session, answer, trace
│           ├── query.js                Submit flow: decompose → prefill → answer → render
│           ├── evidence.js             Evidence drawer, freshness banner, missing data list
│           ├── history.js              Session list, date grouping, search, trace replay entry
│           ├── backtest.js             Counterfactual form + interval results table
│           ├── trace.js                Bitemporal trace replay panel
│           ├── security.js             Client warnings (low conf, stale, blocked, out-of-scope)
│           ├── charts/
│           │   ├── swimlane.js         ECharts: live price line + dashed forecast tail + bands
│           │   ├── analog.js           ECharts: historical analog overlay lines
│           │   └── backtest_pnl.js     Chart.js: cumulative P&L for backtest results
│           └── main.js                 Wires all modules, Alpine component registration
│
├── config/
│   ├── models.yaml                     sovereign / cost_optimized / max_quality profiles
│   ├── tools.yaml                      MCP registry + capability → MCP mapping
│   └── sources.yaml                    Freshness thresholds, rate limits, cache TTLs
│
├── tests/
│   ├── fixtures/
│   │   ├── sample_answers/             Valid/invalid factual verdict JSON fixtures
│   │   ├── sample_market_states/       Stubbed AEMO responses for unit tests
│   │   └── injection_payloads/         Adversarial strings for Security Observer tests
│   ├── test_ltc_cell.py                Unit: LTC cell maths, gradient flow, time constant
│   ├── test_chronograph.py             Unit: ADWIN detects regime shift, t-digest quantiles
│   ├── test_hippograph.py              Unit: graph build, PPR returns ranked analogs
│   ├── test_query_contract.py          Schema: decomposition output is valid
│   ├── test_verdict_contract.py        Schema: every numeric claim has evidence_ref
│   ├── test_backtest_replay.py         Determinism: same snapshot → same answer
│   ├── test_security_observer.py       Injection blocked, retail query refused, etc.
│   └── test_acceptance_matrix.py       The 11 canonical questions — run end of every week
│
├── docker-compose.yml                  PostgreSQL + app + pgAdmin for dev
├── .env.example                        DB_PASSWORD, JWT_SECRET, NEMWEB_CACHE_DIR
├── pyproject.toml                      Dependencies, tool config
└── ARCHITECTURE.md                     How to add a new vertical (reskin guide)
```

---

## 3. aemo_notices.py — Design, Scope, and Split Strategy

### What AEMO Market Notices Cover

AEMO publishes notices at: `https://nemweb.com.au/Reports/Current/Market_Notice/`

| Notice Type | Market Signal | ChronoGraph Action |
|---|---|---|
| `LOR1` | Low reserve warning (≥ 130MW deficit) | Elevate regime score moderately |
| `LOR2` | Lack of reserve imminent (< 130MW) | Elevate regime score high |
| `LOR3` | Actual shortfall occurring | Trigger emergency regime |
| `RECLASSIFY` | Unit trip credible contingency | Trigger analog search for similar trips |
| `INTER_CONSTRAINT` | Interconnector constraint binding | Annotate Why Engine with interconnector context |
| `MARKET_INTERVENTION` | AEMO directs generation | Override normal dispatch signal |
| `DIRECTIONS` | AEMO directs specific generator | High-confidence dispatch signal |
| `MT_PASA` | 2-year projected adequacy | Background context, not real-time |

### MVP: One File, Three Internal Sections

```python
# mcp/aemo_notices.py
# Section 1: Client (fetch + cache + polling)
# Section 2: Parser (raw HTML/XML → MarketNotice dataclass)
# Section 3: Classifier (notice_type → severity + ChronoGraph signal)
```

### Split Trigger: > 350 Lines

When the file grows past 350 lines, split at the section boundaries:

```
mcp/
  aemo_notices_client.py      NEMWeb polling, HTTP, cache, deduplification
  aemo_notices_parser.py      Raw text → structured MarketNotice objects
  aemo_notices_classifier.py  notice_type → severity → ChronoGraph/Why Engine signal
  aemo_notices.py             Thin coordinator — imports the three above, exposes MCP interface
```

### How Notices Flow Into the Pipeline

```
Background scheduler (every 60s)
  → aemo_notices_client fetches new notices
  → aemo_notices_parser normalises them
  → aemo_notices_classifier maps to severity + ChronoGraph signal
  → Stored in DB with tenant_id="system" (visible to all tenants)
  → ChronoGraph receives signal injection → updates regime score

Per-query scatter (Agent A: live_snapshot)
  → Includes active notices in evidence envelope
  → Why Engine can cite: "AEMO LOR2 notice active as of 14:03 — system under stress"
  → Security Observer Pass 3 treats notice content as data, not instruction
```

---

## 4. Framework Boundary — Establish in First Commit

### The Rule (Put This in ARCHITECTURE.md)

> `core/` and `engines/` contain no imports from `data/`, `mcp/`, `domain/`, or `app/api/`.
> They receive and return only types defined in `core/interfaces.py`.
> Domain-specific code adapts INTO those types before passing to engines.

### The Interface Contract (core/interfaces.py)

This is the key file. Every vertical (NEM, shipping, bushfire, healthcare) implements these:

```python
# core/interfaces.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class IngestEvent:
    """Abstract timestamped event from any domain."""
    source: str
    valid_time: datetime       # when this was true in the world
    system_time: datetime      # when we learned about it
    raw_ref: str               # hash or URL to source
    tenant_id: str

@dataclass
class FeatureVector:
    """What ChronoGraph and HippoGraph receive — no domain semantics."""
    event_id: str
    values: dict[str, float]   # named numeric features
    categorical: dict[str, str]
    valid_time: datetime

@dataclass
class EvidenceRef:
    """A citable source for a numeric claim."""
    id: str
    source: str
    field: str
    value: float
    valid_time: datetime
    raw_ref: str

class DomainAdapter(ABC):
    """Every vertical implements this to plug into the framework."""

    @abstractmethod
    def to_feature_vector(self, event: IngestEvent) -> FeatureVector:
        """Convert a domain event into framework-compatible features."""

    @abstractmethod
    def to_evidence_ref(self, event: IngestEvent, field: str) -> EvidenceRef:
        """Produce a citable evidence ref from a domain event."""

    @abstractmethod
    def decomposition_hints(self) -> dict:
        """Domain-specific intent patterns to guide the decomposer."""

    @abstractmethod
    def why_template(self, regime_label: str, analog_count: int) -> str:
        """Plain-English explanation template for the Why Engine."""
```

### NEM Domain Adapter (domain/nem/schema.py)

```python
# domain/nem/schema.py
from app.core.interfaces import IngestEvent, DomainAdapter, FeatureVector, EvidenceRef

@dataclass
class EnergyEvent(IngestEvent):
    """NEM-specific extension of IngestEvent."""
    region: str
    price_rrp: float
    demand_mw: float
    availability_mw: float
    constraint_id: str | None = None
    interconnector: str | None = None
    # ... other NEM fields

class NEMAdapter(DomainAdapter):
    def to_feature_vector(self, event: EnergyEvent) -> FeatureVector:
        return FeatureVector(
            event_id=...,
            values={"price_rrp": event.price_rrp, "demand_mw": event.demand_mw, ...},
            categorical={"region": event.region},
            valid_time=event.valid_time
        )
    # ... etc
```

### How a Future Vertical Plugs In

For shipping, bushfire, healthcare — only these files change:
```
domain/
  shipping/
    schema.py         VesselEvent(IngestEvent) + ShippingAdapter(DomainAdapter)
    features.py       vessel position + weather → FeatureVector
    why_templates.py  "Delay likely because port congestion + weather..."
  
  bushfire/
    schema.py         EnvironmentalEvent(IngestEvent) + BushfireAdapter(DomainAdapter)
    features.py       LiDAR point cloud → animal movement → FeatureVector
    why_templates.py  "Movement corridor shifted because..."
```

The entire `core/`, `engines/`, `agents/`, and `security/` stack is untouched.

---

## 5. Stub Strategy for Future Verticals and Add-ons

### What NOT to Stub

Do not add stub files for video, LiDAR, audio, or satellite imagery.
These require fundamentally different preprocessing pipelines before they become events:
- Video → requires frame extraction + object detection → then becomes IngestEvent
- LiDAR → requires point cloud processing → then becomes IngestEvent
- Audio → requires transcription/classification → then becomes IngestEvent

Adding stubs for these creates misleading dead code that implies an implemented pipeline.
The framework handles them automatically once they emit `IngestEvent` objects.
Document the pattern in ARCHITECTURE.md instead.

### What TO Stub (These Are Low-Noise, High-Value)

Add these as real stubs with clear `# STUB — not yet implemented` markers:

| Stub File | Why Add Now | Size |
|---|---|---|
| `mcp/social_monitor.py` | Architecture is designed for it; shows the Zone 2 approval pattern | ~30 lines |
| `domain/nem/outage_schema.py` | Generator outage events extend EnergyEvent; needed for backtest | ~50 lines |
| `engines/ewma.py` | EWMA signal smoothing used by ChronoGraph; small, high-value | ~40 lines |
| `config/verticals.yaml` | Documents how to register a new vertical domain | ~20 lines |

### The social_monitor.py Stub (Shows the Pattern Correctly)

```python
# mcp/social_monitor.py
# STATUS: STUB — not implemented in MVP
# Zone 2: requires operator approval to enable
# Rationale: high prompt injection risk; AEMO Market Notices covers 80% of signal value
# When built: whitelist-only accounts (AEMO, AER, AEMC, Clean Energy Council only)
#             every retrieved post through Security Observer Pass 3 before entering context
#             output tagged source_trust=unverified_social, never used as primary evidence_ref

class SocialMonitorMCP:
    ZONE = 2  # requires_approval = True
    SOURCE_TRUST = "unverified_social"

    def fetch_energy_signals(self, region: str) -> list:
        raise NotImplementedError("Social monitor not implemented in MVP")
```

This communicates the architectural decision without pretending it works.

---

## 6. Agent Pipeline — Timing and Implementation

```
T+0ms     Query received at POST /query/answer
T+10ms    Security Observer Pass 1
            - Unicode normalisation
            - Prompt injection scan
            - Relevance classification (core_market / out_of_scope / malicious)
            - If blocked: return immediately, no engine budget spent

T+20ms    Decomposition (Tier 1 local model)
            - Intent, entities, time range, output contract
            - Low-confidence → clarifying question instead of guess

T+30ms    Security Observer Pass 2
            - Unsafe intent check (real dispatch execution? private data?)
            - If blocked: return OUT_OF_SCOPE / blocked response

T+30ms    ┌──────────── PARALLEL SCATTER ────────────────────────────┐
          │ Agent A: AEMO live snapshot via mcp.aemo_live            │ ~200ms
          │ Agent B: Active notices via mcp.aemo_notices             │ ~80ms
          │ Agent C: ChronoGraph regime check (in-memory)            │ ~60ms
          │ Agent D: HippoGraph PPR analog retrieval (pre-indexed)   │ ~250ms
          │ Agent E: AEMO forecast comparison via mcp.aemo_live      │ ~150ms
          │ Agent F: Portfolio context load via mcp.portfolio        │ ~40ms
          └──────────────────────────────────────────────────────────┘

T+320ms   Scatter results collected
          Security Observer Pass 3
            - Prompt injection scan on all tool outputs
            - Source trust tagging
            - Stale source detection

T+350ms   Why Engine assembles grounded explanation
            Source 1: Current drivers (from A + B + C)
            Source 2: Forecast drivers (from E + LNN inference)
            Source 3: Historical analogs (from D)
            Source 4: Bitemporal trace reference

T+350ms   ─── TIER 2 GATE (fires only if ANY of these are true) ────
          - decomposition.confidence < 0.72
          - requires_backtest = true
          - requires_counterfactual = true
          - forecast_disagreement > 20%
          - high_value_recommendation = true

T+350ms   [If Tier 2 NOT needed]
          Adversarial Critic (Tier 1) generates counterargument → T+550ms

T+600ms   [If Tier 2 fires]
          Deepening Agent (Tier 2 model) synthesises all scatter + Why Engine → T+2500ms
          Adversarial Critic challenges → T+3000ms

T+600ms   Security Observer Pass 4
            - Every numeric claim has evidence_ref (if not: block before render)
            - No unsupported "because" statements
            - Disclaimer present
            - No overconfidence signals

T+650ms   Factual Verdict schema validation
          Bitemporal trace written to DB

T+700ms   Answer returned

TOTAL: ~700ms standard | ~3s Tier 2 | ~20s backtest (show progress bar)
```

---

## 7. Database Schema — Key Tables (All With tenant_id)

```sql
-- Every table follows this pattern

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    email TEXT NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    region TEXT NOT NULL DEFAULT 'NSW1',
    portfolio_assumption_id UUID,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE queries (
    id TEXT PRIMARY KEY,              -- qry-{uuid}
    tenant_id UUID REFERENCES tenants(id),
    session_id UUID REFERENCES sessions(id),
    raw_query TEXT NOT NULL,
    decomposition JSONB,
    answer JSONB,
    trace_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE traces (
    id TEXT PRIMARY KEY,              -- trace-{uuid}
    tenant_id UUID REFERENCES tenants(id),
    query_id TEXT REFERENCES queries(id),
    valid_time TIMESTAMPTZ NOT NULL,
    system_time TIMESTAMPTZ NOT NULL,
    model_profile TEXT NOT NULL,
    source_manifest JSONB,
    tool_calls JSONB,
    decomposition JSONB,
    prefill JSONB,
    answer JSONB,
    validator JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE market_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000', -- system events
    source TEXT NOT NULL,              -- AEMO_DISPATCH_PRICE | AEMO_NOTICE | etc.
    region TEXT,
    valid_time TIMESTAMPTZ NOT NULL,
    system_time TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    raw_ref TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE observer_events (
    id TEXT PRIMARY KEY,              -- obs-evt-{uuid}
    tenant_id UUID REFERENCES tenants(id),
    query_id TEXT,
    trace_id TEXT,
    phase TEXT NOT NULL,              -- input | decomposition | tool_output | answer
    relevance_class TEXT,
    risk_score INTEGER,
    risk_band TEXT,
    verdict TEXT,
    signals JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## 7b. Forecasting Harness — Directory Subtree (Addendum)

The 12 files from the `Dump/New folder` map into `app/engines/forecasting/`.
See `FORECASTING_INTEGRATION.md` for the full wiring guide.

```
app/engines/forecasting/
  __init__.py
  data/
    __init__.py
    types.py              ~95 lines  FIXED: remove frozen=True, add to_dict()
  features/
    __init__.py
    market_features.py    ~85 lines  NEM skin — production join in Week 3
  models/
    __init__.py
    base.py               ~55 lines  ForecastModel interface — no changes
    baselines.py          ~80 lines  Persistence, SeasonalNaive, AEMOPredispatch — no changes
    gbm_model.py          ~65 lines  LightGBM quantile — no changes
    lnn_model.py          ~90 lines  FIXED: init guards + device handling + save/load
    ltc_scratch.py        ~220 lines NEW Week 4: from-scratch PyTorch LTC
  evaluation/
    __init__.py
    metrics.py            ~75 lines  Pinball, CRPS, skill — no changes
    spike_metrics.py      ~60 lines  Spike P/R/F1 — no changes
    calibration.py        ~45 lines  Coverage + calibration error — no changes
    walk_forward.py       ~65 lines  No-leakage rolling-origin — no changes
    harness.py            ~95 lines  FIXED: real datetimes, skill computation

app/mcp/
  aemo_notices_client.py  ~120 lines NEW: concrete client for news_correlator.py
  news_correlator.py      ~115 lines EXISTS: no changes needed

domain/nem/
  features.py             ~40 lines  NEW: EnergyEvent -> market_features dict adapter
```

Total harness: ~1,150 lines across 15 files. All under 300 lines each.

---

## 8. Week-By-Week Build Plan

### Week 1 — Foundation (Target: `docker compose up` opens a real screen)

**Day 1: Infrastructure**
- [ ] `docker-compose.yml` with PostgreSQL 16 + pgAdmin + app service
- [ ] `.env.example` with DB_PASSWORD, JWT_SECRET, NEMWEB_CACHE_DIR
- [ ] `pyproject.toml` with FastAPI, SQLAlchemy async, Alembic, httpx, pydantic
- [ ] `db/models.py` with all tables above (tenant_id on every table)
- [ ] First Alembic migration
- [ ] `app/api/main.py` — FastAPI init, CORS, health route only
- [ ] `app/api/auth.py` — JWT issue + validate middleware
- [ ] Confirm: `docker compose up` → `GET /health` returns 200

**Day 2: Core Framework Interfaces**
- [ ] `app/core/interfaces.py` — IngestEvent, FeatureVector, EvidenceRef, DomainAdapter
- [ ] `app/core/schema.py` — QueryDecomposition, FactualVerdict, EvidenceRef (Pydantic)
- [ ] `app/core/verdict.py` — deterministic verdict derivation (no LLM)
- [ ] `app/core/trace.py` — BiTemporalTrace write + read
- [ ] `app/core/uncertainty.py` — confidence banding logic
- [ ] `domain/nem/schema.py` — EnergyEvent + NEMAdapter implementing DomainAdapter
- [ ] Unit test: NEMAdapter.to_feature_vector produces valid FeatureVector

**Day 3: AEMO Data Layer (Stubs First, Real Second)**
- [ ] `mcp/registry.py` + `mcp/router.py` — capability routing shell
- [ ] `mcp/aemo_live.py` — stub returning realistic sample data
- [ ] `mcp/aemo_archive.py` — stub + basic NEMWeb zip fetch for one day
- [ ] `mcp/aemo_notices.py` — stub + basic Market Notice fetch + parser
- [ ] `mcp/portfolio.py` — stub loading from config/portfolio.yaml
- [ ] Background scheduler: AEMO live refresh every 5 min
- [ ] `app/api/routes_market.py` — GET /market/state returns stubbed data
- [ ] Confirm: `GET /market/state?region=NSW1` returns valid EnergyEvent JSON

**Day 4: Frontend Shell**
- [ ] `frontend/static/app.html` — shell with ALL stable IDs from PRD, Alpine init
- [ ] `frontend/static/css/theme.css` — dark background, NEM region colour palette
  - NSW = blue, VIC = teal, QLD = orange, SA = purple, TAS = green
- [ ] `frontend/static/css/components.css` — recommendation card, evidence badge, panel
- [ ] `frontend/static/js/api.js` — fetch wrapper with JWT header
- [ ] `frontend/static/js/state.js` — Alpine store
- [ ] `frontend/static/js/main.js` — wires modules
- [ ] Confirm: open app.html, live market state panel shows stubbed NSW price

**Day 5: Test Foundation**
- [ ] `tests/test_acceptance_matrix.py` — all 11 queries wired, all return stubs
- [ ] `tests/test_verdict_contract.py` — schema validation tests
- [ ] `tests/fixtures/` — sample answer JSON, injection payloads
- [ ] Confirm: `pytest tests/` passes (stubs, not real answers — that's fine)

---

### Week 2 — Query Pipeline (Target: type a question, get a structured answer)

**Day 1-2: Decomposition + Prefill**
- [ ] `engines/decomposition.py` — Tier 1 LLM call → QueryDecomposition object
- [ ] `engines/prefill.py` — assembles evidence envelope from scatter stub results
- [ ] `domain/nem/decomposition_hints.py` — NEM intent patterns (LOR, trip, dispatch, etc.)
- [ ] `app/api/routes_query.py` — POST /query/decompose returns valid decomposition
- [ ] Test: all 11 acceptance queries produce valid decomposition objects

**Day 3: Scatter Agent Pipeline**
- [ ] `agents/scatter_agents.py` — 6 parallel agents firing via asyncio.gather
- [ ] `agents/orchestrator.py` — full scatter → collect → pass-3 → why → validate lifecycle
- [ ] `engines/prefill.py` — source coverage scoring, stale source detection

**Day 4: Security Observer Passes 1 + 4**
- [ ] `security/prompt_injection.py` — pattern + heuristic detector
- [ ] `security/security_observer.py` — Pass 1 (input) + Pass 4 (answer) implemented
- [ ] `security/observer_passes.py` — isolated pass logic
- [ ] Test S01-S04 from acceptance test matrix

**Day 5: Market State UI**
- [ ] `js/charts/swimlane.js` — ECharts price line with static data
- [ ] `js/evidence.js` — evidence drawer, freshness badge
- [ ] `js/query.js` — form submit → decompose → prefill → answer → render
- [ ] Recommendation card renders from FactualVerdict schema
- [ ] Confirm: type "Why is NSW expensive?" → structured response card appears

---

### Week 3 — Intelligence Layer (Target: answers grounded in real market data)

**Day 1-2: ChronoGraph**
- [ ] `engines/chronograph/adwin.py` — ADWIN change-point detection implementation
- [ ] `engines/chronograph/tdigest.py` — streaming t-digest quantile estimation
- [ ] `engines/chronograph/regime.py` — compose into regime verdict + confidence
- [ ] Wire to AEMO live data: ChronoGraph updates on every 5-min dispatch tick
- [ ] Test: inject price spike into test data → regime shift detected

**Day 3: HippoGraph + Analog Retrieval**
- [ ] `engines/hippograph/graph.py` — market state graph builder from FeatureVector
- [ ] `engines/hippograph/embedder.py` — local BGE embedder for state vectors
- [ ] `engines/hippograph/ppr.py` — Personalised PageRank over archive graph
- [ ] `engines/analog_retriever.py` — end-to-end: current state → top N historical analogs
- [ ] Test: Q05 "show me when prices looked like this before" returns analog count + outcomes

**Day 4: Why Engine + AEMO Real Data**
- [ ] `engines/why_engine.py` — four grounded sources → plain-English explanation
- [ ] `domain/nem/why_templates.py` — NEM-specific templates
- [ ] Real AEMO live client: fetch actual dispatch price from NEMWeb
- [ ] Real AEMO notices: fetch + parse + classify actual market notices
- [ ] Test Q02: "Why is NSW expensive?" returns grounded explanation with evidence_refs

**Day 5: Bitemporal Trace**
- [ ] `core/trace.py` — full trace write + read + as-of replay
- [ ] Every query writes a trace record
- [ ] `app/api/routes_trace.py` — GET /trace/{trace_id} returns full trace
- [ ] `js/trace.js` — trace replay panel in UI
- [ ] Test Q09: "Why did you recommend dispatch at 14:05?" replays correct trace

---

### Week 4 — LNN + Backtest (Target: distribution forecasts + counterfactual answers)

**Day 1-2: LTC Cell From Scratch**
- [ ] `engines/lnn/ltc_cell.py`
  - Implement the LTC ODE: dx/dt = -x/τ(x,I,θ) + f(x,I,θ)
  - τ is input-modulated: τ(x,I) = τ_0 exp(-|Wx + UI + b_τ|)
  - f is a sigmoid MLP: f(x,I) = σ(Wx + UI + b_f)
  - Euler/RK4 numerical solver for the ODE
  - PyTorch custom cell inheriting from nn.Module
- [ ] `engines/lnn/ltc_model.py` — sequence model: stack of LTC cells
- [ ] `engines/lnn/distribution.py` — quantile output head (p10/p50/p90)
- [ ] `tests/test_ltc_cell.py` — verify gradient flow, time constant modulation
- [ ] Initial training on 30 days of NEM archive data

**Day 3: Forecast Integration**
- [ ] `engines/forecast.py` — LNN inference → FeatureForecast with distribution
- [ ] Integrate with scatter Agent E (AEMO forecast comparison)
- [ ] Why Engine Source 2: forecast drivers from LNN attention/features
- [ ] Q10: "Will price hit $5,000 today?" returns probability band, not false certainty

**Day 4: Backtest Engine**
- [ ] `engines/backtest.py` — historical window simulation
- [ ] Bitemporal: simulate "as of" each past interval using only data available then
- [ ] `app/api/routes_backtest.py` — POST /backtest/run
- [ ] `js/backtest.js` — counterfactual form + interval results table
- [ ] Q06: "What would my battery have earned dispatching above $300 last quarter?"

**Day 5: Full Security Observer**
- [ ] Pass 2 (decomposition unsafe intent) implemented
- [ ] Pass 3 (tool-output hygiene) implemented
- [ ] All 16 security tests from acceptance matrix automated
- [ ] Observer metrics endpoint: GET /security/observer/metrics

---

### Week 5 — Multi-Tenant + Conversation History + Polish

**Day 1: Auth + Multi-Tenant**
- [ ] User registration + login endpoints
- [ ] Tenant creation flow
- [ ] JWT scoped per tenant
- [ ] All DB queries filtered by tenant_id
- [ ] ngrok / Cloudflare Tunnel setup instructions in README

**Day 2: Session History**
- [ ] `js/history.js` — session list with date grouping
- [ ] Left sidebar with ChatGPT-style session list
- [ ] Session search by keyword
- [ ] "New session" clears context, lets user switch region/portfolio

**Day 3: Evidence + Counterargument Polish**
- [ ] Evidence drawer fully implemented with source, interval, field, value, raw_ref
- [ ] "Why might this be wrong?" drawer on every recommendation
- [ ] Freshness banner visible when any source is stale
- [ ] Missing data list visible when coverage is incomplete

**Day 4: Real End-to-End Test**
- [ ] Run all 11 acceptance matrix questions against real AEMO data
- [ ] All 16 security tests pass
- [ ] Backtest test determinism confirmed (same inputs → same output)
- [ ] Performance: standard queries < 1s, Tier 2 < 5s

**Day 5: Demo Polish + README**
- [ ] README with three sections: researcher / energy professional / enterprise AI
- [ ] ARCHITECTURE.md: how to add a new vertical (the DomainAdapter pattern)
- [ ] Demo recording: 90 seconds, starts at live dashboard, asks Q01, traces Q09
- [ ] `docker compose up` → working demo in one command

---

## 9. How to Add a New Vertical (The Reskin Pattern)

This section lives in ARCHITECTURE.md. The pattern for any new domain:

**Step 1: Define your domain event schema**
```python
# domain/shipping/schema.py
from app.core.interfaces import IngestEvent, DomainAdapter

@dataclass
class VesselEvent(IngestEvent):
    vessel_id: str
    lat: float
    lon: float
    speed_knots: float
    port_eta: datetime | None
    weather_state: str

class ShippingAdapter(DomainAdapter):
    def to_feature_vector(self, event: VesselEvent) -> FeatureVector:
        # Map vessel state to numeric features ChronoGraph understands
        ...
    def why_template(self, regime_label: str, analog_count: int) -> str:
        return f"Delay likely because {regime_label}. Similar patterns in {analog_count} historical voyages."
```

**Step 2: Build your data MCP**
```python
# mcp/ais_live.py   (AIS = Automatic Identification System, vessel tracking)
# Zone 1, read-only, public data
```

**Step 3: Register in config/verticals.yaml**
```yaml
verticals:
  nem:
    domain_adapter: domain.nem.schema.NEMAdapter
    primary_mcp: aemo_live
    decomposition_hints: domain.nem.decomposition_hints
  shipping:
    domain_adapter: domain.shipping.schema.ShippingAdapter
    primary_mcp: ais_live
    decomposition_hints: domain.shipping.decomposition_hints
```

**Step 4: Write acceptance tests**
Same pattern as the NEM 11-question matrix, but with domain-specific questions.
The entire framework stack (ChronoGraph, HippoGraph, LNN, Security Observer, bitemporal trace) runs unchanged.

**For video/LiDAR/audio verticals:**
These require a preprocessing step BEFORE the domain adapter:
- Video → run object detection → emit VehicleEvent or AnimalEvent → then DomainAdapter
- LiDAR → run point cloud processing → emit SpatialEvent → then DomainAdapter
- Audio → run transcription/classification → emit AudioEvent → then DomainAdapter

The framework never receives raw pixels or point clouds — it receives events.
Document the preprocessing pipeline in the vertical's README. Do not stub it in core.

---

## 10. Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| LTC from-scratch takes longer than expected | Medium | ncps library is the fallback for MVP; swap in during Week 4 if blocked |
| NEMWeb rate limits hit during archive build | Medium | Polite crawl with 1-2s delay; cache everything; never fetch twice |
| PostgreSQL adds setup friction | Low | docker-compose handles it; pgAdmin for debugging |
| Scatter agents time out on slow queries | Medium | Per-agent timeout (500ms hard limit); return partial evidence with LOW_CONFIDENCE |
| Security Observer adds too much latency | Low | Passes 1 and 4 are mandatory; Passes 2 and 3 can degrade gracefully |
| Multi-tenant row filtering missed on a query | Medium | DB-level row security policy (PostgreSQL RLS) as second layer behind app filtering |
| LLM for decomposition unavailable | Low | Rule-based fallback decomposer in engines/decomposition.py for known query patterns |

---

## 11. First Command On Wakeup

```bash
cd C:\AI\GridVerdict
mkdir -p app/api app/core app/engines/chronograph app/engines/hippograph
mkdir -p app/engines/lnn app/agents app/security app/mcp app/domain/nem app/db
mkdir -p frontend/static/css frontend/static/js/charts
mkdir -p config tests/fixtures/sample_answers tests/fixtures/injection_payloads
mkdir -p domain

# Then start with:
# 1. docker-compose.yml
# 2. pyproject.toml
# 3. app/core/interfaces.py  (the framework contract)
# 4. app/core/schema.py      (the answer/verdict Pydantic models)
# 5. app/api/main.py         (FastAPI init only)
```

The first 3 files written define the architecture for everything else.
Get `core/interfaces.py` right before anything else exists.
