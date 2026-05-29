# GridVerdict

**Evidence-grounded decision-support cockpit for the Australian National Electricity Market (NEM).**

Natural-language questions about live market conditions. Short, cited, defensible answers.  
The LLM decomposes the question. Deterministic code answers it with evidence.

```
docker compose up
```

→ Open `http://localhost:8000`

---

## What makes it technically interesting

| Component | What it does | Why it matters |
|---|---|---|
| **LTC cell** | Liquid Time-Constant ODE implemented from scratch in PyTorch — `dx/dt = -x/τ(x,I,θ) + f(x,I,θ)` | Continuous-time model that adapts its time constant to input volatility — handles SA1's $15k spikes differently to TAS1's flat overnight prices |
| **ChronoGraph** | ADWIN change-point detection + t-digest streaming quantiles | Detects regime transitions in real time without storing the full price history |
| **HippoGraph** | Market-state graph + Personalised PageRank analog retrieval | Finds the 10 most similar historical market states and reports what happened next |
| **TemporalRAG** | Bitemporal archive (valid_time vs system_time) | Answers retrospective questions with what the system *actually knew at the time*, not hindsight |
| **Evidence contract** | Every `SUPPORTED` verdict requires a cited `evidence_ref` | Confidence bands are derived from evidence coverage, never from model self-assessment |
| **Security Observer** | 4-pass hygiene pipeline on every query | Tool outputs are treated as untrusted data, not instructions — prompt injection blocked at pass 3 |
| **Forecast ensemble** | LEAR + QRA + GBM + TCN + LNN meta-ensemble with conformal calibration | P10/P50/P90 intervals; models degrade gracefully when data is insufficient |
| **Drift monitor** | ADWIN on live forecast residuals per region | Triggers early model retraining when error distribution shifts beyond expected NEM volatility |

---

## Three ways to read this project

### If you're a researcher or ML practitioner

GridVerdict is the first application of a reusable **temporal reasoning framework**. The same components — ChronoGraph, HippoGraph, TemporalRAG, the LNN/LTC cell, the Security Observer — can power a new vertical (shipping logistics, ICU monitoring, bushfire tracking) by writing one `DomainAdapter` class and one data MCP. The `core/` and `engines/` packages have zero imports from `data/`, `mcp/`, or domain code — the framework boundary is enforced, not aspirational.

The LTC cell is implemented from scratch in `app/engines/lnn/ltc_cell.py` without relying on the `ncps` library. It implements the full ODE with an Euler/RK4 solver, input-modulated time constant, and a sigmoid MLP output. The distribution head produces quantile outputs (P10/P50/P90) rather than point forecasts. SA1 — South Australia, the most volatile NEM region — gets a larger hidden dimension and lower learning rate than other regions because spike volatility is structurally different from normal price variance.

Relevant files for ML research:
- [`app/engines/lnn/ltc_cell.py`](app/engines/lnn/ltc_cell.py) — LTC ODE implementation
- [`app/engines/lnn/distribution.py`](app/engines/lnn/distribution.py) — quantile head + pinball loss
- [`app/engines/lnn/trainer.py`](app/engines/lnn/trainer.py) — feature normalisation, training loop, region-specific config
- [`app/engines/forecasting/inference.py`](app/engines/forecasting/inference.py) — SA1/other region config, bootstrap from DB
- [`app/engines/chronograph/`](app/engines/chronograph/) — ADWIN + t-digest
- [`app/engines/hippograph/`](app/engines/hippograph/) — PPR analog retrieval
- [`app/engines/drift_monitor.py`](app/engines/drift_monitor.py) — River ADWIN on live residuals
- [`app/engines/forecasting/models/meta_ensemble.py`](app/engines/forecasting/models/meta_ensemble.py) — ensemble weighting

### If you're an energy market professional

GridVerdict answers these questions in under a second from live AEMO data:

- *Why is NSW price elevated right now?* → confirms or denies each driver (constraints, interconnectors, unit dispatch, rebids, weather, AEMO notices) with cited evidence refs and explicit "not confirmed" flags for missing data
- *Have we seen conditions like this before?* → HippoGraph returns top-10 historical analog states and what happened next
- *Is this cheap compared to last year?* → P25/median/P75/P90 for this hour/season window from the dispatch archive
- *What would my BESS have earned last quarter?* → backtest engine with no-lookahead bitemporal replay
- *What are the FCAS opportunity values right now?* → 8 NEM FCAS markets, tight-market detection, combined raise+lower opportunity

What it **cannot** do: execute bids or trades, predict participant bidding strategy (data is confidential until T-5min), cover WA/NT (different grids), or provide financial products data (ASX/RECs).

Data source: public NEMWeb only (free, no credentials). 5-minute dispatch prices, pre-dispatch 30-min intervals, AEMO market notices, BOM weather.

### If you're an AI/ML engineer evaluating architecture

The key design decision: **LLMs route and structure; deterministic code answers.**

The language model produces a `QueryDecomposition` — intent label, entity list, `requested_output` type, `sub_questions` schema, confidence score. It never asserts market facts or generates numbers. The following pipeline is entirely deterministic:

```
ScatterGather (parallel, 8 async tasks, ~330ms)
  → AEMO live price
  → AEMO market notices
  → FCAS prices
  → HippoGraph PPR analogs
  → LNN quantile forecast
  → AEMO predispatch comparison
  → NEM news RSS
  → Weather consensus

WhySources (normalise raw evidence into typed dataclasses)
WhyBuilder (deterministic driver tiers, claim map, evidence refs)
ClaimVerifier (downgrades unsupported causal/forecast claims)
AnswerPlanner (builds 3-5 line answer sections per sub-question type)
SecurityObserver pass 4 (every numeric claim must have evidence_ref)
```

Security Observer runs 4 passes synchronously on every query — input, decomposition, tool output, and answer. Tool outputs are treated as untrusted data; prompt injection is blocked before it reaches the context. The implementation is in [`app/security/`](app/security/).

Bitemporal trace: every query writes a `Trace` row with `valid_time` and `system_time`, enabling full replay of any past decision with exactly the data that was available at that moment.

Compliance surfaces: ISO 27001 control mappings, AESCSF profile, AI risk register, model registry with performance metrics, and audit log are all queryable via the API. See [`app/compliance/`](app/compliance/) and [`app/audit/`](app/audit/).

---

## Quick start

### Requirements

- Docker + Docker Compose
- 4 GB RAM (PyTorch CPU image)

### One command

```bash
cp .env.example .env   # edit DB_PASSWORD and JWT_SECRET
docker compose up
```

Open `http://localhost:8000`

On first startup the LNN bootstraps from the dispatch archive and trains all 5 NEM regions in ~30 seconds. You'll see `LTC[NSW1] trained on 599 intervals, loss=98` in the logs. All five models must show ✓ in the top bar before the forecast panel is meaningful.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `DB_PASSWORD` | yes | PostgreSQL password |
| `JWT_SECRET` | yes | 32+ char secret for JWT signing |
| `GRIDVERDICT_DEV_NO_AUTH` | dev only | Set `true` to skip auth (never in prod) |
| `DECOMPOSER_BACKEND` | no | `ollama` / `claude` / `rule_based` (default: `ollama`) |
| `OLLAMA_BASE_URL` | no | Ollama endpoint (default: `http://localhost:11434`) |
| `ANTHROPIC_API_KEY` | no | Claude API key for LLM decomposer fallback |

The rule-based decomposer runs with no LLM at all — useful for CI and offline demos. For best intent accuracy use Ollama with `qwen2.5:7b` or `mistral`.

### With Ollama (better decomposer accuracy)

```bash
ollama pull qwen2.5:7b
# then in .env:
DECOMPOSER_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:7b
```

---

## Capability map

| Question class | Supported | Notes |
|---|---|---|
| Live causation (why is price X right now?) | ✓ | Tiered driver evidence with explicit missing-data flags |
| 4-hour probabilistic forecast | ✓ | LEAR + QRA + LNN + TCN meta-ensemble, P10/P50/P90 |
| Historical retrospective (any archived event) | ✓ | Bitemporal archive, valid_time-aware retrieval |
| BESS dispatch optimisation | ✓ | Energy + FCAS combined revenue, simulation only |
| Counterfactual ("what if X had not rebid") | ✓ | Rebid detection + modified dispatch replay |
| Seasonal multi-year analysis | ✓ | Hour/season/year bucket comparison |
| Auto-commentary during events | ✓ | ADWIN-triggered commentary events via SSE |
| Compliance audit trail | ✓ | ISO 42001-style decision trace, model provenance |
| Long-horizon forecasting (>4h) | ✗ | Requires structural market model + bidding strategy |
| Participant intent prediction | ✗ | Bid data confidential until T-5min |
| Financial products (futures, RECs, swaps) | ✗ | ASX data, different license |
| Western Australia / Northern Territory | ✗ | Different grids (SWIS/NWIS), out of scope |

---

## Running tests

```bash
# Fast CI suite — no DB or network needed
python -m pytest tests/test_decomposer_quality.py tests/test_answer_planner.py tests/test_why_engine.py -q

# Full offline suite (1982 tests, ~60s)
python -m pytest tests/ --ignore=tests/test_live_mcp_smoke.py --ignore=tests/test_live_qa_regressions.py -q

# Live MCP smoke (requires network + NEMWeb access)
GRIDVERDICT_LIVE_TESTS=1 python -m pytest tests/test_live_mcp_smoke.py -q
```

---

## Project layout

```
app/
  agents/       ScatterGather, WhySources, WhyBuilder, ClaimVerifier, AnswerPlanner
  api/          FastAPI routes — query, market, incidents, models, events, trace
  audit/        Audit logger + export
  compliance/   ISO 27001, AESCSF, AI risk register
  core/         Schema, evidence contracts, bitemporal trace writer
  data/         Scheduler, cache, Redis event bus, AEMO live client
  db/           SQLAlchemy models, sessions, Alembic migrations
  engines/      ChronoGraph, HippoGraph, TemporalRAG, LNN, forecasting ensemble,
                drift monitor, FCAS attribution, incident timeline, fuel mix
  mcp/          Read-only MCP registry — AEMO live/archive/notices, weather, news
  portfolio/    BESS scenario engine, fleet coordinator, dispatch policy
  security/     SecurityObserver — 4-pass hygiene pipeline
frontend/       Alpine.js SPA, ECharts swimlane/analog/backtest, live SSE feed
tests/          Unit, integration, sprint regression, acceptance matrix
docs/           Architecture walkthrough, demo script, model cards, runbooks
config/         Model profiles, MCP tool registry, source freshness thresholds
```

---

## Extending to a new domain

GridVerdict is vertical one of a reusable framework. To add shipping, bushfire monitoring, or healthcare vital signs:

1. **Define your event schema** — extend `core.interfaces.IngestEvent` with domain fields
2. **Implement `DomainAdapter`** — `to_feature_vector()`, `to_evidence_ref()`, `decomposition_hints()`, `why_template()`
3. **Write a data MCP** — read-only, Zone 1 tool
4. **Register in `config/verticals.yaml`**
5. **Write 10 acceptance questions** — same format as `tests/test_acceptance_matrix.py`

The entire `core/`, `engines/`, `agents/`, and `security/` stack runs unchanged.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full reskin guide.

---

## Design constraints

- **No bid execution** — simulation and advisory only, no write path to AEMO or any broker
- **Read-only MCP tools** — external fetches never write to external systems
- **No unsupported causality** — drivers are tiered; unconfirmed causes are labelled, not hidden
- **No forecast authority without model status** — unavailable models show as unavailable
- **Auditability** — every query writes a bitemporal trace; decisions are replayable

---

*Research/demo use only. GridVerdict does not execute bids or trades and makes no representation of fitness for trading or operational decisions.*
