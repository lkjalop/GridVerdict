# GridVerdict

Evidence-grounded decision-support cockpit for the Australian National Electricity Market (NEM).

GridVerdict answers natural-language questions about live market conditions with short, cited, defensible responses. The LLM is used for query decomposition only; prices, model values, causality tiers, confidence, evidence refs, and final answer sections are built by deterministic code.

Research/demo use only. GridVerdict does not execute bids or trades.

---

## What It Does

- **Live NEM monitoring**: 5-minute dispatch prices for NSW1, VIC1, QLD1, SA1, and TAS1.
- **Concise Answer Planner**: chat cards show short sections: Answer, Evidence, Drivers, Continuation, Missing.
- **Evidence refs**: factual claims trace to source records, timestamps, and provenance.
- **Causality tiers**: confirmed, supported, plausible, and unconfirmed drivers are separated explicitly.
- **Incident timeline**: price trajectory, constraints, interconnectors, unit dispatch, rebids, FCAS, outages, and weather context when available.
- **Forecasting**: LEAR, QRA, GBM/TCN/LNN candidates, meta-ensemble output, conformal calibration, and model availability/status.
- **Historical analogs**: HippoGraph retrieves similar market states and summarizes what happened afterwards.
- **TemporalRAG**: distinguishes valid_time from system_time so retrospective answers can respect what was known at the time.
- **Rolling live feed**: SSE-backed commentary events for material price, forecast, data freshness, weather, and source changes.
- **BESS scenario engine**: simulation-only battery dispatch economics and missing-data checklist.
- **Security and compliance surfaces**: SecurityObserver, claim verifier, audit traces, model registry, ISO 42001-style model cards, ISO 27001/AESCSF mappings.
- **Production plumbing**: PostgreSQL, Redis pub/sub, Redis rate limiting, scheduler leader election, Prometheus metrics, and runbooks.

---

## Quick Start

### Prerequisites

```text
Python 3.11+
PostgreSQL via Docker for realistic demos
Redis optional for multi-worker event bus / rate limit / scheduler leader lock
Ollama optional; rule-based decomposer works for local tests
```

### Install

```bash
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and set values appropriate for local use:

```env
DATABASE_URL=postgresql+asyncpg://gv:<password>@localhost:5432/gridverdict
GRIDVERDICT_DEV_NO_AUTH=true
JWT_SECRET=local-demo-secret
DECOMPOSER_BACKEND=rule_based
```

### Run

```bash
docker compose up -d db
uvicorn app.api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/api/docs
```

---

## Demo Questions

Use these to show the current Answer Planner flow:

```text
Why is NSW price elevated right now, what evidence supports it, and is it likely to continue?
```

Shows live dispatch evidence, recent 5m/10m trend, driver tiers, forecast model direction, missing causality blockers, and HippoGraph analog outcome if available.

```text
Have we seen similar NSW price and headroom conditions before, and what happened afterwards?
```

Shows HippoGraph analog count, outcome split, caveat if the graph is cold, and raw analog details in the right panel.

```text
Are live weather conditions, AEMO notices, or recent RSS energy news helping explain the NSW price move?
```

Shows weather/notice/news correlation only when those sources are relevant and fresh; otherwise it says what is missing or stale.

---

## Architecture

```text
User question
  -> SecurityObserver pass 1
  -> Decomposer: intent, region, requested_output, evidence needs
  -> ScatterGather MCP tools: dispatch, notices, RSS, weather, forecast, analogs
  -> WhySources normalize raw evidence
  -> WhyBuilder computes deterministic facts, driver tiers, claim map
  -> Verdict formatter creates FactualVerdict
  -> Claim verifier downgrades unsupported causal/forecast claims
  -> Answer Planner creates short user-facing sections
  -> SecurityObserver pass 4
  -> Trace/audit persistence + SSE events
  -> Frontend chat summary + right-panel evidence tabs
```

The important trust rule is simple:

```text
LLMs route and structure the question.
Deterministic code answers with evidence.
```

---

## Project Structure

```text
app/
  agents/       ScatterGather, WhySources, WhyBuilder, claim verifier, Answer Planner
  api/          FastAPI routes for query, market, incidents, models, data status, events
  compliance/   ISO/AESCSF mappings, risk register, model registry surfaces
  core/         Schema, evidence contracts, trace writer
  data/         Scheduler, cache, Redis client, event bus
  db/           SQLAlchemy models, sessions, migrations
  engines/      ChronoGraph, HippoGraph, TemporalRAG, incident timeline, forecasting
  mcp/          Read-only tool registry and AEMO/RSS/weather/archive tools
  portfolio/    BESS scenario and dispatch policy
  security/     SecurityObserver
frontend/       Alpine.js SPA, ECharts, live feed, answer/evidence panels
tests/          Unit, integration, regression, Playwright e2e tests
docs/           Architecture, assessments, model cards, runbooks, demo script
monitoring/     Prometheus alert rules and Grafana dashboard
```

---

## Testing

Core smoke:

```bash
python -m pytest tests/test_answer_planner.py tests/test_decomposer_quality.py tests/test_professional_question_regressions.py -q
```

Frontend e2e:

```bash
E2E_TESTS=1 python -m pytest tests/e2e/ -q
```

Live MCP smoke, when network and NEMWeb are reachable:

```bash
GRIDVERDICT_LIVE_TESTS=1 python -m pytest tests/test_live_mcp_smoke.py -q
```

Recent development runs have exercised more than 1,800 non-e2e tests plus Playwright browser checks. Exact counts vary as new regression tests are added.

---

## Design Constraints

- **No trading or bid submission**: simulation/advisory only.
- **Read-only MCP tools**: external data fetches do not write to external systems.
- **No hidden unsupported causality**: causes are tiered and downgraded when evidence is missing.
- **No forecast authority without model status**: unavailable LNN/TCN/GBM outputs are shown as unavailable, not implied.
- **Freshness matters**: stale sources reduce confidence and are visible in the UI.
- **Auditability**: traces and model provenance are persisted for replay and review.

---

## Current Readiness

- **Technical demo/MVP**: strong. Suitable for architecture walkthroughs, LinkedIn/GitHub showcase, and live local demos with a warmed DB.
- **Professional pilot**: credible but still needs deeper constraint/interconnector/unit/rebid causality, sharper forecast calibration surfaces, and expert market validation.
- **Commercial production**: not complete. Requires licensing review, security/tenant isolation review, uptime/SLA process, customer onboarding, and operational support.
