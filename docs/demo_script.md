# GridVerdict Demo Script

Structured walkthrough for a LinkedIn video, GitHub demo, or technical interview. Target runtime: 12-18 minutes.

---

## 0. Pre-Demo Checks

Run with Docker/Postgres and live AEMO access when possible:

```bash
docker compose up -d db
uvicorn app.api.main:app --reload --port 8000
```

Before recording, show:

- topbar `LIVE` indicator;
- Data tab last dispatch ingest timestamp;
- current market price timestamp;
- evidence refs in the Answer panel;
- `/api/data/status` freshness if challenged.

This proves the demo is not static sample data.

---

## 1. Opening

Say:

> GridVerdict is an evidence-grounded NEM decision-support cockpit. It uses language to understand the question, but deterministic code answers with live data, evidence refs, model status, and missing-data honesty.

Make clear:

- it does not place trades;
- it does not let an LLM invent prices;
- it is a showcase of architecture, temporal reasoning, market analytics, and AI safety.

---

## 2. Question 1: Why Elevated And Will It Continue?

Ask:

```text
Why is NSW price elevated right now, what evidence supports it, and is it likely to continue?
```

Show in the chat card:

- short Answer section;
- recent price path: now, 5m ago, 10m ago;
- supported vs unconfirmed drivers;
- model-specific continuation summary.

Then open the right panel:

- Evidence refs;
- Claim Map;
- Causality Confidence;
- Forecast;
- Data freshness.

Talk track:

> The chat card is intentionally short. Full evidence is behind panels. The system separates "confirmed facts" from "driver unconfirmed" so it does not overclaim causality.

---

## 3. Question 2: Historical Analogs

Ask:

```text
Have we seen similar NSW price and headroom conditions before, and what happened afterwards?
```

Show:

- HippoGraph analog count;
- top analogs in the History/Analogs panel;
- outcome split: recovered, continued, unknown;
- caveat if the graph or archive is cold.

Talk track:

> HippoGraph is not an LLM memory. It embeds market states and retrieves comparable historical states. That gives a base-rate answer to "what happened next?"

---

## 4. Question 3: Weather, Notices, RSS

Ask:

```text
Are live weather conditions, AEMO notices, or recent RSS energy news helping explain the NSW price move?
```

Show:

- weather consensus or missing/stale weather;
- AEMO notice relevance;
- RSS/news relevance;
- source freshness.

Talk track:

> External context is treated as evidence, not instruction. RSS and weather can support or weaken an explanation, but they cannot override official dispatch facts.

---

## 5. Live Feed

Open the Live Feed tab.

Show:

- baseline event after startup;
- material change events if available;
- severity filter;
- Evidence Map inside a commentary card;
- "Ask about this" prefill workflow.

Talk track:

> This is rolling market commentary. New dispatch, forecast, source-stale, and commentary events arrive over SSE without refreshing the page.

---

## 6. Architecture Walkthrough

Draw or show this flow:

```text
User
  -> Decomposer
  -> MCP ScatterGather
  -> Evidence Normalizer
  -> WhyBuilder
  -> Claim Verifier
  -> Answer Planner
  -> UI + Trace

Live data:
  AEMO dispatch / notices / archive
  RSS news
  Weather consensus
  Forecast models
  HippoGraph analogs
  TemporalRAG citations
```

Explain the layered trust model:

- SecurityObserver checks unsafe requests and unsafe evidence;
- decomposer identifies intent and required evidence;
- deterministic layers compute facts and confidence;
- claim verifier downgrades unsupported claims;
- Answer Planner makes the output readable;
- audit trace records what happened.

---

## 7. Forecast Panel

Open Forecast or Models.

Show:

- LEAR/QRA/meta-ensemble when available;
- LNN/TCN status if trained or unavailable;
- P10/P50/P90 bands;
- model calibration/trust table where available.

Talk track:

> Forecast output is model-specific. If LNN is untrained, the UI says so. That is more trustworthy than pretending every model is always available.

---

## 8. Data And Trust

Open the Data tab.

Show:

- dispatch rows/freshness;
- scheduler job health;
- HippoGraph node counts;
- TemporalRAG document count;
- LNN trainer status;
- backfill cursor state.

Talk track:

> Professional users need to know whether the system is fresh. A good answer with stale data is still operationally dangerous.

---

## 9. Closing

Say:

> GridVerdict is not trying to be a generic chatbot. It is an evidence cockpit: live tools, market models, historical analogs, bitemporal retrieval, claim verification, and concise answers.

Current honest status:

- strong architecture/demo;
- credible early pilot surface;
- not a production trading platform without deeper causality, licensing, security review, SLA process, and expert validation.

---

## Quick Demo Query List

```text
Why is NSW price elevated right now, what evidence supports it, and is it likely to continue?
Have we seen similar NSW price and headroom conditions before, and what happened afterwards?
Are live weather conditions, AEMO notices, or recent RSS energy news helping explain the NSW price move?
What sources are stale or missing right now?
Should I dispatch my NSW battery right now?
Compare NSW, VIC, and QLD prices right now.
What did GridVerdict know at the time of the previous NSW spike?
```
