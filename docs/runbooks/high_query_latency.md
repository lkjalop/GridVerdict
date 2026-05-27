# Runbook: HighQueryLatencyP99 / HighQueryLatencyMedian / HighDecomposeLatency

**Alerts:**
- `HighQueryLatencyP99` — P99 > 5 s for 5 min
- `HighQueryLatencyMedian` — P50 > 2 s for 10 min
- `HighDecomposeLatency` — LLM decompose P95 > 3 s for 5 min

**Metrics:** `gv_query_latency_ms`, `gv_llm_decompose_latency_ms`
**Team:** backend

---

## Query Pipeline Stages (and their latency budgets)

| Stage | Typical | Budget |
|---|---|---|
| Security pass 1 (input hygiene) | <1 ms | 5 ms |
| LLM decompose (Ollama → Claude → rule-based) | 50–2000 ms | 3 s |
| ScatterGather (AEMO + analogs + notices in parallel) | 100–800 ms | 2 s |
| TemporalRAG retrieval | 50–200 ms | 500 ms |
| Why engine + claim verifier | <20 ms | 100 ms |
| DB persist (flush) | <50 ms | 200 ms |

---

## Diagnosis

**P99 latency spike (HighQueryLatencyP99):**
1. Check `gv_llm_decompose_latency_ms` — is LLM the bottleneck?
2. Check `gv_forecast_latency_ms` — did forecast inference spike?
3. Look for DB slow queries in PostgreSQL slow query log.
4. Check if scatter-gather is waiting on a slow/unavailable external source.

**HighDecomposeLatency:**
1. Check Ollama is running: `ollama list` / `ollama ps`
2. Check Claude API availability if Ollama is down (fallback chain)
3. The rule-based fallback activates instantly — if P95 is high, the LLM tier is in use
4. Consider reducing `OLLAMA_TIMEOUT_S` to force faster fallback

---

## Remediation

| Cause | Action |
|---|---|
| Ollama overloaded | Restart Ollama; reduce concurrency; add GPU |
| Claude API slow | Check Anthropic status page; reduce Claude timeout |
| PostgreSQL slow queries | Run `EXPLAIN ANALYZE` on the offending query; add indexes |
| ScatterGather source timeout | Check AEMO NEMWeb latency; the source times out and falls back |
| High ScatterGather fan-out (COMPARISON intent) | Limit comparison regions; these fan-out to N parallel gather calls |

---

## Fast mitigation

The query pipeline has three decomposition tiers: Ollama → Claude → rule-based.
If LLM latency is the issue, temporarily set `OLLAMA_URL=""` to skip to Claude (or rule-based if Claude is also slow).
