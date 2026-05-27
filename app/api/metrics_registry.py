"""Prometheus metric registry for GridVerdict.

All metric objects are defined here as module-level singletons. Import individual
metrics from this module; never construct Counter/Gauge/Histogram elsewhere —
prometheus_client enforces globally unique names and raises on re-registration.

Metrics are instrumented at their natural call sites:
  ingest_success/failure_total — scheduler._record_success / _record_failure
  claim_verifier_downgrades_total — claim_verifier.verify_answer
  last_dispatch_age_seconds — metrics endpoint (pull on scrape)
  sse_subscribers — metrics endpoint (pull on scrape)
  *_latency_ms histograms — routes_query / routes_market (observe on response)
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Ingestion counters ────────────────────────────────────────────────────────

ingest_success_total = Counter(
    "gv_ingest_success_total",
    "Successful scheduler job runs",
    ["job"],
)

ingest_failure_total = Counter(
    "gv_ingest_failure_total",
    "Failed scheduler job runs",
    ["job"],
)

# ── Claim verifier ────────────────────────────────────────────────────────────

claim_verifier_downgrades_total = Counter(
    "gv_claim_verifier_downgrades_total",
    "Claim verifier downgrade findings emitted",
)

# ── Gauges (refreshed on each scrape) ────────────────────────────────────────

last_dispatch_age_seconds = Gauge(
    "gv_last_dispatch_age_seconds",
    "Seconds since the last successful AEMO dispatch snapshot",
)

sse_subscribers = Gauge(
    "gv_sse_subscribers",
    "Current active SSE subscriber count",
)

# ── Infrastructure health gauges ─────────────────────────────────────────────

redis_connected = Gauge(
    "gv_redis_connected",
    "Redis connection health (1 = up, 0 = down)",
)

scheduler_jobs_running = Gauge(
    "gv_scheduler_jobs_running",
    "Number of APScheduler jobs currently executing",
)

# ── Latency histograms ────────────────────────────────────────────────────────

query_latency_ms = Histogram(
    "gv_query_latency_ms",
    "End-to-end query processing latency in milliseconds",
    buckets=[100, 250, 500, 1_000, 2_000, 5_000, 10_000],
)

llm_decompose_latency_ms = Histogram(
    "gv_llm_decompose_latency_ms",
    "LLM query decomposition latency in milliseconds",
    buckets=[100, 250, 500, 1_000, 2_000, 5_000],
)

forecast_latency_ms = Histogram(
    "gv_forecast_latency_ms",
    "Forecast model inference latency in milliseconds",
    buckets=[10, 50, 100, 250, 500, 1_000, 2_000],
)
