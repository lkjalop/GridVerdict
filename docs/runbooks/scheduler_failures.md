# Runbook: SchedulerJobFailure / SchedulerJobCritical

**Alert:** `SchedulerJobFailure` (>0 failures/5 m) | `SchedulerJobCritical` (>0.1/s for 10 m)
**Metric:** `gv_ingest_failure_total{job="..."}`
**Team:** data

---

## Symptoms

One or more APScheduler jobs are failing repeatedly.
Historical data gaps will accumulate; forecast model training windows shrink.

---

## Job Inventory

| Job ID | Interval | Effect of failure |
|---|---|---|
| `dispatch_refresh` | 5 min | Stale live prices — see `stale_dispatch.md` |
| `notices_refresh` | 60 s | Stale AEMO notices in scatter-gather |
| `lnn_retrain` | 1 h | LNN uses last trained weights; eventually stale |
| `predispatch_refresh` | 30 min | Predispatch intervals not ingested; forecast features degrade |
| `archive_backfill` | 1 h | Historical gaps not filled; analog retrieval degrades |
| `nem_news_refresh` | 5 min | No RSS commentary in query responses |
| `weather_refresh` | 5 min | No weather consensus in query responses |

---

## Diagnosis

1. **Identify failing job**
   ```
   GET /api/data/status  →  "scheduler" → per-job last_error, consecutive_failures
   ```

2. **Check logs for job-specific errors**
   ```
   grep "failed" app.log | grep <job_id>
   ```

3. **Check external dependencies** — each job calls an external service:
   - `dispatch_refresh` / `predispatch_refresh`: NEMWeb
   - `notices_refresh`: AEMO notices feed
   - `archive_backfill`: NEMWeb MMSDM archive
   - `nem_news_refresh`: RSS feeds (WattClarity, etc.)
   - `weather_refresh`: weather API

---

## Remediation

Most job failures are transient (network blips, rate limits). The scheduler retries automatically.

| Scenario | Action |
|---|---|
| `consecutive_failures < 3` | Monitor — automatic retry will clear it |
| `consecutive_failures >= 3` | Check `last_error`; may need source or config fix |
| All jobs failing | Likely a Python import error at startup — check full logs |
| `lnn_retrain` persistently failing | Check torch/GPU availability; skip if not available |
| DB connection errors | Check PostgreSQL health; check connection pool settings |

---

## Escalation

If `SchedulerJobCritical` fires for `dispatch_refresh`, treat it the same as `DispatchDataCritical`
— see `stale_dispatch.md`. For other jobs, escalate if `consecutive_failures > 10`.
