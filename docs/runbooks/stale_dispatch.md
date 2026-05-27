# Runbook: StaleDispatchData / DispatchDataCritical

**Alert:** `StaleDispatchData` (>120 s) | `DispatchDataCritical` (>600 s)
**Metric:** `gv_last_dispatch_age_seconds`
**Team:** data

---

## Symptoms

The `gv_last_dispatch_age_seconds` gauge exceeds 120 seconds (warning) or 600 seconds (critical).
Live market panels display stale prices; forecast models receive no fresh input features.

---

## Diagnosis

1. **Check scheduler job health**
   ```
   GET /api/data/status   →  "scheduler" → "dispatch_refresh"
   ```
   Look for `consecutive_failures > 0` and `last_error`.

2. **Check NEMWeb availability**
   - AEMO NEMWeb occasionally returns 503 or redirects. The live client logs `AEMO down` at WARNING.
   - Check app logs: `grep "Dispatch refresh failed"`.

3. **Check Redis connectivity** (if multi-worker)
   - Only the leader runs dispatch_refresh. If the leader died and no failover happened,
     dispatch may silently skip. Check `gv_redis_connected == 1` and `gv_scheduler_jobs_running`.

4. **Check network egress**
   - The app must reach `www.nemweb.com.au`. Check firewall/proxy rules.

---

## Remediation

| Cause | Action |
|---|---|
| Transient AEMO outage | Wait — the scheduler retries every 5 min automatically |
| NEMWeb rate-limit (HTTP 429) | The client uses a browser UA. Check logs for 429; rate limit resets in ~1 min |
| Redis leader lock stuck | Restart the leader instance; follower acquires within `redis_leader_lock_ttl_s` (default 30 s) |
| Persistent AEMO outage | Monitor AEMO status page; activate manual override if >30 min |
| Scheduler not running | Check lifespan startup logs. If missing, restart the app process |

---

## Escalation

If `DispatchDataCritical` fires and the outage exceeds 30 minutes, page the on-call engineer.
All live recommendations are operating on stale data — consider displaying a maintenance banner.
