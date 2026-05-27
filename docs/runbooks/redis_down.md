# Runbook: RedisDown / RedisMetricAbsent

**Alert:** `RedisDown` (`gv_redis_connected == 0`) | `RedisMetricAbsent`
**Metric:** `gv_redis_connected`
**Team:** infrastructure

---

## Symptoms

`gv_redis_connected` reports 0 (or is absent). The following features **degrade to in-process fallbacks**:
- Distributed rate limiting → per-worker in-process limits (horizontal scaling breaks)
- SSE broadcast fanout → no cross-worker fan-out (each worker only sees its own subscribers)
- Scheduler leader election → all workers act as leaders (duplicate job execution)

The app **does not crash** — it degrades gracefully. But horizontal scaling is non-functional.

---

## Diagnosis

1. **Check Redis container / service**
   ```bash
   docker ps | grep redis
   redis-cli -h <host> -p 6379 ping
   ```

2. **Check app logs for connection errors**
   ```
   grep "Redis unavailable" app.log
   ```

3. **Check `TEST_REDIS_URL` or `REDIS_URL` env var** — is it set correctly?

4. **`RedisMetricAbsent` only** — this may fire if the app has not yet connected (cold start).
   Wait 5 minutes before escalating.

---

## Remediation

| Cause | Action |
|---|---|
| Redis container crashed | `docker restart gv_redis` or equivalent |
| Redis OOM | Check `used_memory` via `redis-cli INFO memory`; flush non-critical keys or increase memory limit |
| Network partition | Check VPC/firewall rules between app and Redis host |
| Wrong REDIS_URL | Update the env var and restart the app |
| Single-instance deploy | Disable Redis (unset `REDIS_URL`) — app falls back automatically |

---

## Recovery

After Redis is restored, the app will reconnect automatically on the next Redis call.
The `gv_redis_connected` gauge will return to 1 within the next scrape interval.

The leader lock will be re-acquired within `redis_leader_lock_ttl_s` seconds (default: 30).

---

## Note on single-process deployments

If `REDIS_URL` is not configured, `gv_redis_connected` will never be set and `RedisMetricAbsent`
will fire permanently. Silence this alert or remove it in single-process deployments.
