/**
 * security.js — Security observer status panel.
 *
 * Shows the live risk score from the SecurityObserver:
 *   - Overall status (healthy / elevated / alert)
 *   - Recent signal log with pass/phase, risk score, and description
 *   - Per-query observer summary embedded in the verdict panel
 *
 * Polled every 30s (not real-time — the observer is synchronous/in-process).
 *
 * API:
 *   GET /api/security/status   → SecurityStatusResponse
 *   GET /api/security/signals  → paginated signal log
 */

'use strict';

let _pollHandle = null;

/**
 * Render the security panel and start polling for updates.
 *
 * @param {HTMLElement} container
 * @param {number}      [pollIntervalMs=30000]
 */
export function renderSecurityPanel(container, pollIntervalMs = 30_000) {
  if (!container) return;
  _loadAndRender(container);

  if (_pollHandle) clearInterval(_pollHandle);
  _pollHandle = setInterval(() => _loadAndRender(container), pollIntervalMs);
}

/**
 * Stop the background poll (call when panel is unmounted).
 */
export function stopSecurityPoll() {
  if (_pollHandle) {
    clearInterval(_pollHandle);
    _pollHandle = null;
  }
}

/**
 * Render a compact per-query observer summary (used in the verdict panel).
 *
 * @param {object}      observerResult  ObserverResult.to_dict() from the API
 * @param {HTMLElement} container
 */
export function renderObserverBadge(observerResult, container) {
  if (!observerResult || !container) return;
  container.innerHTML = _buildBadgeHTML(observerResult);
}

// ── Loaders ───────────────────────────────────────────────────────────────

async function _loadAndRender(container) {
  try {
    const [statusResp, signalsResp] = await Promise.all([
      fetch('/api/security/status', { headers: { Accept: 'application/json' } }),
      fetch('/api/security/signals?limit=30', { headers: { Accept: 'application/json' } }),
    ]);

    const statusData = statusResp.ok ? await statusResp.json() : null;
    const signalsData = signalsResp.ok ? await signalsResp.json() : null;

    container.innerHTML = _buildPanelHTML(statusData, signalsData?.items ?? []);
  } catch {
    // Non-fatal — panel stays as-is on transient errors
  }
}

// ── HTML builders ─────────────────────────────────────────────────────────

function _buildPanelHTML(status, signals) {
  const st = status?.status ?? 'unknown';
  const cls = { healthy: 'sec-healthy', elevated: 'sec-elevated', alert: 'sec-alert' }[st] ?? 'sec-unknown';

  const statusRow = `
<div class="sec-status-row ${cls}">
  <span class="sec-status-icon">${st === 'healthy' ? '✔' : st === 'elevated' ? '⚠' : '✖'}</span>
  <span class="sec-status-label">Observer: ${st.toUpperCase()}</span>
  <span class="sec-signal-counts">
    ${status?.recent_halts ?? 0} halts · ${status?.recent_warns ?? 0} warns (last 50)
  </span>
</div>`;

  const signalRows = signals.length
    ? signals.map(s => `
<div class="sec-signal-row sec-verdict-${s.verdict}">
  <span class="sec-phase">${_esc(s.phase)}</span>
  <span class="sec-score">risk ${s.risk_score}</span>
  <span class="sec-band sec-band-${s.risk_band}">${s.risk_band}</span>
  <span class="sec-desc">${_esc((s.signals || []).join('; ') || '—')}</span>
  <span class="sec-time">${_fmtTime(s.recorded_at)}</span>
</div>`).join('')
    : '<div class="sec-empty">No signals recorded yet.</div>';

  return `
<div class="sec-panel">
  <div class="sec-header">
    <h4 class="sec-title">Security observer</h4>
    <span class="sec-total">${status?.total_signals ?? 0} signals total</span>
  </div>
  ${statusRow}
  <div class="sec-log-header">Recent signals</div>
  <div class="sec-log">${signalRows}</div>
</div>`;
}

function _buildBadgeHTML(result) {
  const verdict = result.verdict || 'pass';
  const cls = { halt: 'obs-halt', warn: 'obs-warn', pass: 'obs-pass' }[verdict] ?? 'obs-pass';
  const icon = { halt: '✖', warn: '⚠', pass: '✔' }[verdict] ?? '?';
  const sigs = (result.signals || []).map(s => s.description || s).join('; ');

  return `
<div class="observer-badge ${cls}" title="${_esc(sigs)}">
  <span class="obs-icon">${icon}</span>
  <span class="obs-phase">${_esc(result.phase || '')}</span>
  <span class="obs-score">risk ${result.risk_score ?? 0}</span>
  ${sigs ? `<span class="obs-sigs">${_esc(sigs)}</span>` : ''}
</div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _fmtTime(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleTimeString('en-AU', {
      timeZone: 'Australia/Sydney',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return iso;
  }
}

function _esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
