/**
 * history.js — Query history panel.
 *
 * Renders a paginated list of past queries for the current session,
 * with the ability to click a query to replay its trace.
 *
 * Displayed when viewport_type === "retrospective" or when the user
 * navigates to the History tab.
 *
 * API:
 *   GET /api/sessions/{session_id}/queries  →  QueryListResponse
 *   GET /api/traces/{trace_id}              →  Trace (on click)
 */

'use strict';

/**
 * Render the query history panel for a session.
 *
 * @param {string}      sessionId
 * @param {HTMLElement} container
 * @param {object}      [options]
 * @param {number}      [options.limit=20]
 * @param {function}    [options.onTraceClick]  callback(traceId) when user clicks a row
 */
export async function renderHistoryPanel(sessionId, container, options = {}) {
  if (!sessionId || !container) return;

  const limit = options.limit ?? 20;
  container.innerHTML = '<div class="history-loading">Loading history…</div>';

  let data;
  try {
    const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/queries?limit=${limit}`, {
      headers: { Accept: 'application/json' },
    });
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    data = await resp.json();
  } catch (err) {
    container.innerHTML = `<div class="history-error">Failed to load history: ${_esc(err.message)}</div>`;
    return;
  }

  const queries = Array.isArray(data) ? data : (data.items ?? []);
  container.innerHTML = _buildHistoryHTML(queries, options.onTraceClick);
}

// ── HTML builders ─────────────────────────────────────────────────────────

function _buildHistoryHTML(queries, onTraceClick) {
  if (!queries.length) {
    return `<div class="history-panel">
      <div class="history-empty">No queries in this session yet.</div>
    </div>`;
  }

  const rows = queries.map((q, i) => {
    const ts = _fmtTime(q.created_at);
    const verdict = (q.verdict || '').toLowerCase();
    const intent = (q.intent || '').toLowerCase().replace('_', ' ');
    const hasTrace = !!q.trace_id;

    return `
<div class="history-row" data-trace-id="${_esc(q.trace_id || '')}" role="button" tabindex="0">
  <div class="history-row-left">
    <span class="history-seq">#${queries.length - i}</span>
    <span class="history-query-text" title="${_esc(q.raw_query)}">${_esc(_truncate(q.raw_query, 80))}</span>
  </div>
  <div class="history-row-right">
    <span class="intent-badge intent-${intent}">${intent}</span>
    <span class="verdict-badge verdict-${verdict}">${(q.verdict || '—').toUpperCase()}</span>
    <span class="history-time">${ts}</span>
    ${hasTrace ? '<span class="history-trace-link" title="View decision trace">trace</span>' : ''}
  </div>
</div>`;
  }).join('');

  return `<div class="history-panel">
  <div class="history-header">
    <span class="history-title">Query history</span>
    <span class="history-count">${queries.length} queries</span>
  </div>
  <div class="history-list">${rows}</div>
</div>`;
}

// ── Event delegation — click on a history row to view the trace ───────────

document.addEventListener('click', (e) => {
  const row = e.target.closest('.history-row[data-trace-id]');
  if (!row) return;
  const traceId = row.dataset.traceId;
  if (!traceId) return;

  // Dispatch a custom event; state.js / Alpine can listen for it
  document.dispatchEvent(new CustomEvent('gv:view-trace', { detail: { traceId } }));
});

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const row = e.target.closest('.history-row[data-trace-id]');
  if (!row) return;
  const traceId = row.dataset.traceId;
  if (traceId) document.dispatchEvent(new CustomEvent('gv:view-trace', { detail: { traceId } }));
});

// ── Helpers ───────────────────────────────────────────────────────────────

function _fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('en-AU', {
      timeZone: 'Australia/Sydney',
      dateStyle: 'short',
      timeStyle: 'short',
    });
  } catch {
    return iso;
  }
}

function _truncate(str, maxLen) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
}

function _esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
