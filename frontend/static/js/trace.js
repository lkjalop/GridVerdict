/**
 * Trace replay panel — render a bitemporal decision trace.
 *
 * Called when viewport_type === "trace_replay" or when the user clicks
 * "View trace" on a verdict card.
 *
 * Renders:
 *   - Timeline: valid_time (data truth) vs system_time (when GridVerdict knew)
 *   - Decomposition: intent, entities, confidence
 *   - Tool calls: what data sources were queried
 *   - Answer: the verdict that was produced
 *   - Source manifest: coverage summary
 */

'use strict';

/**
 * Fetch a trace from the API and render it into the given container.
 *
 * @param {string} traceId
 * @param {HTMLElement} container - DOM element to render into
 */
export async function renderTrace(traceId, container) {
  if (!traceId || !container) return;

  container.innerHTML = '<div class="trace-loading">Loading trace…</div>';

  let trace;
  try {
    const resp = await fetch(`/api/traces/${traceId}`, {
      headers: { 'Accept': 'application/json' },
    });
    if (!resp.ok) {
      throw new Error(`${resp.status} ${resp.statusText}`);
    }
    trace = await resp.json();
  } catch (err) {
    container.innerHTML = `<div class="trace-error">Failed to load trace: ${err.message}</div>`;
    return;
  }

  container.innerHTML = _buildTraceHTML(trace);
}

/**
 * Render trace data already fetched (embedded in a query response).
 *
 * @param {object} traceData - pre-fetched trace dict
 * @param {HTMLElement} container
 */
export function renderTraceFromData(traceData, container) {
  if (!traceData || !container) return;
  container.innerHTML = _buildTraceHTML(traceData);
}

// ── HTML builders ─────────────────────────────────────────────────────

function _buildTraceHTML(t) {
  const validTime = _fmtTime(t.valid_time);
  const systemTime = _fmtTime(t.system_time);
  const lagSecs = _lagSeconds(t.valid_time, t.system_time);
  const lagLabel = lagSecs !== null ? `${lagSecs}s lag` : '';

  const decomp = t.decomposition || {};
  const answer = t.answer || {};
  const toolCalls = t.tool_calls || [];
  const manifest = t.source_manifest || {};

  return `
<div class="trace-panel">

  <!-- Bitemporal header -->
  <div class="trace-bitemp">
    <div class="trace-bitemp-row">
      <span class="trace-label">Data valid at</span>
      <span class="trace-value mono">${validTime}</span>
    </div>
    <div class="trace-bitemp-row">
      <span class="trace-label">GridVerdict knew at</span>
      <span class="trace-value mono">${systemTime}</span>
    </div>
    ${lagLabel ? `<div class="trace-bitemp-row trace-lag">
      <span class="trace-label">Bitemporal lag</span>
      <span class="trace-value trace-lag-value">${lagLabel}</span>
    </div>` : ''}
  </div>

  <!-- Decomposition -->
  <div class="trace-section">
    <div class="trace-section-title">Query decomposition</div>
    <div class="trace-kv">
      <span class="trace-key">Intent</span>
      <span class="trace-val intent-badge intent-${decomp.intent || 'unknown'}">${decomp.intent || '—'}</span>
    </div>
    ${decomp.entities?.regions?.length ? `<div class="trace-kv">
      <span class="trace-key">Regions</span>
      <span class="trace-val">${decomp.entities.regions.join(', ')}</span>
    </div>` : ''}
    <div class="trace-kv">
      <span class="trace-key">Decomp confidence</span>
      <span class="trace-val">${_pct(decomp.confidence)}</span>
    </div>
    ${decomp.ambiguities?.length ? `<div class="trace-kv">
      <span class="trace-key">Ambiguities</span>
      <span class="trace-val trace-warn">${decomp.ambiguities.join('; ')}</span>
    </div>` : ''}
  </div>

  <!-- Tool calls / data sources -->
  ${toolCalls.length ? `<div class="trace-section">
    <div class="trace-section-title">Data sources used</div>
    ${toolCalls.map(tc => `
    <div class="trace-tool-call">
      <span class="trace-source-badge">${tc.source || 'unknown'}</span>
      ${tc.price_rrp != null ? `<span class="trace-price">$${Number(tc.price_rrp).toFixed(2)}/MWh</span>` : ''}
      ${tc.demand_mw != null ? `<span class="trace-demand">${Number(tc.demand_mw).toFixed(0)} MW demand</span>` : ''}
    </div>`).join('')}
  </div>` : ''}

  <!-- Source manifest -->
  <div class="trace-section">
    <div class="trace-section-title">Coverage</div>
    <div class="trace-kv">
      <span class="trace-key">Evidence refs</span>
      <span class="trace-val">${manifest.evidence_count ?? 0}</span>
    </div>
    <div class="trace-kv">
      <span class="trace-key">Sources</span>
      <span class="trace-val">${(manifest.sources || []).join(', ') || '—'}</span>
    </div>
  </div>

  <!-- Final answer summary -->
  <div class="trace-section">
    <div class="trace-section-title">Verdict</div>
    <div class="trace-kv">
      <span class="trace-key">Label</span>
      <span class="trace-val verdict-badge verdict-${(answer.verdict || 'unknown').toLowerCase()}">${answer.verdict || '—'}</span>
    </div>
    <div class="trace-kv">
      <span class="trace-key">Action</span>
      <span class="trace-val">${answer.action || '—'}</span>
    </div>
    <div class="trace-kv">
      <span class="trace-key">Confidence</span>
      <span class="trace-val">${_pct(answer.confidence)} <em class="trace-band">(${answer.confidence_band || '—'})</em></span>
    </div>
    ${answer.missing_data?.length ? `<div class="trace-kv">
      <span class="trace-key">Missing data</span>
      <span class="trace-val trace-warn">${answer.missing_data.join(', ')}</span>
    </div>` : ''}
  </div>

  <div class="trace-id">Trace ID: <span class="mono">${t.id}</span></div>
</div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────

function _fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('en-AU', {
      timeZone: 'Australia/Sydney',
      dateStyle: 'short',
      timeStyle: 'medium',
    }) + ' AEST';
  } catch {
    return iso;
  }
}

function _lagSeconds(validIso, systemIso) {
  try {
    const lag = Math.round((new Date(systemIso) - new Date(validIso)) / 1000);
    return isNaN(lag) ? null : lag;
  } catch {
    return null;
  }
}

function _pct(v) {
  if (v == null) return '—';
  return `${Math.round(Number(v) * 100)}%`;
}
