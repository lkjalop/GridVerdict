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

const _STEP_META = {
  QUERY_RECEIVED:     { label: 'Query Received',      color: '#6366f1' },
  SECURITY_INPUT:     { label: 'Security Scan',        color: '#0ea5e9' },
  DECOMPOSE:          { label: 'Decomposition',        color: '#8b5cf6' },
  SCATTER_GATHER:     { label: 'Scatter Gather',       color: '#f59e0b' },
  EVIDENCE_ASSEMBLED: { label: 'Evidence Assembled',  color: '#10b981' },
  VERDICT:            { label: 'Verdict',              color: '#ef4444' },
  ANSWER_PLAN:        { label: 'Answer Plan',          color: '#3b82f6' },
  COVERAGE_AUDIT:     { label: 'Coverage Audit',       color: '#a855f7' },
  SECURITY_OUTPUT:    { label: 'Output Scan',          color: '#0ea5e9' },
  COMPLETE:           { label: 'Complete',             color: '#22c55e' },
};

function _buildPipelineTimeline(events) {
  if (!events || !events.length) return '';

  const rows = events.map((ev, i) => {
    const meta = _STEP_META[ev.step] || { label: ev.step, color: '#6b7280' };
    const details = _buildEventDetails(ev);
    return `
<div class="dt-event">
  <div class="dt-event-left">
    <div class="dt-event-dot" style="background:${meta.color}"></div>
    ${i < events.length - 1 ? '<div class="dt-event-line"></div>' : ''}
  </div>
  <div class="dt-event-body">
    <div class="dt-event-header">
      <span class="dt-step-badge" style="background:${meta.color}20;color:${meta.color};border:1px solid ${meta.color}40">${meta.label}</span>
      <span class="dt-event-time">+${ev.t_ms}ms</span>
    </div>
    ${details ? `<div class="dt-event-details">${details}</div>` : ''}
  </div>
</div>`;
  }).join('');

  return `<div class="dt-timeline">${rows}</div>`;
}

function _buildEventDetails(ev) {
  const parts = [];
  switch (ev.step) {
    case 'QUERY_RECEIVED':
      if (ev.region) parts.push(`region: <b>${_esc(ev.region)}</b>`);
      if (ev.text_len) parts.push(`${ev.text_len} chars`);
      break;
    case 'SECURITY_INPUT':
    case 'SECURITY_OUTPUT':
      parts.push(`result: <b style="color:${ev.result === 'clean' ? '#22c55e' : '#ef4444'}">${_esc(ev.result)}</b>`);
      if (ev.signals) parts.push(`${ev.signals} signal(s)`);
      break;
    case 'DECOMPOSE':
      if (ev.intent) parts.push(`intent: <b>${_esc(ev.intent)}</b>`);
      if (ev.regions?.length) parts.push(`regions: <b>${ev.regions.join(', ')}</b>`);
      if (ev.requested_output) parts.push(`planner: <b>${_esc(ev.requested_output)}</b>`);
      if (ev.confidence != null) parts.push(`conf: ${Math.round(ev.confidence * 100)}%`);
      if (ev.sub_questions?.length) parts.push(`sub-Qs: ${ev.sub_questions.join(', ')}`);
      if (ev.clarifying) parts.push(`<span class="dt-clarify">${_esc(ev.clarifying)}</span>`);
      break;
    case 'SCATTER_GATHER':
      if (ev.sources?.length) parts.push(ev.sources.map(s => `<span class="dt-source-chip">${_esc(s)}</span>`).join(' '));
      if (ev.dispatch_price != null) parts.push(`spot: <b>$${ev.dispatch_price}/MWh</b>`);
      if (ev.dispatch_fresh != null) parts.push(ev.dispatch_fresh ? '<span style="color:#22c55e">fresh</span>' : '<span style="color:#f59e0b">stale</span>');
      break;
    case 'EVIDENCE_ASSEMBLED':
      if (ev.temporal_docs) parts.push(`TemporalRAG: ${ev.temporal_docs} docs`);
      if (ev.fuel_sources) parts.push(`fuel sources: ${ev.fuel_sources}`);
      if (ev.hist_dist) parts.push(`hist dist: median $${ev.hist_dist.median} (n=${ev.hist_dist.n_rows})`);
      break;
    case 'VERDICT':
      if (ev.verdict) parts.push(`<b>${_esc(ev.verdict)}</b>`);
      if (ev.action) parts.push(`action: <b>${_esc(ev.action)}</b>`);
      if (ev.confidence != null) parts.push(`conf: ${Math.round(ev.confidence * 100)}%`);
      if (ev.band) parts.push(`band: ${_esc(ev.band)}`);
      break;
    case 'ANSWER_PLAN':
      if (ev.planner) parts.push(`planner: <b>${_esc(ev.planner)}</b>`);
      if (ev.claim_findings) parts.push(`${ev.claim_findings} claim finding(s)`);
      break;
    case 'COVERAGE_AUDIT':
      parts.push(ev.re_routed ? `re-routed → <b>${_esc(ev.suggested_planner)}</b>` : 'no re-route needed');
      break;
    case 'COMPLETE':
      parts.push(`total: <b>${ev.t_ms}ms</b>`);
      break;
  }
  return parts.join(' &nbsp;·&nbsp; ');
}

function _buildTraceHTML(t) {
  const validTime = _fmtTime(t.valid_time);
  const systemTime = _fmtTime(t.system_time);
  const lagSecs = _lagSeconds(t.valid_time, t.system_time);
  const lagLabel = lagSecs !== null ? `${lagSecs}s lag` : '';

  const decomp = t.decomposition || {};
  const answer = t.answer || {};
  const toolCalls = t.tool_calls || [];
  const manifest = t.source_manifest || {};
  const pipelineEvents = t.prefill?.pipeline_events || [];

  return `
<div class="trace-panel">

  <!-- Pipeline event timeline -->
  ${pipelineEvents.length ? `
  <div class="trace-section trace-section--timeline">
    <div class="trace-section-title">Decision trace</div>
    ${_buildPipelineTimeline(pipelineEvents)}
  </div>` : ''}

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
