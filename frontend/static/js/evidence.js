/**
 * Evidence drawer — renders evidence_refs, freshness banner, and missing data list.
 *
 * Used in:
 *   - The verdict panel (evidence citations below the verdict hero)
 *   - The why panel (inline citations per sentence)
 *   - The trace replay panel (source manifest)
 *
 * Design rules (from Factual Verdict Contract):
 *   - Every numeric claim must have a cited evidence_ref
 *   - Stale data (is_stale=true or staleness_seconds > 600) shows a warning banner
 *   - Missing data fields are listed as a checklist with explanation
 */

'use strict';

/**
 * Render the evidence drawer into a container element.
 *
 * @param {object} verdict - FactualVerdict dict from the API
 * @param {HTMLElement} container
 * @param {object} [marketState] - optional MarketStateResponse for freshness banner
 */
export function renderEvidenceDrawer(verdict, container, marketState) {
  if (!verdict || !container) return;
  container.innerHTML = _buildEvidenceHTML(verdict, marketState);
}

/**
 * Render only the freshness banner (used in the top bar).
 *
 * @param {object} marketState - MarketStateResponse
 * @param {HTMLElement} bannerEl
 */
export function renderFreshnessBanner(marketState, bannerEl) {
  if (!marketState || !bannerEl) return;
  bannerEl.innerHTML = _buildFreshnessBanner(marketState);
}

// ── HTML builders ─────────────────────────────────────────────────────

function _buildEvidenceHTML(verdict, marketState) {
  const refs = verdict.evidence_refs || [];
  const missing = verdict.missing_data || [];
  const isStale = marketState?.is_stale;
  const staleSecs = marketState?.staleness_seconds ?? 0;

  const parts = [];

  // Freshness banner
  if (isStale || staleSecs > 600) {
    parts.push(_buildFreshnessBanner(marketState));
  }

  // Evidence citations
  if (refs.length > 0) {
    parts.push(`
<div class="evidence-section">
  <div class="evidence-section-title">Evidence citations</div>
  <table class="evidence-table">
    <thead>
      <tr>
        <th>Source</th><th>Field</th><th>Value</th><th>Interval</th><th>Ref</th>
      </tr>
    </thead>
    <tbody>
      ${refs.map(ref => `
      <tr class="evidence-row">
        <td class="ev-source">${ref.source || '—'}</td>
        <td class="ev-field">${ref.field || '—'}</td>
        <td class="ev-value">${_formatValue(ref.field, ref.value)}</td>
        <td class="ev-interval">${_fmtInterval(ref.interval)}</td>
        <td class="ev-ref mono" title="${ref.raw_ref || ''}">${(ref.id || '').slice(-8)}</td>
      </tr>`).join('')}
    </tbody>
  </table>
</div>`);
  } else if (verdict.verdict !== 'SUPPORTED') {
    parts.push(`<div class="evidence-empty">No evidence references (${verdict.verdict} verdict)</div>`);
  }

  // Missing data checklist
  if (missing.length > 0) {
    const descriptions = {
      live_dispatch_price: 'Live dispatch price unavailable or stale',
      aemo_market_notice: 'No active AEMO market notices found',
      historical_analogs: 'Insufficient historical analog periods (need ≥3)',
      predispatch_forecast: 'Pre-dispatch forecast not available',
    };
    parts.push(`
<div class="evidence-section">
  <div class="evidence-section-title">Missing data</div>
  <ul class="missing-data-list">
    ${missing.map(key => `
    <li class="missing-item">
      <span class="missing-icon">⚠</span>
      <span class="missing-key">${key}</span>
      ${descriptions[key] ? `<span class="missing-desc">— ${descriptions[key]}</span>` : ''}
    </li>`).join('')}
  </ul>
</div>`);
  }

  // Counterargument
  if (verdict.counterargument) {
    parts.push(`
<div class="evidence-section evidence-counterarg">
  <div class="evidence-section-title">Adversarial critique</div>
  <p class="counterarg-text">${_escapeHtml(verdict.counterargument)}</p>
</div>`);
  }

  // Disclaimer
  if (verdict.disclaimer) {
    parts.push(`<div class="evidence-disclaimer">${_escapeHtml(verdict.disclaimer)}</div>`);
  }

  return `<div class="evidence-drawer">${parts.join('')}</div>`;
}

function _buildFreshnessBanner(marketState) {
  if (!marketState) return '';

  const secs = marketState.staleness_seconds ?? 0;
  const isStale = marketState.is_stale;

  if (!isStale && secs <= 60) return '';

  let cls, msg;
  if (secs > 600 || isStale) {
    cls = 'freshness-stale';
    msg = `Data is ${_humanDuration(secs)} old — may not reflect the latest dispatch interval.`;
  } else {
    cls = 'freshness-warn';
    msg = `Data is ${_humanDuration(secs)} old.`;
  }

  return `<div class="freshness-banner ${cls}">${msg}</div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────

function _formatValue(field, value) {
  if (value == null) return '—';
  const n = Number(value);
  if (isNaN(n)) return String(value);
  if (field === 'price_rrp') return `$${n.toFixed(2)}/MWh`;
  if (field === 'demand_mw' || field === 'availability_mw' || field === 'headroom_mw') {
    return `${n.toFixed(0)} MW`;
  }
  return n.toFixed(2);
}

function _fmtInterval(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleTimeString('en-AU', {
      timeZone: 'Australia/Sydney',
      hour: '2-digit',
      minute: '2-digit',
    }) + ' AEST';
  } catch {
    return iso;
  }
}

function _humanDuration(secs) {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  return `${Math.round(secs / 3600)}h`;
}

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
