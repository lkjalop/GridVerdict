/**
 * backtest.js — Backtest panel: submit jobs, poll results, render report.
 *
 * Rendered when viewport_type === "backtest" or the user navigates to
 * the Backtest tab.  Alpine state is owned by state.js; this module
 * exports pure functions that Alpine calls.
 *
 * Panel layout:
 *   1. Config form (region, lookback_days, horizon_intervals)
 *   2. Job status + spinner while running
 *   3. Results table (CRPS, pinball, spike F1, calibration, skill vs AEMO)
 *   4. Empty state message before first run
 */

'use strict';

// ── Public API ────────────────────────────��───────────────────────────────

/**
 * Render the backtest panel into a container element.
 * Call once; subsequent updates go through refreshJobStatus().
 *
 * @param {HTMLElement} container
 * @param {object} state - Alpine state proxy (for two-way binding)
 */
export function renderBacktestPanel(container, state) {
  if (!container) return;
  container.innerHTML = _buildPanelHTML(state);
}

/**
 * Submit a backtest job and return the job_id.
 *
 * @param {object} config  { region, lookback_days, horizon_intervals, include_lnn }
 * @returns {Promise<string>} job_id
 */
export async function submitBacktest(config) {
  const resp = await fetch('/api/backtest/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  const job = await resp.json();
  return job.job_id;
}

/**
 * Poll a job until it reaches done or error.
 * Calls onUpdate(job) on each poll.
 *
 * @param {string} jobId
 * @param {function} onUpdate  callback(job)
 * @param {number}  [intervalMs=2000]
 */
export function pollJob(jobId, onUpdate, intervalMs = 2000) {
  const handle = setInterval(async () => {
    try {
      const resp = await fetch(`/api/backtest/${jobId}`);
      if (!resp.ok) return;
      const job = await resp.json();
      onUpdate(job);
      if (job.status === 'done' || job.status === 'error') {
        clearInterval(handle);
      }
    } catch {
      // transient network error — keep polling
    }
  }, intervalMs);
  return handle;   // caller can clearInterval() to cancel
}

/**
 * Render a completed BacktestReport result dict into a container.
 *
 * @param {object} result  BacktestReport.to_dict() from the API
 * @param {HTMLElement} container
 */
export function renderBacktestResult(result, container) {
  if (!result || !container) return;
  container.innerHTML = _buildResultHTML(result);
}

// ── HTML builders ───────────────────────────────────────���─────────────────

function _buildPanelHTML(state) {
  const regions = ['NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1'];
  const regionOptions = regions
    .map(r => `<option value="${r}"${r === 'NSW1' ? ' selected' : ''}>${r}</option>`)
    .join('');

  return `
<div class="backtest-panel">
  <div class="backtest-header">
    <h3 class="backtest-title">Walk-forward backtest</h3>
    <p class="backtest-subtitle">
      Rolling-origin evaluation: baselines vs LNN. No lookahead — training data
      strictly precedes each test window.
    </p>
  </div>

  <form class="backtest-form" id="backtest-form">
    <div class="backtest-form-row">
      <label class="backtest-label">Region</label>
      <select class="backtest-select" name="region">${regionOptions}</select>
    </div>
    <div class="backtest-form-row">
      <label class="backtest-label">Lookback days</label>
      <input class="backtest-input" type="number" name="lookback_days"
             value="7" min="1" max="90">
    </div>
    <div class="backtest-form-row">
      <label class="backtest-label">Forecast horizon</label>
      <select class="backtest-select" name="horizon_intervals">
        <option value="1">1 interval (5 min)</option>
        <option value="6" selected>6 intervals (30 min)</option>
        <option value="12">12 intervals (1 hr)</option>
        <option value="36">36 intervals (3 hr)</option>
      </select>
    </div>
    <div class="backtest-form-row backtest-checkbox-row">
      <label class="backtest-label">
        <input type="checkbox" name="include_lnn" checked> Include LNN model
      </label>
    </div>
    <button class="backtest-submit-btn" type="submit">Run backtest</button>
  </form>

  <div class="backtest-status" id="backtest-status"></div>
  <div class="backtest-result" id="backtest-result"></div>
</div>`;
}

function _buildResultHTML(result) {
  const scores = result.scores || [];
  if (!scores.length) return '<div class="backtest-empty">No model scores returned.</div>';

  const rows = scores
    .slice()
    .sort((a, b) => a.crps - b.crps)
    .map(s => {
      const skill = s.skill_vs_aemo;
      const skillStr = isFinite(skill)
        ? `<span class="${skill > 0 ? 'skill-positive' : 'skill-negative'}">${_fmt(skill, 3, true)}</span>`
        : '<span class="skill-na">n/a</span>';
      return `
  <tr class="backtest-row">
    <td class="bt-model">${s.model}</td>
    <td class="bt-crps">${_fmt(s.crps, 2)}</td>
    <td class="bt-pinball">${_fmt(s.pinball, 2)}</td>
    <td class="bt-spikef1">${_fmt(s.spike_f1, 3)}</td>
    <td class="bt-calib">${_fmt(s.calibration_error, 3)}</td>
    <td class="bt-skill">${skillStr}</td>
  </tr>`;
    }).join('');

  const summary = `
<div class="backtest-meta">
  ${result.horizon_min} min horizon &middot;
  ${result.n_origins} origins &middot;
  spike threshold $${result.spike_threshold}/MWh
</div>`;

  return `
<div class="backtest-results">
  ${summary}
  <table class="backtest-table">
    <thead>
      <tr>
        <th>Model</th>
        <th title="Continuous Ranked Probability Score — lower is better">CRPS ↓</th>
        <th title="Mean pinball loss across quantiles">Pinball ↓</th>
        <th title="F1 score on spike detection (price above threshold)">Spike F1 ↑</th>
        <th title="Calibration error — how often the true value falls inside the predicted interval">Calib err ↓</th>
        <th title="Skill score vs AEMO pre-dispatch (positive = beats AEMO)">vs AEMO</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
</div>`;
}

function _buildStatusHTML(job) {
  if (job.status === 'running' || job.status === 'pending') {
    return `<div class="backtest-running">
      <span class="spinner"></span> Running backtest for ${job.region}…
    </div>`;
  }
  if (job.status === 'error') {
    return `<div class="backtest-error">Backtest failed: ${_escapeHtml(job.error || 'unknown error')}</div>`;
  }
  return '';  // done — result will be shown by renderBacktestResult
}

// ── Helpers ─────────────────────────��─────────────────────────────────────

function _fmt(v, decimals, signed = false) {
  if (v == null || !isFinite(v)) return '—';
  const s = Number(v).toFixed(decimals);
  return signed && v > 0 ? `+${s}` : s;
}

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Form wiring — auto-attaches when the panel is injected into the DOM ──
document.addEventListener('click', (e) => {
  const btn = e.target.closest('#backtest-form button[type="submit"]');
  if (!btn) return;
  e.preventDefault();

  const form = document.getElementById('backtest-form');
  if (!form) return;

  const fd = new FormData(form);
  const config = {
    region: fd.get('region') || 'NSW1',
    lookback_days: parseInt(fd.get('lookback_days') || '7', 10),
    horizon_intervals: parseInt(fd.get('horizon_intervals') || '6', 10),
    include_lnn: form.querySelector('[name="include_lnn"]')?.checked ?? true,
  };

  const statusEl = document.getElementById('backtest-status');
  const resultEl = document.getElementById('backtest-result');
  if (statusEl) statusEl.innerHTML = '<div class="backtest-running"><span class="spinner"></span> Submitting…</div>';
  if (resultEl) resultEl.innerHTML = '';

  submitBacktest(config).then(jobId => {
    pollJob(jobId, (job) => {
      if (statusEl) statusEl.innerHTML = _buildStatusHTML(job);
      if (job.status === 'done' && job.result && resultEl) {
        renderBacktestResult(job.result, resultEl);
      }
    });
  }).catch(err => {
    if (statusEl) statusEl.innerHTML = `<div class="backtest-error">Error: ${_escapeHtml(err.message)}</div>`;
  });
});
