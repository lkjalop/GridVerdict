/**
 * main.js — Entry point. Wires all modules and registers Alpine components.
 *
 * Load order matters:
 *  1. Alpine is loaded as a deferred script in app.html
 *  2. This module runs as type="module" before Alpine initialises
 *  3. We expose gridverdictApp on window so Alpine's x-data can find it
 *  4. After Alpine is ready, we initialise charts
 */

import { gridverdictApp } from './state.js';
import { api } from './api.js';
import { getSwimLaneChart, destroySwimLaneChart } from './charts/swimlane.js';
import { renderTrace, renderTraceFromData } from './trace.js';
import { renderEvidenceDrawer, renderFreshnessBanner } from './evidence.js';
import { renderBacktestPanel, submitBacktest, pollJob, renderBacktestResult } from './backtest.js';
import { renderHistoryPanel } from './history.js';
import { renderSecurityPanel, stopSecurityPoll, renderObserverBadge } from './security.js';
import { getAnalogChart, destroyAnalogChart } from './charts/analog.js';
import { getBacktestChart, destroyBacktestChart } from './charts/backtest_pnl.js';
import { renderFuelMix, destroyFuelMixChart } from './charts/fuel_mix.js';

// ── Expose to Alpine ──────────────────────────────────────────────────
window.gridverdictApp = gridverdictApp;
window.gvApi = api;
window.gvTrace = { renderTrace, renderTraceFromData };
window.gvEvidence = { renderEvidenceDrawer, renderFreshnessBanner };
window.gvBacktest = { renderBacktestPanel, submitBacktest, pollJob, renderBacktestResult };
window.gvHistory = { renderHistoryPanel };
window.gvSecurity = { renderSecurityPanel, stopSecurityPoll, renderObserverBadge };

// ── Chart bridge — called from Alpine's x-init and market refresh ─────
window.gvCharts = {
  swimlane: null,
  analog: { getAnalogChart, destroyAnalogChart },
  backtestPnl: { getBacktestChart, destroyBacktestChart },
  fuelMixChart: { renderFuelMix, destroyFuelMixChart },

  renderFuelMix(data) {
    renderFuelMix(data, 'fuel-mix-chart');
  },

  initSwimlane(region = 'NSW1') {
    if (this.swimlane) {
      this.swimlane.setRegion(region);
      return;
    }
    this.swimlane = getSwimLaneChart('chart-price');
    this.swimlane.setRegion(region);
  },

  /**
   * Push a new dispatch interval into the live chart.
   * Called after every market refresh.
   */
  pushDispatch(point) {
    // point: { time, price, demand, regime }
    if (!this.swimlane) this.initSwimlane();
    this.swimlane?.push(point);
  },

  /**
   * Update forecast bands from LNN output (Week 4).
   */
  setForecast(forecast) {
    this.swimlane?.setForecast(forecast);
  },

  setHistory(points) {
    if (!this.swimlane) this.initSwimlane();
    this.swimlane?.setHistory(points);
  },

  changeRegion(region) {
    this.swimlane?.setRegion(region);
  },

  destroy() {
    destroySwimLaneChart();
    this.swimlane = null;
  },
};

// ── Patch state.js refreshMarket to also push to chart ───────────────
// We monkey-patch after Alpine is ready so we don't depend on import order.
document.addEventListener('alpine:init', () => {
  // Alpine is initialising — components are registered
  // Chart init happens lazily when the market panel is first shown
});

document.addEventListener('alpine:initialized', () => {
  // DOM is live; initialise chart on the market panel if it's visible
  const chartEl = document.getElementById('chart-price');
  if (chartEl && chartEl.offsetParent !== null) {
    window.gvCharts.initSwimlane();
  }

  // Intercept region changes to update chart
  const regionSelect = document.querySelector('select[x-model="region"]');
  if (regionSelect) {
    regionSelect.addEventListener('change', (e) => {
      window.gvCharts.changeRegion(e.target.value);
    });
  }
});

// ── Market snapshot → chart bridge ────────────────────────────────────
// state.js calls window.gvCharts.pushDispatch after refreshMarket succeeds.
// We hook this by wrapping the fetch response in state.js via a custom event.
window.addEventListener('gv:market-updated', (e) => {
  const { region, price_rrp, demand_mw, regime, valid_time } = e.detail;
  const time = new Date(valid_time).toLocaleTimeString('en-AU', {
    hour: '2-digit', minute: '2-digit', timeZone: 'Australia/Sydney',
  });
  window.gvCharts.pushDispatch({ time, price: price_rrp, demand: demand_mw, regime });
});

// ── Keyboard shortcuts ────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + K — focus query input
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    document.getElementById('query-input')?.focus();
  }
  // Escape — blur input
  if (e.key === 'Escape') {
    document.getElementById('query-input')?.blur();
  }
});
