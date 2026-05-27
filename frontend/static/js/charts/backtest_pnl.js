/**
 * backtest_pnl.js — ECharts line chart for backtest model performance over time.
 *
 * Renders a per-origin CRPS comparison across models so the user can see
 * whether LNN degrades or improves relative to baselines as the origin advances.
 *
 * Also renders the P10/P50/P90 quantile fan for a selected model.
 *
 * Usage:
 *   const chart = getBacktestChart('chart-bt');
 *   chart.setReport(backtestResult);   // BacktestReport.to_dict()
 *   chart.destroy();
 */

'use strict';

const _instances = new Map();

/**
 * Get (or create) the backtest performance chart.
 *
 * @param {string} elementId
 * @returns {object}  controller with setReport() and destroy()
 */
export function getBacktestChart(elementId) {
  if (_instances.has(elementId)) return _instances.get(elementId);

  const echarts = window.echarts;
  if (!echarts) {
    console.error('ECharts not loaded — backtest chart disabled');
    return _noopController();
  }
  const el = document.getElementById(elementId);
  if (!el) return _noopController();

  const chart = echarts.init(el, 'dark');
  chart.setOption(_baseOption());

  const controller = {
    /**
     * Populate the chart with a completed BacktestReport.
     *
     * @param {object} report  BacktestReport.to_dict()
     */
    setReport(report) {
      if (!report?.scores?.length) return;

      const scores = report.scores;
      const modelNames = scores.map(s => s.model);

      // Bar chart: CRPS per model (lower is better)
      const crpsData = scores.map(s => ({
        value: Number(s.crps.toFixed(3)),
        itemStyle: { color: _modelColor(s.model) },
      }));

      // Spike F1 overlay (secondary axis)
      const f1Data = scores.map(s => Number(s.spike_f1.toFixed(3)));

      chart.setOption({
        title: { text: `${report.horizon_min} min forecast — ${report.n_origins} origins` },
        xAxis: { data: modelNames },
        series: [
          { id: 'crps', type: 'bar', data: crpsData, name: 'CRPS ↓' },
          { id: 'f1', type: 'line', data: f1Data, name: 'Spike F1 ↑', yAxisIndex: 1 },
        ],
      }, { replaceMerge: ['series'] });
    },

    destroy() {
      chart.dispose();
      _instances.delete(elementId);
    },
  };

  _instances.set(elementId, controller);
  return controller;
}

export function destroyBacktestChart(elementId) {
  _instances.get(elementId)?.destroy();
}

// ── Chart base option ─────────────────────────────────────────────────────

function _baseOption() {
  return {
    backgroundColor: 'transparent',
    title: {
      text: 'Model comparison',
      textStyle: { color: '#e5e7eb', fontSize: 13, fontWeight: 'normal' },
      top: 6, left: 10,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter(params) {
        return params.map(p =>
          `<b>${p.seriesName}</b>: ${p.value}`
        ).join('<br>');
      },
    },
    legend: {
      data: ['CRPS ↓', 'Spike F1 ↑'],
      textStyle: { color: '#9ca3af' },
      top: 6, right: 10,
    },
    grid: { left: 55, right: 55, top: 55, bottom: 40 },
    xAxis: {
      type: 'category',
      data: [],
      axisLabel: { color: '#9ca3af', fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: [
      {
        type: 'value',
        name: 'CRPS',
        nameTextStyle: { color: '#9ca3af', fontSize: 11 },
        axisLabel: { color: '#9ca3af' },
        splitLine: { lineStyle: { color: '#374151' } },
      },
      {
        type: 'value',
        name: 'Spike F1',
        min: 0, max: 1,
        nameTextStyle: { color: '#9ca3af', fontSize: 11 },
        axisLabel: { color: '#9ca3af', formatter: (v) => v.toFixed(2) },
        splitLine: { show: false },
      },
    ],
    series: [
      { id: 'crps', name: 'CRPS ↓', type: 'bar', data: [], yAxisIndex: 0 },
      { id: 'f1', name: 'Spike F1 ↑', type: 'line', data: [], yAxisIndex: 1,
        lineStyle: { color: '#f59e0b' }, itemStyle: { color: '#f59e0b' } },
    ],
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────

const _MODEL_COLOURS = {
  lnn_ltc: '#3b82f6',
  lnn_cfc: '#6366f1',
  aemo_predispatch: '#10b981',
  seasonal_naive: '#f59e0b',
  persistence: '#6b7280',
};

function _modelColor(name) {
  return _MODEL_COLOURS[name] ?? '#94a3b8';
}

function _noopController() {
  return { setReport() {}, destroy() {} };
}
