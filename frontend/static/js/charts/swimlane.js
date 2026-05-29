/**
 * swimlane.js — ECharts live NEM price chart with regime band colouring.
 *
 * Renders a timeseries of dispatch prices with:
 *  - Coloured area fill by regime (normal/elevated/spike/extreme)
 *  - Dashed forecast tail (P10/P50/P90 bands) when LNN data available
 *  - 30-interval rolling window (2.5 hours of 5-min dispatch data)
 *  - Regime threshold reference lines
 *  - Tooltip showing price, demand, regime at each interval
 *
 * Usage:
 *   import { SwimLaneChart } from './charts/swimlane.js';
 *   const chart = new SwimLaneChart('chart-price');
 *   chart.push({ time: '14:30', price: 247.5, demand: 8420, regime: 'spike' });
 *   chart.setForecast([...]);  // optional LNN forecast bands
 */

const REGIME_COLOURS = {
  normal:   '#22c55e',
  elevated: '#f59e0b',
  spike:    '#ef4444',
  extreme:  '#ff0033',
};

const MODEL_COLOURS = {
  qra: '#f59e0b',
  lear: '#38bdf8',
  experimental_lnn: '#a78bfa',
};

const REGIME_THRESHOLDS = {
  NSW1: { elevated: 100, spike: 300, extreme: 1000 },
  VIC1: { elevated: 100, spike: 300, extreme: 1000 },
  QLD1: { elevated: 100, spike: 300, extreme: 1000 },
  SA1:  { elevated: 120, spike: 400, extreme: 1500 },
  TAS1: { elevated:  90, spike: 250, extreme:  800 },
};

const MAX_POINTS = 48;   // 4 hours of history — matches the 48-interval (4h) forecast horizon

export class SwimLaneChart {
  constructor(domId) {
    this._dom = document.getElementById(domId);
    this._data = [];        // { time, price, demand, regime }[]
    this._forecast = null;  // { times, p10, p50, p90 } | null
    this._region = 'NSW1';
    if (!this._dom) return;
    this._chart = window.echarts.init(this._dom, 'dark', { renderer: 'canvas' });
    this._render();
    window.addEventListener('resize', () => this._chart?.resize());
  }

  setRegion(region) {
    this._region = region;
    this._render();
  }

  push(point) {
    // point: { time: string (HH:MM), price: number, demand: number, regime: string }
    this._data.push(point);
    if (this._data.length > MAX_POINTS) this._data.shift();
    this._render();
  }

  setHistory(points) {
    this._data = (points || []).map(p => ({
      time: _formatTime(p.time),
      price: Number(p.price_rrp ?? p.price ?? 0),
      demand: Number(p.demand_mw ?? p.demand ?? 0),
      regime: p.regime || 'normal',
      samples: p.samples || 1,
    })).filter(p => Number.isFinite(p.price));
    if (this._data.length > MAX_POINTS && this._data[0]?.samples === 1) {
      this._data = this._data.slice(-MAX_POINTS);
    }
    this._render();
  }

  setForecast(forecast) {
    // forecast: null | { times: string[], p10: number[], p50: number[], p90: number[] }
    this._forecast = forecast;
    // Extract spike risk series from primary model's spike_probs_series
    this._spikeRisk = null;
    if (forecast?.forecasts) {
      const primary = forecast.primary_model;
      const fc = forecast.forecasts.find(f => f.model === primary) || forecast.forecasts[0];
      if (fc?.spike_probs_series?.gt_300) {
        this._spikeRisk = {
          times: fc.target_times || [],
          gt_300: fc.spike_probs_series.gt_300,
          gt_1000: fc.spike_probs_series.gt_1000 || [],
          lt_0: fc.spike_probs_series.lt_0 || [],
        };
      }
    }
    this._render();
  }

  setSpikeRisk(riskData) {
    // riskData: null | { times: string[], gt_300: number[], gt_1000: number[], lt_0: number[] }
    this._spikeRisk = riskData;
    this._render();
  }

  clear() {
    this._data = [];
    this._forecast = null;
    this._render();
  }

  _render() {
    if (!this._chart) return;

    const thresholds = REGIME_THRESHOLDS[this._region] || REGIME_THRESHOLDS.NSW1;
    const times = this._data.map(d => d.time);
    const prices = this._data.map(d => d.price);
    const regimes = this._data.map(d => d.regime || 'normal');

    // Build per-segment coloured price line using markArea pieces
    const markAreas = this._buildRegimeAreas(regimes, times);

    const forecastTimes = this._forecastTimes();
    const allTimes = [...times];
    for (const t of forecastTimes) {
      if (!allTimes.includes(t)) allTimes.push(t);
    }

    // Forecast series
    const forecastSeries = this._buildForecastSeries();

    const option = {
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 60, right: 24, top: 16, bottom: 32 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1a1d27',
        borderColor: '#2a2f45',
        textStyle: { color: '#e4e8f0', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' },
        formatter: (params) => {
          const p = params.find(s => s.seriesName === 'Price' && s.value !== null && s.value !== undefined);
          const d = params.find(s => s.seriesName === 'Demand');
          if (!p) {
            const rows = params
              .filter(s => s.value !== null && s.value !== undefined)
              .map(s => {
                const value = Array.isArray(s.value) ? s.value[1] : s.value;
                return `${s.marker || ''} ${s.seriesName}: <b>$${Number(value).toFixed(2)}/MWh</b>`;
              });
            return rows.length ? `<div style="line-height:1.8"><b>${params[0]?.axisValue || ''}</b><br/>${rows.join('<br/>')}</div>` : '';
          }
          const pt = this._data[p.dataIndex] || {};
          const regime = (pt.regime || 'normal').toUpperCase();
          const col = REGIME_COLOURS[pt.regime || 'normal'];
          return `<div style="line-height:1.8">
            <b>${p.axisValue}</b><br/>
            <span style="color:${col}">●</span> ${regime}<br/>
            Price: <b>$${Number(p.value).toFixed(2)}/MWh</b><br/>
            ${d ? `Demand: ${Number(d.value).toFixed(0)} MW` : ''}
          </div>`;
        },
      },
      xAxis: {
        type: 'category',
          data: allTimes,
        axisLabel: {
          color: '#555e78', fontSize: 10,
          interval: Math.max(0, Math.floor(times.length / 6) - 1),
        },
        axisLine: { lineStyle: { color: '#2a2f45' } },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: 'value',
          name: '$/MWh',
          // NOTE: yAxisIndex 2 = risk % added below dynamically
          nameTextStyle: { color: '#555e78', fontSize: 10 },
          axisLabel: { color: '#555e78', fontSize: 10, formatter: v => `$${v}` },
          axisLine: { show: false },
          splitLine: { lineStyle: { color: '#2a2f45', type: 'dashed' } },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              { yAxis: thresholds.elevated, lineStyle: { color: '#f59e0b', type: 'dashed', width: 1 }, label: { formatter: 'Elevated', color: '#f59e0b', fontSize: 10 } },
              { yAxis: thresholds.spike,    lineStyle: { color: '#ef4444', type: 'dashed', width: 1 }, label: { formatter: 'Spike',    color: '#ef4444', fontSize: 10 } },
              { yAxis: thresholds.extreme,  lineStyle: { color: '#ff0033', type: 'dashed', width: 1 }, label: { formatter: 'Extreme',  color: '#ff0033', fontSize: 10 } },
            ],
          },
        },
        {
          type: 'value',
          name: 'MW',
          nameTextStyle: { color: '#555e78', fontSize: 10 },
          axisLabel: { color: '#555e78', fontSize: 10, formatter: v => `${v / 1000}k` },
          axisLine: { show: false },
          splitLine: { show: false },
        },
        {
          type: 'value',
          name: 'Risk%',
          min: 0,
          max: 100,
          position: 'right',
          offset: 40,
          nameTextStyle: { color: '#f59e0b', fontSize: 9 },
          axisLabel: { color: '#f59e0b', fontSize: 9, formatter: v => `${v}%` },
          axisLine: { show: false },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: 'Price',
          type: 'line',
          data: prices.concat(Array(Math.max(0, allTimes.length - prices.length)).fill(null)),
          smooth: false,
          symbol: 'none',
          lineStyle: { width: 2, color: this._priceLineColour(regimes) },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(99,102,241,0.25)' },
                { offset: 1, color: 'rgba(99,102,241,0.02)' },
              ],
            },
          },
          markArea: { silent: true, itemStyle: { opacity: 0.08 }, data: markAreas },
          z: 3,
        },
        {
          name: 'Demand',
          type: 'line',
          yAxisIndex: 1,
          data: this._data.map(d => d.demand).concat(Array(Math.max(0, allTimes.length - prices.length)).fill(null)),
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 1, color: '#6366f1', type: 'dotted', opacity: 0.6 },
          z: 2,
        },
        ...forecastSeries,
      ],
    };

    this._chart.setOption(option, { notMerge: true });
  }

  _priceLineColour(regimes) {
    const last = regimes[regimes.length - 1] || 'normal';
    return REGIME_COLOURS[last] || REGIME_COLOURS.normal;
  }

  _buildRegimeAreas(regimes, times) {
    const areas = [];
    if (!regimes.length) return areas;

    let start = 0;
    let curRegime = regimes[0];

    for (let i = 1; i <= regimes.length; i++) {
      if (i === regimes.length || regimes[i] !== curRegime) {
        if (curRegime !== 'normal') {
          areas.push([
            { xAxis: times[start], itemStyle: { color: REGIME_COLOURS[curRegime] || '#555' } },
            { xAxis: times[i - 1] || times[times.length - 1] },
          ]);
        }
        if (i < regimes.length) {
          curRegime = regimes[i];
          start = i;
        }
      }
    }
    return areas;
  }

  _buildForecastSeries() {
    if (!this._forecast) return [];
    const models = Array.isArray(this._forecast.forecasts)
      ? this._forecast.forecasts
      : [{
          model: this._forecast.model || 'forecast',
          target_times: this._forecast.times,
          p10: this._forecast.p10,
          p50: this._forecast.p50,
          p90: this._forecast.p90,
        }];
    const primary = this._forecast.primary_model || models[0]?.model;
    const series = [];
    for (const m of models) {
      const times = (m.target_times || m.times || []).map(_formatTime);
      const colour = MODEL_COLOURS[m.model] || '#818cf8';
      if (m.model === primary && m.p90?.length) {
        series.push({
          name: `${m.model} P90`,
          type: 'line',
          data: times.map((t, i) => [t, m.p90[i]]),
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 0 },
          areaStyle: { color: 'rgba(245,158,11,0.12)', origin: 'auto' },
          z: 1,
        });
        series.push({
          name: `${m.model} P10`,
          type: 'line',
          data: times.map((t, i) => [t, m.p10[i]]),
          smooth: true,
          symbol: 'none',
          lineStyle: { width: 1, color: colour, type: 'dotted', opacity: 0.55 },
          z: 2,
        });
      }
      if (m.p50?.length) {
        series.push({
          name: `${m.model} P50`,
          type: 'line',
          data: times.map((t, i) => [t, m.p50[i]]),
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { width: m.model === primary ? 2 : 1, color: colour, type: 'dashed' },
          z: 5,
        });
      }
    }
    // Spike risk overlay — secondary axis (right, 0–100%)
    if (this._spikeRisk?.gt_300?.length) {
      const riskTimes = (this._spikeRisk.times || []).map(_formatTime);
      series.push({
        name: 'P(>$300)',
        type: 'line',
        yAxisIndex: 2,
        data: riskTimes.map((t, i) => [t, Math.round((this._spikeRisk.gt_300[i] || 0) * 100)]),
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: '#f59e0b', type: 'dotted' },
        z: 6,
      });
    }

    return series;
  }

  _forecastTimes() {
    if (!this._forecast) return [];
    const models = Array.isArray(this._forecast.forecasts)
      ? this._forecast.forecasts
      : [{ target_times: this._forecast.times }];
    const seen = new Set();
    const times = [];
    for (const m of models) {
      for (const raw of (m.target_times || m.times || [])) {
        const t = _formatTime(raw);
        if (t && !seen.has(t)) {
          seen.add(t);
          times.push(t);
        }
      }
    }
    return times;
  }

  dispose() {
    this._chart?.dispose();
  }
}

// Singleton chart instance managed by state.js
let _instance = null;

export function getSwimLaneChart(domId = 'chart-price') {
  if (!_instance) _instance = new SwimLaneChart(domId);
  return _instance;
}

export function destroySwimLaneChart() {
  _instance?.dispose();
  _instance = null;
}

function _formatTime(value) {
  if (!value) return '';
  try {
    return new Date(value).toLocaleTimeString('en-AU', {
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'Australia/Sydney',
    });
  } catch {
    return String(value);
  }
}
