/**
 * analog.js — ECharts scatter chart showing historical analog periods.
 *
 * Renders HippoGraph PPR analog results as a scatter plot:
 *   X axis: valid_time (chronological)
 *   Y axis: price_rrp ($/MWh)
 *   Colour: ppr_score (cold → warm gradient)
 *   Size:   proportional to ppr_score
 *
 * The current query point is highlighted as a distinct marker.
 *
 * Usage:
 *   const chart = getAnalogChart('container-id');
 *   chart.setData(analogs, currentPoint);
 *   chart.destroy();
 */

'use strict';

const _instances = new Map();   // elementId → ECharts instance

/**
 * Get (or create) the analog scatter chart for a container element.
 *
 * @param {string} elementId  ID of the container div
 * @returns {object}          chart controller with setData() and destroy()
 */
export function getAnalogChart(elementId) {
  if (_instances.has(elementId)) {
    return _instances.get(elementId);
  }

  const echarts = window.echarts;
  if (!echarts) {
    console.error('ECharts not loaded — analog chart disabled');
    return _noopController();
  }

  const el = document.getElementById(elementId);
  if (!el) return _noopController();

  const chart = echarts.init(el, 'dark');
  chart.setOption(_baseOption());

  const controller = {
    /**
     * @param {Array}  analogs      list of AnalogResult dicts from the API
     * @param {object} current      { valid_time, price_rrp, regime }
     */
    setData(analogs, current) {
      if (!analogs?.length) return;

      const maxScore = Math.max(...analogs.map(a => a.ppr_score ?? 0), 0.01);

      const scatterData = analogs.map(a => ({
        value: [
          new Date(a.valid_time).getTime(),
          Number(a.price_rrp ?? 0),
          Number(a.ppr_score ?? 0),
        ],
        itemStyle: { color: _scoreColor(a.ppr_score / maxScore) },
        symbolSize: 6 + 14 * (a.ppr_score / maxScore),
        name: a.regime || '',
      }));

      const currentData = current
        ? [{
            value: [new Date(current.valid_time).getTime(), Number(current.price_rrp ?? 0), 1],
            itemStyle: { color: '#ffd700', borderColor: '#fff', borderWidth: 2 },
            symbolSize: 18,
            name: 'Current',
          }]
        : [];

      chart.setOption({
        series: [
          { id: 'analogs', data: scatterData },
          { id: 'current', data: currentData },
        ],
      }, { replaceMerge: ['series'] });
      chart.resize();
    },

    destroy() {
      chart.dispose();
      _instances.delete(elementId);
    },
  };

  _instances.set(elementId, controller);
  return controller;
}

export function destroyAnalogChart(elementId) {
  _instances.get(elementId)?.destroy();
}

// ── Chart base option ─────────────────────────────────────────────────────

function _baseOption() {
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter(params) {
        const [ts, price, score] = params.value;
        const dt = new Date(ts).toLocaleString('en-AU', {
          timeZone: 'Australia/Sydney',
          dateStyle: 'short',
          timeStyle: 'short',
        });
        return `<b>${params.name || 'Analog'}</b><br>
          ${dt}<br>
          Price: $${Number(price).toFixed(2)}/MWh<br>
          PPR score: ${Number(score).toFixed(3)}`;
      },
    },
    grid: { left: 55, right: 20, top: 30, bottom: 45 },
    xAxis: {
      type: 'time',
      axisLabel: {
        formatter(val) {
          return new Date(val).toLocaleDateString('en-AU', {
            month: 'short', day: 'numeric',
          });
        },
        color: '#9ca3af',
        fontSize: 11,
      },
      splitLine: { lineStyle: { color: '#374151' } },
    },
    yAxis: {
      type: 'value',
      name: '$/MWh',
      nameTextStyle: { color: '#9ca3af', fontSize: 11 },
      axisLabel: { color: '#9ca3af', formatter: (v) => `$${v}` },
      splitLine: { lineStyle: { color: '#374151' } },
    },
    series: [
      {
        id: 'analogs',
        type: 'scatter',
        name: 'Analogs',
        data: [],
        emphasis: { scale: 1.3 },
      },
      {
        id: 'current',
        type: 'scatter',
        name: 'Current',
        data: [],
        symbol: 'diamond',
        emphasis: { scale: 1.2 },
      },
    ],
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _scoreColor(ratio) {
  // Cold (low PPR) → blue; warm (high PPR) → orange
  const r = Math.round(50 + 180 * ratio);
  const g = Math.round(100 + 80 * (1 - ratio));
  const b = Math.round(220 - 170 * ratio);
  return `rgb(${r},${g},${b})`;
}

function _noopController() {
  return { setData() {}, destroy() {} };
}
