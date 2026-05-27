/**
 * fuel_mix.js — ECharts donut + bar chart for fuel type mix and source recommendation.
 *
 * Renders:
 *   1. A donut chart showing capacity MW by fuel type
 *   2. Colour-coded fuel type badges with price/volatility info
 */

const FUEL_COLORS = {
  wind:    '#4ade80',   // green
  solar:   '#fbbf24',   // amber
  coal:    '#94a3b8',   // slate
  hydro:   '#38bdf8',   // sky blue
  gas:     '#fb923c',   // orange
  battery: '#a78bfa',   // purple
  other:   '#64748b',   // gray
};

const FUEL_ICONS = {
  wind: '🌬', solar: '☀', coal: '⬛', hydro: '💧', gas: '🔥', battery: '🔋', other: '⚡',
};

let _chart = null;

export function getFuelMixChart(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return null;
  if (_chart) {
    const existing = echarts.getInstanceByDom(el);
    if (existing) return existing;
  }
  _chart = echarts.init(el, 'dark', { renderer: 'canvas' });
  return _chart;
}

export function destroyFuelMixChart() {
  if (_chart) { _chart.dispose(); _chart = null; }
}

export function renderFuelMix(fuelMixData, containerId = 'fuel-mix-chart') {
  if (!fuelMixData || !fuelMixData.sources) return;
  const chart = getFuelMixChart(containerId);
  if (!chart) return;

  const sources = fuelMixData.sources.filter(
    s => (s.mw_capacity || 0) > 0 || (s.mw_dispatched || 0) > 0
  );

  // Donut data — prefer dispatched MW, fall back to capacity
  const donutData = sources.map(s => ({
    name: s.fuel_type,
    value: Math.round(s.mw_dispatched ?? s.mw_capacity ?? 0),
    itemStyle: { color: FUEL_COLORS[s.fuel_type] || '#64748b' },
    label: { show: true },
  })).filter(d => d.value > 0);

  // Bar chart — marginal cost range
  const barCategories = sources.map(s => `${FUEL_ICONS[s.fuel_type] || '⚡'} ${s.fuel_type}`);
  const barLow  = sources.map(s => s.marginal_cost_low ?? 0);
  const barHigh = sources.map(s => s.marginal_cost_high ?? 0);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        if (p.seriesType === 'pie') {
          const src = sources.find(s => s.fuel_type === p.name);
          return [
            `<b>${p.name}</b>`,
            `Dispatched: ${p.value} MW`,
            src?.unit_count ? `Units: ${src.unit_count}` : '',
            src?.effective_price_mwh != null ? `Spot: $${src.effective_price_mwh}/MWh` : '',
          ].filter(Boolean).join('<br>');
        }
        return `${p.name}: $${p.value}/MWh`;
      },
    },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '52%', containLabel: true },
    xAxis: {
      type: 'category',
      data: barCategories,
      axisLabel: { color: '#94a3b8', fontSize: 10 },
      axisLine: { lineStyle: { color: '#334155' } },
    },
    yAxis: {
      type: 'value',
      name: '$/MWh',
      nameTextStyle: { color: '#64748b', fontSize: 10 },
      axisLabel: { color: '#94a3b8', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    series: [
      {
        name: 'Fuel Mix',
        type: 'pie',
        radius: ['25%', '42%'],
        center: ['50%', '24%'],
        data: donutData,
        label: {
          formatter: '{b}\n{d}%',
          fontSize: 10,
          color: '#94a3b8',
        },
        emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' } },
      },
      {
        name: 'Marginal Cost Low',
        type: 'bar',
        stack: 'cost',
        data: barLow,
        itemStyle: { color: 'rgba(0,0,0,0)', borderColor: 'transparent' },
        tooltip: { show: false },
      },
      {
        name: 'Cost Range',
        type: 'bar',
        stack: 'cost',
        data: sources.map((s, i) => ({
          value: (s.marginal_cost_high ?? 0) - (s.marginal_cost_low ?? 0),
          itemStyle: {
            color: FUEL_COLORS[s.fuel_type] || '#64748b',
            opacity: 0.8,
          },
        })),
        label: { show: false },
      },
    ],
  };

  chart.setOption(option, true);
  window.addEventListener('resize', () => chart.resize());
}
