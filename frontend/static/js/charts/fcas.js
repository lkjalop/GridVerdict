/**
 * FCAS multi-line chart — 8 ancillary service price series.
 * Raise services: warm orange palette. Lower services: cool blue palette.
 * Threshold lines at $50/MWh (elevated) and $200/MWh (extreme).
 */

const _RAISE_COLORS = ['#f97316', '#fb923c', '#fdba74', '#fed7aa'];
const _LOWER_COLORS = ['#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe'];

export function renderFcasChart(data, containerId = 'fcas-chart') {
  const el = document.getElementById(containerId);
  if (!el || !data?.series?.length) return;

  // Dispose previous instance if region/hours changed
  const existing = echarts.getInstanceByDom(el);
  if (existing) existing.dispose();

  const chart = echarts.init(el, null, { renderer: 'canvas' });

  const raiseSeries = data.series.filter(s => s.direction === 'raise');
  const lowerSeries = data.series.filter(s => s.direction === 'lower');
  const allSeries = [...raiseSeries, ...lowerSeries];

  // Timestamps from first series (all share same timestamps)
  const timestamps = data.series[0]?.timestamps || [];
  // Show every Nth label to avoid crowding
  const labelEvery = Math.max(1, Math.floor(timestamps.length / 12));

  const echartsData = allSeries.map((s, idx) => {
    const isRaise = s.direction === 'raise';
    const colorPalette = isRaise ? _RAISE_COLORS : _LOWER_COLORS;
    const colorIdx = isRaise
      ? raiseSeries.indexOf(s)
      : lowerSeries.indexOf(s);
    return {
      name: s.label,
      type: 'line',
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 1.5 },
      itemStyle: { color: colorPalette[colorIdx % colorPalette.length] },
      data: s.values,
      emphasis: { lineStyle: { width: 3 } },
    };
  });

  // Mark elevated ($50) and extreme ($200) thresholds
  const markLines = [
    { yAxis: 50,  name: 'Elevated',  lineStyle: { color: '#f59e0b', type: 'dashed', width: 1 },
      label: { formatter: '$50', color: '#f59e0b', fontSize: 10 } },
    { yAxis: 200, name: 'Extreme',   lineStyle: { color: '#ef4444', type: 'dashed', width: 1 },
      label: { formatter: '$200', color: '#ef4444', fontSize: 10 } },
  ];

  chart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'var(--bg-panel)',
      borderColor: 'var(--border)',
      textStyle: { color: 'var(--text-primary)', fontSize: 11 },
      formatter: (params) => {
        const ts = timestamps[params[0].dataIndex];
        const time = ts ? new Date(ts).toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' }) : '';
        const rows = params.map(p =>
          `<div style="display:flex;justify-content:space-between;gap:12px;">
            <span>${p.marker}${p.seriesName}</span>
            <span style="font-family:monospace;font-weight:600;">$${p.value?.toFixed(1) ?? '—'}/MWh</span>
          </div>`
        ).join('');
        return `<div style="font-size:11px;">${time}</div>${rows}`;
      },
    },
    legend: {
      data: allSeries.map(s => s.label),
      bottom: 0,
      textStyle: { color: 'var(--text-secondary)', fontSize: 10 },
      itemWidth: 12, itemHeight: 8,
    },
    grid: { left: 55, right: 12, top: 12, bottom: 45 },
    xAxis: {
      type: 'category',
      data: timestamps,
      axisLabel: {
        color: 'var(--text-muted)',
        fontSize: 10,
        interval: labelEvery - 1,
        formatter: v => {
          try { return new Date(v).toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' }); }
          catch { return v; }
        },
      },
      axisLine: { lineStyle: { color: 'var(--border)' } },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '$/MWh',
      nameTextStyle: { color: 'var(--text-muted)', fontSize: 10 },
      axisLabel: { color: 'var(--text-muted)', fontSize: 10, formatter: v => `$${v}` },
      splitLine: { lineStyle: { color: 'var(--border-subtle)', type: 'dashed' } },
      markLine: {
        silent: true,
        symbol: 'none',
        data: markLines,
      },
    },
    series: echartsData,
  });

  // Responsive resize
  const ro = new ResizeObserver(() => chart.resize());
  ro.observe(el);
}
