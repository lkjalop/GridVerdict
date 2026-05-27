/**
 * api.js — thin wrapper over fetch for all GridVerdict API calls.
 * Reads token from localStorage. Throws on non-2xx responses.
 */

const API_BASE = '/api';

function getToken() {
  return localStorage.getItem('gv_token') || '';
}

function setToken(token) {
  localStorage.setItem('gv_token', token);
}

function clearToken() {
  localStorage.removeItem('gv_token');
}

async function _request(method, path, body = null) {
  const headers = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const opts = { method, headers };
  if (body !== null) opts.body = JSON.stringify(body);

  const resp = await fetch(API_BASE + path, opts);
  if (resp.status === 401) {
    clearToken();
    window.dispatchEvent(new CustomEvent('gv:unauthorized'));
    throw new Error('Unauthorized — please log in');
  }
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

export const api = {
  getToken,
  // Auth
  async login(email, password) {
    const form = new URLSearchParams({ username: email, password });
    const resp = await fetch(`${API_BASE}/auth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || 'Login failed');
    }
    const data = await resp.json();
    setToken(data.access_token);
    return data;
  },
  async register(email, password, tenantName = 'local') {
    return _request('POST', '/auth/register', { email, password, tenant_name: tenantName });
  },
  async me() {
    return _request('GET', '/auth/me');
  },
  logout() {
    clearToken();
  },

  // Health
  async health() {
    return _request('GET', '/health');
  },

  // Market
  async marketState(region = 'NSW1') {
    return _request('GET', `/market/state?region=${region}`);
  },
  async marketRegions() {
    return _request('GET', '/market/regions');
  },
  async marketRefresh() {
    return _request('POST', '/market/refresh');
  },
  async marketCoverage() {
    return _request('GET', '/market/coverage');
  },
  async marketHistory(region = 'NSW1', range = '24h', bucket = '5m') {
    return _request('GET', `/market/history?region=${region}&range=${range}&bucket=${bucket}`);
  },
  async marketForecast(region = 'NSW1') {
    return _request('GET', `/market/forecast?region=${region}`);
  },

  // Generic GET helper (used by state.js)
  async get(path) {
    return _request('GET', path.replace(/^\/api/, ''));
  },

  // Fuel mix
  async fuelMix(region = 'NSW1') {
    return _request('GET', `/market/fuel-mix?region=${region}`);
  },

  // Rebids
  async rebids(region = 'NSW1', date = '') {
    const q = date ? `&date=${date}` : '';
    return _request('GET', `/market/rebids?region=${region}${q}`);
  },

  // Constraints / interconnectors
  async constraints(region = 'NSW1', hours = 4, driverType = 'all') {
    return _request('GET', `/market/constraints?region=${region}&hours=${hours}&driver_type=${driverType}`);
  },

  // Model status
  async modelStatus(region = 'NSW1') {
    return _request('GET', `/models/status?region=${region}`);
  },

  // Data quality status
  async dataStatus() {
    return _request('GET', '/data/status');
  },

  // Incident timeline
  async incidentTimeline(region = 'NSW1', interval = '', lookbackMinutes = 30) {
    const params = new URLSearchParams({ region, lookback_minutes: lookbackMinutes });
    if (interval) params.set('interval', interval);
    return _request('GET', `/incidents/timeline?${params}`);
  },

  // Incident brief (full situational report)
  async incidentBrief(region = 'NSW1', anchor = null) {
    const params = new URLSearchParams();
    if (anchor) params.set('anchor', anchor);
    const qs = params.toString() ? `?${params}` : '';
    return _request('GET', `/incidents/brief/${encodeURIComponent(region)}${qs}`);
  },

  // Forecast trust panel
  async forecastTrust(region = 'NSW1') {
    return _request('GET', `/market/trust?region=${region}`);
  },

  // Sessions
  async createSession(region = 'NSW1', title = null) {
    return _request('POST', '/sessions', { region, title });
  },
  async listSessions() {
    return _request('GET', '/sessions');
  },
  async getSession(id) {
    return _request('GET', `/sessions/${id}`);
  },
  async renameSession(id, title) {
    return _request('PATCH', `/sessions/${id}/title`, { title });
  },

  // Query
  async submitQuery(sessionId, text, region = 'NSW1') {
    return _request('POST', `/sessions/${sessionId}/query`, { text, region });
  },

  // Portfolio
  async bessScenario(position, market) {
    return _request('POST', '/portfolio/bess/scenario', { position, market });
  },
  async bessMarketPrefill(region) {
    return _request('GET', `/portfolio/bess/market-prefill?region=${region}`);
  },

  // Sprint Q: Commentary (Live Feed)
  async getCommentaryRecent(region = 'NSW1', limit = 20, minSeverity = null) {
    const params = new URLSearchParams({ region, limit });
    if (minSeverity) params.set('min_severity', minSeverity);
    return _request('GET', `/commentary/recent?${params}`);
  },
  async getCommentaryEvent(id) {
    return _request('GET', `/commentary/${id}`);
  },
  async getCommentaryStats(region = 'NSW1', hours = 24) {
    return _request('GET', `/commentary/stats?region=${region}&hours=${hours}`);
  },
};
