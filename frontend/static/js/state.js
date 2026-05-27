/**
 * state.js — Alpine.js root store for GridVerdict.
 *
 * All UI state lives here. Components read from and write to this object.
 * The viewport panel that renders depends on `viewport.type` which is
 * set by the query response.
 */

import { api } from './api.js';

export function gridverdictApp() {
  return {
    // ── Auth ──────────────────────────────────────────────────────
    auth: {
      loggedIn: false,
      email: '',
      tenantId: '',
      showLogin: true,
      loginEmail: '',
      loginPassword: '',
      loginError: '',
      loading: false,
    },

    // ── Region selector ───────────────────────────────────────────
    region: 'NSW1',
    regions: ['NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1'],
    coverage: {},   // { NSW1: { status, days, intervals, earliest, latest }, ... }
    dataSources: {},  // { "AEMO_DISPATCH_PRICE": { status, caveat }, ... }

    // ── Market strip (topbar) ─────────────────────────────────────
    market: {
      price_rrp: null,
      regime: null,
      staleness_seconds: null,
      is_stale: false,
      loading: false,
    },
    chartRange: '24h',
    chartBucket: '5m',
    chartRanges: ['1h', '3h', '6h', '12h', '24h'],
    chartBuckets: ['5m', '1h', '3h', '6h', '12h', '24h'],
    chartLoading: false,
    forecastLoading: false,
    forecastStatus: null,

    // ── Sessions ──────────────────────────────────────────────────
    sessions: [],
    currentSessionId: null,
    sessionsOpen: false,

    // ── Chat messages ─────────────────────────────────────────────
    messages: [],
    queryText: '',
    queryLoading: false,

    // ── Viewport ──────────────────────────────────────────────────
    viewport: {
      type: 'market',
      loading: false,
      data: null,             // last QueryResponse from API
      activeTab: 'market',
    },

    // ── Answer panel collapsible sections ─────────────────────────
    answerExpanded: {
      evidence: true,
      analogs: false,
      temporal: false,
      missing: false,
      causalityTiers: true,
      claimMap: true,
      incidentTimeline: true,
      provenance: false,
    },

    // ── Incident timeline ─────────────────────────────────────────
    incidentTimeline: null,    // { events, verdict, coverage_grade, anchor_price, ... }
    incidentTimelineLoading: false,

    // ── Fuel mix / source recommendation ─────────────────────────
    fuelMix: null,       // { sources, recommendation, spot_price_rrp }
    fuelMixLoading: false,

    // ── Model status ──────────────────────────────────────────────
    modelStatus: null,   // { models: { lear, qra, lnn, ... } }
    modelStatusLoading: false,

    // ── Data quality status ───────────────────────────────────────
    dataStatus: null,    // full /api/data/status response
    dataStatusLoading: false,

    // ── Constraint / interconnector timeline ──────────────────────
    constraintData: null,  // { events, summary }
    constraintLoading: false,
    constraintHours: 4,

    // ── Incident Brief ────────────────────────────────────────────
    incidentBrief: null,        // full /incidents/brief/{region} response
    incidentBriefLoading: false,
    incidentBriefError: null,

    // ── Portfolio / BESS ──────────────────────────────────────────
    portfolio: {
      loading: false,
      result: null,
      error: null,
      position: {
        capacity_mwh: 100,
        soc_pct: 62,
        max_discharge_mw: 50,
        max_charge_mw: 25,
        efficiency_pct: 88,
        degradation_cost_per_mwh: 8,
        min_reserve_soc_pct: 20,
        contract_type: 'merchant',
        fcas_enabled: false,
        risk_limit_dollar: null,
        site_export_limit_mw: null,
      },
      market: {
        region: 'NSW1',
        price_rrp: null,
        price_regime: 'normal',
        headroom_mw: null,
        forecast_direction: null,
        fcas_raise_6sec_rrp: null,
        fcas_raise_reg_rrp: null,
        rebid_evidence_tier: null,
        outage_evidence_tier: null,
        evidence_quality: 'plausible',
      },
    },

    // ── SSE connection state ──────────────────────────────────────
    sseConnected: false,
    _sse: null,
    _sseRetryMs: 5000,

    // ── Sprint Q: Live Feed (Rolling Market Commentary) ───────────
    commentary: {
      events: [],          // list of CommentaryEvent objects
      loading: false,
      error: null,
      unreadCount: 0,
      minSeverity: 'LOW',  // filter: LOW | MEDIUM | HIGH | CRITICAL
      paused: false,       // pause auto-scroll when user is reading
    },

    // ── Toast ─────────────────────────────────────────────────────
    toasts: [],

    // ── Lifecycle ─────────────────────────────────────────────────
    async init() {
      window.addEventListener('gv:unauthorized', () => this.doLogout());
      await this.loadRegions();
      // Dev mode: try /api/auth/me silently
      try {
        const me = await api.me();
        this.auth.loggedIn = true;
        this.auth.email = me.email;
        this.auth.tenantId = me.tenant_id;
        this.auth.showLogin = false;
        await this.loadSessions();
        await this.refreshMarket();
        await this.refreshMarketHistory();
        await this.refreshForecast();
        await this.loadCoverage();
        await this.refreshFuelMix();
        await this.refreshModelStatus();
        this._startMarketPoll();
      } catch (_) {
        this.auth.showLogin = true;
      }
    },

    async loadCoverage() {
      try {
        const data = await api.marketCoverage();
        this.coverage = data.coverage || {};
        this.dataSources = data.sources || {};
      } catch (_) {
        // Non-fatal — UI just omits chips
      }
    },

    coverageChipClass(region) {
      const s = (this.coverage[region] || {}).status || 'unknown';
      return `coverage-chip coverage-chip--${s}`;
    },

    coverageLabel(region) {
      const c = this.coverage[region];
      if (!c) return '?';
      if (c.status === 'operational') return `${c.days}d`;
      if (c.status === 'partial')     return `${c.days}d partial`;
      if (c.status === 'not_ingested' || c.status === 'unavailable') return 'no data';
      return '?';
    },

    sourceChipClass(key) {
      const s = (this.dataSources[key] || {}).status || 'unknown';
      return `coverage-chip coverage-chip--${s}`;
    },

    sourceLabel(key) {
      const src = this.dataSources[key];
      if (!src) return key;
      const label = { operational: 'operational', partial: 'partial',
                      scaffolded: 'scaffolded', unavailable: 'unavailable' };
      return label[src.status] || src.status || '?';
    },

    async loadRegions() {
      try {
        const data = await api.marketRegions();
        if (Array.isArray(data.regions) && data.regions.length) {
          this.regions = data.regions;
          if (!this.regions.includes(this.region)) this.region = this.regions[0];
        }
      } catch (_) {
        this.regions = ['NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1'];
      }
    },

    // ── Auth actions ──────────────────────────────────────────────
    async doLogin() {
      this.auth.loading = true;
      this.auth.loginError = '';
      try {
        await api.login(this.auth.loginEmail, this.auth.loginPassword);
        const me = await api.me();
        this.auth.loggedIn = true;
        this.auth.email = me.email;
        this.auth.tenantId = me.tenant_id;
        this.auth.showLogin = false;
        await this.loadSessions();
        await this.refreshMarket();
        await this.refreshMarketHistory();
        await this.refreshForecast();
        await this.loadCoverage();
        await this.refreshFuelMix();
        await this.refreshModelStatus();
        this._startMarketPoll();
      } catch (err) {
        this.auth.loginError = err.message;
      } finally {
        this.auth.loading = false;
      }
    },
    doLogout() {
      api.logout();
      this.auth.loggedIn = false;
      this.auth.showLogin = true;
      this.sessions = [];
      this.messages = [];
      this.currentSessionId = null;
    },

    // ── Session actions ───────────────────────────────────────────
    async loadSessions() {
      try {
        this.sessions = await api.listSessions();
      } catch (err) {
        this.showToast(err.message, 'error');
      }
    },
    async newSession() {
      try {
        const s = await api.createSession(this.region);
        this.sessions.unshift(s);
        this.switchSession(s.id);
      } catch (err) {
        this.showToast(err.message, 'error');
      }
    },
    async switchSession(id) {
      this.currentSessionId = id;
      this.sessionsOpen = false;
      this.messages = [];
      try {
        const detail = await api.getSession(id);
        for (const q of detail.queries) {
          this.messages.push({ role: 'user', text: q.raw_query, id: q.id });
          if (q.answer) {
            this.messages.push({
              role: 'assistant',
              text: this.formatVerdictSummary(q.answer),
              verdict: q.answer,
              id: q.id + '-a',
            });
          }
        }
        this._scrollMessages();
      } catch (err) {
        this.showToast(err.message, 'error');
      }
    },

    // ── Query submission ──────────────────────────────────────────
    async submitQuery() {
      const text = this.queryText.trim();
      if (!text || this.queryLoading) return;

      if (!this.currentSessionId) {
        await this.newSession();
      }

      this.queryText = '';
      this.queryLoading = true;
      this.viewport.loading = true;

      this.messages.push({ role: 'user', text, id: 'tmp-' + Date.now() });
      this._scrollMessages();

      try {
        const resp = await api.submitQuery(this.currentSessionId, text, this.region);

        // Render concise planner summary in chat; full narrative stays in the Answer/Audit panels.
        const why = this.formatVerdictSummary(resp.verdict);
        this.messages.push({
          role: 'assistant',
          text: why,
          verdict: resp.verdict,
          viewportType: resp.viewport_type,
          id: resp.query_id + '-a',
        });

        // Switch viewport — map legacy type names to 5-tab layout
        const mappedTab = this._mapViewportType(resp.viewport_type);
        this.viewport.type = mappedTab;
        this.viewport.activeTab = mappedTab;
        this.viewport.data = resp;
        // Auto-expand analogs if enough matches came back
        if (resp.analogs?.length >= 3) this.answerExpanded.analogs = true;
        // Render analog chart in audit panel if visible
        if (resp.analogs?.length && mappedTab === 'audit') {
          this.$nextTick(() => this.renderAnalogChart());
        }

        // Auto-fetch incident timeline for explanation/action queries with elevated+ price
        const regime = resp.verdict?.as_of
          ? null   // not stored on verdict directly
          : null;
        const shouldFetchTimeline = resp.intent &&
          ['explanation', 'action_recommendation', 'retrospective'].includes(resp.intent);
        if (shouldFetchTimeline) {
          const anchorInterval = resp.verdict?.as_of || '';
          this.refreshIncidentTimeline(anchorInterval);
        }

        // Update session list (updated_at + title)
        await this.loadSessions();
        this._scrollMessages();
      } catch (err) {
        this.messages.push({ role: 'system', text: `Error: ${err.message}`, id: 'err-' + Date.now() });
        this.showToast(err.message, 'error');
      } finally {
        this.queryLoading = false;
        this.viewport.loading = false;
      }
    },

    handleInputKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.submitQuery();
      }
    },

    // ── Market refresh ────────────────────────────────────────────
    async refreshMarket() {
      this.market.loading = true;
      try {
        const state = await api.marketState(this.region);
        this.market.price_rrp = state.price_rrp;
        this.market.demand_mw = state.demand_mw;
        this.market.availability_mw = state.availability_mw;
        this.market.headroom_mw = state.headroom_mw;
        this.market.regime = state.regime;
        this.market.staleness_seconds = state.staleness_seconds;
        this.market.is_stale = state.is_stale;
        this.market.valid_time = state.valid_time;
        this.market.price_percentile = state.price_percentile;
        this.viewport.marketState = state;
        // Fire chart bridge event
        window.dispatchEvent(new CustomEvent('gv:market-updated', { detail: state }));
        // Init chart on first load
        if (window.gvCharts && !window.gvCharts.swimlane) {
          window.gvCharts.initSwimlane(this.region);
        }
      } catch (_) {
        this.market.is_stale = true;
      } finally {
        this.market.loading = false;
      }
    },
    async refreshMarketHistory() {
      this.chartLoading = true;
      try {
        const data = await api.marketHistory(this.region, this.chartRange, this.chartBucket);
        window.gvCharts?.setHistory(data.points || []);
      } catch (err) {
        this.showToast(`Chart history unavailable: ${err.message}`, 'error');
      } finally {
        this.chartLoading = false;
      }
    },
    async refreshForecast() {
      this.forecastLoading = true;
      try {
        const data = await api.marketForecast(this.region);
        this.forecastStatus = data;
        window.gvCharts?.setForecast(data.available && data.forecasts?.length ? data : null);
      } catch (err) {
        this.forecastStatus = { available: false, reason: err.message };
        window.gvCharts?.setForecast(null);
      } finally {
        this.forecastLoading = false;
      }
    },
    async setChartRange(range) {
      this.chartRange = range;
      await this.refreshMarketHistory();
    },
    async setChartBucket(bucket) {
      this.chartBucket = bucket;
      await this.refreshMarketHistory();
    },
    _startMarketPoll() {
      this._startSSE();
      // Fallback poll every 5 min for history/forecast data SSE doesn't cover
      setInterval(async () => {
        await this.refreshMarketHistory();
        await this.refreshForecast();
        await this.refreshFuelMix();
      }, 5 * 60 * 1000);
    },

    _startSSE() {
      if (this._sse) { this._sse.close(); this._sse = null; }
      this.sseConnected = false;
      const token = api.getToken?.() || '';
      const url = `/api/events/stream?region=${encodeURIComponent(this.region)}${token ? '&token=' + encodeURIComponent(token) : ''}`;
      const es = new EventSource(url);

      es.onopen = () => {
        this.sseConnected = true;
        this._sseRetryMs = 5000;
      };

      // dispatch_updated → refresh market state
      es.addEventListener('dispatch_updated', () => {
        this.refreshMarket().catch(() => {});
      });

      // forecast_updated → refresh forecast overlay
      es.addEventListener('forecast_updated', () => {
        this.refreshForecast().catch(() => {});
      });

      // data_status_changed → refresh data panel if it's loaded
      es.addEventListener('data_status_changed', () => {
        if (this.dataStatus !== null) {
          this.refreshDataStatus().catch(() => {});
        }
      });

      // scheduler_failure → toast warning
      es.addEventListener('scheduler_failure', (ev) => {
        try {
          const d = JSON.parse(ev.data);
          this.showToast(
            `Scheduler "${d.job_id}" failed ${d.consecutive_failures}x: ${d.error}`,
            'error',
          );
        } catch (_) {}
      });

      // source_stale → mark market stale and toast
      es.addEventListener('source_stale', (ev) => {
        try {
          const d = JSON.parse(ev.data);
          this.market.is_stale = true;
          this.showToast(`Market data source stale: ${d.source || 'AEMO'}`, 'error');
        } catch (_) {}
      });

      // spike_alert → urgent toast + refresh
      es.addEventListener('spike_alert', (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (!d.region || d.region === this.region) {
            this.showToast(
              `SPIKE ALERT ${d.region}: $${Number(d.price_rrp).toFixed(0)}/MWh (${(d.regime||'spike').toUpperCase()})`,
              'error',
            );
            this.refreshMarket().catch(() => {});
          }
        } catch (_) {}
      });

      // spike_resolved → info toast + refresh
      es.addEventListener('spike_resolved', (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (!d.region || d.region === this.region) {
            this.showToast(
              `${d.region} price normalised (${(d.regime||'normal').toUpperCase()}, $${Number(d.price_rrp).toFixed(0)}/MWh)`,
              'info',
            );
            this.refreshMarket().catch(() => {});
          }
        } catch (_) {}
      });

      // incident_timeline_updated → refresh if timeline is open
      es.addEventListener('incident_timeline_updated', (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (this.incidentTimeline && (!d.region || d.region === this.region)) {
            this.refreshIncidentTimeline().catch(() => {});
          }
        } catch (_) {}
      });

      // trace_written — no UI action needed
      es.addEventListener('trace_written', () => {});

      // commentary_created — Sprint Q: push event to Live Feed
      es.addEventListener('commentary_created', (e) => {
        try {
          const payload = JSON.parse(e.data);
          this.onCommentaryEvent(payload);
        } catch (_) {}
      });

      es.onerror = () => {
        this.sseConnected = false;
        es.close();
        this._sse = null;
        // Exponential back-off capped at 60s
        this._sseRetryMs = Math.min((this._sseRetryMs || 5000) * 2, 60_000);
        setTimeout(() => this._startSSE(), this._sseRetryMs);
      };

      this._sse = es;
    },

    // ── Fuel mix ──────────────────────────────────────────────────
    async refreshFuelMix() {
      this.fuelMixLoading = true;
      try {
        this.fuelMix = await api.fuelMix(this.region);
        this.$nextTick(() => window.gvCharts?.renderFuelMix?.(this.fuelMix));
      } catch (err) {
        this.fuelMix = null;
      } finally {
        this.fuelMixLoading = false;
      }
    },

    // ── Incident Brief ────────────────────────────────────────────
    async refreshIncidentBrief() {
      this.incidentBriefLoading = true;
      this.incidentBriefError = null;
      try {
        this.incidentBrief = await api.incidentBrief(this.region);
      } catch (err) {
        this.incidentBriefError = err.message;
        this.incidentBrief = null;
      } finally {
        this.incidentBriefLoading = false;
      }
    },

    // ── Model status ──────────────────────────────────────────────
    async refreshModelStatus() {
      this.modelStatusLoading = true;
      try {
        this.modelStatus = await api.modelStatus(this.region);
      } catch (err) {
        this.modelStatus = null;
      } finally {
        this.modelStatusLoading = false;
      }
    },

    async refreshDataStatus() {
      this.dataStatusLoading = true;
      try {
        this.dataStatus = await api.dataStatus();
      } catch (err) {
        this.dataStatus = null;
      } finally {
        this.dataStatusLoading = false;
      }
    },

    async refreshIncidentTimeline(interval = '', lookbackMinutes = 30) {
      this.incidentTimelineLoading = true;
      try {
        this.incidentTimeline = await api.incidentTimeline(this.region, interval, lookbackMinutes);
      } catch (err) {
        this.incidentTimeline = null;
      } finally {
        this.incidentTimelineLoading = false;
      }
    },

    // ── Constraint timeline ───────────────────────────────────────
    async refreshConstraints() {
      this.constraintLoading = true;
      try {
        this.constraintData = await api.constraints(this.region, this.constraintHours);
      } catch (err) {
        this.constraintData = null;
      } finally {
        this.constraintLoading = false;
      }
    },

    // ── Portfolio / BESS ──────────────────────────────────────────
    async runBessScenario() {
      if (this.portfolio.loading) return;
      this.portfolio.loading = true;
      this.portfolio.error = null;
      try {
        const pf = this.portfolio;
        this.portfolio.result = await api.bessScenario(
          { ...pf.position },
          { ...pf.market, region: this.region },
        );
      } catch (e) {
        this.portfolio.error = e.message;
        this.showToast(`Scenario failed: ${e.message}`, 'error');
      } finally {
        this.portfolio.loading = false;
      }
    },

    async prefillBessMarket() {
      try {
        const data = await api.bessMarketPrefill(this.region);
        const m = this.portfolio.market;
        if (data.price_rrp !== null && data.price_rrp !== undefined) m.price_rrp = data.price_rrp;
        if (data.price_regime) m.price_regime = data.price_regime;
        if (data.headroom_mw !== null && data.headroom_mw !== undefined) m.headroom_mw = data.headroom_mw;
        m.region = this.region;
        this.showToast('Market context pre-filled from live data', 'info');
      } catch (_) {
        // Non-fatal
      }
    },

    bessActionLabel(action) {
      return {
        dispatch_full: 'DISPATCH NOW (FULL)',
        dispatch_partial: 'DISPATCH (PARTIAL)',
        hold: 'HOLD',
        charge: 'CHARGE',
        reserve_fcas: 'RESERVE FOR FCAS',
        avoid_insufficient_data: 'AVOID — INSUFFICIENT DATA',
      }[action] || action;
    },

    bessActionClass(action) {
      return {
        dispatch_full: 'bess-action--dispatch',
        dispatch_partial: 'bess-action--dispatch-partial',
        hold: 'bess-action--hold',
        charge: 'bess-action--charge',
        reserve_fcas: 'bess-action--fcas',
        avoid_insufficient_data: 'bess-action--avoid',
      }[action] || '';
    },

    // ── Sprint Q: Live Feed ───────────────────────────────────────
    async refreshCommentary() {
      this.commentary.loading = true;
      this.commentary.error = null;
      try {
        const data = await api.getCommentaryRecent(this.region, 30, this.commentary.minSeverity);
        this.commentary.events = data.events || [];
        this.commentary.unreadCount = 0;
      } catch (e) {
        this.commentary.error = e.message;
      } finally {
        this.commentary.loading = false;
      }
    },

    onCommentaryEvent(payload) {
      if (payload.region && payload.region !== this.region) return;
      if (this.commentary.paused) {
        this.commentary.unreadCount++;
        return;
      }
      this.commentary.events.unshift(payload);
      if (this.commentary.events.length > 50) this.commentary.events.pop();
      if (this.viewport.activeTab !== 'livefeed') {
        this.commentary.unreadCount++;
      }
    },

    dismissCommentaryEvent(id) {
      this.commentary.events = this.commentary.events.filter(e => e.id !== id);
    },

    askAboutCommentaryEvent(evt) {
      const time = evt.valid_time ? evt.valid_time.substring(11, 16) : '';
      const type = (evt.event_type || '').replace(/_/g, ' ');
      this.queryText = `What caused the ${type} in ${this.region} at ${time}?`;
      this.switchViewportTab('answer');
      this.$nextTick(() => document.querySelector('#query-input')?.focus());
    },

    severityColor(severity) {
      return { LOW: '#6b7280', MEDIUM: '#d97706', HIGH: '#dc2626', CRITICAL: '#7c3aed' }[severity] || '#6b7280';
    },

    tierIcon(tier) {
      return { confirmed: '✓', supported: '~', plausible: '?', unconfirmed: '·' }[tier] || '·';
    },

    // ── Viewport tab switching ────────────────────────────────────
    switchViewportTab(tab) {
      this.viewport.activeTab = tab;
      this.viewport.type = tab;
      if (tab === 'livefeed') {
        this.commentary.unreadCount = 0;
        this.refreshCommentary();
      }
      if (tab === 'audit') {
        this.$nextTick(() => this.renderAnalogChart());
      }
      if (tab === 'sources') {
        this.$nextTick(() => {
          if (this.fuelMix) window.gvCharts?.renderFuelMix?.(this.fuelMix);
        });
      }
    },

    toggleSection(key) {
      this.answerExpanded[key] = !this.answerExpanded[key];
    },

    _mapViewportType(type) {
      return {
        verdict: 'answer',
        why: 'answer',
        market_state: 'market',
        retrospective: 'answer',
        counterfactual: 'audit',
        trace_replay: 'audit',
        out_of_scope: 'answer',
        comparison: 'answer',
        answer: 'answer',
        market: 'market',
        audit: 'audit',
      }[type] || 'answer';
    },
    renderAnalogChart() {
      const analogs = this.viewport.data?.analogs || [];
      const el = document.getElementById('analog-chart');
      if (!el || !analogs.length || !window.gvCharts?.analog) return;
      const chart = window.gvCharts.analog.getAnalogChart('analog-chart');
      chart.setData(analogs, {
        valid_time: this.market.valid_time || new Date().toISOString(),
        price_rrp: this.market.price_rrp,
        regime: this.market.regime,
      });
    },

    // ── Helpers ───────────────────────────────────────────────────
    showToast(msg, type = 'info') {
      if (!Array.isArray(this.toasts)) this.toasts = [];
      const id = Date.now();
      this.toasts.push({ id, msg, type });
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 4000);
    },
    _scrollMessages() {
      this.$nextTick(() => {
        const el = document.getElementById('messages');
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    // ── Computed display helpers ──────────────────────────────────
    regimeBadgeClass(regime) {
      return `regime-badge--${regime || 'normal'}`;
    },
    verdictBadgeClass(verdict) {
      const map = {
        SUPPORTED: 'badge--supported',
        LOW_CONFIDENCE: 'badge--low',
        INSUFFICIENT_DATA: 'badge--insufficient',
        NEEDS_CLARIFICATION: 'badge--clarify',
        OUT_OF_SCOPE: 'badge--oos',
      };
      return map[verdict] || 'badge--oos';
    },
    actionChipClass(action) {
      const map = {
        dispatch_now: 'action-chip--dispatch',
        hold: 'action-chip--hold',
        charge: 'action-chip--charge',
        monitor: 'action-chip--monitor',
        refuse: 'action-chip--refuse',
      };
      return map[action] || 'action-chip--monitor';
    },
    confFillClass(band) {
      return `conf-bar__fill--${(band || 'very_low').replace('_', '_')}`;
    },
    confPct(confidence) {
      return `${Math.round((confidence || 0) * 100)}%`;
    },
    metricValue(value, suffix = '') {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
      return `${Number(value).toFixed(1)}${suffix}`;
    },
    evidenceValue(ev) {
      const v = Number(ev?.value);
      if (!Number.isFinite(v)) return '—';
      if (ev.field === 'price_rrp') return `$${v.toFixed(2)}/MWh`;
      if (ev.field === 'temperature_c') return `${v.toFixed(1)} C`;
      if (ev.field === 'wind_speed_kmh') return `${v.toFixed(1)} km/h`;
      if ((ev.field || '').includes('pct')) return `${v.toFixed(1)}%`;
      return `${v.toFixed(1)} MW`;
    },
    formatVerdictSummary(verdict) {
      if (!verdict) return '';
      const sections = verdict.answer_sections || [];
      if (!sections.length) return verdict.why_plain_english || '';
      const preferred = ['Answer', 'Drivers', 'Continuation'];
      const lines = [];
      for (const title of preferred) {
        const section = sections.find(s => s.title === title);
        if (!section || !Array.isArray(section.items)) continue;
        const limit = title === 'Answer' ? 2 : 1;
        for (const item of section.items.slice(0, limit)) {
          if (item) lines.push(item);
        }
      }
      return lines.slice(0, 5).join('\n');
    },
    priceColor(regime) {
      return {
        normal: 'var(--regime-normal)',
        elevated: 'var(--regime-elevated)',
        spike: 'var(--regime-spike)',
        extreme: 'var(--regime-extreme)',
      }[regime] || 'var(--text-primary)';
    },
    formatTime(iso) {
      if (!iso) return '—';
      return new Date(iso).toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit', timeZone: 'Australia/Sydney' }) + ' AEST';
    },
    formatDateTime(iso) {
      if (!iso) return '—';
      return new Date(iso).toLocaleString('en-AU', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Australia/Sydney',
      });
    },
  };
}
