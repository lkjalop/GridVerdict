# Forecasting Harness — Integration Guide
# Generated: 2026-05-23. Read alongside ROADMAP.md and WHY_ENGINE_DESIGN.md.

---

## 1. What You Have — Quality Assessment

All 12 files are well-structured and follow the scaffold/skin discipline correctly.
Six are production-ready as written. Four have specific bugs to fix before wiring.
Two need concrete implementations added.

| File | Status | Issue |
|---|---|---|
| `types.py` | Production-ready | Minor: QuantileForecast frozen + np.ndarray not hashable — use `field(compare=False)` |
| `base.py` | Production-ready | None |
| `baselines.py` | Production-ready | None |
| `walk_forward.py` | Production-ready | None — leakage assertion is correct |
| `metrics.py` | Production-ready | None |
| `spike_metrics.py` | Production-ready | None |
| `calibration.py` | Production-ready | None |
| `market_features.py` | Production-ready (reference) | Reference impl only — needs multi-stream join for production |
| `gbm_model.py` | Production-ready | None |
| `lnn_model.py` | Bug | `_cell`/`_head` uninitialized — AttributeError if predict before fit |
| `harness.py` | Bug | `target_times` are integers not datetimes; self-skill scoring bug |
| `news_correlator.py` | Stub needed | Protocol defined but no concrete AEMO client implementation |

---

## 2. LNN Decision — Resolution

The `lnn_model.py` uses `ncps` library (`from ncps.torch import CfC`).
Earlier we decided to implement LTC from scratch for the research showcase.

**Resolution: Keep both. They serve different purposes.**

```
engines/forecasting/models/
  lnn_model.py          ncps-based CfC — MVP battery in the ensemble
                        Used immediately in Week 2 harness
                        Interchangeable with GBM via ForecastModel interface

  ltc_scratch.py        From-scratch LTC implementation (Week 4)
                        The research showcase centrepiece
                        Also implements ForecastModel — same interface
                        Demonstrates the ODE: dx/dt = -x/τ(x,I,θ) + f(x,I,θ)
```

The harness runs both through identical walk-forward splits.
The research narrative: "The GBM wins on CRPS. The LTC demonstrates the mathematical
architecture. The evaluation harness is what makes you trust either."
This is stronger than the LTC alone because the harness proves the LTC is honestly scored.

---

## 3. Directory Mapping — Where Each File Goes

```
Source file               →  GridVerdict path                           Notes
─────────────────────────────────────────────────────────────────────────────────────
types.py                  →  app/engines/forecasting/data/types.py       Fix frozen+ndarray
base.py                   →  app/engines/forecasting/models/base.py      No changes
baselines.py              →  app/engines/forecasting/models/baselines.py No changes
gbm_model.py              →  app/engines/forecasting/models/gbm_model.py No changes
lnn_model.py              →  app/engines/forecasting/models/lnn_model.py Fix init guard
                           +  app/engines/forecasting/models/ltc_scratch.py  NEW (Week 4)
market_features.py        →  app/engines/forecasting/features/market_features.py  NEM skin
metrics.py                →  app/engines/forecasting/evaluation/metrics.py   No changes
spike_metrics.py          →  app/engines/forecasting/evaluation/spike_metrics.py No changes
calibration.py            →  app/engines/forecasting/evaluation/calibration.py  No changes
walk_forward.py           →  app/engines/forecasting/evaluation/walk_forward.py  No changes
harness.py                →  app/engines/forecasting/evaluation/harness.py  Fix datetimes+skill
news_correlator.py        →  app/mcp/news_correlator.py                    + concrete client
```

**Full subtree after mapping:**

```
app/engines/forecasting/
  __init__.py
  data/
    __init__.py
    types.py              ~95 lines  (fixed)
  features/
    __init__.py
    market_features.py    ~85 lines  (reference — production join in Week 3)
  models/
    __init__.py
    base.py               ~55 lines
    baselines.py          ~80 lines
    gbm_model.py          ~65 lines
    lnn_model.py          ~90 lines  (fixed)
    ltc_scratch.py        ~220 lines (Week 4 — from scratch PyTorch LTC)
  evaluation/
    __init__.py
    metrics.py            ~75 lines
    spike_metrics.py      ~60 lines
    calibration.py        ~45 lines
    walk_forward.py       ~65 lines
    harness.py            ~95 lines  (fixed)

app/mcp/
  news_correlator.py      ~115 lines  (existing correlation logic)
  aemo_notices_client.py  ~120 lines  (NEW — concrete client for news_correlator)
```

Total forecasting harness: ~950 lines across 12 files + 2 new.
Every file under 300 lines. Scaffold/skin boundary intact.

---

## 4. Production Fixes — Specific

### Fix 1: types.py — Frozen Dataclass + NumPy

`QuantileForecast` is `frozen=True` but contains `np.ndarray` which is mutable and unhashable.
Python will not complain at definition time, but `__hash__` will fail at runtime.

```python
# BEFORE (types.py line 30):
@dataclass(frozen=True)
class QuantileForecast:
    ...
    values: np.ndarray  # (n_targets, n_quantiles)

# AFTER:
from dataclasses import dataclass, field
import numpy as np

@dataclass
class QuantileForecast:
    """A probabilistic forecast. Not frozen — np.ndarray is mutable."""
    target_times: Sequence[datetime]
    quantiles: Sequence[float]
    values: np.ndarray = field(repr=False)  # (n_targets, n_quantiles)

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.target_times), len(self.quantiles)):
            raise ValueError(
                f"values shape {self.values.shape} != "
                f"({len(self.target_times)}, {len(self.quantiles)})"
            )
```

Same fix applies to `ForecastWindow` — remove `frozen=True`.

### Fix 2: lnn_model.py — Uninitialized Attributes

`_cell` and `_head` are set in `fit()` but accessed in `predict_quantiles()` without guard.
Calling `predict_quantiles` before `fit` raises `AttributeError: 'LNNQuantileModel' has no
attribute '_cell'`. Add explicit init guards and device handling.

See the fixed file written below.

### Fix 3: harness.py — Integer Target Times + Self-Skill Bug

Bug 1: `target_times = list(range(sp.test_start, sp.test_end))` passes integers to
`predict_quantiles` which expects `Sequence[datetime]`. The walk_forward split gives
integer indices, not timestamps. Fix: pass real timestamps or use a sentinel.

Bug 2: The `skill` dict is computed as:
```python
for base_name, base_crps in raw_crps.items():
    if base_name != name:
        skill[base_name] = skill_score(raw_crps[name], base_crps)
```
This means model A gets `skill_vs_B = 1 - CRPS_A/CRPS_B` AND
model B gets `skill_vs_A = 1 - CRPS_B/CRPS_A` — so both appear in each other's skill dict.
This is intentional for full comparison but confusing. Clarify with a comment
and ensure `skill_vs_aemo_predispatch` is always present for the headline number.

See the fixed file written below.

### Fix 4: news_correlator.py — No Concrete Client

The `NewsMCPClient` Protocol has no implementation. The correlator is unusable without one.
The concrete `AEMOMarketNoticesClient` is written below in `aemo_notices_client.py`.

---

## 5. Integration Points — How This Wires Into GridVerdict

### 5.1 The 5th Evidence Source (Why Engine Update)

The news correlator adds a 5th evidence source to the Why Engine. Update WHY_ENGINE_DESIGN.md:

```
Evidence sources for the Why Engine:
  Source 1: Current causal drivers    (ChronoGraph + AEMO live data)
  Source 2: Forecast drivers          (LNN/GBM distribution + AEMO deviation)
  Source 3: Historical analogs        (HippoGraph PPR over archive)
  Source 4: Bitemporal trace          (replay of prior decisions)
  Source 5: News/market notice        (news_correlator.py — cited official cause)  ← NEW
```

**Impact on why_sources.py** — add one new call:
```python
# why_sources.py — add to WhyEvidence collection:

from app.mcp.news_correlator import correlate_price_event
from app.mcp.aemo_notices_client import AEMOMarketNoticesClient

@dataclass
class WhyEvidence:
    current_drivers: dict           # ChronoGraph + live AEMO
    forecast_context: dict          # LNN distribution
    analog_candidates: list         # HippoGraph PPR results
    bitemporal_ref: str | None      # trace_id
    news_correlation: CorrelationResult | None   # NEW — 5th source

# In the collection function, add:
client = AEMOMarketNoticesClient()
news = correlate_price_event(client, event_time=as_of, region=region)
evidence.news_correlation = news
```

**Impact on scatter agents** — add Agent G:
```python
# agents/scatter_agents.py — add to PARALLEL SCATTER:

async def _scatter_news(region: str, as_of: datetime) -> CorrelationResult:
    """Agent G: market notice and news correlation."""
    client = AEMOMarketNoticesClient()
    return correlate_price_event(client, as_of, region, lookback_min=60)
```

**Impact on why_formatter.py** — the formatter now has a 5th source to cite:
```python
# If news_correlation.explained:
#   Prefix the why with the cited notice
#   "An AEMO Market Notice at {time} reported: {title}. This correlates with..."
# If not explained:
#   The critic notes: "No public explanation found — treat as lower-confidence"
#   And the confidence score is reduced (add to uncertainty.py logic)
```

### 5.2 Backtest Route

```python
# app/api/routes_backtest.py

from app.engines.forecasting.evaluation.harness import run_backtest
from app.engines.forecasting.models.baselines import (
    PersistenceModel, SeasonalNaiveModel, AEMOPredispatchModel
)
from app.engines.forecasting.models.gbm_model import GBMQuantileModel
from app.engines.forecasting.models.lnn_model import LNNQuantileModel
from app.engines.forecasting.features.market_features import build_features, COL_LAST_PRICE, COL_SEASONAL, COL_AEMO
from app.engines.forecasting.data.types import BacktestReport
from app.core.trace import write_trace

@router.post("/api/v1/backtest/run")
async def run_backtest_endpoint(request: BacktestRequest, tenant: Tenant = Depends(get_tenant)):
    # 1. Fetch archive data for the requested window
    events = await archive_client.fetch_window(
        region=request.region,
        start=request.start,
        end=request.end,
        tenant_id=tenant.id
    )

    # 2. Build feature matrix
    X, y = build_features(events, target_times=[e["valid_time"] for e in events])

    # 3. Register models (always include baselines for skill scoring)
    models = {
        "persistence": PersistenceModel(last_value_col=COL_LAST_PRICE),
        "seasonal_naive": SeasonalNaiveModel(season_col=COL_SEASONAL),
        "aemo_predispatch": AEMOPredispatchModel(predispatch_col=COL_AEMO),
        "gbm_quantile": GBMQuantileModel(),
        "lnn_cfc": LNNQuantileModel(),
    }

    # 4. Run evaluation harness
    report: BacktestReport = run_backtest(
        models=models,
        X=X, y=y,
        horizon=request.horizon_intervals,
        step=1,
        min_train=288,          # 24 hours of 5-min intervals minimum training
        spike_threshold=request.spike_threshold or 300.0,
        reference="aemo_predispatch",
    )

    # 5. Write to bitemporal trace
    trace_id = await write_trace(
        tenant_id=tenant.id,
        query_id=request.query_id,
        valid_time=request.end,
        system_time=datetime.utcnow(),
        backtest_report=report.to_dict(),
    )

    # 6. Return structured result for frontend
    return BacktestResponse(
        trace_id=trace_id,
        summary_table=report.summary_table(),
        headline_skill=_headline_skill(report),
        scores=[s.__dict__ for s in report.scores],
        horizon_min=report.horizon_min,
        n_origins=report.n_origins,
    )


def _headline_skill(report: BacktestReport) -> dict:
    """Extract the CRPS skill vs AEMO for every model — the headline number."""
    best = report.best_by_crps()
    if not best:
        return {}
    return {
        "best_model": best.model_name,
        "crps": best.crps,
        "skill_vs_aemo": best.skill_vs.get("aemo_predispatch"),
        "spike_f1": best.spike_f1,
        "calibration_error": best.calibration_error,
    }
```

### 5.3 BacktestReport → Factual Verdict Contract

When a backtest is requested via natural language (Q06: "what would my battery have earned..."),
the backtest result must be wrapped in the factual verdict schema before returning.

```python
# In the answer synthesis path (engines/prefill.py or orchestrator.py):

if decomposition.requires_backtest:
    report = await run_backtest_for_query(decomposition, tenant)
    # Wrap in verdict
    verdict = FactualVerdict(
        verdict="SUPPORTED" if report.best_by_crps() else "INSUFFICIENT_DATA",
        action="monitor",    # backtest doesn't produce a dispatch action
        confidence=_confidence_from_report(report),
        as_of=decomposition.time_range.end,
        why_plain_english=_narrate_backtest(report),    # LLM narrates the table
        evidence_refs=_evidence_from_report(report),    # each ModelScore is an evidence_ref
        counterargument=_critique_backtest(report),     # always: "assumes perfect foresight"
        missing_data=[],
        disclaimer=SIMULATION_DISCLAIMER,
    )
```

### 5.4 Market Features → Canonical Schema Alignment

`market_features.py` currently accepts raw `dict` records. In GridVerdict these come from
`EnergyEvent` objects (the NEM domain adapter). Add a thin adapter in `domain/nem/features.py`:

```python
# domain/nem/features.py
from app.engines.forecasting.features.market_features import build_features
from app.domain.nem.schema import EnergyEvent

def build_nem_features(events: list[EnergyEvent], target_times):
    """Adapt EnergyEvent list to the dict format market_features.py expects."""
    raw = [
        {
            "valid_time": e.valid_time,
            "last_price": e.price_rrp,
            "seasonal_price": e.price_rrp,      # filled by archive join in production
            "aemo_predispatch": e.forecast.get("predispatch_rrp", e.price_rrp),
            "demand": e.demand_mw,
            "demand_forecast": e.forecast.get("demand_mw", e.demand_mw),
            "available_gen": e.availability_mw,
            "interconnector_room": e.forecast.get("interconnector_room", 0.0),
            "renewable_frac": e.forecast.get("renewable_frac", 0.0),
            "roll_vol_12": 0.0,                 # computed in production from rolling window
            "price": e.price_rrp,               # the target
        }
        for e in events
    ]
    return build_features(raw, target_times)
```

### 5.5 Frontend Wiring (backtest_chart.js)

The `BacktestReport.summary_table()` method returns a formatted string.
The frontend needs the structured `scores` list, not the string.
Add `to_dict()` to `BacktestReport` in types.py:

```python
# Add to BacktestReport in types.py:
def to_dict(self) -> dict:
    return {
        "horizon_min": self.horizon_min,
        "n_origins": self.n_origins,
        "spike_threshold": self.spike_threshold,
        "scores": [
            {
                "model": s.model_name,
                "crps": round(s.crps, 3),
                "pinball": round(s.pinball, 3),
                "spike_f1": round(s.spike_f1, 3),
                "calibration_error": round(s.calibration_error, 3),
                "skill_vs_aemo": round(s.skill_vs.get("aemo_predispatch", float("nan")), 3),
                "coverage": {str(k): round(v, 3) for k, v in s.coverage.items()},
            }
            for s in self.scores
        ],
    }
```

`js/charts/backtest_pnl.js` renders:
- Table: model / CRPS / pinball / spike_f1 / skill_vs_aemo
- Chart: reliability diagram (nominal vs empirical quantile coverage) per model
- Headline: best model's skill_vs_aemo — positive means "beat AEMO pre-dispatch"

---

## 6. New Files Needed

Three new files are required for production wiring. They are written below in the repo.

```
app/mcp/aemo_notices_client.py       ~120 lines  Concrete AEMO Market Notice fetcher
app/engines/forecasting/             ~5 lines each  Package __init__.py files (5 files)
app/engines/forecasting/models/ltc_scratch.py  ~220 lines  Week 4 from-scratch LTC
```

---

## 7. Updated Build Sequence

### Week 2 additions (forecasting harness arrives early):
- [ ] Copy all 12 files to their target directories
- [ ] Apply fixes to types.py, lnn_model.py, harness.py
- [ ] Write `__init__.py` files
- [ ] Write `aemo_notices_client.py` stub
- [ ] Wire `routes_backtest.py` with stub data
- [ ] Test: harness runs on 30 days of AEMO archive data with baselines only
- [ ] Test: `summary_table()` contains AEMO pre-dispatch skill score

### Week 3 additions (Why Engine 5th source):
- [ ] Wire news correlator into `why_sources.py`
- [ ] Add Agent G (news correlator) to `scatter_agents.py`
- [ ] Update `why_formatter.py` to cite news source when present
- [ ] Update adversarial critic: "no public explanation → lower confidence"
- [ ] Test Q02: "Why is NSW expensive?" answer cites AEMO Market Notice when one is active

### Week 4 additions (real models):
- [ ] Add `domain/nem/features.py` — EnergyEvent → market feature dict adapter
- [ ] Train GBM on 90 days of AEMO archive data
- [ ] Train ncps LNN on same data
- [ ] Run full harness: walk-forward over 30-day test set
- [ ] Report: summary_table printed, skill_vs_aemo computed
- [ ] Write `ltc_scratch.py` — from-scratch LTC cell and model
- [ ] Train LTC on same data, add to harness as 6th battery
- [ ] Compare: does LTC beat GBM on spike F1? (It probably doesn't — report honestly)

---

## 8. The Honest Reporting Contract

This comes from the `evaluation_harness.md` spec and must be enforced in code, not prose.

```python
# In routes_backtest.py or the NLP narration layer:

def _narrate_skill(report: BacktestReport) -> str:
    """Honest skill narrative — never hides a negative skill score."""
    best = report.best_by_crps()
    if not best:
        return "Insufficient data to evaluate model performance."
    
    aemo_skill = best.skill_vs.get("aemo_predispatch")
    if aemo_skill is None:
        return "AEMO pre-dispatch benchmark not available for comparison."
    
    if aemo_skill > 0:
        return (
            f"{best.model_name} achieved CRPS skill of +{aemo_skill:.1%} vs AEMO pre-dispatch "
            f"over {report.n_origins} walk-forward origins. Spike detection F1: {best.spike_f1:.3f}."
        )
    else:
        # Negative skill — report honestly, pivot to explainability
        return (
            f"{best.model_name} did not beat AEMO pre-dispatch on average CRPS "
            f"(skill: {aemo_skill:.1%}). AEMO's forecast is the strong baseline on smooth data. "
            f"GridVerdict's value is in explaining WHY spikes are likely and flagging "
            f"deviation risk the AEMO forecast doesn't surface. Spike F1: {best.spike_f1:.3f}."
        )
```

The `summary_table()` string is always included in the trace record, regardless of whether
the skill is positive or negative. A reviewer who re-runs the backtest sees the same numbers.

---

## 9. Package __init__.py Contents

```python
# app/engines/forecasting/__init__.py
from .evaluation.harness import run_backtest
from .data.types import BacktestReport, QuantileForecast, ModelScore

__all__ = ["run_backtest", "BacktestReport", "QuantileForecast", "ModelScore"]
```

```python
# app/engines/forecasting/models/__init__.py
from .base import ForecastModel
from .baselines import PersistenceModel, SeasonalNaiveModel, AEMOPredispatchModel
from .gbm_model import GBMQuantileModel
from .lnn_model import LNNQuantileModel

__all__ = [
    "ForecastModel",
    "PersistenceModel", "SeasonalNaiveModel", "AEMOPredispatchModel",
    "GBMQuantileModel", "LNNQuantileModel",
]
```

```python
# app/engines/forecasting/evaluation/__init__.py
from .harness import run_backtest
from .metrics import pinball_loss, crps_from_quantiles, skill_score
from .spike_metrics import spike_scores
from .calibration import empirical_coverage, calibration_error, reliability_points
from .walk_forward import walk_forward_splits

__all__ = [
    "run_backtest",
    "pinball_loss", "crps_from_quantiles", "skill_score",
    "spike_scores",
    "empirical_coverage", "calibration_error", "reliability_points",
    "walk_forward_splits",
]
```
