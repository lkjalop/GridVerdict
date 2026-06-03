# GridVerdict NLP v2.0 — Architecture Slides

Three-slide update of "GridVerdict NLP v1.0.pdf".
Audience: AI architects and hiring managers.
Goal: show secure evidence-grounded NLP, live source orchestration, model harnessing,
and useful answers for both energy professionals and non-energy users.

---

# Slide 1 — From Natural Language to Auditable Market Answer

```
GridVerdict   NSW1 ▾   ● $66.53/MWh   NORMAL   LNN ✓   LEAR ✓   QRA ✓   TCN ✓   LNN_LTC ✓   META ✓   ● LIVE
```

```
┌──────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────────────────┐
│ CHAT                                 │   │ HOW THE ANSWER IS BUILT — 5 DETERMINISTIC LAYERS                       │
│                                      │   │                                                                        │
│ You                                  │   │  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐         │
│ > What are prices on Monday          │   │  │  QUERY   │  │ DECOMPOSE  │  │    GATHER    │  │  DEEPEN  │         │
│ > June 8th? Which to buy?            │   │  │          │→ │            │→ │  (parallel)  │→ │(seq/safe)│ ──┐     │
│                                      │   │  │ natural  │  │rules first │  │              │  │          │   │     │
│ ┌──────────────────────────────────┐ │   │  │ language │  │LLM enriches│  │· AEMO live + │  │· Temporal│   │     │
│ │ LOW CONFIDENCE   HOLD   75%      │ │   │  │          │  │            │  │  notices     │  │  RAG 4h  │   │     │
│ └──────────────────────────────────┘ │   │  │· region +│  │· regex →   │  │· LNN·LEAR·   │  │· fuel mix│   │     │
│                                      │   │  │  raw text│  │  intent    │  │  QRA·TCN·    │  │  split   │   │     │
│ GridVerdict does not have live data  │   │  │· session │  │· entity +  │  │  LNN_LTC·    │  │· BOM 7d  │   │     │
│ for June 8 — that date has not       │   │  │  carry-  │  │  flags     │  │  META        │  │· hist    │   │     │
│ occurred yet.                        │   │  │  fwd     │  │· LLM adds  │  │· HippoGraph  │  │  P10/    │   │     │
│                                      │   │  │· last 3  │  │  sub-Qs    │  │  analogs     │  │  P50/P90 │   │     │
│ Evidence-grounded estimate built     │   │  │  Q&A     │  │· routing   │  │· FCAS prices │  │· OpenNEM │   │     │
│ from historical June archive and     │   │  │  merged  │  │  key set   │  │· unit        │  │  trend   │   │     │
│ BOM forecast — with every gap named. │   │  │          │  │            │  │  dispatch    │  │· causal  │   │     │
│                                      │   │  │          │  │  adjacent  │  │· market      │  │  chain   │   │     │
│ See answer →                         │   │  │          │  │  path ──►  │  │  drivers     │  │          │   │     │
│                                      │   │  │          │  │  (exits    │  │· live feed   │  │          │   │     │
│                                      │   │  └──────────┘  │  here for  │  │· weather     │  └──────────┘   │     │
│                                      │   │                │  non-NEM)  │  └──────────────┘                  │     │
│                                      │   │                └────────────┘                                    │     │
│                                      │   │                                                                   ↓     │
│                                      │   │                              ┌──────────────────┐  ┌───────────────────┐│
│                                      │   │                              │      PLAN        │  │      VERDICT      ││
│                                      │   │                              │ routes by output │→ │                   ││
│                                      │   │                              │                  │  │       HOLD        ││
│                                      │   │                              │· 14 planners     │  │    75% confidence ││
│                                      │   │                              │· 8 adjacent      │  │                   ││
│                                      │   │                              │  handlers        │  │ + claim verifier  ││
│                                      │   │                              │· evidence +      │  │   guard           ││
│                                      │   │                              │  missing[]       │  │                   ││
│                                      │   │                              │· claim map built │  │                   ││
│                                      │   │                              └──────────────────┘  └───────────────────┘│
└──────────────────────────────────────┘   └────────────────────────────────────────────────────────────────────────┘
```

```
A raw LLM gives a fluent guess.
GridVerdict gives a verdict + a confidence score + a named list of what's missing.

Not "what sounds right" — but "what can we prove from live data, and what can't we?"
```

---

# Slide 2 — Four Security Gates Ride the Pipeline

```
GridVerdict   NSW1 ▾   ● $66.53/MWh   NORMAL   LNN ✓   LEAR ✓   QRA ✓   TCN ✓   LNN_LTC ✓   META ✓   ● LIVE
```

```
┌──────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────────────────┐
│ CHAT                                 │   │ CHECKED AT EVERY STEP — FOUR SECURITY GATES RIDE THE PIPELINE          │
│                                      │   │                                                                        │
│ You                                  │   │  ①②③④  =  a regex + rule pass that can HALT on risk (score ≥ 80)      │
│ > Why pay double for coal now?       │   │                                                                        │
│ > Wind was $15–25 this morning.      │   │    ① input        ② adjacent      ③ tool output   ④ final answer      │
│                                      │   │       ↓               ↓                 ↓                ↓            │
│ ┌──────────────────────────────────┐ │   │  ┌─────────┐   ┌───────────┐   ┌───────────────┐  ┌──────────────┐   │
│ │ INSUFFICIENT DATA  MONITOR  35%  │ │   │  │  QUERY  │   │ DECOMPOSE │   │    GATHER     │  │   VERDICT    │   │
│ └──────────────────────────────────┘ │   │  │  input  │ → │  intent   │ → │   parallel    │→ │   + guard    │   │
│                                      │   │  │         │   │           │   │               │  │              │   │
│ The price shift is the NEM's normal  │   │  │· raw    │   │· regex    │   │· AEMO · LNN   │  │ INSUFF.      │   │
│ diurnal cycle — not a market fault.  │   │  │  text   │   │  first    │   │· LEAR · QRA   │  │ DATA 35%     │   │
│                                      │   │  │· region │   │· LLM      │   │· unit · fuel  │  │              │   │
│ Coal modelled at $73. Low confidence │   │  │· session│   │  sub-Qs   │   │· FCAS · wx    │  │· claim chk   │   │
│ = a data gap, not a model failure.   │   │  │         │   │· conf     │   │· analogs      │  │              │   │
│ The system says so instead of        │   │  │         │   │  score    │   │· drivers      │  │              │   │
│ guessing.                            │   │  └─────────┘   └───────────┘   └───────────────┘  └──────────────┘   │
│                                      │   │                      │                                                │
│ See answer →                         │   │              (adjacent path)                                          │
│                                      │   │              gate ② fires here                                        │
│                                      │   │              before non-NEM answer                                    │
│                                      │   │              is returned                                              │
│                                      │   │                                                                        │
│                                      │   │  ── Rules decide routing. LLM only enriches wording — never the answer │
│                                      │   │  ── No source for a claim → marked MISSING, not guessed               │
│                                      │   │  ── Every model down → runs on rules alone. All gates log to audit    │
│                                      │   │  ── Tool output from adjacent clients (gas, ISP, fiscal) passes ③      │
└──────────────────────────────────────┘   └────────────────────────────────────────────────────────────────────────┘
```

```
① input  ② adjacent-path answer  ③ external tool output  ④ final answer
— four checks, in-process, all logged to the audit table.
```

---

# Slide 3 — Same Pipeline, Three Honest Outcomes

```
GridVerdict   NSW1 ▾   ● $66.53/MWh   NORMAL   LNN ✓   LEAR ✓   QRA ✓   TCN ✓   LNN_LTC ✓   META ✓   ● LIVE
```

```
┌──────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────────────────┐
│ CHAT                                 │   │ SAME PIPELINE, THREE HONEST OUTCOMES                                   │
│                                      │   │                                                                        │
│ You                                  │   │  "June 8th prices?"                                                    │
│ > What are prices on June 8th?       │   │                                                                        │
│                                      │   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌───────┐  │
│ ┌──────────────────────────────────┐ │   │  │DECOMPOSE │→ │  GATHER  │→ │    DEEPEN    │→ │  PLAN  │→ │ HOLD  │  │
│ │ LOW CONFIDENCE   HOLD   75%      │ │   │  │ future   │  │ AEMO +   │  │ BOM 7d limit │  │ Winter │  │  75%  │  │
│ └──────────────────────────────────┘ │   │  │ date ref │  │ hist arch│  │ named gap    │  │ table  │  │       │  │
│                                      │   │  └──────────┘  └──────────┘  └──────────────┘  └────────┘  └───────┘  │
│ EVIDENCE shown. MISSING shown.       │   │                                                                        │
│                                      │   │  "Why pay double — wind was $15?"                                      │
│ The product names the gap between    │   │                                                                        │
│ its answer and the better answer     │   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌───────┐  │
│ it could give with more data.        │   │  │DECOMPOSE │→ │  GATHER  │→ │    DEEPEN    │→ │  PLAN  │→ │INSUFF │  │
│                                      │   │  │ fuel     │  │ AEMO +   │  │ fuel mix +   │  │diurnal │  │DATA   │  │
│ Recheck closer to the date.          │   │  │ source   │  │ forecast │  │ history      │  │ cycle  │  │ 35%   │  │
│                                      │   │  └──────────┘  └──────────┘  └──────────────┘  └────────┘  └───────┘  │
│ See answer →                         │   │                                                                        │
│                                      │   │  "Is rooftop solar worth it in Adelaide?"                              │
│                                      │   │                                                                        │
│                                      │   │  ┌──────────┐  ┌─────────────────────────────────────────┐  ┌───────┐  │
│                                      │   │  │DECOMPOSE │→ │  ADJACENT HANDLER (no scatter-gather)    │→ │PARTIAL│  │
│                                      │   │  │ adjacent │  │ LCOE sensitivity + SA1 wholesale +        │  │SCOPE  │  │
│                                      │   │  │ LCOE     │  │ BOM irradiance · explicit scope boundary  │  │ 70%   │  │
│                                      │   │  └──────────┘  └─────────────────────────────────────────┘  └───────┘  │
│                                      │   │                                                                        │
│                                      │   │  ┌──────────────────────────────────────────────────────────────────┐  │
│                                      │   │  │ The second answer is low-confidence on purpose.                   │  │
│                                      │   │  │ The data to prove the fuel rank was not there — so it says so,    │  │
│                                      │   │  │ instead of inventing a number.                                    │  │
│                                      │   │  │ The third exits at DECOMPOSE — 8 adjacent handlers, no gather.   │  │
│                                      │   │  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┘   └────────────────────────────────────────────────────────────────────────┘
```

```
Built for ANZ energy desks & regulated industries — verdicts you can take to an audit.
```

---

## Verified claims (codebase audit June 2026)

| Claim | Source | Status |
|---|---|---|
| 14 planners | `answer_planner.py` — 14 dispatched routes in `plan_answer()` | ✓ confirmed |
| 8 adjacent handlers | `adjacent_handlers.py` — 8 `handle_*` functions | ✓ confirmed |
| LNN (lnn_cfc) | `live_forecast.py` — `lnn_cfc` model class, shown as "LNN" in UI | ✓ confirmed |
| LNN_LTC | `live_forecast.py` — `lnn_ltc` Liquid Time-Constant variant | ✓ confirmed |
| LEAR / QRA / META | `live_forecast.py` — `lear`, `qra`, `meta_ensemble` | ✓ confirmed |
| TCN | `live_forecast.py` — experimental, gated by `enable_experimental_sequence_forecasters` | ⚠ conditional |
| TemporalRAG | `app/engines/temporalrag.py` + `routes_temporalrag.py` | ✓ confirmed |
| HippoGraph analogs | `scatter_gather.py` T3 — PPR retrieval | ✓ confirmed |
| Claim verifier guard | `claim_verifier.py` — `verify_answer()` + Prometheus counter | ✓ confirmed |
| 4 security gates | `routes_query.py` — `pass_input`, adj `pass_answer`, `pass_tool_output`, final `pass_answer` | ✓ confirmed |
| FCAS prices in gather | `scatter_gather.py` — `_task_fcas` parallel task | ✓ confirmed |
| "3yr" history | `historical_price.py` — default 365d; archive 2022-08 to 2024-07 (~2yr actual) | ⚠ label only — say "historical archive" |
| Causal chain inference | `why_evidence_chains.py` + `why_builder.py` | ✓ confirmed |
| 11 parallel gather sources | `scatter_gather.py` — 9 always-on + 2 conditional | ✓ confirmed |
