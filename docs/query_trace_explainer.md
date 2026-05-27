# GridVerdict — From NLP Question to Evidence-Grounded Answer

**One slide. Three real queries. Every step explained.**

This document uses three live questions from the GridVerdict chat interface to show exactly
how the system moves from natural language to a structured, evidence-referenced answer —
and why it is architecturally incapable of inventing facts.

---

## The Slide

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  GRIDVERDICT  ·  From NLP Question to Evidence-Grounded Answer  ·  NSW1  ·  2026-05-27                               ║
╠════════════════════════╦══════════════════════╦═══════════════════════════╦═════════════════════╦═════════════════════╣
║  💬  CHAT THREAD       ║  ① NLP DECOMPOSE     ║  ② GATHER  (MCP tools)   ║  ③ REASON           ║  ④ ANSWER PLANNER  ║
║  (left-panel, UI)      ║  SecurityObserver    ║  ScatterGather +          ║  WhyBuilder +        ║  AnswerSections    ║
║                        ║  → Decomposer        ║  TemporalRAG              ║  ClaimVerifier       ║  + Critic output   ║
╠════════════════════════╬══════════════════════╬═══════════════════════════╬═════════════════════╬═════════════════════╣
║  Q1                    ║  intent: EXPLANATION ║  ✅ Dispatch  CONF        ║  LEAR  ↓ P90 $1153  ║  "now $143;        ║
║  "Why is NSW1          ║  region: NSW1        ║  🟡 Analogs(9) PLAU       ║  QRA   ↓ P90  $458  ║   60m ago $164     ║
║  elevated and will     ║  requires_why: T     ║  ⚪ Weather    UNCO       ║  Meta  ↓ P90  $756  ║   (-21)            ║
║  it continue?"         ║  requires_fcst: T    ║  ⚪ Notices(0) UNCO       ║  LNN   ✗ untrained  ║   10 analogs →     ║
║                        ║  output:             ║  🟡 TRAG(12)  PLAU        ║  Conf: 73%          ║   7/10 recovered"  ║
║  → now $143            ║   causal+forecast    ║                           ║  Observer: ✅ PASS  ║  Models lean lower ║
║  73% · HOLD            ║                      ║                           ║  Critic: countered  ║  VERDICT: HOLD     ║
╠════════════════════════╬══════════════════════╬═══════════════════════════╬═════════════════════╬═════════════════════╣
║  Q2                    ║  intent: EXPLANATION ║  ✅ Dispatch   CONF       ║  Fuel merit order:  ║  "Hydro ranks #1   ║
║  "Why is coal best     ║  tech: [coal,        ║  🟡 Fuel mix   PLAU       ║   hydro > wind      ║   at $167 spot.    ║
║  to buy instead of     ║         solar,hydro] ║  ⚪ Weather    UNCO       ║   > gas > coal      ║   Coal marginal    ║
║  solar or hydro?"      ║  output:             ║  ⚫ Unit disp  N/A        ║  ⚠ prior tier only  ║   ~$55/MWh.        ║
║                        ║   fuel_source_rec    ║  ⚫ Rebid stk  N/A        ║  Conf: 55%          ║   Missing: unit    ║
║  → now $167            ║                      ║                           ║  Observer: ✅ PASS  ║   dispatch + bids" ║
║  55% · HOLD            ║                      ║                           ║  ⚠ PRIOR caveat     ║  NOT financial adv ║
╠════════════════════════╬══════════════════════╬═══════════════════════════╬═════════════════════╬═════════════════════╣
║  Q3                    ║  intent: EXPLANATION ║  ✅ Dispatch   CONF       ║  Query prices parsed║  "You described    ║
║  "Why did price go     ║  req_history: T      ║  🔵 Query text QEXT       ║  $143→$167→$165→    ║   $143→$167→$140.  ║
║  143→167→165→140?      ║  output:             ║    $143→$167→$165→$140   ║   $140               ║   Swing $27 =      ║
║  Which fuel source?"   ║   price_fluctuation  ║  ⚫ Recent DB  EMPTY      ║  Swing: $27 MINOR   ║   minor variation. ║
║                        ║   _attribution       ║  🟡 Fuel mix   PLAU       ║  No unit dispatch   ║   Driver needs     ║
║  → now $140            ║                      ║  ⚫ Unit disp  N/A        ║  Conf: 71%          ║   unit disp +      ║
║  71% · HOLD            ║                      ║                           ║  Observer: ✅ PASS  ║   rebid evidence"  ║
╠════════════════════════╩══════════════════════╩═══════════════════════════╩═════════════════════╩═════════════════════╣
║  EVIDENCE TIERS:  ✅ CONF = direct AEMO telemetry  ·  🟡 PLAU = pattern/model signal  ·  ⚪ UNCO = no backing data   ║
║                   🔵 QEXT = extracted from user's query text  ·  ⚫ N/A / EMPTY = source unavailable                  ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  SAFETY:  4-pass SecurityObserver (input → decomp → tool_output → answer)  ·  ClaimVerifier + adversarial critic     ║
║           LLM never produces numbers or asserts facts  ·  All claims have evidence_ref IDs  ·  Missing always surfaced ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Why This Order? The Architecture Rationale

Before walking row by row, it is worth understanding why the pipeline is sequenced the way
it is. Each stage is a deliberate constraint on what the next stage is allowed to do.

### Stage ①  NLP Decompose — *classify intent before touching any data*

The user's raw text is sandboxed in this stage. A language model (Ollama → Claude → rule-based
fallback) reads the query and produces a structured JSON object: intent, region, what kind of
evidence to seek, and what shape the answer should take. It produces **no facts**. It makes
**no market claims**. Its only job is to label the question.

This matters because it separates *what the user is asking* from *what the market data says*.
If you let the evidence gathering happen before intent classification, the system might pull
dispatch data for the wrong region, look for a forecast when the user wanted a causal explanation,
or run a comparison when the user wanted attribution.

The SecurityObserver runs its first pass here — checking the raw input for injection attempts,
out-of-scope queries (non-NEM topics), or unsafe action requests before any data is fetched.

### Stage ②  Gather (MCP tools) — *pull only what the decomposition licensed*

The `ScatterGather` function fires a set of parallel async fetches, each guarded by whether
the decomposition requested it. Dispatch price always runs. Analogs only run if `requires_why`
is true. Weather only runs if weather context is relevant to the query. TemporalRAG only runs
if the query needs bitemporal document retrieval. Fuel mix only runs for fuel-source queries.

Every data source comes back with a `source_status`: fresh, stale, or unavailable. These statuses
directly determine the evidence tier (CONF, PLAU, UNCO, N/A) attached to every downstream claim.

This is the MCP (Model Context Protocol) layer — structured, typed evidence ingestion. Not
retrieval-augmented generation over uncontrolled text. Each tool call produces a typed object
that the reasoning layer reads deterministically.

The SecurityObserver runs its second pass here on the tool outputs — checking that the data
returned from AEMO and other sources does not contain anomalous values or injected payloads.

### Stage ③  Reason — *derive claims from evidence, not from language model imagination*

The `WhyBuilder` takes the assembled evidence bundle and runs deterministic causal logic:
- Which evidence is fresh enough to be CONF tier?
- Which evidence is present but not direct (pattern match only) → PLAU?
- Which sources were sought but unavailable → UNCO?
- What is the confidence score, computed as a weighted function of tier mix?

The forecasting models (LEAR, QRA, Meta-ensemble, LNN/TCN) each produce a probability
distribution over the next dispatch interval — a P10, P50, P90 range, not a point estimate.
These models do not agree on a single number; the system explicitly shows where they diverge
and by how much.

The `ClaimVerifier` runs the adversarial critic: for every verdict, it constructs a
counterargument. This is not decoration. If the counterargument is strong enough to invalidate
the primary claim, confidence is downgraded. Every answer that reaches the user has already
been challenged.

The SecurityObserver runs a third pass here on the assembled answer object.

### Stage ④  Answer Planner — *narrate only what the evidence already established*

The `AnswerPlanner` is a deterministic routing function, not a language model. It reads
`requested_output` from the decomposition (set in Stage ①) and calls the matching planner
function. The planner constructs sentences from the evidence bundle — it formats what Stage ③
already computed. It does not infer, summarize, or generate.

This is the architectural firewall against hallucination: the language model that classified
the query in Stage ① never sees the evidence. The planner that writes the answer in Stage ④
never has access to a language model. Every number in the answer comes from a `evidence_ref`
object created in Stage ② or ③.

The SecurityObserver runs its fourth and final pass on the formatted answer before it reaches
the user.

---

## Row-by-Row Commentary

---

### Q1 — "Why is NSW1 elevated and will it continue?"

**What the user wants to know:**
Is this price elevation real, what is driving it, and is it about to resolve or persist?
These are two questions embedded in one: a causal explanation (present state) and a
continuation forecast (near future).

**① Decompose**

The decomposer identifies `intent = EXPLANATION` because the query contains "why" and
asks about a present market condition. The "will it continue" clause triggers
`requires_forecast = true` alongside `requires_why = true`. The region NSW1 is identified
from context. The `requested_output` is set to `causal_explanation_with_forecast` —
the most evidence-rich answer path.

This single structured object now controls everything that follows. No stage downstream
re-reads the raw query text for intent. The classification is done once, deterministically.

**② Gather**

Five parallel evidence fetches fire simultaneously:

- **Dispatch** (`CONF`): Live AEMO price ($143/MWh), demand (8,106 MW), availability
  (12,060 MW), headroom (3,954 MW), regime classification (ELEVATED). This is the only
  CONF-tier source for this query — direct telemetry with sub-30-second staleness.

- **Analogs (9 matches, `PLAU`)**: HippoGraph searches its index of every dispatch interval
  since platform inception, finding 9 historical moments where price, demand, headroom, and
  regime were similar to the current state. 7 of those 9 recovered within 30 minutes.
  This is PLAU tier because it is a pattern match, not a confirmed causal statement.

- **Weather (`UNCO`)**: The weather consensus aggregator contacts multiple sources and returns
  a reading, but no market-weather correlation is confirmed for this interval. The source
  exists but does not explain the price move — so it is UNCO, not dropped.

- **AEMO Notices (0, `UNCO`)**: No market notice from AEMO covers the current interval.
  An AEMO notice would be the clearest confirmation of a known market event (forced outage,
  constraint binding, RERT activation). Absence is noted explicitly, not silently.

- **TemporalRAG (12 docs, `PLAU`)**: Bitemporal document retrieval finds 12 relevant
  dispatch-price documents from the past 4 hours. These supplement the analogs with
  recent interval history. PLAU because they inform context, not direct causation.

**The price path** ("now $143; 60m ago $164; -21") comes from the `recent_dispatch` table —
actual stored dispatch rows queried from the database. The planner formats these as a
human-readable trend line. This is not generated by a model.

**③ Reason**

Three forecasting models all independently lean falling:
- **LEAR** (Least Absolute shrinkage Regression): P50 $81, P90 $1,153. Wide P90 means
  LEAR sees elevated spike risk even while its central estimate is falling. The $1,153
  P90 is not an error — LEAR includes long tail behaviour from historical spike events.
- **QRA** (Quantile Regression Averaging): P50 $96, P90 $458. Tighter than LEAR,
  still sees elevated but falling trajectory.
- **Meta-ensemble**: P50 $104, P90 $756. The meta model blends LEAR and QRA outputs
  with calibrated weights. When LEAR and QRA agree on direction (both falling), the
  meta ensemble's central estimate reflects that agreement.
- **LNN**: Not trained yet on this instance. Surfaced as explicitly unavailable, not
  silently absent.

Confidence is computed as 73%: CONF dispatch + PLAU analogs + PLAU TRAG
offset by two UNCO sources (weather, notices) and one unavailable model (LNN).

The adversarial critic raises the counterargument: "The price may not continue falling
if a late-afternoon demand peak materialises or if a generator rebid occurs in the next
dispatch interval." This counterargument is embedded in the verdict before the user sees
the answer.

**④ Answer**

The `_plan_explanation` function (with `include_forecast=True`) formats:
1. The price path from DB rows — not generated, read directly from `recent_dispatch`
2. The analog outcome summary — "7/10 recovered" comes from HippoGraph's structured output
3. The model call — "Available models lean lower" comes from comparing LEAR, QRA, Meta
   direction labels, not from a language model summarising the numbers

The answer says HOLD because confidence is 73% (below the ACTION threshold), the missing
data (constraints, unit dispatch, AEMO notice) means the driver is unconfirmed, and the
adversarial critic produced a valid counterargument.

---

### Q2 — "Why is coal best to buy instead of solar or hydro?"

**What the user wants to know:**
Given current market conditions, which generation fuel type should be preferred for
procurement — is coal genuinely the best option, or do solar and hydro have advantages
at this price level?

This is a **procurement and merit-order question**, not a pure causal question. It requires
understanding the NEM's marginal cost stack and how the current spot price relates to each
fuel's opportunity cost.

**① Decompose**

The decomposer identifies:
- `intent = EXPLANATION` (the "why" framing)
- `technologies = [coal, solar, hydro]` (extracted from the query text)
- `requested_output = fuel_source_recommendation` (triggered by "best to buy", "instead of",
  plus fuel-type keywords co-occurring in the same query)

The `requested_output` routing is important here. The system does not treat this as a generic
explanation query. It routes to the fuel-source planner, which knows that answering this
question requires: (a) the current spot price as a reference, (b) the marginal cost model
for each fuel type, and (c) explicit disclaimers that NEM spot prices clear uniformly —
*all* generators receive the regional spot price regardless of their fuel cost.

**② Gather**

- **Dispatch (`CONF`)**: Spot price $167/MWh. This is the reference price against which
  the fuel-cost model compares each source. The $167 figure is direct telemetry — CONF.

- **Fuel mix model (`PLAU`)**: This is where the merit order is computed. The system holds
  marginal cost *priors* for each fuel type based on NEM historical data:
  - Solar/Wind: ~$0/MWh fuel cost (zero marginal fuel cost, sunk capital cost)
  - Hydro: ~$0–50/MWh (opportunity cost model, varies by storage level and season)
  - Black coal (NSW): ~$30–55/MWh (fuel + operational)
  - CCGT gas: ~$80–120/MWh
  - OCGT gas peakers: ~$150–300/MWh
  At a spot price of $167/MWh, the marginal generator is likely CCGT gas — the price-setting
  generator is gas, not coal. Hydro ranks above coal in the merit order because its
  effective dispatch cost at current water storage is below coal's fuel cost, and it can
  ramp faster.
  This source is **PLAU tier** because these are prior estimates — no live unit dispatch
  data has confirmed which generators are actually running at this interval.

- **Unit dispatch (`N/A`)**: The system sought generator-level dispatch data (which unit,
  which fuel type, what MW, what bid price) but it was unavailable. This is the single most
  important missing data item for this query. Without it, the merit order is a model
  estimate, not an observation.

- **Rebid stack (`N/A`)**: The rebid history shows if any generator moved its bid price
  significantly during this interval — a common cause of the $143→$167 move (see Q3).
  Unavailable.

**The data tier warning** (`⚠ prior tier only`) propagates directly into the confidence
score. The system computes: "fuel mix confidence is low because the data tier is PRIOR,
not DISPATCH. The answer can give directional guidance but not confirmed attribution."

**③ Reason**

The fuel merit order ranking (hydro > wind > gas > coal) is produced by the deterministic
fuel mix engine, not by a language model. At $167/MWh spot:
- Hydro and wind generators are well in-the-money (their fuel cost is effectively zero)
- Coal is also in-the-money but has slower ramping, higher emissions costs, and its
  typical marginal cost of ~$55/MWh leaves less margin than hydro/wind
- Coal is NOT the cheapest option for the buyer despite being "cheap" fuel — the NEM
  clears at the spot price regardless of what the generator's fuel actually cost

The adversarial critic flags the `⚠ PRIOR tier only` warning explicitly. The critic's
counterargument is: "Without live unit dispatch, we cannot confirm whether hydro is
actually dispatching at this interval. Storage constraints, seasonal water limits, or
operator decisions could make hydro unavailable. The ranking is a marginal-cost model,
not an observed dispatch fact."

Confidence drops to 55% — the lowest of the three queries — because two critical data
sources (unit dispatch, rebid stack) are absent, and the remaining fuel mix source is
explicitly PLAU tier.

**④ Answer**

The `_plan_fuel_source` function routes here because `requested_output = fuel_source_recommendation`.
It formats:
1. The merit order ranking — from the fuel mix engine output, not generated
2. The spot price reference — from CONF dispatch data
3. The data tier caveat — from the `source_status` object
4. The explicit NOT-financial-advice disclaimer — hardcoded into the planner, never skippable

The answer does not say "coal is bad" or "hydro is best". It says: "the marginal-cost
model ranks hydro first at this spot price, with low confidence because unit dispatch
is missing." The system refuses to give a stronger answer than the evidence supports.

---

### Q3 — "Why did price fluctuate from 143→167→165→140? Which fuel source?"

**What the user wants to know:**
This is the most complex of the three queries because it asks for two things simultaneously:
(a) a causal explanation of a specific price sequence the user observed, and (b) a fuel
source attribution for that sequence. It is also the hardest query to answer because it
requires recent historical data that may not yet be in the database.

**Understanding what causes 143→167→165→140 in the NEM**

Before explaining how the system handles this, it helps to understand what genuinely causes
these kinds of multi-interval price movements in the Australian NEM:

1. **Generator rebidding**: Under AEMO's 5-minute dispatch intervals, any generator can
   rebid its price-quantity pairs between dispatch intervals. A gas peaker moving one unit
   from $0/MWh to $14,500/MWh (the market price cap) causes an immediate price spike at
   that interval. This is the most common cause of sharp, short-duration spikes. The
   recovery ($167→$165→$140) happens when the rebidder returns to its normal price or
   when demand drops slightly in the next interval.

2. **Demand surge across intervals**: NSW afternoon peaks (roughly 4–7pm AEST) push demand
   through successive generators in the merit order. If the 12:15 interval was at the gas
   CCGT threshold (~$143), the 12:20 interval pushed into an OCGT peaker (~$167), and by
   12:30 demand eased back below the OCGT threshold ($140). The fuel source causing $167
   would be the OCGT gas peaker, not coal.

3. **Interconnector congestion / flow reversal**: NSW can be a net importer (from QLD or VIC)
   or exporter. A sudden constraint on the QNI (Queensland–NSW) or VIC–NSW interconnector
   removes cheap interstate supply, forcing NSW to dispatch more expensive local generators.
   Flow reversal can cause sharp price movements that recover within 2–3 intervals.

4. **FCAS market interaction**: Frequency Control Ancillary Services can cause price
   movements that appear in dispatch price but are driven by the frequency regulation market,
   not energy dispatch itself.

5. **Battery BESS dispatch**: A large battery (like Waratah Super Battery) can set the price
   in a dispatch interval. BESS bids at its opportunity cost — when charging, it competes at
   the low end; when discharging, it may bid at near-VoLL prices if storage is valuable.
   The $167 spike may be a BESS discharge interval.

**The fuel source answer for a $140–$167 range:**
At this price range in NSW, the *price-setting generator* (the marginal unit that determines
the spot price) is almost certainly **CCGT gas or OCGT gas**. Solar and wind are below this
range in marginal cost. Coal (black coal in NSW) typically bids between $30–80/MWh — it
would be the price-setter at $80–100/MWh ranges but not typically at $140+. Hydro in NSW
(Snowy scheme) may or may not be dispatching depending on water storage and the operator's
opportunity cost model.

To confirm which fuel actually caused the $167 peak requires unit-level dispatch data —
the exact generator that was the marginal price-setter at 12:20, what fuel type it is,
and whether there was a rebid between 12:15 and 12:20.

**① Decompose**

The decomposer identifies:
- `intent = EXPLANATION` (the "why" framing + "what is causing")
- `requires_history = true` (triggered by "fluctuate", "back down", "143→167")
- `requested_output = price_fluctuation_attribution` (triggered by "fluctuate/fluctuation/
  back down" keyword match — the most specific routing path for this query type)

The `requires_history = true` flag tells the evidence gathering stage to query the
`recent_dispatch` database table for the last 70 minutes of dispatch intervals. This is
what would allow the system to reconstruct the actual price path from stored data rather
than relying on what the user described in the query text.

**② Gather**

- **Dispatch (`CONF`)**: Live current reading $140.25/MWh. This confirms the "back down to
  140" end-state. CONF tier — direct telemetry.

- **Query text price extraction (`QEXT` — a special tier)**: Because `recent_dispatch` is
  empty (the database has not yet accumulated enough interval history at this point in the
  platform's operation), the system falls back to reading the numbers the user themselves
  provided in the question. The `_extract_query_price_path()` function uses a regex to
  find numbers following price transition words ("from", "to", "back down to", "$"):
  "from 143 to 167 to 165 and then back down to 140" → `[143.0, 167.0, 165.0, 140.0]`.
  This is `QEXT` tier — the user's own words as evidence. It is real (the user witnessed
  this price sequence) but it is unverifiable from the database at this moment.

- **Recent dispatch DB (`EMPTY`)**: The query for the last 70 minutes of stored NSW1
  dispatch intervals returned zero rows. This is surfaced explicitly as a missing data item
  ("recent price trend from database") in the Missing section of the answer. The system
  does not pretend the data exists or estimate it.

- **Fuel mix model (`PLAU`)**: The same prior marginal-cost model from Q2 applies here.
  At the $143–167 range, gas (CCGT/OCGT) is the most plausible price-setter. PLAU because
  it is a prior model, not confirmed by unit dispatch.

- **Unit dispatch (`N/A`)**: Not available. This is the single piece of evidence that would
  answer "which fuel source" definitively. Without it, the system cannot confirm whether
  the $167 peak was caused by a gas peaker rebid, a BESS discharge, a demand surge,
  or an interconnector constraint.

**③ Reason**

The `WhyBuilder` runs the price-path analysis:
- Extracts the sequence [143, 167, 165, 140] from the QEXT evidence
- Computes the observed swing: max(167) − min(140) = **$27/MWh**
- Labels the swing: $27 is "minor price variation" (below $100 threshold for "moderate",
  below $500 for "significant")
- Notes the sequence shape: spike up ($143→$167), partial recovery ($167→$165),
  full recovery ($165→$140) — consistent with either a single-interval rebid event
  or a demand pulse that resolved

Confidence is 71%: CONF dispatch + QEXT query prices (weighted lower than DB dispatch)
offset by EMPTY recent DB, N/A unit dispatch. Slightly lower than Q1 because the price
path comes from query text rather than database verification.

The adversarial critic flags: "A $27/MWh swing across 4 intervals is within normal
intraday variation for elevated-regime NSW1. This may not represent a discrete market
event — it could be routine merit-order movement as demand crosses generation thresholds.
Treat as minor unless unit dispatch confirms a rebid or constraint event."

**④ Answer**

The `_plan_price_fluctuation` function routes here because
`requested_output = price_fluctuation_attribution`.

When `recent_dispatch` is empty, the planner acknowledges the user's described price path
directly rather than returning a bare snapshot:

> "You described a price path of $143 → $167 → $165 → $140/MWh; live reading is $140.25/MWh.
>  The observed swing of $27/MWh is consistent with a minor price variation event.
>  Confirmed driver and fuel-source attribution requires unit dispatch, bids/rebids,
>  and constraint evidence."

This is the key intelligence gap the system makes explicit: **it knows what it doesn't know**.
It cannot tell the user whether a gas peaker, a BESS, or an interconnector event caused
the $167 peak — but it tells the user exactly what data would be needed to answer that
question and why it is currently absent.

**On the fuel source question specifically:**
Based on the price range ($140–$167), the most likely price-setting fuel types in NSW1 are:
- **CCGT gas** (Combined Cycle Gas Turbine): Probable price-setter at $143 (baseline
  elevated price). Typical marginal cost $80–120/MWh, bids higher when capacity is tight.
- **OCGT gas peaker** (Open Cycle Gas Turbine): Most likely price-setter at $167 (the peak
  interval). OCGTs have higher fuel costs ($150–300/MWh) and are used only for short
  duration peaks — their presence in the dispatch stack would explain the $167 spike and
  fast recovery as demand drops back below their cost threshold.
- **BESS (Battery Energy Storage)**: May be the $167 price-setter if a large battery was
  discharging and bidding near its opportunity cost. BESS can move price rapidly because
  they can ramp instantaneously.

Solar and coal are NOT plausible price-setters at $167 in NSW1. Solar's marginal cost is
effectively zero — it never sets a high price. NSW black coal typically bids $30–80/MWh.
Neither would be the marginal generator at $167.

This is the answer the system would give if unit dispatch data were available. Without it,
it can only say: "based on the price range and marginal cost priors, OCGT gas or BESS is
the most likely cause of the $167 peak — but this is unconfirmed."

---

## The Confidence Numbers Are Not Decorative

A critical property of this architecture is that the confidence scores (73%, 55%, 71%)
are computed deterministically from the evidence tier mix, not estimated by a language model:

| Query | CONF sources | PLAU sources | UNCO/N/A sources | Confidence |
|-------|-------------|-------------|------------------|-----------|
| Q1    | Dispatch    | Analogs, TRAG | Weather, Notices | 73%      |
| Q2    | Dispatch    | Fuel mix    | Weather, Unit disp, Rebids | 55% |
| Q3    | Dispatch    | Fuel mix    | Unit disp, Recent DB (EMPTY) | 71% |

Q2 is the lowest because two categories of evidence that are directly relevant to the
question (unit dispatch, rebid stack) are absent. The system knows it cannot answer
"which fuel is best" with high confidence without knowing what is actually dispatching.
This degradation is not a limitation — it is the system working correctly.

---

## Why the Order Cannot Be Reversed

It is worth being explicit about why each stage must come before the next:

**You cannot Gather before you Decompose** because you would not know which region to fetch,
which time range to query, whether to run analogs or forecasts, or which fuel types to
price. The decomposition provides the fetch specification. Without it, every query would
need to fetch everything — expensive, slow, and full of irrelevant noise.

**You cannot Reason before you Gather** because the WhyBuilder has no evidence to reason
over. The causal logic is deterministic: it computes tiers and confidence from the structured
evidence bundle. That bundle must exist before reasoning begins.

**You cannot Answer before you Reason** because the AnswerPlanner formats outputs from the
WhyBuilder's computed claims. It has no independent knowledge. It is a formatter of
already-computed results. If it ran before reasoning, it would have nothing to format except
the raw query text — which is exactly how hallucination happens.

**The SecurityObserver must run at every stage** because the attack surface changes at each
stage. Input hygiene catches injection in the raw query. Decomposition hygiene catches
adversarial intent labels. Tool output hygiene catches anomalous market data or API
responses that could corrupt the reasoning layer. Answer hygiene catches any claim that
should not reach the user (price assertions, financial advice, out-of-scope verdicts).
A single final check would miss the intermediate attack vectors.

---

## What This Architecture Cannot Do (By Design)

- It cannot say "the $167 price was caused by coal" without a `CONF` evidence_ref from
  unit dispatch data showing a coal generator was the marginal price-setter.
- It cannot give a probability above 80% when a CONF-tier source is absent.
- It cannot skip the adversarial critic — every verdict has a counterargument whether or
  not it is flattering to the answer.
- It cannot answer "should I bid $X in the next interval" — such queries are blocked at
  Stage ① (SecurityObserver input pass, unsafe action detection).
- It cannot produce a number that is not in an `evidence_ref` object traceable to a
  specific data source, interval, and field.

These constraints are not guardrails added after the fact. They are structural properties
of the pipeline: the planner cannot invent numbers because it only reads from the evidence
bundle; the evidence bundle cannot have numbers without source refs; the security observer
blocks any answer that asserts unverified facts.

That is the architecture. Every answer is a claim audit trail.
