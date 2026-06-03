# GridVerdict — 90-Second Demo Script

**Target audience:** LinkedIn video, GitHub README GIF, technical interview opener.  
**Setup:** App running at localhost:8000, DB warmed, LNN ✓ in top bar, at least 1 day of dispatch data.

---

## Pre-flight checklist (run before recording)

```bash
docker compose up -d
# Wait for: "LTC[NSW1] trained ... LTC[SA1] trained"
```

Browser tab open at `http://localhost:8000`.  
Confirm in top bar: `LIVE · NSW1 · LNN ✓`  
If LNN shows ✗: check `docker compose logs app | grep "LTC\["` — wait for training to complete (~30s).

---

## Scene 1 — Live dashboard (0:00–0:15)

**[screen shows live price swimlane chart]**

Say:
> "This is GridVerdict — a live NEM decision-support cockpit. The swimlane updates every 5 minutes from public NEMWeb data. NSW spot is [current price]. The dashed tail is the LNN probabilistic forecast — P10, P50, P90."

Point to the Evidence strip: `Dispatch · Notices · Analogs · LNN ✓`

---

## Scene 2 — Ask a question (0:15–0:45)

Type into the query box:
```
Why is NSW price elevated right now, and is it likely to continue?
```

**[answer card appears in ~1s]**

Say:
> "The language model decomposed the question. Deterministic code answered it. Watch what comes back."

Point to the answer card:
- **Answer** section: current price, demand vs availability headroom
- **Evidence** section: dispatch freshness timestamp, AEMO notice if active  
- **Drivers** section: supported vs unconfirmed (this is the key — it says *what it doesn't know*)
- **Continuation** section: model agreement across LEAR / QRA / LNN

Say:
> "It separates confirmed facts from unconfirmed drivers. The LLM never invented a price or asserted causality — that's all deterministic."

---

## Scene 3 — Decision Path (0:45–1:10)

Click **Decision Path** (collapsed audit table below the answer card).

Say:
> "Every step is traced — security pass, decomposition intent, scatter gather timing, evidence coverage. This is the full audit trail."

Point to the pipeline steps visible in the table: `SECURITY_INPUT → DECOMPOSE → SCATTER_GATHER → WHY_ENGINE → ANSWER`

Then click the **Data Provenance** panel:
> "Each source has a freshness timestamp and confidence. If AEMO data is stale, the confidence band drops automatically — the system tells you when to distrust it."

---

## Scene 4 — FCAS dashboard / NEM schematic (1:10–1:25)

*[optional, use if time permits — pick one or both]*

Click the **FCAS** tab in the viewport:
> "FCAS ancillary service prices — all 8 services, colour-coded raise/lower. Threshold lines at $50 and $200/MWh. Elevated FCAS means frequency stress — often precedes price spikes."

Or click **NEM Network**:
> "Circuit-diagram view of the NEM — current price, demand, and headroom per region. Interconnector arrows show live flow direction and % of transfer limit. SA to VIC through Heywood."

Or type:
```
Have we seen NSW conditions like this before?
```
> "HippoGraph finds the 10 closest historical market states by price, demand, headroom, and regime. It reports what happened next in each case."

---

## Scene 5 — Close (1:25–1:30)

Say:
> "Full test suite, Docker one-command start, reusable framework that maps to any time-series domain. Link in the description."

---

## Recording tips

- Use OBS or Loom at 1920×1080
- Font size: browser zoom at 110%
- Disable browser notifications
- Use a dark terminal if showing logs — the dark theme matches the app
- The query response is ~1s — no pause needed, it comes back fast
- If going live on LinkedIn: trim dead air in the 1s after submitting the query

---

## Alternate 60-second cut (for GIF / LinkedIn carousel)

1. **Scene 1** (0:00–0:10): swimlane chart, LIVE indicator, LNN ✓
2. **Scene 2** (0:10–0:45): type the question, show answer card with Answer/Evidence/Drivers
3. **Scene 5** (0:45–1:00): close

Cut everything else. The answer card alone is the strongest single moment.
