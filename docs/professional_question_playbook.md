# GridVerdict Professional Question Playbook

Purpose: turn real energy-market workflows into simple evidence-backed answer patterns. These questions should become decomposer evals, answer-format tests, and demo prompts.

## Simple / Entry-Level Questions

1. What is the current price in NSW?
Answer simply: price, interval time, stale/live status, source.
Evidence: AEMO dispatch price.

2. Is the current price high or normal?
Answer simply: current price, regime, recent percentile if meaningful.
Evidence: dispatch price, ChronoGraph regime state.

3. Has the price moved in the last hour?
Answer simply: now, 5m, 10m, 60m, net move.
Evidence: persisted dispatch history.

4. Is the data live?
Answer simply: last successful ingest, interval time, age, scheduler status.
Evidence: data status endpoint, source freshness.

5. Why is the app saying confidence is low?
Answer simply: list missing evidence and stale sources.
Evidence: evidence_quality, missing_data, claim verifier findings.

## Market Analyst Questions

6. Why is NSW price elevated right now?
Answer simply: current state, recent trajectory, confirmed/supporting/plausible drivers, missing causal data.
Evidence: dispatch, constraints, interconnectors, unit dispatch, notices, weather, analogs.

7. What changed between the last two dispatch intervals?
Answer simply: price delta, demand delta, headroom delta, regime change.
Evidence: latest two dispatch rows.

8. Was demand the main driver?
Answer simply: demand level and delta; do not say cause unless supported by headroom/constraints/unit data.
Evidence: dispatch demand, weather, historical demand analogs.

9. Was low supply/headroom the main driver?
Answer simply: availability, demand, headroom, headroom delta.
Evidence: dispatch availability and demand.

10. Was a constraint binding?
Answer simply: constraint ID, marginal value, affected region/flow if available.
Evidence: DISPATCHCONSTRAINT / market_driver_events.

11. Did interconnector congestion matter?
Answer simply: flow, limit/proximity, direction, affected neighboring region.
Evidence: DISPATCHINTERCONNECTORRES / market_driver_events.

12. Did any unit trip or withdraw capacity?
Answer simply: DUID, MW change, timing, fuel/participant if known.
Evidence: unit_dispatch_events and outage detector.

13. Did rebidding contribute?
Answer simply: changed DUIDs, MW withdrawn from cheap bands, timing before spike.
Evidence: BIDDAYOFFER/BIDPEROFFER via rebid engine.

14. Are FCAS markets showing stress?
Answer simply: highest FCAS service price, threshold crossing, whether energy price also moved.
Evidence: fcas_price_events.

15. Are AEMO notices relevant?
Answer simply: notice type, time, credibility tier, region relevance.
Evidence: AEMO market notices.

## Forecast / Risk Questions

16. Is the high price likely to continue?
Answer simply: LEAR says X, QRA says Y, LNN says Z/unavailable, AEMO predispatch says W; conclude model agreement/disagreement.
Evidence: live_quantile_forecast, predispatch.

17. What is the next 30-minute price range?
Answer simply: model P10/P50/P90 bands and primary model.
Evidence: LEAR/QRA/LNN/meta-ensemble forecast output.

18. Are models agreeing or disagreeing?
Answer simply: compare direction and P50/P90 spread across models.
Evidence: per-model forecast detail.

19. How trustworthy is the forecast today?
Answer simply: recent CRPS, calibration error, spike recall, skill vs persistence/AEMO.
Evidence: forecast trust endpoint and model registry.

20. What would make the forecast wrong?
Answer simply: missing constraints, rebids, outages, weather forecast error, stale source.
Evidence: missing_data, claim verifier, data status.

21. What should I watch next?
Answer simply: headroom threshold, price threshold, notice/constraint/rebid trigger, model P90.
Evidence: next_watch.

## Historical / HippoGraph Questions

22. Have we seen similar price/headroom conditions before?
Answer simply: analog count, top 3 analogs, outcome split.
Evidence: HippoGraph PPR analogs.

23. What happened 30 minutes after similar events?
Answer simply: recovered/continued/unknown counts and examples.
Evidence: analog outcome labels.

24. Are these analogs good matches?
Answer simply: price delta, demand delta, headroom delta, regime match, driver similarity.
Evidence: analog match_reason and quality_score.

25. Which historical period looks most similar?
Answer simply: timestamp, price, demand, headroom, outcome.
Evidence: top analog.

26. Did similar events have the same driver?
Answer simply: compare driver tags; say unconfirmed if driver data missing.
Evidence: reranked analog driver_types.

## Weather / News / External Context

27. Is hot weather increasing demand?
Answer simply: temperature, demand level/delta, historical sensitivity if available; avoid causal certainty if only weather exists.
Evidence: weather consensus and dispatch demand.

28. Is low wind or cloud cover affecting renewable output?
Answer simply: weather reading, renewable dispatch/curtailment if available.
Evidence: weather consensus, unit dispatch by fuel.

29. Are RSS/news items relevant to this price move?
Answer simply: titles that match keywords, explain contextual-only status.
Evidence: NEM news RSS.

30. Are weather sources agreeing?
Answer simply: source count, confidence, spread/consensus.
Evidence: weather MCP consensus.

## Battery / Portfolio Questions

31. Should I dispatch my battery now?
Answer simply: action, dispatch MW/MWh, net value, missing-before-action list.
Evidence: BESS scenario engine, market snapshot, forecast, FCAS.

32. Should I hold for a higher price?
Answer simply: current price vs forecast P50/P90 and SOC constraints.
Evidence: BESS policy + forecast.

33. Should I reserve capacity for FCAS?
Answer simply: FCAS tightness and opportunity value vs energy dispatch.
Evidence: FCAS prices and BESS economics.

34. What is the risk of dispatching now?
Answer simply: stale data, forecast disagreement, missing portfolio/contract constraints.
Evidence: evidence_quality and BESS missing_before_action.

## Operational / Trust Questions

35. What sources are stale right now?
Answer simply: source, last success, age, consequence.
Evidence: /api/data/status.

36. Did the scheduler fail?
Answer simply: job name, consecutive failures, last error.
Evidence: scheduler job health and metrics.

37. Can I replay the evidence for this answer?
Answer simply: trace ID and bitemporal evidence known before query time.
Evidence: trace, TemporalRAG valid_time/system_time.

38. Which claims are confirmed vs plausible?
Answer simply: claim_map grouped by tier.
Evidence: claim verifier and driver tiers.

39. Is the system guessing?
Answer simply: show missing data and downgrade status.
Evidence: claim verifier, missing_data, evidence_quality.

40. What did the system know at the time?
Answer simply: valid_time, system_time, prior/post badges.
Evidence: TemporalRAG.

## Technical / Expert Questions

41. How does GridVerdict decompose my query?
Answer simply: intent, region, required sources, causal targets, output contract.
Evidence: decomposition object.

42. Why does the LLM not produce numbers?
Answer simply: LLM routes; tools and DB produce facts; verifier checks claims.
Evidence: architecture contract.

43. How does HippoGraph find analogs?
Answer simply: state vector, similarity edges, PPR, outcome lookahead.
Evidence: analog metadata.

44. How do LEAR, QRA, and LNN differ?
Answer simply: LEAR linear autoregression; QRA ensemble combiner; LNN sequence/regime specialist.
Evidence: model registry and model cards.

45. What prevents hallucinated causal claims?
Answer simply: driver tiers, evidence refs, claim verifier downgrades.
Evidence: claim_map and verifier findings.

46. How is stale data handled?
Answer simply: freshness affects confidence, labels, missing data, and source status.
Evidence: data status and evidence manifest.

47. What happens if RSS/news includes malicious text?
Answer simply: untrusted evidence, tool-output observer, title/link citation only.
Evidence: SecurityObserver events.

48. How do you avoid duplicate schedulers?
Answer simply: Redis leader lock and heartbeat; fallback single-process behavior.
Evidence: metrics and scheduler state.

49. How do you prove a forecast is useful?
Answer simply: walk-forward backtest, CRPS, pinball, calibration, spike recall, skill vs persistence/AEMO.
Evidence: forecast trust panel.

50. What are the biggest blockers before professional use?
Answer simply: live freshness, constraints, interconnectors, unit dispatch, rebids, FCAS, portfolio state, model calibration.
Evidence: data status and missing-data flags.

## Sources Used To Shape This Playbook

- AEMO NEMWeb current reports are the live source for dispatch, predispatch, and market notices:
  https://nemweb.com.au/Reports/Current/
- AEMO explains NEM operation and dispatch/process concepts:
  https://aemo.com.au/en/energy-systems/electricity/national-electricity-market-nem
- AER wholesale market monitoring and reporting highlights price events, rebidding, network constraints, and forecast variance:
  https://www.aer.gov.au/industry/registers/resources/reports/wholesale-markets-quarterly
- AEMC explains NEM dispatch and market design context:
  https://www.aemc.gov.au/energy-system/electricity/electricity-system/NEM
- WattClarity analysis repeatedly inspects demand, outages, constraints, interconnectors, rebidding, predispatch, and weather-driven uncertainty:
  https://wattclarity.com.au/
- Montel EnAppSys is an example of a commercial short-term market analytics platform with forecasts, weather, interconnectors, alerts, and trader workflows:
  https://montel.energy/platforms/enappsys
