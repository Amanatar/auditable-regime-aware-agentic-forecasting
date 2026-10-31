# Paper proposal

## Working title

**TRACE-Fin: Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting for Financial Returns**

## Research question and hypothesis

**Research question:** Can a forecasting system that enforces timestamp-valid evidence retrieval, detects market regimes, and calibrates uncertainty online improve decision quality during regime shifts without relying on information unavailable at forecast time?

**Primary hypothesis:** Relative to numeric TSFMs and unconstrained retrieval-augmented agents, TRACE-Fin will improve post-cost directional utility and interval coverage specifically in detected regime-shift windows, while producing no systematic advantage in calm periods.

The claim is intentionally narrower than “agents create alpha.” The paper will test whether auditability and regime-conditioned calibration make forecasts more trustworthy and economically useful.

## Proposed method

TRACE-Fin is a four-stage pipeline:

1. **Timestamp-safe data layer.** At forecast cutoff t, construct a feature record containing prices and macro observations whose publication/release time is ≤ t. Retrieve news/event documents with publication time ≤ t and store their stable URL, timestamp, query, and content hash. No future revisions are silently substituted: macro series use vintage/release timestamps where available, and every row carries an availability timestamp.

2. **Regime detector.** Fit an online Bayesian change-point detector to market volatility, cross-asset correlation, drawdown, and macro-surprise features. The detector emits a soft regime vector (calm, trend, high-volatility, event-shock) and a shift probability. A rolling-origin protocol prevents the detector from seeing future labels.

3. **Forecast ensemble and evidence agent.**
   - Numeric experts: random walk, DLinear, iTransformer, and one pretrained TSFM (Chronos or Moirai).
   - Evidence expert: a small open-weight instruction model receives only timestamp-valid retrieved snippets and structured event fields; it outputs an event direction, horizon relevance, confidence, and cited evidence IDs, not a free-form unconstrained price target.
   - Gated combiner: a lightweight ridge/MLP gate uses regime state, forecast disagreement, evidence confidence, and recent calibration to weight numeric and evidence experts. The gate is trained only on past folds.

4. **Adaptive calibration and audit ledger.** Apply rolling conformal residual calibration to the ensemble interval, with regime-conditional residual pools when there are enough historical examples. Persist one JSON ledger per forecast: cutoff time, model versions, input hashes, retrieved document IDs, forecast distribution, regime state, calibration set, and final decision. An evaluator can replay the forecast and verify that no document arrived after the cutoff.

The key ablations are: no regime detector; no evidence; unrestricted retrieval; no conformal recalibration; fixed equal-weight ensemble; and a placebo evidence shuffle. The placebo should remove any apparent benefit if the agent is merely producing persuasive rationales.

## Required data and acquisition

The initial study will use five liquid U.S. equities (AAPL, AMZN, GOOG, JPM, META) to align with the nearest financial TSFM benchmark, with an optional out-of-sample ETF extension.

- **Prices and corporate actions:** daily OHLCV from a reproducible public market-data source; downloaded once and checksummed.
- **Macro releases:** FRED/ALFRED series such as VIX, 3-month Treasury, CPI, unemployment, and credit spreads. FRED documents series observations and release/vintage access through its API: https://fred.stlouisfed.org/docs/api/fred/fred/
- **News and events:** GDELT DOC 2.0 / Event data, retaining publication timestamps, source URLs, and query logs. Official API documentation: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ and https://gdeltproject.org/data.html
- **Company disclosures (extension):** SEC EDGAR submissions and XBRL data through data.sec.gov; official developer guidance: https://www.sec.gov/about/developer-resources

The minimum viable dataset is daily prices plus GDELT; FRED and SEC features are controlled extensions. All data acquisition scripts will save raw responses, timestamps, hashes, and a manifest so the benchmark can be rerun.

## Evaluation protocol

Use expanding-window rolling-origin evaluation. Each fold has a training interval, a validation interval for gate/calibration selection, and a strictly later test interval. Report results separately for:

- all days;
- the first 5 and 20 trading days after a detected regime shift;
- high-volatility versus calm regimes;
- event days with at least one timestamp-valid document.

Primary metrics:

- point accuracy: MAE, RMSE, and directional accuracy;
- probabilistic quality: CRPS, weighted interval score, empirical coverage, interval width, and calibration slope;
- statistical tests: Diebold–Mariano against random walk and block-bootstrap confidence intervals;
- economic utility: net return after declared transaction costs, annualized Sharpe, turnover, maximum drawdown, hit rate, and certainty-equivalent return;
- audit quality: percentage of forecasts replayable from the ledger, percentage of retrieved documents passing cutoff validation, and evidence-ablation sensitivity.

The headline result is a preregistered composite: calibrated net utility during post-shift windows, with a secondary constraint that 90% intervals achieve 85–95% empirical coverage. No single favorable RMSE table will be treated as evidence of alpha.

## Baselines

Nearest matrix papers and baselines:

1. **Pretrained Time-Series Foundation Models for Financial Return Forecasting** — direct financial benchmark and random-walk comparison: https://arxiv.org/abs/2606.27100
2. **Nexus** — closest multi-agent contextual forecasting comparison: https://arxiv.org/abs/2605.14389
3. **Bridging the Last Mile of Time Series Forecasting with LLM Agents** — closest forecast-revision comparison: https://arxiv.org/abs/2606.02497
4. **Chronos** — probabilistic TSFM: https://arxiv.org/abs/2403.07815
5. **Unified Training of Universal Time Series Forecasting Transformers (Moirai)** — universal probabilistic TSFM: https://arxiv.org/abs/2402.02592
6. **iTransformer** and **DLinear** — strong supervised numeric baselines: https://arxiv.org/abs/2310.06625 and https://arxiv.org/abs/2205.13504
7. **FOIL** — regime/OOD adaptation comparator: https://arxiv.org/abs/2406.09130

The fair comparison fixes the same price history, forecast horizon, fold boundaries, and information cutoff for every method. Agent baselines receive the same retrieval corpus but are not given the proposed ledger constraints unless that is the ablation being tested.

## Novelty statement

The closest existing papers each solve only part of the problem: financial TSFM work evaluates numeric zero-shot models; Nexus and Last Mile add contextual agent decomposition or revision; FOIL addresses OOD forecasting; and conformal prediction supplies generic interval calibration. TRACE-Fin is different because it combines all four into a replayable financial protocol: timestamp-restricted retrieval, explicit regime-conditioned gating, online conformal calibration, and an evidence ledger evaluated with post-cost utility and shift-window analysis. The novelty is the auditable evaluation-and-control interface, not a claim that a larger language model is intrinsically better.

## Expected contribution

This project should contribute a reproducible benchmark and a lightweight forecasting method that makes the reliability failures of agentic financial forecasting measurable. If the hypothesis is supported, the result will show that gains concentrate in regime-shift windows and come from calibrated, timestamp-valid evidence integration rather than generic reasoning text. If it is not supported, the negative result is still valuable: it would demonstrate that agentic context does not improve economically meaningful, calibrated forecasts once leakage and costs are controlled. The released ledger schema, replay checker, fold definitions, and ablations would make subsequent agentic forecasting papers easier to audit.
