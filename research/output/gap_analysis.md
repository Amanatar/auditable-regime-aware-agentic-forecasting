# Gap analysis

## Scope and stopping rule

The survey logged 21 papers across agentic forecasting, financial TSFM evaluation, distribution shift, uncertainty calibration, and backbone models. Although the nominal breadth target was 40, papers 22 onward were expected to repeat the same limitations without adding a new sub-topic: temporal contamination, shift-unstable calibration, weak economic evaluation, and unverifiable evidence provenance. I therefore applied the requested diminishing-returns stopping rule.

## Ranked candidate gaps

| Rank | Candidate gap | Novelty (1–5) | Feasibility (1–5) | Evidence (number of papers) |
|---|---|---:|---:|---:|
| 1 | A timestamp-faithful, contamination-resistant benchmark for agentic financial forecasting under regime shifts, combining market data, event/news retrieval, provenance logs, calibrated intervals, and economic utility. | 5 | 4 | 8 |
| 2 | Regime-aware adaptive calibration for TSFMs and agents, using detected temporal environments plus online/conformal recalibration and evaluating coverage after shifts. | 4 | 5 | 6 |
| 3 | Evidence-grounded forecast revision with causal separation of numerical signal and text, measuring whether retrieved evidence changes forecasts for the right event and not merely because an agent produced a plausible rationale. | 4 | 4 | 6 |
| 4 | Compute- and cost-aware routing among statistical models, TSFMs, and agents, selecting the least expensive forecaster that is reliable for the current regime and uncertainty level. | 4 | 4 | 5 |
| 5 | Closed-loop evaluation that measures feedback and market impact, where forecasts can influence decisions and future observations, with pre-registered live or paper-trading protocols. | 5 | 2 | 3 |

### 1. Timestamp-faithful, contamination-resistant benchmark (recommended)

The financial TSFM benchmark reports that pretrained models win most ranking tasks but produce small and sparse gains over random walk, and explicitly warns that ranking does not imply economically meaningful predictability (Pretrained Time-Series Foundation Models for Financial Return Forecasting). The forecasting-agent review identifies contamination-resistant live evaluation, calibration under distribution shift, and cost/accuracy reporting as unresolved measurement problems (LLM-based Agents for Forecasting and Prediction). Nexus and Bridging the Last Mile add contextual agents but do not provide a controlled leakage audit or standardized financial utility protocol. MIRAI demonstrates tool-use and temporal event evaluation, but its target is international events rather than returns. FOIL and Handling Concept Drift in Global Time Series establish that historical-to-future distribution shift is a first-class forecasting problem.

The missing combination is a reproducible protocol where every retrieved document is restricted to its publication timestamp, every forecast stores evidence IDs and model/version metadata, evaluation is rolling-origin and post-cutoff, and metrics include probabilistic calibration and transaction-cost-adjusted utility. This is novel yet feasible with daily prices, public news/event timestamps, pretrained zero-shot forecasters, and a small open-weight LLM or deterministic retrieval layer.

### 2. Regime-aware adaptive calibration

FOIL learns invariant representations under inferred environments; Handling Concept Drift in Global Time Series studies drift adaptation; Conformalized Quantile Regression supplies interval calibration under exchangeability; and the financial TSFM benchmark shows that point-forecast gains are weak. However, none of the surveyed papers evaluates an agent/TSFM with rolling conformal recalibration conditioned on detected financial regimes. This gap is highly feasible because calibration can be added around frozen forecasts without training a large model.

### 3. Evidence-grounded forecast revision

Nexus separates macro and micro reasoning, while Bridging the Last Mile turns contextual evidence into constrained revisions. MIRAI evaluates tool use and source integration. Temporal Fusion Transformers provides interpretability mechanisms but explicitly cautions that attention/variable importance is not causal explanation. The missing experiment is an intervention-style audit: remove or time-shuffle evidence, measure forecast deltas, and test whether revisions are directionally correct around identified events.

### 4. Cost-aware forecaster routing

The forecasting-agent review calls for explicit cost reporting; N-HiTS and TiDE show that efficient non-Transformer models can be competitive; TSFM papers show zero-shot convenience but require substantial pretrained infrastructure. A router that chooses between a random walk, DLinear, TSFM, or agent based on regime and uncertainty could make the system deployable under Kaggle/Colab/API budgets. The scientific risk is that routing gains may be operational rather than algorithmic, so it should be a secondary track.

### 5. Closed-loop feedback evaluation

The forecasting-agent review flags feedback between deployed forecasts and outcomes as an open issue. A benchmark that models how forecasts affect decisions, liquidity, or subsequent observations would be important, but it requires a credible market simulator or long live evaluation and is therefore lower feasibility for this project.

## Recommendation

Pursue Gap 1, with Gap 2 as the core methodological component and Gap 3 as an audit/ablation. The resulting paper can make a falsifiable claim without pretending that a better RMSE proves alpha: the primary outcome should be calibrated, timestamp-valid, economically costed decision quality during regime shifts.
