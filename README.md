# TRACE-Fin

Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting for Financial Returns.

This repository contains the research survey, gap analysis, paper proposal, and a runnable implementation scaffold. The method is deliberately conservative: timestamp validity, uncertainty calibration, and transaction-cost-adjusted utility are first-class outcomes. The current evidence is a reproducible research milestone, not a claim of publishable alpha.

## Quick start

From the repository root, run:

    python implementation/experiments/run_experiment.py

The smoke run writes implementation/results/smoke_results.json and exercises both the random-walk baseline and the TRACE-Fin method end-to-end on deterministic synthetic data.

## Real-data bootstrap

Download daily prices and run the first rolling-origin evaluation:

    python implementation/data/download_prices.py --output-dir data/raw --start 2020-01-01 --end 2026-09-08
    python implementation/experiments/run_real_experiment.py data/raw/prices_daily.csv

The measured results are documented in research/output/real_data_report.md. The first run is intentionally numeric-only; timestamp-safe GDELT/FRED/SEC retrieval is the next research milestone.

## TSFM.ai API benchmark

The project also supports hosted Chronos inference through TSFM.ai. Keep the credential in TSFM_API_KEY and run:

    python implementation/experiments/run_tsfm_api.py data/raw/prices_daily.csv amazon/chronos-bolt-tiny
    python implementation/experiments/run_tsfm_api_eval.py --prices data/raw/prices_daily.csv --model amazon/chronos-bolt-tiny --origins 8 --horizon 3

See research/output/tsfm_api_report.md. No API key is stored in this repository.

## Publication-strengthening runs

Run the leakage-safe learned SEC evidence ablation and the pre-specified
multiple-testing correction locally:

    python implementation/experiments/run_learned_evidence.py data/raw/prices_daily.csv data/raw/sec_features.jsonl 60
    python implementation/experiments/run_multiple_testing.py implementation/results/real_price_results.json

For hosted TSFM evaluation, set `TSFM_API_KEY` only in the process environment
and run the multi-model evaluator. It defaults to Chronos-Bolt Tiny, TimesFM
2.5 200M, and Moirai 2.0 Small over 20 origins per ticker:

    python implementation/experiments/run_tsfm_api_eval.py --origins 20 --horizon 3
    python implementation/experiments/run_multiple_testing.py implementation/results/tsfm_api_multi_eval.json

Run the expanded ten-asset final contiguous holdout and its HAC/FDR audit:

    python implementation/data/download_prices.py --output-dir data/raw_expanded --start 2020-01-01 --end 2026-09-10 --symbols AAPL AMZN GOOG JPM META MSFT NVDA TSLA SPY QQQ
    python implementation/experiments/run_holdout_experiment.py --prices data/raw_expanded/prices_daily.csv --holdout-origins 20 --horizon 3
    python implementation/experiments/run_multiple_testing.py implementation/results/holdout_results.json implementation/results/holdout_multiple_testing.json

The holdout report is `research/output/holdout_report.md`. SEC evidence is currently
available only for the original five symbols; the added five assets are a price-only
stress test and do not establish cross-asset evidence generalization.

Unavailable hosted models are retained as explicit errors; no result is
silently imputed. The key is never written to JSON, logs, or Git.

## Layout

- research/papers/ — paper notes
- research/matrix.csv — survey matrix
- research/output/ — gap analysis and proposal
- implementation/baseline/ — simple comparison models
- implementation/method/ — TRACE-Fin scaffold
- implementation/experiments/ — runnable experiments
- implementation/results/ — generated results
- docs/process.md — milestone plan and reproducibility gates
