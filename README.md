# TRACE-Fin

Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting for Financial Returns.

This repository contains the research survey, gap analysis, paper proposal, and a runnable dependency-free implementation scaffold. The method is deliberately conservative: timestamp validity, uncertainty calibration, and transaction-cost-adjusted utility are first-class outcomes.

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

## Layout

- research/papers/ — paper notes
- research/matrix.csv — survey matrix
- research/output/ — gap analysis and proposal
- implementation/baseline/ — simple comparison models
- implementation/method/ — TRACE-Fin scaffold
- implementation/experiments/ — runnable experiments
- implementation/results/ — generated results
- docs/process.md — milestone plan and reproducibility gates
