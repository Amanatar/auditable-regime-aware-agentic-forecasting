# TRACE-Fin

Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting for Financial Returns.

This repository contains the research survey, gap analysis, paper proposal, and a runnable dependency-free implementation scaffold. The method is deliberately conservative: timestamp validity, uncertainty calibration, and transaction-cost-adjusted utility are first-class outcomes.

## Quick start

From the repository root, run:

    python implementation/experiments/run_experiment.py

The smoke run writes implementation/results/smoke_results.json and exercises both the random-walk baseline and the TRACE-Fin method end-to-end on deterministic synthetic data.

## Layout

- research/papers/ — paper notes
- research/matrix.csv — survey matrix
- research/output/ — gap analysis and proposal
- implementation/baseline/ — simple comparison models
- implementation/method/ — TRACE-Fin scaffold
- implementation/experiments/ — runnable experiments
- implementation/results/ — generated results
- docs/process.md — milestone plan and reproducibility gates
