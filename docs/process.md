# TRACE-Fin process plan

## Compute and operating assumptions

The first pass is designed for a Kaggle kernel or Colab-tier CPU/single-GPU session. The smoke experiment is dependency-free and runs locally. Full experiments should use small pretrained TSFM checkpoints, cached data, and one fold at a time; no GPU cluster is assumed. Unattended sweeps belong in Kaggle, while interactive debugging belongs in Colab with a persistent terminal/tmux session.

## Milestones

1. **Data preparation (1–2 days):** download daily prices, FRED/ALFRED release-aware series, GDELT/SEC records; normalize publication timestamps; create checksummed manifests; implement a cutoff validator.
2. **Baseline reproduction (1 day):** run random walk, DLinear, and iTransformer/Chronos baselines on identical rolling-origin folds; save predictions, metrics, and environment metadata.
3. **TRACE-Fin method (2–3 days):** implement online regime features, timestamp-safe retrieval, constrained evidence schema, gated ensemble, and conformal calibration.
4. **Ablations (1–2 days):** remove regime detector, evidence, cutoff restriction, conformal layer, and replace evidence with shuffled/placebo snippets.
5. **Results and stress tests (2 days):** evaluate calm versus shift windows, compute calibration and cost-adjusted utility, run Diebold–Mariano and block-bootstrap tests, and inspect ledger replay failures.
6. **Write-up (2–3 days):** document protocol, limitations, negative results, data licenses, reproducibility commands, and a table of all pre-registered metrics.

## Reproducibility gates

- Every forecast has a cutoff timestamp and input/evidence hash.
- No training or calibration row may occur after the forecast cutoff.
- Every baseline and ablation uses the same folds and transaction-cost assumptions.
- Results are written as JSON/CSV plus the exact configuration and package versions.
- A replay checker must reconstruct at least 99% of sampled forecasts before a result is considered publishable.

## Immediate next milestone

Replace the synthetic loader with a small timestamped price CSV and run the same smoke protocol on three tickers for 30 rolling origins. Do not add an LLM dependency until the cutoff validator and ledger replay test pass.
