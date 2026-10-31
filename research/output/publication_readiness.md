# Publication-readiness audit (2026-09-10)

## Completed in this pass

- **Leakage-safe learned evidence:** `run_learned_evidence.py` fits a seven-feature ridge model online. At each origin it uses only SEC rows with `available_at <= cutoff` and only earlier origins whose next-day return is already observed. The 60-origin output is `implementation/results/learned_evidence_results.json`; the cutoff audit reports zero failures for all five symbols.
- **Multiple testing:** `run_multiple_testing.py` applies Benjamini–Hochberg FDR across the 15 local method-by-symbol comparisons (numeric TRACE-Fin, lexical evidence, and learned evidence) and moving-block bootstrap intervals for the mean squared-loss difference. The output is `implementation/results/multiple_testing.json`.
- **Hosted TSFM protocol:** `run_tsfm_api_eval.py` now supports Chronos, TimesFM, and Moirai model IDs, retries transient failures, retains explicit model errors, and stores per-origin losses without persisting the API key. The completed expanded run covers 3 models × 5 tickers × 20 origins in `implementation/results/tsfm_api_multi_eval.json`.
- **Out-of-time holdout:** `run_holdout_experiment.py` evaluates ten assets on a final contiguous 20-origin, three-day holdout with HAC DM statistics. Mean MAE is 6.0447 for random walk, 7.4433 for numeric TRACE-Fin, and 7.9866 for lexical evidence; no comparison survives BH correction. See `research/output/holdout_report.md`.
- **Independent market windows:** `run_multi_window_holdout.py` evaluates three separated 60-origin windows (180 origins per asset). Mean MAE is 6.2390 for random walk, 7.0585 for numeric TRACE-Fin, and 7.1585 for lexical evidence; all 13 significant BH comparisons favor random walk. See `research/output/multi_window_holdout_report.md`.

## What the corrected local evidence says

The strongest uncorrected local p-values (AAPL and META numeric/evidence comparisons) do not survive the 15-test BH family: every q-value is above 0.05. The block-bootstrap intervals are also generally positive for those comparisons, meaning higher squared error than the random walk. The learned evidence ablation loses to the random walk in MAE on all five symbols in the current 60-origin sample; this is a negative result, not support for the hypothesis.

## Still required before calling the paper publication-ready

1. Increase the hosted sweep from 20 to at least 60 origins per ticker and add independent market origins before making a final claim. The completed 20-origin family is still an interim power check.
2. Replace the current three-document-per-symbol lexical sample (15 documents total) with a larger archived SEC/news corpus and pre-register the vocabulary, training window, and target horizon.
3. Add a fresh-data replication and pre-register the window selection; the three-window analysis is retrospective. SEC evidence is still absent for five of the ten assets.
4. Archive raw responses, environment versions, and a replay ledger, then write the final manuscript only after every cited URL and every reported result is re-run from a clean checkout.

The repository is therefore **research-complete as an auditable, runnable milestone**, but **not yet a publication-ready final paper** until the larger out-of-time validation and stronger evidence corpus are completed.
