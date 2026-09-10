# Independent multi-window holdout report

## Protocol

To test whether the final-window result was specific to one recent market episode, this run evaluates three separated final contiguous windows ending 2023-12-29, 2025-01-03, and 2026-09-10. Each window contains 60 rolling origins and a three-day horizon, giving 180 origins per asset and 1,800 asset-origin forecasts across the ten-asset price universe. No window is used for tuning another window. HAC Diebold–Mariano diagnostics use lag 2, and the complete 20-comparison method-by-symbol family is corrected with Benjamini–Hochberg at q=0.05.

The machine-readable outputs are [multi_window_holdout_results.json](../../implementation/results/multi_window_holdout_results.json) and [multi_window_holdout_multiple_testing.json](../../implementation/results/multi_window_holdout_multiple_testing.json).

## Results

Mean MAE pooled over the three windows and ten assets is:

| Method | Mean MAE | Assets beating random walk |
| --- | ---: | ---: |
| Random walk | 6.2390 | — |
| Numeric TRACE-Fin | 7.0585 | 0/10 |
| Lexical-evidence TRACE-Fin | 7.1585 | 0/10 |

Thirteen of the 20 BH-adjusted comparisons are significant at q<0.05. Every significant loss difference is positive (method squared error exceeds random walk), so the corrected result is evidence against the current scaffold improving forecasts, not evidence of a profitable signal. This is the appropriate negative-control outcome for the present method design.

## Coverage and interpretation

SEC evidence is retained for only AAPL, AMZN, GOOG, JPM, and META. MSFT, NVDA, SPY, QQQ, and TSLA therefore exercise the price-only fallback and should not be interpreted as evidence-channel generalization. The three-window protocol is substantially stronger than a single recent holdout, but it remains a retrospective public-data stress test rather than a preregistered live-trading study.

## Reproduction

```powershell
python implementation/experiments/run_multi_window_holdout.py --origins-per-window 60 --horizon 3
python implementation/experiments/run_multiple_testing.py implementation/results/multi_window_holdout_results.json implementation/results/multi_window_holdout_multiple_testing.json
```
