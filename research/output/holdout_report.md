# Final contiguous holdout report

## Protocol

This stress test evaluates the local TRACE-Fin scaffold on a final contiguous window that was not used for feature design. Prices were downloaded with the existing Yahoo bootstrap downloader for ten symbols (AAPL, AMZN, GOOG, JPM, META, MSFT, NVDA, SPY, QQQ, TSLA), using daily closes from 2020-01-01 through 2026-09-10. Each symbol uses 20 rolling origins and a three-day forecast horizon. The comparator is a random-walk forecast. Diebold–Mariano diagnostics use a Newey–West/Bartlett HAC variance with lag 2, matching the three-day horizon.

The machine-readable output is [holdout_results.json](../../implementation/results/holdout_results.json). Multiple-testing correction over the 20 method-by-symbol comparisons is in [holdout_multiple_testing.json](../../implementation/results/holdout_multiple_testing.json), using Benjamini–Hochberg at q=0.05 and moving-block bootstrap confidence intervals.

## Results

Across the ten-symbol price universe, mean MAE was 6.0447 for random walk, 7.4433 for the numeric TRACE-Fin variant, and 7.9866 for the lexical-evidence variant. Numeric TRACE-Fin beat random walk on 1/10 symbols (JPM); lexical evidence beat it on 0/10. After Benjamini–Hochberg correction, no comparison was significant at q<0.05.

These results are a useful negative control: the current small-sample scaffold does not support a claim that either local variant improves forecasting out of time. The final holdout is a stronger audit than the expanding-window development result, but it is still only 20 origins per symbol and therefore not a publication-scale power analysis.

## Evidence-coverage limitation

SEC evidence rows are available only for the original five symbols (AAPL, AMZN, GOOG, JPM, META). The five added assets (MSFT, NVDA, SPY, QQQ, TSLA) have no retained SEC rows because the live SEC endpoint was unavailable during collection; their lexical-evidence forecast therefore falls back to the numeric signal. The expanded universe is consequently a price-universe stress test, not evidence that the textual channel generalizes across all ten assets.

## Reproduction

```powershell
python implementation/experiments/run_holdout_experiment.py --prices data/raw_expanded/prices_daily.csv --holdout-origins 20 --horizon 3
python implementation/experiments/run_multiple_testing.py implementation/results/holdout_results.json implementation/results/holdout_multiple_testing.json
```
