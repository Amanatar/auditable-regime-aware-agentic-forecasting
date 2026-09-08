# Real-data milestone report

## What was run

Daily close data for AAPL, AMZN, GOOG, JPM, and META were downloaded from the Yahoo Finance chart endpoint for 2020-01-01 through 2026-09-08. The raw table and retrieval manifest are stored under data/raw/. The experiment used 60 expanding-window rolling origins and a 3-trading-day horizon per ticker.

This milestone evaluates the numeric portion of TRACE-Fin only. The evidence agent, GDELT retrieval, FRED vintage alignment, SEC extension, transaction costs, and formal Diebold–Mariano tests are not yet enabled.

## Results

| Symbol | Random-walk MAE | TRACE-Fin numeric MAE | H1 interval coverage |
|---|---:|---:|---:|
| AAPL | 6.6093 | 7.4408 | 88.3% |
| AMZN | 5.9213 | 6.6038 | 88.3% |
| GOOG | 8.2890 | 8.4117 | 88.3% |
| JPM | 4.4584 | 4.1859 | 85.0% |
| META | 17.2501 | 21.5609 | 86.7% |

## Interpretation

The numeric-only TRACE-Fin scaffold does not beat the random walk consistently. It improves one ticker (JPM) and loses on four. That is not evidence for the research hypothesis; it is a useful falsification checkpoint showing that the proposed contribution cannot be justified by the current gate alone.

Coverage is close to the 85–95% target on this small sample, but intervals are based on a short rolling residual buffer and are not yet regime-conditional in a statistically powered test. The next method work should therefore focus on timestamp-valid evidence and a stronger regime detector, while retaining the random walk as the primary hurdle.

## Reproducibility

Run:

    python implementation/data/download_prices.py --output-dir data/raw --start 2020-01-01 --end 2026-09-08
    python implementation/experiments/run_real_experiment.py data/raw/prices_daily.csv

The result JSON is implementation/results/real_price_results.json. Yahoo Finance is used as a bootstrap source; a publication release should archive raw responses and replace it with a licensed or explicitly archived vendor feed.

## Economic and statistical diagnostics

The rolling evaluation now reports 10 bps transaction-cost-adjusted trading metrics and a dependency-free Diebold–Mariano approximation. The DM loss difference is method squared error minus random-walk squared error, so a positive statistic means the method lost to the random walk.

- AAPL: numeric TRACE-Fin annualized Sharpe -1.02; evidence variant -2.73; numeric DM p=0.034.
- AMZN: numeric Sharpe -2.85; evidence variant 0.67; neither DM comparison is below 0.05.
- GOOG: numeric Sharpe 0.31; evidence variant -0.52; neither DM comparison is significant.
- JPM: numeric Sharpe 1.52; evidence variant 0.61; numeric DM p=0.922.
- META: numeric Sharpe -1.57; evidence variant -1.66; numeric DM p=0.018.

These figures are exploratory: only 60 origins per ticker, no multiple-testing correction, and a simple long/short sign rule. They do not establish deployable alpha.

The optional Chronos checkpoint was not included in the headline table. Torch now imports successfully, but downloading amazon/chronos-t5-tiny from Hugging Face did not complete and no local checkpoint was cached. Reporting a TSFM number would therefore be fabricated.
