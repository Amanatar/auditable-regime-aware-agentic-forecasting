# TSFM.ai hosted-forecast benchmark

## Authentication and model

The supplied credential validated successfully against TSFM.ai in the earlier smoke run. The API catalog exposed Chronos-Bolt Tiny, TimesFM 2.0/2.5, and several Moirai variants; Chronos-Bolt Small returned model-access denial, so only Chronos-Bolt Tiny has a measured result so far. The key was supplied only through the TSFM_API_KEY process environment and is not stored in the repository or result files.

## Protocol

The completed run used the same Yahoo Finance daily-close table as the numeric experiment, with 8 expanding rolling origins per ticker, a 64-point context, a 3-trading-day horizon, and 10 bps trading-cost accounting. Results are exploratory because 8 origins is too small for a reliable significance claim. The evaluator now supports a 20-origin multi-model sweep and stores per-origin loss differences for correction, but that expanded sweep is not yet run in the current shell because TSFM_API_KEY is not present.

| Symbol | Chronos-Bolt MAE | Random-walk MAE | TSFM net Sharpe | DM p-value |
|---|---:|---:|---:|---:|
| AAPL | 3.5493 | 3.0387 | 10.36 | ~0.000 |
| AMZN | 6.9013 | 4.0512 | -2.21 | ~0.000 |
| GOOG | 4.0667 | 3.6613 | -8.74 | 0.202 |
| JPM | 2.4409 | 1.7100 | 3.58 | ~0.000 |
| META | 20.0093 | 6.7688 | -11.59 | <0.000001 |

The model loses to the random walk on all five MAE comparisons in this short window. The positive-looking AAPL/JPM trading figures are not evidence of generalizable alpha because the sample is small and the API model is being evaluated on a single recent slice.

## Reproduce

Set the key in the shell only:

    $env:TSFM_API_KEY = "your-key"
    python implementation/experiments/run_tsfm_api.py data/raw/prices_daily.csv amazon/chronos-bolt-tiny
    python implementation/experiments/run_tsfm_api_eval.py --prices data/raw/prices_daily.csv --model amazon/chronos-bolt-tiny --origins 8 --horizon 3

Outputs:

- implementation/results/tsfm_api_sample.json
- implementation/results/tsfm_api_eval.json
- implementation/results/tsfm_api_multi_eval.json (after the expanded sweep)
- implementation/results/multiple_testing.json (after the expanded sweep)

The expanded command is:

    python implementation/experiments/run_tsfm_api_eval.py --models amazon/chronos-bolt-tiny,google/timesfm-2.5-200m-pytorch,Salesforce/moirai-2.0-R-small --origins 20 --horizon 3
    python implementation/experiments/run_multiple_testing.py implementation/results/tsfm_api_multi_eval.json

After use, rotate the supplied API key because it was pasted into a chat transcript.
