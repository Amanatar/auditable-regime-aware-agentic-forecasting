# SEC evidence-audit milestone

The SEC submissions API produced 348 timestamped records across AAPL, AMZN, GOOG, JPM, and META for forms 8-K, 10-K, and 10-Q. Each record stores available_at, accession number, primary document, and a source URL.

The cutoff audit sampled 10–11 historical cutoffs per ticker. Every accepted evidence set passed the no-future check. The real-price rolling experiment also attached up to 20 accession IDs to each forecast ledger.

This is an evidence-provenance result, not an accuracy result. The current numeric method assigns no directional signal to filings, so the MAE numbers remain the numeric-only results in real_data_report.md. The next scientific step is to parse filing text into preregistered event features and test them against a placebo/shuffled-evidence control.

The GDELT news connector remains optional and is not included in the historical headline result because the public DOC endpoint's rolling coverage window does not provide a clean six-year backtest archive.
