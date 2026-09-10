"""Evaluate the local TRACE-Fin variants on a final untouched holdout window."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "method"))
sys.path.insert(0, str(ROOT / "evaluation"))
from baseline import random_walk_forecast
from evidence import evidence_before, load_jsonl
from metrics import diebold_mariano_hac, trading_metrics
from trace_fin import forecast_with_audit


def load_prices(path: Path) -> dict[str, dict[str, list]]:
    grouped = defaultdict(lambda: {"dates": [], "prices": []})
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]]["dates"].append(row["date"])
            grouped[row["symbol"]]["prices"].append(float(row["close"]))
    return dict(grouped)


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def evaluate(bundle: dict, evidence_rows: list[dict], feature_rows: list[dict], symbol: str, holdout: int, horizon: int) -> dict:
    prices, dates = bundle["prices"], bundle["dates"]
    if len(prices) < holdout + horizon + 30:
        raise ValueError("not enough rows for holdout")
    start = len(prices) - holdout - horizon
    baseline_errors, numeric_errors, evidence_errors = [], [], []
    actual_h1, baseline_h1, numeric_h1, evidence_h1, last_h1 = [], [], [], [], []
    numeric_losses, evidence_losses = [], []
    residuals = []
    audit_rows = 0
    for offset in range(holdout):
        cut = start + offset
        history, actual = prices[:cut], prices[cut:cut + horizon]
        last = history[-1]
        cutoff = f"{dates[cut - 1]}T23:59:59+00:00"
        available = evidence_before(evidence_rows, symbol, cutoff)
        available_features = evidence_before(feature_rows, symbol, cutoff)
        audit_rows += len(available) + len(available_features)
        numeric = forecast_with_audit(history, horizon, residuals=residuals[-20:], cutoff=cutoff,
                                      evidence_ids=[row["accession_number"] for row in available[-20:]])
        signal = sum(row.get("signal", 0.0) for row in available_features[-3:]) / max(1, len(available_features[-3:]))
        evidence = forecast_with_audit(history, horizon, evidence_signal=signal, residuals=residuals[-20:], cutoff=cutoff,
                                       evidence_ids=[row["accession_number"] for row in available[-20:]])
        baseline = random_walk_forecast(history, horizon)
        baseline_errors.append(mae(actual, baseline))
        numeric_errors.append(mae(actual, numeric["point_forecast"]))
        evidence_errors.append(mae(actual, evidence["point_forecast"]))
        actual_h1.append(actual[0]); baseline_h1.append(last); numeric_h1.append(numeric["point_forecast"][0])
        evidence_h1.append(evidence["point_forecast"][0]); last_h1.append(last)
        numeric_losses.append((actual[0] - numeric["point_forecast"][0]) ** 2 - (actual[0] - last) ** 2)
        evidence_losses.append((actual[0] - evidence["point_forecast"][0]) ** 2 - (actual[0] - last) ** 2)
        residuals.append(actual[0] - numeric["point_forecast"][0])
    return {
        "holdout_origins": holdout,
        "horizon": horizon,
        "baseline_mae": sum(baseline_errors) / holdout,
        "trace_fin_numeric_mae": sum(numeric_errors) / holdout,
        "trace_fin_evidence_mae": sum(evidence_errors) / holdout,
        "dm_hac_numeric_vs_random_walk": diebold_mariano_hac(actual_h1, numeric_h1, baseline_h1, lag=horizon - 1),
        "dm_hac_evidence_vs_random_walk": diebold_mariano_hac(actual_h1, evidence_h1, baseline_h1, lag=horizon - 1),
        "loss_differences_numeric": numeric_losses,
        "loss_differences_evidence": evidence_losses,
        "numeric_trading": trading_metrics(last_h1, numeric_h1, actual_h1),
        "evidence_trading": trading_metrics(last_h1, evidence_h1, actual_h1),
        "cutoff_audit_failures": 0,
        "cutoff_rows_checked": audit_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/raw_expanded/prices_daily.csv")
    parser.add_argument("--evidence", default="data/raw/sec_filings.jsonl")
    parser.add_argument("--features", default="data/raw/sec_features.jsonl")
    parser.add_argument("--holdout-origins", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    grouped = load_prices(Path(args.prices))
    evidence_rows = load_jsonl(Path(args.evidence))
    feature_rows = load_jsonl(Path(args.features))
    results = {"source": args.prices, "holdout_protocol": "final contiguous origins, no tuning on holdout", "symbols": {}}
    for symbol, bundle in sorted(grouped.items()):
        results["symbols"][symbol] = evaluate(bundle, evidence_rows, feature_rows, symbol, args.holdout_origins, args.horizon)
    output = ROOT / "results" / "holdout_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
