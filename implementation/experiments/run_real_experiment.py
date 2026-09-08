"""Evaluate the baseline and numeric TRACE-Fin scaffold on downloaded prices."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "method"))

from baseline import random_walk_forecast
from trace_fin import forecast_with_audit


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def load_prices(path: Path) -> dict[str, list[float]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]].append(float(row["close"]))
    return dict(grouped)


def evaluate_series(prices: list[float], origins: int = 60, horizon: int = 3) -> dict:
    if len(prices) < origins + horizon + 30:
        raise ValueError("not enough rows for requested rolling origins")
    start = len(prices) - origins - horizon
    baseline_errors, method_errors, coverage = [], [], []
    residuals = []
    for offset in range(origins):
        cut = start + offset
        history, actual = prices[:cut], prices[cut:cut + horizon]
        baseline_errors.append(mae(actual, random_walk_forecast(history, horizon)))
        result = forecast_with_audit(history, horizon, residuals=residuals[-20:], cutoff=f"row-{cut}")
        method_errors.append(mae(actual, result["point_forecast"]))
        coverage.append(float(result["interval"][0] <= actual[0] <= result["interval"][1]))
        residuals.append(actual[0] - result["point_forecast"][0])
    return {
        "origins": origins,
        "horizon": horizon,
        "baseline_mae": sum(baseline_errors) / len(baseline_errors),
        "trace_fin_numeric_mae": sum(method_errors) / len(method_errors),
        "interval_coverage_h1": sum(coverage) / len(coverage),
    }


if __name__ == "__main__":
    parser_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/prices_daily.csv")
    grouped = load_prices(parser_path)
    results = {"source": str(parser_path), "symbols": {}}
    for symbol, prices in grouped.items():
        results["symbols"][symbol] = evaluate_series(prices)
    output_path = ROOT / "results" / "real_price_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
