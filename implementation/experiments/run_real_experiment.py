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
sys.path.insert(0, str(ROOT / "evaluation"))

from baseline import random_walk_forecast
from evidence import evidence_before, load_jsonl
from metrics import diebold_mariano_squared, trading_metrics
from trace_fin import forecast_with_audit


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def load_prices(path: Path) -> dict[str, list[float]]:
    grouped = defaultdict(lambda: {"dates": [], "prices": []})
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]]["dates"].append(row["date"])
            grouped[row["symbol"]]["prices"].append(float(row["close"]))
    return dict(grouped)


def evaluate_series(prices: list[float], dates: list[str], evidence_rows: list[dict], feature_rows: list[dict], symbol: str, origins: int = 60, horizon: int = 3) -> dict:
    if len(prices) < origins + horizon + 30:
        raise ValueError("not enough rows for requested rolling origins")
    start = len(prices) - origins - horizon
    baseline_errors, method_errors, evidence_errors, coverage, evidence_counts = [], [], [], [], []
    h1_actual, h1_baseline, h1_numeric, h1_evidence, h1_last = [], [], [], [], []
    residuals = []
    for offset in range(origins):
        cut = start + offset
        history, actual = prices[:cut], prices[cut:cut + horizon]
        baseline_errors.append(mae(actual, random_walk_forecast(history, horizon)))
        h1_actual.append(actual[0])
        h1_baseline.append(history[-1])
        h1_last.append(history[-1])
        cutoff = f"{dates[cut - 1]}T23:59:59+00:00"
        available = evidence_before(evidence_rows, symbol, cutoff)
        available_features = evidence_before(feature_rows, symbol, cutoff)
        evidence_counts.append(len(available))
        numeric_result = forecast_with_audit(
            history, horizon, residuals=residuals[-20:], cutoff=cutoff,
            evidence_ids=[row["accession_number"] for row in available[-20:]],
        )
        signal = sum(row["signal"] for row in available_features[-3:]) / max(1, len(available_features[-3:]))
        evidence_result = forecast_with_audit(
            history, horizon, evidence_signal=signal, residuals=residuals[-20:], cutoff=cutoff,
            evidence_ids=[row["accession_number"] for row in available[-20:]],
        )
        method_errors.append(mae(actual, numeric_result["point_forecast"]))
        evidence_errors.append(mae(actual, evidence_result["point_forecast"]))
        h1_numeric.append(numeric_result["point_forecast"][0])
        h1_evidence.append(evidence_result["point_forecast"][0])
        coverage.append(float(evidence_result["interval"][0] <= actual[0] <= evidence_result["interval"][1]))
        residuals.append(actual[0] - numeric_result["point_forecast"][0])
    return {
        "origins": origins,
        "horizon": horizon,
        "baseline_mae": sum(baseline_errors) / len(baseline_errors),
        "trace_fin_numeric_mae": sum(method_errors) / len(method_errors),
        "trace_fin_evidence_mae": sum(evidence_errors) / len(evidence_errors),
        "interval_coverage_h1": sum(coverage) / len(coverage),
        "origins_with_sec_evidence": sum(count > 0 for count in evidence_counts),
        "mean_sec_evidence_count": sum(evidence_counts) / len(evidence_counts),
        "dm_numeric_vs_random_walk": diebold_mariano_squared(h1_actual, h1_numeric, h1_baseline),
        "dm_evidence_vs_random_walk": diebold_mariano_squared(h1_actual, h1_evidence, h1_baseline),
        "loss_differences_numeric": [(a - n) ** 2 - (a - b) ** 2 for a, n, b in zip(h1_actual, h1_numeric, h1_baseline)],
        "loss_differences_evidence": [(a - e) ** 2 - (a - b) ** 2 for a, e, b in zip(h1_actual, h1_evidence, h1_baseline)],
        "numeric_trading": trading_metrics(h1_last, h1_numeric, h1_actual),
        "evidence_trading": trading_metrics(h1_last, h1_evidence, h1_actual),
    }


if __name__ == "__main__":
    parser_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/prices_daily.csv")
    sec_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw/sec_filings.jsonl")
    feature_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/raw/sec_features.jsonl")
    grouped = load_prices(parser_path)
    evidence_rows = load_jsonl(sec_path)
    feature_rows = load_jsonl(feature_path)
    results = {"source": str(parser_path), "evidence_source": str(sec_path), "feature_source": str(feature_path), "symbols": {}}
    for symbol, bundle in grouped.items():
        results["symbols"][symbol] = evaluate_series(bundle["prices"], bundle["dates"], evidence_rows, feature_rows, symbol)
    output_path = ROOT / "results" / "real_price_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
