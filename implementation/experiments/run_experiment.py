"""Run a deterministic end-to-end smoke experiment for TRACE-Fin."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "method"))

from baseline import random_walk_forecast
from trace_fin import forecast_with_audit


def synthetic_prices(n: int = 180, seed: int = 7) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    prices = [100.0]
    events = [0.0] * n
    for t in range(1, n):
        drift = 0.0005 if t < n // 2 else -0.0002
        shock = rng.gauss(0.0, 0.006 if t < n // 2 else 0.018)
        if t in (n // 2, n // 2 + 1, n // 2 + 2):
            shock += 0.012
            events[t] = 0.01
        prices.append(prices[-1] * math.exp(drift + shock))
    return prices, events


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def run(origins: int = 24, horizon: int = 3) -> dict:
    prices, events = synthetic_prices()
    baseline_errors, method_errors, coverage_hits, ledgers = [], [], [], []
    residuals: list[float] = []
    start = len(prices) - origins - horizon
    for offset in range(origins):
        cut = start + offset
        history = prices[:cut]
        actual = prices[cut:cut + horizon]
        baseline_errors.append(mae(actual, random_walk_forecast(history, horizon)))
        signal = events[cut - 1] if cut > 0 else 0.0
        result = forecast_with_audit(
            history, horizon, evidence_signal=signal, residuals=residuals[-20:],
            cutoff=f"synthetic-{cut}", evidence_ids=[f"event-{cut - 1}"] if signal else [],
        )
        method_errors.append(mae(actual, result["point_forecast"]))
        coverage_hits.append(float(result["interval"][0] <= actual[0] <= result["interval"][1]))
        residuals.append(actual[0] - result["point_forecast"][0])
        ledgers.append(result)
    return {
        "status": "smoke_pass",
        "origins": origins,
        "horizon": horizon,
        "baseline_mae": sum(baseline_errors) / len(baseline_errors),
        "trace_fin_mae": sum(method_errors) / len(method_errors),
        "interval_coverage_h1": sum(coverage_hits) / len(coverage_hits),
        "ledger_replayable": all("input_hash" in item for item in ledgers),
        "first_ledger": ledgers[0],
    }


if __name__ == "__main__":
    output = run()
    result_path = ROOT / "results" / "smoke_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
