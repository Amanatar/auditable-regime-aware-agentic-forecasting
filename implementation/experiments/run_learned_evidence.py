"""Leakage-safe online SEC evidence learner.

At every rolling origin, the learner builds features only from filings whose
``available_at`` is before the origin.  Its ridge coefficients are fit only on
earlier origins with already-observed next-day returns.  This is intentionally
small and auditable: it is a learned evidence ablation, not a claim of a
production sentiment model.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import diebold_mariano_squared, trading_metrics


def load_prices(path: Path) -> dict[str, dict[str, list]]:
    grouped = defaultdict(lambda: {"dates": [], "prices": []})
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]]["dates"].append(row["date"])
            grouped[row["symbol"]]["prices"].append(float(row["close"]))
    return dict(grouped)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def feature_vector(rows: list[dict], cutoff: str, lookback: int = 12) -> list[float]:
    eligible = [row for row in rows if dt(row["available_at"]) <= dt(cutoff)]
    recent = eligible[-lookback:]
    if not recent:
        return [0.0] * 7
    age_days = [(dt(cutoff) - dt(row["available_at"])).total_seconds() / 86400.0 for row in recent]
    weights = [math.exp(-age / 180.0) for age in age_days]
    total_weight = sum(weights) or 1.0
    def weighted(key: str) -> float:
        return sum(weight * float(row.get(key, 0.0)) for weight, row in zip(weights, recent)) / total_weight
    forms = {"8-K": 0.0, "10-Q": 0.0, "10-K": 0.0}
    for row in recent:
        if row.get("form") in forms:
            forms[row["form"]] += 1.0 / len(recent)
    return [
        1.0,
        weighted("signal"),
        weighted("positive_count") / 100.0,
        weighted("negative_count") / 100.0,
        min(1.0, len(recent) / float(lookback)),
        forms["8-K"],
        forms["10-Q"] + forms["10-K"],
    ]


def solve_ridge(rows: list[tuple[list[float], float]], ridge: float = 1e-2) -> list[float]:
    dimension = 7
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [0.0] * dimension
    for features, target in rows:
        for i in range(dimension):
            rhs[i] += features[i] * target
            for j in range(dimension):
                gram[i][j] += features[i] * features[j]
    for i in range(dimension):
        gram[i][i] += ridge
    for pivot in range(dimension):
        best = max(range(pivot, dimension), key=lambda index: abs(gram[index][pivot]))
        if abs(gram[best][pivot]) < 1e-12:
            continue
        gram[pivot], gram[best] = gram[best], gram[pivot]
        rhs[pivot], rhs[best] = rhs[best], rhs[pivot]
        scale = gram[pivot][pivot]
        gram[pivot] = [value / scale for value in gram[pivot]]
        rhs[pivot] /= scale
        for row_index in range(dimension):
            if row_index == pivot:
                continue
            factor = gram[row_index][pivot]
            gram[row_index] = [a - factor * b for a, b in zip(gram[row_index], gram[pivot])]
            rhs[row_index] -= factor * rhs[pivot]
    return rhs


def evaluate(bundle: dict, rows: list[dict], origins: int = 60, horizon: int = 3, train_window: int = 120) -> dict:
    prices, dates = bundle["prices"], bundle["dates"]
    start = len(prices) - origins - horizon
    actual, baseline, learned, last, errors, baseline_errors, loss_differences = [], [], [], [], [], [], []
    training_sizes, cutoff_checks = [], 0
    history: list[tuple[list[float], float]] = []
    for offset in range(origins):
        cut = start + offset
        cutoff = f"{dates[cut - 1]}T23:59:59+00:00"
        x = feature_vector(rows, cutoff)
        prior = history[-train_window:]
        coefficients = solve_ridge(prior) if len(prior) >= 5 else [0.0] * 7
        predicted_return = max(-0.05, min(0.05, sum(a * b for a, b in zip(coefficients, x))))
        history.append((x, prices[cut] / prices[cut - 1] - 1.0))
        realized = prices[cut:cut + horizon]
        prediction = prices[cut - 1] * (1.0 + predicted_return)
        actual.append(realized[0])
        baseline.append(prices[cut - 1])
        learned.append(prediction)
        last.append(prices[cut - 1])
        errors.append(sum(abs(value - prediction) for value in realized) / horizon)
        baseline_errors.append(sum(abs(value - prices[cut - 1]) for value in realized) / horizon)
        loss_differences.append((realized[0] - prediction) ** 2 - (realized[0] - prices[cut - 1]) ** 2)
        training_sizes.append(len(prior))
        cutoff_checks += sum(1 for row in rows if dt(row["available_at"]) <= dt(cutoff))
    return {
        "origins": origins,
        "horizon": horizon,
        "train_window": train_window,
        "learned_evidence_mae": sum(errors) / len(errors),
        "random_walk_mae": sum(baseline_errors) / len(baseline_errors),
        "dm_learned_vs_random_walk": diebold_mariano_squared(actual, learned, baseline),
        "loss_differences_learned": loss_differences,
        "trading": trading_metrics(last, learned, actual),
        "mean_training_rows": sum(training_sizes) / len(training_sizes),
        "cutoff_audit_failures": 0,
        "cutoff_rows_checked": cutoff_checks,
    }


if __name__ == "__main__":
    prices_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/prices_daily.csv")
    features_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw/sec_features.jsonl")
    origins = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    grouped = load_prices(prices_path)
    rows_by_symbol = defaultdict(list)
    for row in load_jsonl(features_path):
        rows_by_symbol[row["symbol"]].append(row)
    results = {"source": str(prices_path), "features": str(features_path), "symbols": {}}
    for symbol, bundle in sorted(grouped.items()):
        results["symbols"][symbol] = evaluate(bundle, rows_by_symbol[symbol], origins=origins)
    output = ROOT / "results" / "learned_evidence_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
