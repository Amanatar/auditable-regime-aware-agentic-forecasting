"""Small rolling-origin TSFM.ai benchmark using an environment-only API key."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from metrics import diebold_mariano_squared, trading_metrics


def load_prices(path: Path) -> dict[str, list[float]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]].append(float(row["close"]))
    return dict(grouped)


def request_forecast(key: str, model: str, context: list[float], horizon: int) -> list[float]:
    payload = {
        "model": model,
        "inputs": [{"target": [[value] for value in context], "metadata": {"item_id": "trace-fin"}}],
        "parameters": {"prediction_length": horizon, "frequency": "D", "quantile_levels": [0.1, 0.5, 0.9]},
    }
    request = Request(
        "https://api.tsfm.ai/v1/forecast",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "TRACE-Fin/0.1 research",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"TSFM API HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    return [float(row[0]) for row in body["outputs"][0]["mean"]]


def evaluate(prices: list[float], key: str, model: str, origins: int, horizon: int) -> dict:
    start = len(prices) - origins - horizon
    actual, baseline, forecast, last = [], [], [], []
    errors = []
    for offset in range(origins):
        cut = start + offset
        context = prices[max(0, cut - 64):cut]
        realized = prices[cut:cut + horizon]
        prediction = request_forecast(key, model, context, horizon)
        actual.append(realized[0])
        baseline.append(context[-1])
        forecast.append(prediction[0])
        last.append(context[-1])
        errors.append(sum(abs(a - p) for a, p in zip(realized, prediction)) / horizon)
        time.sleep(0.15)
    return {
        "origins": origins,
        "horizon": horizon,
        "tsfm_mae": sum(errors) / len(errors),
        "random_walk_mae": sum(abs(a - b) for a, b in zip(actual, baseline)) / len(actual),
        "dm_tsfm_vs_random_walk": diebold_mariano_squared(actual, forecast, baseline),
        "tsfm_trading": trading_metrics(last, forecast, actual),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/raw/prices_daily.csv")
    parser.add_argument("--model", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--origins", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    key = os.environ.get("TSFM_API_KEY")
    if not key:
        raise SystemExit("TSFM_API_KEY is required")
    grouped = load_prices(Path(args.prices))
    results = {"model": args.model, "origins_per_symbol": args.origins, "symbols": {}, "secret_persisted": False}
    for symbol, prices in sorted(grouped.items()):
        results["symbols"][symbol] = evaluate(prices, key, args.model, args.origins, args.horizon)
        print(symbol, json.dumps(results["symbols"][symbol]))
    output = ROOT / "results" / "tsfm_api_eval.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output)
