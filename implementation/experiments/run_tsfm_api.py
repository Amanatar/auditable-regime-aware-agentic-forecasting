"""Run TSFM.ai inference without persisting the API key.

Set TSFM_API_KEY in the process environment. The key is never written to
results, logs, or Git.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def load_latest(path: Path, context: int = 64) -> dict[str, dict]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["symbol"]].append(row)
    return {
        symbol: {
            "rows": rows[-context:],
            "target": [[float(row["close"])] for row in rows[-context:]],
            "start": f"{rows[-context:][0]['date']}T00:00:00Z",
        }
        for symbol, rows in grouped.items()
    }


def forecast(api_key: str, model: str, item_id: str, series: dict, horizon: int = 3) -> dict:
    payload = {
        "model": model,
        "inputs": [{
            "start": series["start"],
            "target": series["target"],
            "metadata": {"item_id": item_id},
        }],
        "parameters": {
            "prediction_length": horizon,
            "frequency": "D",
            "quantile_levels": [0.1, 0.5, 0.9],
        },
        "metadata": {"surface": "trace-fin-api-smoke"},
    }
    request = Request(
        "https://api.tsfm.ai/v1/forecast",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "TRACE-Fin/0.1 research"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TSFM API HTTP {exc.code}: {detail}") from exc
    return {
        "item_id": item_id,
        "model": result.get("model", model),
        "object": result.get("object"),
        "prediction_length": result.get("prediction_length"),
        "outputs": result.get("outputs"),
        "usage": result.get("usage"),
        "latency_ms": result.get("latency_ms"),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    key = os.environ.get("TSFM_API_KEY")
    if not key:
        raise SystemExit("TSFM_API_KEY is required")
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/prices_daily.csv")
    model = sys.argv[2] if len(sys.argv) > 2 else "amazon/chronos-bolt-tiny"
    series = load_latest(data_path)
    results = []
    for symbol in sorted(series):
        results.append(forecast(key, model, symbol, series[symbol]))
    output = Path(__file__).resolve().parents[1] / "results" / "tsfm_api_sample.json"
    output.write_text(json.dumps({"model": model, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "items": len(results), "output": str(output)}, indent=2))
