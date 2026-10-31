"""Download daily adjusted-close data from Yahoo Finance's chart endpoint.

The endpoint is used only as a reproducible public-data bootstrap. The raw
JSON response is saved with a manifest so the source and retrieval time remain
auditable. For publication, pin a licensed vendor or archive the raw file.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def fetch_symbol(symbol: str, start: str, end: str) -> list[dict]:
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )
    request = Request(url, headers={"User-Agent": "TRACE-Fin/0.1 research bootstrap"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    rows = []
    for i, stamp in enumerate(timestamps):
        close = quote["close"][i]
        if close is not None:
            rows.append({
                "symbol": symbol,
                "date": datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat(),
                "close": float(close),
                "volume": int(quote["volume"][i] or 0),
            })
    return rows


def download(output_dir: Path, symbols: list[str], start: str, end: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for symbol in symbols:
        all_rows.extend(fetch_symbol(symbol, start, end))
        time.sleep(0.2)
    all_rows.sort(key=lambda row: (row["symbol"], row["date"]))
    data_path = output_dir / "prices_daily.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "date", "close", "volume"])
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {
        "source": "Yahoo Finance chart endpoint",
        "url_template": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "start": start,
        "end": end,
        "rows": len(all_rows),
    }
    (output_dir / "prices_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-09-08")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "AMZN", "GOOG", "JPM", "META"])
    args = parser.parse_args()
    path = download(Path(args.output_dir), args.symbols, args.start, args.end)
    print(path)
