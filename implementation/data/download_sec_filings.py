"""Download timestamped SEC filing metadata for the TRACE-Fin evidence ledger."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

CIKS = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "GOOG": "0001652044",
    "JPM": "0000019617",
    "META": "0001326801",
}


def fetch_filings(symbol: str, cik: str) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    request = Request(url, headers={"User-Agent": "TRACE-Fin research research@example.com"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    recent = payload["filings"]["recent"]
    rows = []
    for i, form in enumerate(recent["form"]):
        if form not in {"8-K", "10-K", "10-Q"}:
            continue
        accepted = recent["acceptanceDateTime"][i]
        if not accepted:
            accepted = recent["filingDate"][i] + "T00:00:00+00:00"
        else:
            accepted = accepted.replace("Z", "+00:00")
        rows.append({
            "symbol": symbol,
            "form": form,
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "available_at": accepted,
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{recent['accessionNumber'][i].replace('-', '')}/{recent['primaryDocument'][i]}",
        })
    return rows


def download(output_dir: Path, symbols: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        rows.extend(fetch_filings(symbol, CIKS[symbol]))
        time.sleep(0.25)
    rows.sort(key=lambda row: (row["symbol"], row["available_at"]))
    path = output_dir / "sec_filings.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "source": "SEC EDGAR submissions API",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "forms": ["8-K", "10-K", "10-Q"],
        "rows": len(rows),
    }
    (output_dir / "sec_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--symbols", nargs="+", default=list(CIKS))
    args = parser.parse_args()
    print(download(Path(args.output_dir), args.symbols))
