"""Extract a transparent lexical feature from a small SEC filing sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

POSITIVE = {"growth", "strong", "increase", "increased", "record", "improved", "profit", "positive", "revenue"}
NEGATIVE = {"decline", "decrease", "decreased", "loss", "risk", "risks", "litigation", "impairment", "uncertainty"}


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "TRACE-Fin research research@example.com"})
    with urlopen(request, timeout=10) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    parser = TextParser()
    parser.feed(raw)
    return parser.text()


def score(text: str) -> tuple[int, int, float]:
    tokens = re.findall(r"[a-z]+", text.lower())
    positive = sum(token in POSITIVE for token in tokens)
    negative = sum(token in NEGATIVE for token in tokens)
    denom = max(1, positive + negative)
    return positive, negative, max(-0.03, min(0.03, (positive - negative) / denom * 0.01))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/sec_filings.jsonl")
    parser.add_argument("--output", default="data/raw/sec_features.jsonl")
    parser.add_argument("--per-symbol", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=1000, help="maximum uncached fetches in this run")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    symbols = sorted({row["symbol"] for row in rows})
    for symbol in symbols:
        selected.extend([row for row in rows if row["symbol"] == symbol][-args.per_symbol:])
    output_path = Path(args.output)
    cached = {}
    if output_path.exists():
        cached = {row.get("accession_number"): row for row in (json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    metadata = {row["accession_number"]: row for row in rows}
    features = []
    new_fetches = 0
    for row in selected:
        if row["accession_number"] in cached and cached[row["accession_number"]].get("text_sha256"):
            existing = cached[row["accession_number"]]
            existing.update({"form": row.get("form"), "filing_date": row.get("filing_date"), "report_date": row.get("report_date")})
            features.append(existing)
            continue
        if new_fetches >= args.max_new:
            continue
        try:
            text = fetch_text(row["source_url"])
        except Exception as exc:
            print(json.dumps({"status": "fetch_error", "accession_number": row["accession_number"], "error": str(exc)[:200]}))
            continue
        new_fetches += 1
        positive, negative, signal = score(text)
        features.append({
            "symbol": row["symbol"],
            "form": row.get("form"),
            "filing_date": row.get("filing_date"),
            "report_date": row.get("report_date"),
            "available_at": row["available_at"],
            "accession_number": row["accession_number"],
            "source_url": row["source_url"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "positive_count": positive,
            "negative_count": negative,
            "signal": signal,
        })
        time.sleep(0.25)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in features), encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(features), "new_fetches": new_fetches, "output": args.output}, indent=2))
