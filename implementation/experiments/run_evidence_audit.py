"""Audit SEC evidence coverage and replayability without making price claims."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "method"))

from evidence import audit_no_future, evidence_before, load_jsonl


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/sec_filings.jsonl")
    rows = load_jsonl(path)
    report = {"source": str(path), "rows": len(rows), "symbols": {}, "status": "no_data" if not rows else "pass"}
    for symbol in sorted({row["symbol"] for row in rows}):
        symbol_rows = [row for row in rows if row["symbol"] == symbol]
        step = max(1, len(symbol_rows) // 10)
        cutoffs = sorted({row["available_at"] for row in symbol_rows})[::step]
        accepted = [evidence_before(rows, symbol, cutoff) for cutoff in cutoffs]
        report["symbols"][symbol] = {
            "rows": len(symbol_rows),
            "cutoff_checks": len(cutoffs),
            "all_checks_pass": all(audit_no_future(batch, cutoff) for batch, cutoff in zip(accepted, cutoffs)),
        }
    output = ROOT / "results" / "evidence_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
