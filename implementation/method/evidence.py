"""Evidence loading and cutoff-audit helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_before(rows: list[dict], symbol: str, cutoff: str) -> list[dict]:
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    accepted = []
    for row in rows:
        if row.get("symbol") != symbol:
            continue
        available = datetime.fromisoformat(row["available_at"].replace("Z", "+00:00"))
        if available <= cutoff_dt:
            accepted.append(row)
    return accepted


def audit_no_future(rows: list[dict], cutoff: str) -> bool:
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    return all(datetime.fromisoformat(row["available_at"].replace("Z", "+00:00")) <= cutoff_dt for row in rows)
