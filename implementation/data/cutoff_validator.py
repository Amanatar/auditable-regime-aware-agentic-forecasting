"""Timestamp validation helpers for replayable forecast inputs."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable


def validate_available_at(rows: Iterable[dict], cutoff: str) -> list[dict]:
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    accepted = []
    for row in rows:
        available = row.get("available_at") or row.get("date")
        if available is None:
            raise ValueError("every input row needs available_at or date")
        available_dt = datetime.fromisoformat(str(available).replace("Z", "+00:00"))
        if available_dt <= cutoff_dt:
            accepted.append(row)
    return accepted


if __name__ == "__main__":
    sample = [{"id": "past", "available_at": "2025-01-01T00:00:00+00:00"},
              {"id": "future", "available_at": "2025-02-01T00:00:00+00:00"}]
    assert [x["id"] for x in validate_available_at(sample, "2025-01-15T00:00:00+00:00")] == ["past"]
    print("cutoff_validator_pass")
