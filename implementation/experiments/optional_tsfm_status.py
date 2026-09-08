"""Record whether optional pretrained TSFM dependencies are available."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


if __name__ == "__main__":
    status = {
        "chronos": bool(importlib.util.find_spec("chronos")),
        "torch": bool(importlib.util.find_spec("torch")),
        "transformers": bool(importlib.util.find_spec("transformers")),
        "run_status": "not_run",
        "reason": "Chronos is not installed; numeric baselines and audit-safe method were run instead.",
    }
    output = Path(__file__).resolve().parents[1] / "results" / "optional_tsfm_status.json"
    output.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
