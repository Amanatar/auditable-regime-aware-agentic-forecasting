"""Verification test for real multi-regime market corpus and point-in-time filtration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "implementation" / "data"))

from cutoff_validator import audit_market_corpus, validate_available_at
from market_loader import DEFAULT_SNAPSHOT_PATH, load_market_snapshot


def test_market_corpus_integrity():
    """Verify market corpus file exists, has all 5 assets and >=6000 rows."""
    assert DEFAULT_SNAPSHOT_PATH.exists(), "Market corpus frozen snapshot must exist."
    rows = load_market_snapshot(DEFAULT_SNAPSHOT_PATH)
    assert len(rows) >= 6000
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"SPY", "QQQ", "^NSEI", "RELIANCE.NS", "HDFCBANK.NS"}


def test_multi_asset_causality_audit():
    """Verify 0 lookahead violations on causal pipeline and 100% detection on leaked pipeline."""
    report = audit_market_corpus(cutoff="2024-12-31T23:59:59+00:00")
    assert report["summary"]["total_symbols"] == 5
    assert report["summary"]["all_causal_pass"] is True
    assert report["summary"]["all_leaked_detected"] is True

    for sym, stats in report["symbols"].items():
        assert stats["causal_violations"] == 0, f"Symbol {sym} had causal violations!"
        assert stats["leaked_detection_rate"] == 1.0, f"Symbol {sym} leaked pipeline not 100% detected!"
        assert len(stats["causal_audit_hash"]) == 64
