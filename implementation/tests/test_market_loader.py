"""Unit tests for multi-regime market ingestion, frozen snapshot, and point-in-time features."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from implementation.data.cutoff_validator import (
    audit_lookahead_violations,
    compute_cutoff_sha256,
    validate_available_at,
    verify_asset_pipelines,
)
from implementation.data.market_loader import (
    ASSET_METADATA,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_SYMBOLS,
    compute_causal_rolling_zscore,
    compute_close_to_close_volatility,
    compute_drawdown_from_peak,
    compute_trailing_log_returns,
    compute_yang_zhang_volatility,
    engineer_point_in_time_features,
    load_market_snapshot,
)


class TestMarketLoaderAndCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure frozen snapshot is accessible
        cls.snapshot_path = DEFAULT_SNAPSHOT_PATH
        cls.manifest_path = DEFAULT_MANIFEST_PATH
        cls.rows = load_market_snapshot(cls.snapshot_path)

    def test_frozen_snapshot_and_manifest_exist(self):
        """Verify frozen dataset and manifest are present and populated."""
        self.assertTrue(self.snapshot_path.exists(), "Snapshot CSV data/eval/multi_asset_daily_2021_2026.csv must exist.")
        self.assertTrue(self.manifest_path.exists(), "Manifest JSON must exist.")
        self.assertGreater(len(self.rows), 6000, "Corpus must contain 6,000+ daily trading records.")

    def test_asset_universe_coverage(self):
        """Verify all 5 required US & Indian assets exist with ~1,250 rows each."""
        symbols = {r["symbol"] for r in self.rows}
        for expected in DEFAULT_SYMBOLS:
            self.assertIn(expected, symbols)
            sym_rows = [r for r in self.rows if r["symbol"] == expected]
            self.assertGreaterEqual(len(sym_rows), 1200, f"Asset {expected} must have at least 1,200 rows.")
            self.assertLessEqual(len(sym_rows), 1300, f"Asset {expected} row count should be ~1,250.")
            self.assertTrue(sym_rows[0]["date"].startswith("2021-01"))
            self.assertTrue(sym_rows[-1]["date"].startswith("2025-12"))

    def test_causal_trailing_log_returns(self):
        """Verify trailing log returns use only past prices (none before lag)."""
        closes = [100.0, 102.0, 101.0, 105.0, 110.0]
        rets = compute_trailing_log_returns(closes, lags=(1, 2))
        self.assertIsNone(rets[1][0])
        self.assertAlmostEqual(rets[1][1], math.log(102.0 / 100.0), places=5)
        self.assertIsNone(rets[2][0])
        self.assertIsNone(rets[2][1])
        self.assertAlmostEqual(rets[2][2], math.log(101.0 / 100.0), places=5)

    def test_yang_zhang_volatility_mathematical_invariants(self):
        """Verify Yang-Zhang volatility returns non-negative annualized numbers."""
        closes = [100.0 + (i * 0.2) + ((i % 3) * 0.1) for i in range(30)]
        opens = [c * 0.998 for c in closes]
        highs = [c * 1.008 for c in closes]
        lows = [c * 0.992 for c in closes]
        vols = compute_yang_zhang_volatility(opens, highs, lows, closes, window=10)
        self.assertIsNone(vols[5])
        self.assertIsNotNone(vols[10])
        self.assertGreater(vols[10], 0.0)

    def test_causal_rolling_zscores_invariance_to_future(self):
        """Strict causality invariant: perturbing future data must NOT change historical z-scores."""
        series = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0]
        z_orig = compute_causal_rolling_zscore(series, window=5)

        # Perturb future data point at index 8 and 9
        perturbed = list(series)
        perturbed[8] = 999.0
        perturbed[9] = 10000.0
        z_pert = compute_causal_rolling_zscore(perturbed, window=5)

        # Historical z-scores at index <= 7 must be identical
        for i in range(8):
            self.assertEqual(z_orig[i], z_pert[i], f"Z-score at historical index {i} altered by future perturbation!")

    def test_causal_pipeline_zero_lookahead_violations(self):
        """Verify causal feature pipeline produces 0 violations across all 5 assets."""
        for sym in DEFAULT_SYMBOLS:
            sym_records = [r for r in self.rows if r["symbol"] == sym]
            features = engineer_point_in_time_features(sym_records, leaked_pipeline_mode=False)
            audit = audit_lookahead_violations(features, cutoff="2024-12-31T23:59:59+00:00")
            self.assertEqual(audit["violations_count"], 0, f"Found lookahead violations in {sym} causal pipeline!")
            self.assertTrue(audit["is_clean"])
            self.assertEqual(audit["violation_rate"], 0.0)

    def test_deliberately_leaked_pipeline_detected_100_percent(self):
        """Verify deliberately leaked pipeline is detected with 100% violation rate."""
        for sym in DEFAULT_SYMBOLS:
            sym_records = [r for r in self.rows if r["symbol"] == sym]
            leaked_features = engineer_point_in_time_features(sym_records, leaked_pipeline_mode=True)
            audit = audit_lookahead_violations(leaked_features)
            self.assertGreater(audit["violations_count"], 0, f"Leaked pipeline for {sym} failed to detect lookahead!")
            self.assertEqual(audit["violation_rate"], 1.0, f"Expected 100% detection rate for {sym} leaked pipeline.")

    def test_deterministic_sha256_audit_hash(self):
        """Verify SHA-256 filtration audit hash is deterministic and changes upon data mutation."""
        spy_rows = [r for r in self.rows if r["symbol"] == "SPY"]
        causal_spy = engineer_point_in_time_features(spy_rows, leaked_pipeline_mode=False)
        cutoff = "2024-06-30T23:59:59+00:00"

        hash1 = compute_cutoff_sha256(causal_spy, cutoff=cutoff)
        hash2 = compute_cutoff_sha256(causal_spy, cutoff=cutoff)
        self.assertEqual(hash1, hash2, "Filtration hash must be 100% deterministic.")
        self.assertEqual(len(hash1), 64, "SHA-256 hex digest length must be 64 characters.")

        # Mutate one historical value and ensure hash changes
        mutated_spy = [dict(r) for r in causal_spy]
        mutated_spy[10]["close"] = mutated_spy[10]["close"] + 0.05
        mutated_hash = compute_cutoff_sha256(mutated_spy, cutoff=cutoff)
        self.assertNotEqual(hash1, mutated_hash, "Mutating historical data must invalidate the audit hash!")


if __name__ == "__main__":
    unittest.main()
