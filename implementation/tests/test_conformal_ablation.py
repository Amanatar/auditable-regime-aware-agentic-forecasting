"""Unit tests for Phase 2: Walk-Forward Regime Detection & Conformal Ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from implementation.models.conformal_calibrator import (
    LeakedNaiveCalibrator,
    StaticSplitConformalCalibrator,
    TraceFinRegimeACICalibrator,
    winkler_interval_score,
)
from implementation.models.forecasting_engine import RegimeForecastingEnsemble
from implementation.models.regime_detector import CausalRegimeDetector

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestConformalAblation(unittest.TestCase):
    def test_winkler_interval_score_invariants(self):
        """Verify Winkler proper scoring rule math."""
        alpha = 0.10
        # 1. Realized inside interval [0.0, 0.04]: score must equal width 0.04
        self.assertAlmostEqual(winkler_interval_score(0.0, 0.04, 0.02, alpha=alpha), 0.04)

        # 2. Miss below lower bound: actual = -0.01 -> width 0.04 + (2/0.1)*(0.0 - (-0.01)) = 0.04 + 0.20 = 0.24
        self.assertAlmostEqual(winkler_interval_score(0.0, 0.04, -0.01, alpha=alpha), 0.24)

        # 3. Miss above upper bound: actual = 0.06 -> width 0.04 + (2/0.1)*(0.06 - 0.04) = 0.04 + 0.40 = 0.44
        self.assertAlmostEqual(winkler_interval_score(0.0, 0.04, 0.06, alpha=alpha), 0.44)

    def test_regime_detector_causal_invariance(self):
        """Verify regime classification uses strictly historical data."""
        det = CausalRegimeDetector(lookback_window=50)
        # Synthetic sequence of 60 records
        records = [
            {"date": f"2024-01-{i+1:02d}", "realized_vol_yz_21d": 0.10 + (0.01 * (i % 5)), "drawdown_from_peak_63d": -0.01}
            for i in range(60)
        ]
        res_orig = det.classify_at_index(records, 40)

        # Mutate future records at index 50..59
        perturbed = [dict(r) for r in records]
        for j in range(50, 60):
            perturbed[j]["realized_vol_yz_21d"] = 0.95
            perturbed[j]["drawdown_from_peak_63d"] = -0.30

        res_pert = det.classify_at_index(perturbed, 40)
        self.assertEqual(res_orig["regime"], res_pert["regime"])
        self.assertEqual(res_orig["stress_probability"], res_pert["stress_probability"])

    def test_aci_online_adaptation_dynamics(self):
        """Verify online ACI updates alpha dynamically upon miscoverage."""
        aci = TraceFinRegimeACICalibrator(target_alpha=0.10, gamma=0.05)
        self.assertEqual(aci.current_alpha, 0.10)

        # Case 1: Missed outcome (err = 1) -> alpha should decrease to widen future interval
        new_alpha = aci.update_online(actual=0.10, lower=0.0, upper=0.05)
        # alpha_1 = 0.10 + 0.05 * (0.10 - 1.0) = 0.10 - 0.045 = 0.055
        self.assertAlmostEqual(new_alpha, 0.055, places=4)

        # Case 2: Covered outcome (err = 0) -> alpha should increase towards target
        new_alpha_2 = aci.update_online(actual=0.03, lower=0.0, upper=0.05)
        # alpha_2 = 0.055 + 0.05 * (0.10 - 0.0) = 0.055 + 0.005 = 0.060
        self.assertAlmostEqual(new_alpha_2, 0.060, places=4)

    def test_conformal_ablation_results_file_and_stress_advantage(self):
        """Verify results JSON exists and trace_fin_regime_aci outperforms static split on stress coverage."""
        results_path = REPO_ROOT / "results" / "conformal_ablation.json"
        self.assertTrue(results_path.exists(), "results/conformal_ablation.json must exist.")
        data = json.loads(results_path.read_text(encoding="utf-8"))

        summary_1d = data["pooled_summary"]["horizon_1d"]
        self.assertIn("leaked_naive", summary_1d)
        self.assertIn("static_split_conformal", summary_1d)
        self.assertIn("trace_fin_regime_aci", summary_1d)

        # Key scientific findings to assert:
        # 1. Leaked naive has artificially high IC / Hit rate due to lookahead peeking
        self.assertGreater(summary_1d["leaked_naive"]["mean_information_coefficient"], 0.80)

        # 2. Static split conformal coverage drops during stress regimes
        self.assertLess(summary_1d["static_split_conformal"]["mean_stress_regime_coverage_pct"], 90.0)

        # 3. TRACE-Fin regime ACI maintains robust coverage (>= 90%) even under stress!
        self.assertGreaterEqual(summary_1d["trace_fin_regime_aci"]["mean_stress_regime_coverage_pct"], 90.0)


if __name__ == "__main__":
    unittest.main()
