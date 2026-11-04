"""Unit tests for Phase 3: Portfolio Construction & Execution."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from implementation.portfolio.backtest_engine import compute_portfolio_metrics
from implementation.portfolio.position_sizer import (
    EqualWeightPositionSizer,
    UncalibratedPositionSizer,
    UncertaintyGatedPositionSizer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPortfolioConstruction(unittest.TestCase):
    def test_uncertainty_inverse_position_scaling(self):
        """Verify position sizing scales inversely to conformal interval width."""
        sizer = UncertaintyGatedPositionSizer(turnover_hurdle=0.0)

        # Baseline scenario: W = 0.04
        w_narrow = sizer.compute_target_weights(
            forecasts={"SPY": 0.004},
            widths={"SPY": 0.04},
            regimes={"SPY": "transition"},
            stress_probs={"SPY": 0.30},
        )

        # Stress scenario: W doubles to 0.08
        w_wide = sizer.compute_target_weights(
            forecasts={"SPY": 0.004},
            widths={"SPY": 0.08},
            regimes={"SPY": "high_volatility_stress"},
            stress_probs={"SPY": 0.80},
        )

        # Exposure should be cut significantly when uncertainty widens
        self.assertGreater(w_narrow["SPY"], w_wide["SPY"])
        self.assertLessEqual(w_wide["SPY"], 0.55 * w_narrow["SPY"])

    def test_no_trade_cost_buffer_turnover_gate(self):
        """Verify small weight changes below hurdle threshold are suppressed."""
        sizer = UncertaintyGatedPositionSizer(turnover_hurdle=0.03)

        current_w = {"SPY": 0.20, "QQQ": 0.15}
        # Proposed changes: SPY changes by +0.015 (< 0.03 hurdle), QQQ changes by +0.05 (> 0.03 hurdle)
        target_w = {"SPY": 0.215, "QQQ": 0.20}

        rebalanced = sizer.apply_turnover_gate(target_w, current_w)
        # SPY must not trade (preserving capital)
        self.assertEqual(rebalanced["SPY"], 0.20)
        # QQQ must trade
        self.assertEqual(rebalanced["QQQ"], 0.20)

    def test_max_drawdown_circuit_breaker(self):
        """Verify circuit breaker cuts gross exposure in half when drawdown <= -10%."""
        sizer = UncertaintyGatedPositionSizer(drawdown_circuit_breaker=-0.10, circuit_breaker_scale=0.50)
        weights = {"SPY": 0.30, "QQQ": 0.20}

        # Case 1: Drawdown = -5% (within tolerance) -> no change
        w_safe = sizer.apply_circuit_breaker(weights, current_drawdown=-0.05)
        self.assertEqual(w_safe["SPY"], 0.30)

        # Case 2: Drawdown = -12% (breach) -> halved
        w_tripped = sizer.apply_circuit_breaker(weights, current_drawdown=-0.12)
        self.assertAlmostEqual(w_tripped["SPY"], 0.15)
        self.assertAlmostEqual(w_tripped["QQQ"], 0.10)

    def test_portfolio_metrics_math(self):
        """Verify portfolio performance accounting and cost deduction."""
        # 252 daily returns with +0.1% gross per day, 0.02 turnover paying 10 bps
        daily_gross = [0.001] * 252
        daily_turnover = [0.02] * 252
        daily_net = [g - (t * 0.001) for g, t in zip(daily_gross, daily_turnover)]

        metrics = compute_portfolio_metrics(daily_gross, daily_net, daily_turnover, cost_bps=10.0)
        self.assertGreater(metrics["annualized_return_gross_pct"], metrics["annualized_return_net_pct"])
        self.assertGreater(metrics["gross_sharpe_ratio"], metrics["net_sharpe_ratio"])
        self.assertEqual(metrics["total_trading_days"], 252)

    def test_results_md_and_json_summary_exist(self):
        """Verify RESULTS.md and results/portfolio_backtest_summary.json exist and are populated."""
        results_md = REPO_ROOT / "RESULTS.md"
        summary_json = REPO_ROOT / "results" / "portfolio_backtest_summary.json"

        self.assertTrue(results_md.exists(), "RESULTS.md must be generated in repo root.")
        self.assertTrue(summary_json.exists(), "portfolio_backtest_summary.json must exist.")

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertIn("portfolio_strategies", data)
        self.assertIn("flagship_c_trace_fin_regime_conformal", data["portfolio_strategies"])
        self.assertIn("contamination_audit", data)


if __name__ == "__main__":
    unittest.main()
