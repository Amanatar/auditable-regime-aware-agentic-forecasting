"""Integration tests for portfolio execution and lookahead contamination decay."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_portfolio_backtest_audit_invariants():
    """Verify that portfolio backtest was completed and lookahead decay is audited."""
    res_path = REPO_ROOT / "results" / "portfolio_backtest_summary.json"
    assert res_path.exists(), "results/portfolio_backtest_summary.json must exist."
    data = json.loads(res_path.read_text(encoding="utf-8"))

    strats = data["portfolio_strategies"]
    assert "baseline_a_buy_and_hold" in strats
    assert "baseline_b_uncalibrated_point_forecast" in strats
    assert "flagship_c_trace_fin_regime_conformal" in strats

    # Strategy C must achieve lower max drawdown than Buy-and-Hold
    dd_c = strats["flagship_c_trace_fin_regime_conformal"]["max_drawdown_pct"]
    dd_a = strats["baseline_a_buy_and_hold"]["max_drawdown_pct"]
    assert dd_c < dd_a, f"Strategy C drawdown ({dd_c}%) should be lower than Baseline A ({dd_a}%)"

    # Strategy C must dramatically reduce turnover vs Baseline B
    turn_c = strats["flagship_c_trace_fin_regime_conformal"]["annualized_turnover"]
    turn_b = strats["baseline_b_uncalibrated_point_forecast"]["annualized_turnover"]
    assert turn_c < turn_b, f"Strategy C turnover ({turn_c}x) should be lower than Baseline B ({turn_b}x)"

    # Lookahead audit: leaked in-sample Sharpe must be artificially high
    leak = data["contamination_audit"]["leaked_pipeline"]
    assert leak["in_sample_sharpe"] > 5.0, "Leaked in-sample Sharpe should demonstrate severe lookahead inflation."
