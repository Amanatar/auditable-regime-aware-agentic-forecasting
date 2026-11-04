"""Integration tests for walk-forward regime and conformal prediction."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_conformal_ablation_results_exist():
    """Verify that the empirical conformal ablation benchmark was generated."""
    res_path = REPO_ROOT / "results" / "conformal_ablation.json"
    assert res_path.exists(), "results/conformal_ablation.json must be present."
    data = json.loads(res_path.read_text(encoding="utf-8"))

    assert "pooled_summary" in data
    assert "horizon_1d" in data["pooled_summary"]
    assert "horizon_5d" in data["pooled_summary"]

    h1 = data["pooled_summary"]["horizon_1d"]
    # Empirical coverage of flagship regime ACI should be close to 90%
    assert abs(h1["trace_fin_regime_aci"]["mean_empirical_coverage_pct"] - 90.0) < 3.0
    # Stress regime coverage of static conformal should drop below 90%
    assert h1["static_split_conformal"]["mean_stress_regime_coverage_pct"] < 90.0
    # Flagship regime ACI should hold >= 90% coverage during stress
    assert h1["trace_fin_regime_aci"]["mean_stress_regime_coverage_pct"] >= 90.0
