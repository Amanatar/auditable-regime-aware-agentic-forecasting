"""Conformal prediction and uncertainty quantification benchmarks for TRACE-Fin.

Benchmarks 3 distinct uncertainty methods at target 90% coverage (alpha = 0.10):
1. Method 1: leaked_naive (Contaminated Baseline) - in-sample variance from lookahead features
2. Method 2: static_split_conformal (Standard Split Conformal Baseline) - exchangeable static quantiles
3. Method 3: trace_fin_regime_aci (Flagship Regime-Conditional + Adaptive Conformal Inference)
   - Volatility-scaled non-conformity scores conditioned on causal regime risk
   - Online Adaptive Conformal Inference (Gibbs & Candes) updating alpha_t dynamically to withstand volatility spikes
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr


def winkler_interval_score(lower: float, upper: float, actual: float, alpha: float = 0.10) -> float:
    """Compute the Winkler (1972) interval score.
    
    A proper scoring rule that penalizes interval width plus 2/alpha penalty for coverage misses:
    Score = (U - L) + (2 / alpha) * (L - actual) * I(actual < L) + (2 / alpha) * (actual - U) * I(actual > U).
    Lower is better.
    """
    width = max(0.0, upper - lower)
    penalty_factor = 2.0 / alpha
    if actual < lower:
        return width + penalty_factor * (lower - actual)
    elif actual > upper:
        return width + penalty_factor * (actual - upper)
    return width


class LeakedNaiveCalibrator:
    """Method 1: Contaminated baseline using in-sample variance from leaked features."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.z_score = 1.64485  # Normal critical value for 90% coverage

    def predict_interval(
        self,
        point_forecast: float,
        in_sample_residuals: np.ndarray,
        step_regime_vol: float = 0.15,
    ) -> Tuple[float, float]:
        if len(in_sample_residuals) < 5:
            radius = self.z_score * 0.01
        else:
            std_in = float(np.std(in_sample_residuals))
            radius = self.z_score * max(std_in, 1e-4)
        return point_forecast - radius, point_forecast + radius


class StaticSplitConformalCalibrator:
    """Method 2: Classic split conformal prediction assuming stationary exchangeability."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha

    def calibrate_quantile(self, calibration_errors: List[float]) -> float:
        if not calibration_errors:
            return 0.02
        n = len(calibration_errors)
        # Standard finite-sample conformal quantile level: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, max(0.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n))
        return float(np.quantile(calibration_errors, min(0.999, level)))

    def predict_interval(
        self,
        point_forecast: float,
        calibration_quantile: float,
    ) -> Tuple[float, float]:
        return point_forecast - calibration_quantile, point_forecast + calibration_quantile


class TraceFinRegimeACICalibrator:
    """Method 3: Flagship Regime-Conditional + Adaptive Conformal Inference (ACI).
    
    1. Heteroskedasticity scaling: Non-conformity score s_i = |e_i| / (sigma_vol_i * (1 + 0.6 * P(stress)))
    2. Online ACI update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    """

    def __init__(
        self,
        target_alpha: float = 0.10,
        gamma: float = 0.05,
        alpha_min: float = 0.01,
        alpha_max: float = 0.40,
    ):
        self.target_alpha = target_alpha
        self.current_alpha = target_alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_history: List[float] = [target_alpha]

    def calibrate_normalized_quantile(
        self,
        calibration_errors: List[float],
        calibration_vols: List[float],
        calibration_stress_probs: List[float],
    ) -> float:
        """Calibrate quantile on regime-volatility normalized non-conformity scores."""
        scores = []
        for err, vol, sp in zip(calibration_errors, calibration_vols, calibration_stress_probs):
            eff_vol = max(1e-4, vol * (1.0 + 0.60 * sp))
            scores.append(abs(err) / eff_vol)
        if not scores:
            return 1.645
        n = len(scores)
        level = min(1.0, max(0.0, math.ceil((n + 1) * (1.0 - self.current_alpha)) / n))
        return float(np.quantile(scores, min(0.999, level)))

    def predict_interval(
        self,
        point_forecast: float,
        norm_quantile: float,
        current_vol: float,
        stress_prob: float,
    ) -> Tuple[float, float]:
        """Compute regime-scaled prediction interval."""
        eff_vol = max(1e-4, current_vol * (1.0 + 0.60 * stress_prob))
        radius = norm_quantile * eff_vol
        return point_forecast - radius, point_forecast + radius

    def update_online(self, actual: float, lower: float, upper: float) -> float:
        """Gibbs & Candes (2021) Online ACI Step.
        
        err_t = 1 if missed else 0
        alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)
        """
        err_t = 1.0 if (actual < lower or actual > upper) else 0.0
        self.current_alpha = self.current_alpha + self.gamma * (self.target_alpha - err_t)
        self.current_alpha = max(self.alpha_min, min(self.alpha_max, self.current_alpha))
        self.alpha_history.append(round(self.current_alpha, 4))
        return self.current_alpha


def compute_ablation_metrics(
    records: List[Dict[str, Any]],
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """Compute empirical coverage, Winkler scores, interval width, directional hit rate, and IC."""
    if not records:
        return {}

    n = len(records)
    hits = [1.0 if r["lower"] <= r["actual"] <= r["upper"] else 0.0 for r in records]
    widths = [max(0.0, r["upper"] - r["lower"]) for r in records]
    winkler_scores = [winkler_interval_score(r["lower"], r["upper"], r["actual"], alpha=alpha) for r in records]

    # Sub-regime coverages
    stress_records = [r for r in records if r.get("regime") == "high_volatility_stress"]
    stress_coverage = (
        sum(1.0 for r in stress_records if r["lower"] <= r["actual"] <= r["upper"]) / len(stress_records) * 100.0
        if stress_records
        else 0.0
    )

    low_records = [r for r in records if r.get("regime") == "low_volatility"]
    low_coverage = (
        sum(1.0 for r in low_records if r["lower"] <= r["actual"] <= r["upper"]) / len(low_records) * 100.0
        if low_records
        else 0.0
    )

    trans_records = [r for r in records if r.get("regime") == "transition"]
    trans_coverage = (
        sum(1.0 for r in trans_records if r["lower"] <= r["actual"] <= r["upper"]) / len(trans_records) * 100.0
        if trans_records
        else 0.0
    )

    # Directional hit rate
    directional_correct = sum(
        1.0 for r in records if (r["point_forecast"] >= 0 and r["actual"] >= 0) or (r["point_forecast"] < 0 and r["actual"] < 0)
    )
    hit_rate = (directional_correct / n) * 100.0

    # Information Coefficient (Spearman Rank Correlation)
    preds = [r["point_forecast"] for r in records]
    actuals = [r["actual"] for r in records]
    ic_val = 0.0
    if len(set(preds)) > 1 and len(set(actuals)) > 1:
        corr, _ = spearmanr(preds, actuals)
        ic_val = float(corr) if not math.isnan(corr) else 0.0

    return {
        "out_of_sample_steps": n,
        "empirical_coverage_pct": round(sum(hits) / n * 100.0, 2),
        "target_coverage_pct": round((1.0 - alpha) * 100.0, 2),
        "stress_regime_coverage_pct": round(stress_coverage, 2),
        "stress_steps_count": len(stress_records),
        "low_regime_coverage_pct": round(low_coverage, 2),
        "transition_regime_coverage_pct": round(trans_coverage, 2),
        "mean_interval_width": round(float(np.mean(widths)), 6),
        "mean_winkler_score": round(float(np.mean(winkler_scores)), 6),
        "directional_hit_rate_pct": round(hit_rate, 2),
        "information_coefficient": round(ic_val, 4),
    }
