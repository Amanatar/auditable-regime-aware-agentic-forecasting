"""Strictly causal market regime detector for TRACE-Fin.

Classifies market state into:
- low_volatility: calm, trending, low-dispersion regime
- transition: moderate, shifting volatility regime
- high_volatility_stress: turbulent, tail-risk, volatility clustering regime

Guarantees 100% causal point-in-time classification (zero lookahead).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_quantile_causal(values: List[float], q: float) -> float:
    """Compute empirical quantile strictly on historical observations."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(q * (len(sorted_v) - 1))
    return sorted_v[max(0, min(len(sorted_v) - 1, idx))]


class CausalRegimeDetector:
    """Online, causal market regime detector using trailing volatility & drawdown quantiles."""

    def __init__(
        self,
        lookback_window: int = 252,
        low_vol_quantile: float = 0.35,
        high_vol_quantile: float = 0.75,
        stress_drawdown_threshold: float = -0.07,
        stress_vol_ratio_threshold: float = 1.35,
    ):
        self.lookback_window = lookback_window
        self.low_vol_quantile = low_vol_quantile
        self.high_vol_quantile = high_vol_quantile
        self.stress_drawdown_threshold = stress_drawdown_threshold
        self.stress_vol_ratio_threshold = stress_vol_ratio_threshold

    def classify_at_index(
        self,
        records: List[Dict[str, Any]],
        t: int,
    ) -> Dict[str, Any]:
        """Classify the regime at historical index t using ONLY records <= t.
        
        Args:
            records: List of feature row dictionaries.
            t: Current step index.
            
        Returns:
            Regime classification dictionary with regime label and posterior probability.
        """
        if t < 21:
            return {
                "regime": "transition",
                "stress_probability": 0.25,
                "current_vol": 0.15,
                "drawdown": 0.0,
                "q_low": 0.10,
                "q_high": 0.20,
            }

        curr = records[t]
        # Realized vol at t (preference: Yang-Zhang 21d, fallback: C2C 21d)
        curr_vol = curr.get("realized_vol_yz_21d") or curr.get("realized_vol_c2c_21d") or 0.15
        dd = curr.get("drawdown_from_peak_63d") or curr.get("drawdown_from_peak_21d") or 0.0
        vol_ratio = curr.get("vol_of_vol_yz_ratio_10_63") or curr.get("vol_of_vol_ratio_10_63") or 1.0

        # Trailing reference window [max(0, t - lookback_window + 1) ... t]
        hist_slice = records[max(0, t - self.lookback_window + 1) : t + 1]
        hist_vols = [
            float(r.get("realized_vol_yz_21d") or r.get("realized_vol_c2c_21d") or 0.15)
            for r in hist_slice
            if (r.get("realized_vol_yz_21d") or r.get("realized_vol_c2c_21d")) is not None
        ]

        if not hist_vols:
            hist_vols = [curr_vol]

        q_low = compute_quantile_causal(hist_vols, self.low_vol_quantile)
        q_high = compute_quantile_causal(hist_vols, self.high_vol_quantile)

        # Causal Classification Logic
        is_stress = (
            curr_vol >= q_high
            or dd <= self.stress_drawdown_threshold
            or vol_ratio >= self.stress_vol_ratio_threshold
        )
        is_low = (
            curr_vol <= q_low
            and dd >= -0.025
            and vol_ratio <= 1.05
        )

        if is_stress:
            regime = "high_volatility_stress"
            # Probability scaled by distance above q_high or severity of drawdown
            vol_boost = max(0.0, (curr_vol - q_high) / max(q_high, 1e-4))
            dd_boost = max(0.0, abs(dd) / abs(self.stress_drawdown_threshold) - 1.0)
            stress_prob = min(0.99, 0.70 + 0.20 * vol_boost + 0.10 * dd_boost)
        elif is_low:
            regime = "low_volatility"
            stress_prob = max(0.02, 0.10 * (curr_vol / max(q_low, 1e-4)))
        else:
            regime = "transition"
            stress_prob = 0.35

        return {
            "regime": regime,
            "stress_probability": round(stress_prob, 4),
            "current_vol": round(curr_vol, 6),
            "drawdown": round(dd, 6),
            "vol_ratio": round(vol_ratio, 4),
            "q_low": round(q_low, 6),
            "q_high": round(q_high, 6),
        }


def classify_regimes_series(
    records: List[Dict[str, Any]],
    detector: Optional[CausalRegimeDetector] = None,
) -> List[Dict[str, Any]]:
    """Classify regime across all rows in a series causally."""
    det = detector or CausalRegimeDetector()
    return [det.classify_at_index(records, t) for t in range(len(records))]
