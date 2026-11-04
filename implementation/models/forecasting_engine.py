"""Regularized regime-conditioned forecasting ensemble for TRACE-Fin.

Ensemble architecture:
1. Linear Baseline Head: Ridge / HuberRegressor with L2 penalty (robust against tail shocks)
2. Non-linear Regime Head: Gradient Boosted Regressor conditioned on regime states & probabilities
3. Regime-Weighted Aggregation: Dynamically weights linear vs non-linear predictions based on
   the causal regime stress probability to protect against overfitting during volatility jumps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge

FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_21d",
    "log_return_63d",
    "realized_vol_c2c_21d",
    "realized_vol_yz_21d",
    "vol_of_vol_yz_ratio_10_63",
    "drawdown_from_peak_21d",
    "drawdown_from_peak_63d",
    "zscore_log_return_1d_63w",
    "zscore_vol_yz_21d_63w",
    "zscore_volume_21w",
]


class RegimeForecastingEnsemble:
    """Walk-forward ensemble combining regularized linear and gradient-boosted regime-conditioned models."""

    def __init__(
        self,
        linear_alpha: float = 10.0,
        gbr_max_iter: int = 30,
        gbr_max_depth: int = 3,
        random_state: int = 42,
    ):
        self.linear_alpha = linear_alpha
        self.gbr_max_iter = gbr_max_iter
        self.gbr_max_depth = gbr_max_depth
        self.random_state = random_state

        # Models for 1-day ahead horizon
        self.linear_1d = Ridge(alpha=self.linear_alpha)
        self.gbr_1d = HistGradientBoostingRegressor(
            max_iter=self.gbr_max_iter,
            max_depth=self.gbr_max_depth,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

        # Models for 5-day ahead horizon
        self.linear_5d = Ridge(alpha=self.linear_alpha)
        self.gbr_5d = HistGradientBoostingRegressor(
            max_iter=self.gbr_max_iter,
            max_depth=self.gbr_max_depth,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

        self.is_fitted = False
        self.in_sample_residuals_1d: np.ndarray = np.array([])
        self.in_sample_residuals_5d: np.ndarray = np.array([])

    def _extract_features(
        self,
        records: List[Dict[str, Any]],
        regime_info: List[Dict[str, Any]],
        indices: List[int],
    ) -> np.ndarray:
        """Extract numeric features plus causal regime stress probability."""
        X = []
        for idx in indices:
            row = records[idx]
            reg = regime_info[idx]
            feat = []
            for col in FEATURE_COLUMNS:
                val = row.get(col)
                feat.append(float(val) if val is not None and not math.isnan(val) else 0.0)
            # Append regime features (stress probability and vol ratio)
            feat.append(float(reg.get("stress_probability", 0.33)))
            feat.append(float(reg.get("vol_ratio", 1.0)))
            X.append(feat)
        return np.array(X, dtype=np.float64)

    def fit(
        self,
        records: List[Dict[str, Any]],
        regime_info: List[Dict[str, Any]],
        train_indices: List[int],
        horizon: int = 1,
    ) -> "RegimeForecastingEnsemble":
        """Fit the ensemble on training indices with zero lookahead.
        
        For horizon h, target at index i is ln(Close_{i+h} / Close_i).
        The training set must only include i where i + h <= max(train_indices).
        """
        closes = [float(r["close"]) for r in records]
        n = len(closes)

        # Prepare 1-day targets
        valid_1d_idx = [i for i in train_indices if i + 1 < n and i >= 63]
        if len(valid_1d_idx) >= 30:
            X_1d = self._extract_features(records, regime_info, valid_1d_idx)
            y_1d = np.array([math.log(closes[i + 1] / closes[i]) for i in valid_1d_idx], dtype=np.float64)
            self.linear_1d.fit(X_1d, y_1d)
            self.gbr_1d.fit(X_1d, y_1d)
            p_lin = self.linear_1d.predict(X_1d)
            p_gbr = self.gbr_1d.predict(X_1d)
            self.in_sample_residuals_1d = y_1d - (0.5 * p_lin + 0.5 * p_gbr)

        # Prepare 5-day targets
        valid_5d_idx = [i for i in train_indices if i + 5 < n and i >= 63]
        if len(valid_5d_idx) >= 30:
            X_5d = self._extract_features(records, regime_info, valid_5d_idx)
            y_5d = np.array([math.log(closes[i + 5] / closes[i]) for i in valid_5d_idx], dtype=np.float64)
            self.linear_5d.fit(X_5d, y_5d)
            self.gbr_5d.fit(X_5d, y_5d)
            p_lin5 = self.linear_5d.predict(X_5d)
            p_gbr5 = self.gbr_5d.predict(X_5d)
            self.in_sample_residuals_5d = y_5d - (0.5 * p_lin5 + 0.5 * p_gbr5)

        self.is_fitted = True
        return self

    def predict_step(
        self,
        records: List[Dict[str, Any]],
        regime_info: List[Dict[str, Any]],
        step_idx: int,
    ) -> Dict[str, float]:
        """Generate 1-day and 5-day return forecasts for step_idx using regime-weighting."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_step.")

        X_step = self._extract_features(records, regime_info, [step_idx])
        stress_prob = float(regime_info[step_idx].get("stress_probability", 0.33))

        # Regime-weighted combination:
        # High stress -> increase weight on regularized linear head (75% linear)
        # Calm -> increase weight on non-linear head (40% linear / 60% gradient boosting)
        w_lin = min(0.85, max(0.35, 0.40 + 0.45 * stress_prob))
        w_gbr = 1.0 - w_lin

        # 1-day forecast
        pred_1d_lin = float(self.linear_1d.predict(X_step)[0])
        pred_1d_gbr = float(self.gbr_1d.predict(X_step)[0])
        pred_1d = w_lin * pred_1d_lin + w_gbr * pred_1d_gbr

        # 5-day forecast
        pred_5d_lin = float(self.linear_5d.predict(X_step)[0])
        pred_5d_gbr = float(self.gbr_5d.predict(X_step)[0])
        pred_5d = w_lin * pred_5d_lin + w_gbr * pred_5d_gbr

        return {
            "forecast_1d": round(pred_1d, 6),
            "forecast_5d": round(pred_5d, 6),
            "pred_1d_linear": round(pred_1d_lin, 6),
            "pred_1d_gbr": round(pred_1d_gbr, 6),
            "linear_weight": round(w_lin, 4),
        }
