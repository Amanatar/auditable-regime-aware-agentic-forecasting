"""Uncertainty-gated position sizing and turnover buffer policies for TRACE-Fin."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class UncertaintyGatedPositionSizer:
    """Flagship Strategy C: Sizing inversely proportional to conformal prediction interval width,
    with a no-trade cost hurdle gate and a max drawdown circuit breaker.
    """

    def __init__(
        self,
        base_leverage: float = 1.0,
        max_asset_long: float = 0.35,
        max_asset_short: float = 0.20,
        min_interval_width: float = 0.005,
        ratio_scale: float = 2.5,
        turnover_hurdle: float = 0.02,
        drawdown_circuit_breaker: float = -0.10,
        circuit_breaker_scale: float = 0.50,
    ):
        self.base_leverage = base_leverage
        self.max_asset_long = max_asset_long
        self.max_asset_short = max_asset_short
        self.min_interval_width = min_interval_width
        self.ratio_scale = ratio_scale
        self.turnover_hurdle = turnover_hurdle
        self.drawdown_circuit_breaker = drawdown_circuit_breaker
        self.circuit_breaker_scale = circuit_breaker_scale

    def compute_target_weights(
        self,
        forecasts: Dict[str, float],
        widths: Dict[str, float],
        regimes: Dict[str, str],
        stress_probs: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute uncertainty-inverse weights: w_i ~ mu_hat_i / W_i."""
        raw_weights: Dict[str, float] = {}

        for sym, mu in forecasts.items():
            w_width = max(self.min_interval_width, widths.get(sym, 0.03))
            # Uncertainty-inverse ratio: predicted edge divided by conformal uncertainty
            ratio = mu / w_width

            # Regime conditioning: during stress, damp exposure
            sp = stress_probs.get(sym, 0.33)
            stress_discount = max(0.40, 1.0 - 0.45 * sp)
            raw = ratio * self.ratio_scale * stress_discount

            # Clip individual limits
            clipped = min(self.max_asset_long, max(-self.max_asset_short, raw))
            raw_weights[sym] = clipped

        # Normalize total gross leverage to base_leverage if exceeded
        gross = sum(abs(v) for v in raw_weights.values())
        if gross > self.base_leverage and gross > 0:
            scale = self.base_leverage / gross
            raw_weights = {k: v * scale for k, v in raw_weights.items()}

        return raw_weights

    def apply_turnover_gate(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply no-trade cost hurdle: only rebalance if proposed change exceeds threshold."""
        rebalanced: Dict[str, float] = {}
        for sym, target in target_weights.items():
            curr = current_weights.get(sym, 0.0)
            if abs(target - curr) < self.turnover_hurdle:
                rebalanced[sym] = curr  # Do not trade; preserve capital
            else:
                rebalanced[sym] = target
        return rebalanced

    def apply_circuit_breaker(
        self,
        weights: Dict[str, float],
        current_drawdown: float,
    ) -> Dict[str, float]:
        """Halve gross exposure if trailing portfolio drawdown crosses risk limit."""
        if current_drawdown <= self.drawdown_circuit_breaker:
            return {k: v * self.circuit_breaker_scale for k, v in weights.items()}
        return weights


class UncalibratedPositionSizer:
    """Baseline B: Point-forecast sizing without uncertainty division or cost gating."""

    def __init__(
        self,
        base_leverage: float = 1.0,
        max_asset_long: float = 0.35,
        max_asset_short: float = 0.20,
        scale: float = 50.0,
    ):
        self.base_leverage = base_leverage
        self.max_asset_long = max_asset_long
        self.max_asset_short = max_asset_short
        self.scale = scale

    def compute_weights(
        self,
        forecasts: Dict[str, float],
    ) -> Dict[str, float]:
        raw_weights: Dict[str, float] = {}
        for sym, mu in forecasts.items():
            # Linear scaling without dividing by conformal width W
            raw = mu * self.scale
            clipped = min(self.max_asset_long, max(-self.max_asset_short, raw))
            raw_weights[sym] = clipped

        gross = sum(abs(v) for v in raw_weights.values())
        if gross > self.base_leverage and gross > 0:
            scale = self.base_leverage / gross
            raw_weights = {k: v * scale for k, v in raw_weights.items()}
        return raw_weights


class EqualWeightPositionSizer:
    """Baseline A: Static Equal-Weight Buy-and-Hold Benchmark."""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.weight = 1.0 / len(symbols) if symbols else 0.0

    def compute_weights(self) -> Dict[str, float]:
        return {s: self.weight for s in self.symbols}
