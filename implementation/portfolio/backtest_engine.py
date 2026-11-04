"""Multi-asset walk-forward portfolio execution and backtesting engine for TRACE-Fin."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_portfolio_metrics(
    gross_returns: List[float],
    net_returns: List[float],
    turnover_series: List[float],
    cost_bps: float = 10.0,
) -> Dict[str, float]:
    """Compute institutional quantitative performance and risk metrics."""
    n = len(net_returns)
    if n < 5:
        return {}

    # Wealth paths
    wealth_gross = [1.0]
    wealth_net = [1.0]
    for g, nt in zip(gross_returns, net_returns):
        wealth_gross.append(wealth_gross[-1] * (1.0 + g))
        wealth_net.append(wealth_net[-1] * (1.0 + nt))

    # Annualization factor
    years = max(0.1, n / 252.0)

    # 1. Annualized Return (compounded)
    ann_ret_gross = (wealth_gross[-1] ** (1.0 / years) - 1.0) * 100.0
    ann_ret_net = (wealth_net[-1] ** (1.0 / years) - 1.0) * 100.0

    # 2. Annualized Volatility
    vol_gross = float(np.std(gross_returns)) * math.sqrt(252.0) * 100.0
    vol_net = float(np.std(net_returns)) * math.sqrt(252.0) * 100.0

    # 3. Sharpe Ratios
    sharpe_gross = ann_ret_gross / vol_gross if vol_gross > 0 else 0.0
    sharpe_net = ann_ret_net / vol_net if vol_net > 0 else 0.0

    # 4. Sortino Ratio (Downside deviation)
    downside_returns = [min(0.0, r) for r in net_returns]
    downside_vol = float(np.std(downside_returns)) * math.sqrt(252.0) * 100.0
    sortino_net = ann_ret_net / downside_vol if downside_vol > 0 else 0.0

    # 5. Max Drawdown %
    peak_net = 1.0
    max_dd_net = 0.0
    for w in wealth_net:
        if w > peak_net:
            peak_net = w
        dd = (peak_net - w) / peak_net
        if dd > max_dd_net:
            max_dd_net = dd
    max_dd_pct = max_dd_net * 100.0

    # 6. Calmar Ratio
    calmar_net = ann_ret_net / max_dd_pct if max_dd_pct > 0 else 0.0

    # 7. Annualized Turnover
    total_turnover = sum(turnover_series)
    ann_turnover = total_turnover / years

    # Total transaction costs paid
    total_cost_paid_pct = (wealth_gross[-1] - wealth_net[-1]) / max(wealth_gross[-1], 1e-4) * 100.0

    return {
        "annualized_return_gross_pct": round(ann_ret_gross, 2),
        "annualized_return_net_pct": round(ann_ret_net, 2),
        "annualized_volatility_gross_pct": round(vol_gross, 2),
        "annualized_volatility_net_pct": round(vol_net, 2),
        "gross_sharpe_ratio": round(sharpe_gross, 3),
        "net_sharpe_ratio": round(sharpe_net, 3),
        "sortino_ratio": round(sortino_net, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "calmar_ratio": round(calmar_net, 3),
        "annualized_turnover": round(ann_turnover, 2),
        "cost_drag_bps_cumulative": round(ann_turnover * cost_bps * years, 1),
        "total_cost_drag_pct": round(total_cost_paid_pct, 2),
        "final_wealth_net": round(wealth_net[-1], 4),
        "total_trading_days": n,
    }


class PortfolioBacktestEngine:
    """Simulates daily execution of the 3 strategies across multi-asset timeline."""

    def __init__(
        self,
        cost_bps: float = 10.0,
    ):
        self.cost_rate = cost_bps / 10000.0

    def run_strategy_simulation(
        self,
        dates: List[str],
        asset_returns: Dict[str, List[float]],
        weight_series: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Run daily portfolio accounting given a pre-computed weight trajectory."""
        n_days = len(dates)
        gross_returns: List[float] = []
        net_returns: List[float] = []
        turnover_series: List[float] = []
        equity_curve: List[float] = [1.0]

        symbols = list(asset_returns.keys())
        prev_weights: Dict[str, float] = {s: 0.0 for s in symbols}

        for t in range(n_days):
            weights = weight_series[t]
            # Turnover from prev_weights to weights
            turnover_t = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in symbols)
            cost_t = turnover_t * self.cost_rate

            # Daily return earned by holding weights from day t
            r_gross = sum(weights.get(s, 0.0) * asset_returns[s][t] for s in symbols)
            r_net = r_gross - cost_t

            gross_returns.append(r_gross)
            net_returns.append(r_net)
            turnover_series.append(turnover_t)
            equity_curve.append(equity_curve[-1] * (1.0 + r_net))

            prev_weights = dict(weights)

        metrics = compute_portfolio_metrics(
            gross_returns=gross_returns,
            net_returns=net_returns,
            turnover_series=turnover_series,
            cost_bps=self.cost_rate * 10000.0,
        )
        metrics["equity_curve_sample"] = [round(x, 4) for x in equity_curve[::max(1, len(equity_curve) // 10)]]
        return metrics
