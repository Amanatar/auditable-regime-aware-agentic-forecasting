"""Dependency-free statistical and trading metrics for rolling forecasts."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Sequence


def diebold_mariano_hac(actual: Sequence[float], first: Sequence[float], second: Sequence[float], lag: int = 1) -> dict:
    """Diebold–Mariano-style statistic with Bartlett Newey–West variance.

    This remains dependency-free and is intended for transparent diagnostics;
    publication analysis should cross-check it against a validated statistics
    package. The loss is squared-error difference (first minus second).
    """
    losses = [(a - x) ** 2 - (a - y) ** 2 for a, x, y in zip(actual, first, second)]
    if len(losses) < 3:
        return {"stat": 0.0, "p_value": 1.0, "n": len(losses)}
    avg = mean(losses)
    n = len(losses)
    centered = [x - avg for x in losses]
    variance = sum(x * x for x in centered) / n
    max_lag = min(max(0, lag), n - 1)
    for k in range(1, max_lag + 1):
        covariance = sum((losses[i] - avg) * (losses[i - k] - avg) for i in range(k, len(losses))) / len(losses)
        variance += 2.0 * (1.0 - k / (max_lag + 1.0)) * covariance
    if variance <= 1e-12:
        statistic = 0.0 if abs(avg) <= 1e-12 else math.copysign(float("inf"), avg)
        p_value = 1.0 if statistic == 0.0 else 0.0
        return {"stat": statistic, "p_value": p_value, "n": n, "lag": max_lag}
    statistic = avg / math.sqrt(variance / n)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {"stat": statistic, "p_value": p_value, "n": n, "lag": max_lag}


def diebold_mariano_squared(actual: Sequence[float], first: Sequence[float], second: Sequence[float], lag: int = 1) -> dict:
    """Backward-compatible alias for the HAC implementation."""
    return diebold_mariano_hac(actual, first, second, lag=lag)


def trading_metrics(last_prices: Sequence[float], forecasts: Sequence[float], actual: Sequence[float], cost_bps: float = 10.0) -> dict:
    positions, pnl = [], []
    previous = 0.0
    cost = cost_bps / 10000.0
    for last, prediction, realized in zip(last_prices, forecasts, actual):
        position = 1.0 if prediction > last else (-1.0 if prediction < last else 0.0)
        pnl.append(position * (realized / last - 1.0) - cost * abs(position - previous))
        positions.append(position)
        previous = position
    avg = mean(pnl) if pnl else 0.0
    sd = pstdev(pnl) if len(pnl) > 1 else 0.0
    wealth, peak, max_drawdown = 1.0, 1.0, 0.0
    for value in pnl:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
    turnover = mean([abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))]) if len(positions) > 1 else 0.0
    return {
        "cost_bps": cost_bps,
        "mean_daily_net_return": avg,
        "annualized_sharpe": (math.sqrt(252.0) * avg / sd) if sd > 0 else 0.0,
        "cumulative_net_return": wealth - 1.0,
        "max_drawdown": max_drawdown,
        "mean_turnover": turnover,
    }
