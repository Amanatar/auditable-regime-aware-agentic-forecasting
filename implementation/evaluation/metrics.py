"""Dependency-free statistical and trading metrics for rolling forecasts."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Sequence


def diebold_mariano_squared(actual: Sequence[float], first: Sequence[float], second: Sequence[float], lag: int = 1) -> dict:
    losses = [(a - x) ** 2 - (a - y) ** 2 for a, x, y in zip(actual, first, second)]
    if len(losses) < 3:
        return {"stat": 0.0, "p_value": 1.0, "n": len(losses)}
    avg = mean(losses)
    variance = sum((x - avg) ** 2 for x in losses) / len(losses)
    for k in range(1, min(lag, len(losses) - 1) + 1):
        covariance = sum((losses[i] - avg) * (losses[i - k] - avg) for i in range(k, len(losses))) / len(losses)
        variance += 2.0 * covariance
    statistic = avg / math.sqrt(max(variance, 1e-12) / len(losses))
    return {"stat": statistic, "p_value": math.erfc(abs(statistic) / math.sqrt(2.0)), "n": len(losses)}


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
