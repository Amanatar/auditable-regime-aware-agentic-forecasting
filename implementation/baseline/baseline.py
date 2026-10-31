"""Small, dependency-free forecasting baselines for TRACE-Fin."""

from __future__ import annotations

from typing import Iterable, List


def random_walk_forecast(history: Iterable[float], horizon: int) -> List[float]:
    values = list(history)
    if not values:
        raise ValueError("history must contain at least one value")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return [float(values[-1])] * horizon


def linear_trend_forecast(history: Iterable[float], horizon: int, window: int = 20) -> List[float]:
    values = list(history)[-window:]
    if len(values) < 2:
        return random_walk_forecast(values, horizon)
    x_bar = (len(values) - 1) / 2.0
    y_bar = sum(values) / len(values)
    denom = sum((i - x_bar) ** 2 for i in range(len(values)))
    slope = sum((i - x_bar) * (y - y_bar) for i, y in enumerate(values)) / denom
    intercept = y_bar - slope * x_bar
    start = len(values)
    return [float(intercept + slope * i) for i in range(start, start + horizon)]
