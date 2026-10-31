"""Dependency-free TRACE-Fin smoke implementation.

The production retriever can replace the deterministic evidence_signal while
preserving the forecast and audit interfaces.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Iterable, List, Sequence, Tuple

from baseline import linear_trend_forecast, random_walk_forecast


def returns(prices: Sequence[float]) -> List[float]:
    return [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices))]


def online_regime(prices: Sequence[float], window: int = 20) -> Tuple[str, float]:
    rs = returns(prices)
    if len(rs) < 4:
        return "unknown", 0.0
    current = rs[-window:]
    current_vol = math.sqrt(sum(x * x for x in current) / len(current))
    prior_vols = []
    for end in range(max(4, len(rs) - 5 * window), len(rs) - window + 1):
        chunk = rs[max(0, end - window):end]
        if len(chunk) >= 4:
            prior_vols.append(math.sqrt(sum(x * x for x in chunk) / len(chunk)))
    baseline = median(prior_vols) if prior_vols else current_vol
    ratio = current_vol / max(baseline, 1e-9)
    if ratio >= 1.8:
        return "high_volatility", min(1.0, (ratio - 1.0) / 2.0)
    if ratio <= 0.65:
        return "calm", min(1.0, (1.0 - ratio) / 0.65)
    return "normal", 0.1


def conformal_interval(point: float, residuals: Sequence[float], alpha: float = 0.1) -> Tuple[float, float]:
    if not residuals:
        return point, point
    errors = sorted(abs(float(x)) for x in residuals)
    index = min(len(errors) - 1, max(0, math.ceil((len(errors) + 1) * (1.0 - alpha)) - 1))
    radius = errors[index]
    return point - radius, point + radius


def forecast_with_audit(
    history: Iterable[float],
    horizon: int,
    evidence_signal: float = 0.0,
    residuals: Sequence[float] = (),
    cutoff: str = "synthetic",
    evidence_ids: Sequence[str] = (),
) -> dict:
    prices = [float(x) for x in history]
    if len(prices) < 2:
        raise ValueError("history must contain at least two values")
    regime, shift_probability = online_regime(prices)
    rw = random_walk_forecast(prices, horizon)
    trend = linear_trend_forecast(prices, horizon)
    scale = max(abs(prices[-1]), 1e-9)
    bounded_signal = max(-0.03, min(0.03, float(evidence_signal)))
    trend_weight = 0.15 if regime == "high_volatility" else (0.55 if regime == "calm" else 0.35)
    points = [
        (1.0 - trend_weight) * rw[i] + trend_weight * trend[i] + bounded_signal * scale * (i + 1)
        for i in range(horizon)
    ]
    lower, upper = conformal_interval(points[0], residuals)
    payload = {
        "cutoff": cutoff,
        "regime": regime,
        "shift_probability": round(shift_probability, 6),
        "evidence_ids": list(evidence_ids),
        "point_forecast": points,
        "interval": [lower, upper],
    }
    payload["input_hash"] = hashlib.sha256(
        json.dumps({"history": prices, "evidence_signal": bounded_signal}, sort_keys=True).encode()
    ).hexdigest()
    return payload
