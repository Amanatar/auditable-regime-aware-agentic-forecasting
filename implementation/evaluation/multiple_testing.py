"""Small, dependency-free multiple-testing and block-bootstrap utilities."""

from __future__ import annotations

import random
from statistics import mean


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH FDR-adjusted q-values in the original order."""
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank in range(len(order), 0, -1):
        index = order[rank - 1]
        value = min(1.0, p_values[index] * len(order) / rank)
        running = min(running, value)
        adjusted[index] = running
    return adjusted


def moving_block_bootstrap_ci(values: list[float], block_length: int = 5, replicates: int = 4000, seed: int = 7) -> dict:
    """95% moving-block bootstrap interval for the mean of dependent losses."""
    if not values:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "n": 0, "block_length": block_length}
    if len(values) < 3:
        value = mean(values)
        return {"mean": value, "lower_95": value, "upper_95": value, "n": len(values), "block_length": 1}
    block_length = max(1, min(block_length, len(values)))
    blocks = [values[start:start + block_length] for start in range(len(values) - block_length + 1)]
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        sample = []
        while len(sample) < len(values):
            sample.extend(rng.choice(blocks))
        samples.append(mean(sample[:len(values)]))
    samples.sort()
    lower = samples[max(0, int(0.025 * len(samples)) - 1)]
    upper = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {"mean": mean(values), "lower_95": lower, "upper_95": upper, "n": len(values), "block_length": block_length}
