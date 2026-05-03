from __future__ import annotations

import random
from math import sqrt
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return sum(data) / len(data)


def stdev(values: Iterable[float]) -> float:
    data = list(values)
    n = len(data)
    if n <= 1:
        return 0.0
    m = mean(data)
    variance = sum((value - m) ** 2 for value in data) / (n - 1)
    return sqrt(variance)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    index = q * (len(sorted_values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = index - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def paired_bootstrap_mean_diff_ci(
    left: list[float],
    right: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    if not left:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(left)
    sample_means: list[float] = []
    for _ in range(n_resamples):
        diffs = []
        for _ in range(n):
            idx = rng.randrange(n)
            diffs.append(left[idx] - right[idx])
        sample_means.append(mean(diffs))

    sample_means.sort()
    alpha = (1.0 - ci) / 2.0
    low = percentile(sample_means, alpha)
    high = percentile(sample_means, 1.0 - alpha)
    return (low, high)
