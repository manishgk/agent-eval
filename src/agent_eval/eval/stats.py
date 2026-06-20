"""Reliability statistics — the heart of the framework.

These functions turn N repeated pass/fail outcomes into the metrics that make
non-determinism legible:

* **reliability**   – observed success rate (successes / n)
* **Wilson CI**     – a confidence interval for that rate that stays valid for
                      small N and near 0/1, unlike the naive normal approximation
* **flake rate**    – how inconsistent the case is: the fraction of reps that
                      disagree with the majority outcome (0 = perfectly stable,
                      0.5 = a coin flip)
* **pass^k**        – the probability that *all* of k independent reps pass,
                      estimated without replacement from the observed reps
"""

from __future__ import annotations

import math

from scipy.stats import norm


def reliability(successes: int, n: int) -> float:
    """Observed success rate: successes / n."""
    return successes / n if n else 0.0


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def flake_rate(successes: int, n: int) -> float:
    """Fraction of reps disagreeing with the majority outcome.

    0 when every rep agrees; approaches 0.5 as outcomes split evenly. A case is
    "flaky" whenever ``0 < successes < n``.
    """
    if n == 0:
        return 0.0
    return min(successes, n - successes) / n


def is_flaky(successes: int, n: int) -> bool:
    """True when reps disagree (neither all-pass nor all-fail)."""
    return 0 < successes < n


def pass_hat_k(successes: int, n: int, k: int) -> float:
    """Unbiased estimate of P(all k of k sampled reps pass), sampled without
    replacement from the n observed reps. Returns 0.0 when k > successes and
    1.0 when every rep passed."""
    if k <= 0 or k > n:
        return math.nan
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(n, k)
