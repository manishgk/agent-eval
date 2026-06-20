"""Tests for the reliability statistics (eval/stats.py)."""

import math

import pytest

from agent_eval.eval import stats


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [(8, 10, 0.8), (0, 0, 0.0)],
)
def test_reliability(successes: int, n: int, expected: float) -> None:
    """reliability() is successes/n, and 0.0 when n is 0."""
    assert stats.reliability(successes, n) == pytest.approx(expected)


def test_wilson_interval_brackets_estimate() -> None:
    """The Wilson interval contains the point estimate."""
    low, high = stats.wilson_interval(8, 10, 0.95)
    assert low == pytest.approx(0.490, abs=0.01)
    assert high == pytest.approx(0.943, abs=0.01)
    assert low < 0.8 < high


def test_wilson_interval_at_100pct_is_not_certain() -> None:
    """10/10 passes still leaves a wide lower bound, not certainty."""
    low, high = stats.wilson_interval(10, 10, 0.95)
    assert high == pytest.approx(1.0)
    assert low < 0.9


def test_wilson_interval_empty() -> None:
    """n=0 returns a degenerate (0.0, 0.0) interval."""
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [(10, 10, 0.0), (8, 10, 0.2), (5, 10, 0.5), (0, 0, 0.0)],
)
def test_flake_rate(successes: int, n: int, expected: float) -> None:
    """flake_rate() is the minority fraction, 0 when all reps agree."""
    assert stats.flake_rate(successes, n) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [(7, 10, True), (10, 10, False), (0, 10, False)],
)
def test_is_flaky(successes: int, n: int, expected: bool) -> None:
    """is_flaky() is true only when reps disagree."""
    assert stats.is_flaky(successes, n) is expected


@pytest.mark.parametrize(
    ("successes", "n", "k", "expected"),
    [(8, 10, 1, 0.8), (10, 10, 5, 1.0), (3, 10, 5, 0.0), (8, 10, 3, 56 / 120)],
)
def test_pass_hat_k(successes: int, n: int, k: int, expected: float) -> None:
    """pass_hat_k() matches the hypergeometric probability of k passes."""
    assert stats.pass_hat_k(successes, n, k) == pytest.approx(expected)


def test_pass_hat_k_nan_when_k_exceeds_n() -> None:
    """pass_hat_k() is NaN when k exceeds the number of reps."""
    assert math.isnan(stats.pass_hat_k(5, 3, 4))
