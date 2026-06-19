import math

import pytest

from agent_eval.eval import stats


def test_reliability() -> None:
    assert stats.reliability(8, 10) == pytest.approx(0.8)
    assert stats.reliability(0, 0) == 0.0


def test_wilson_interval_brackets_estimate() -> None:
    low, high = stats.wilson_interval(8, 10, 0.95)
    assert low == pytest.approx(0.490, abs=0.01)
    assert high == pytest.approx(0.943, abs=0.01)
    assert low < 0.8 < high


def test_wilson_interval_at_100pct_is_not_certain() -> None:
    # The headline talking point: 10/10 passes is NOT proof of 100% reliability.
    low, high = stats.wilson_interval(10, 10, 0.95)
    assert high == pytest.approx(1.0)
    assert low < 0.9  # ~0.72 — wide uncertainty at small N


def test_wilson_interval_empty() -> None:
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


def test_flake_rate() -> None:
    assert stats.flake_rate(10, 10) == 0.0
    assert stats.flake_rate(8, 10) == pytest.approx(0.2)
    assert stats.flake_rate(5, 10) == pytest.approx(0.5)
    assert stats.flake_rate(0, 0) == 0.0


def test_is_flaky() -> None:
    assert stats.is_flaky(7, 10) is True
    assert stats.is_flaky(10, 10) is False
    assert stats.is_flaky(0, 10) is False


def test_pass_hat_k() -> None:
    assert stats.pass_hat_k(8, 10, 1) == pytest.approx(0.8)
    assert stats.pass_hat_k(10, 10, 5) == pytest.approx(1.0)
    assert stats.pass_hat_k(3, 10, 5) == 0.0  # can't draw 5 passes from 3
    assert stats.pass_hat_k(8, 10, 3) == pytest.approx(56 / 120)
    assert math.isnan(stats.pass_hat_k(5, 3, 4))  # k > n
