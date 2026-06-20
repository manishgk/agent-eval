"""Tests for the repeat-run engine (run_case/run_suite)."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

import asyncio

import pytest

from agent_eval.agent.tool_agent import ToolAgent
from agent_eval.eval.case import EvalCase, EvalSuite
from agent_eval.eval.runner import run_case, run_suite
from agent_eval.providers.base import ProviderResponse, ToolCall


class CountingProvider:
    """First `successes` calls return the correct tool; the rest are wrong.
    Single-threaded asyncio means the counter increment is race-free, giving a
    deterministic success count regardless of concurrency.
    """

    model = "stub-model"

    def __init__(self, successes: int) -> None:
        self.successes = successes
        self.calls = 0

    # system/tools/temperature are part of the LLMProvider protocol signature
    # but this stub only needs to count calls.
    # pylint: disable=unused-argument
    async def complete_with_tools(
        self, *, prompt, system, tools, temperature
    ) -> ProviderResponse:
        """Return get_weather for Chicago until `successes` is exhausted."""
        self.calls += 1
        city = "Chicago" if self.calls <= self.successes else "Seattle"
        return ProviderResponse(
            model=self.model,
            tool_calls=[ToolCall(name="get_weather", arguments={"city": city})],
            latency_ms=1.0,
        )


@pytest.fixture
def case() -> EvalCase:
    """An EvalCase expecting get_weather(city=Chicago)."""
    return EvalCase(
        id="weather",
        prompt="weather in Chicago?",
        expected_tool="get_weather",
        expected_args={"city": "Chicago"},
    )


def test_run_case_computes_reliability(case: EvalCase) -> None:
    """run_case aggregates reps into reliability/flake/Wilson-CI metrics."""
    result = asyncio.run(
        run_case(
            ToolAgent(CountingProvider(successes=7)),
            case,
            reps=10,
            semaphore=asyncio.Semaphore(5),
        )
    )
    assert result.n == 10
    assert result.successes == 7
    assert result.reliability == pytest.approx(0.7)
    assert result.flaky is True
    assert result.flake_rate == pytest.approx(0.3)
    assert result.wilson_low < 0.7 < result.wilson_high


def test_run_suite_aggregates(case: EvalCase) -> None:
    """run_suite aggregates per-case results into suite-level metrics."""
    result = asyncio.run(
        run_suite(
            ToolAgent(CountingProvider(successes=10)),
            EvalSuite(name="s", cases=[case]),
            reps=5,
            concurrency=3,
        )
    )
    assert len(result.cases) == 1
    assert result.mean_reliability == pytest.approx(1.0)
    assert result.flaky_cases == []
    assert result.total_reps == 5
    assert result.finished_at != ""


def test_provider_error_counts_as_failure(case: EvalCase) -> None:
    """A provider exception is captured as a failed rep, not raised."""

    class BoomProvider:
        """Provider stub that always raises, to exercise the failure path."""

        model = "boom"

        async def complete_with_tools(self, **kwargs) -> ProviderResponse:
            """Always fail."""
            raise RuntimeError("api down")

    result = asyncio.run(
        run_case(
            ToolAgent(BoomProvider()), case, reps=3, semaphore=asyncio.Semaphore(2)
        )
    )
    assert result.successes == 0
    assert all(r.error and "api down" in r.error for r in result.reps)
