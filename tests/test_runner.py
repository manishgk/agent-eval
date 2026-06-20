"""Tests for the repeat-run engine (run_case/run_suite)."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_eval.agent.tool_agent import ToolAgent
from agent_eval.eval.case import EvalCase, EvalSuite
from agent_eval.eval.judge import LLMJudge
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


async def test_run_case_computes_reliability(case: EvalCase) -> None:
    """run_case aggregates reps into reliability/flake/Wilson-CI metrics."""
    result = await run_case(
        ToolAgent(CountingProvider(successes=7)),
        case,
        reps=10,
        semaphore=asyncio.Semaphore(5),
    )
    assert result.n == 10
    assert result.successes == 7
    assert result.reliability == pytest.approx(0.7)
    assert result.flaky is True
    assert result.flake_rate == pytest.approx(0.3)
    assert result.wilson_low < 0.7 < result.wilson_high


async def test_run_suite_aggregates(case: EvalCase) -> None:
    """run_suite aggregates per-case results into suite-level metrics."""
    result = await run_suite(
        ToolAgent(CountingProvider(successes=10)),
        EvalSuite(name="s", cases=[case]),
        reps=5,
        concurrency=3,
    )
    assert len(result.cases) == 1
    assert result.mean_reliability == pytest.approx(1.0)
    assert result.flaky_cases == []
    assert result.total_reps == 5
    assert result.finished_at != ""


async def test_provider_error_counts_as_failure(case: EvalCase) -> None:
    """A provider exception is captured as a failed rep, not raised."""

    class BoomProvider:
        """Provider stub that always raises, to exercise the failure path."""

        model = "boom"

        async def complete_with_tools(self, **kwargs) -> ProviderResponse:
            """Always fail."""
            raise RuntimeError("api down")

    result = await run_case(
        ToolAgent(BoomProvider()), case, reps=3, semaphore=asyncio.Semaphore(2)
    )
    assert result.successes == 0
    assert all(r.error and "api down" in r.error for r in result.reps)


async def test_run_case_defers_to_judge_when_rubric_present() -> None:
    """When a judge and judge_rubric are both present, the judge's verdict wins
    over the deterministic assertion, and used_judge is recorded as True."""
    case = EvalCase(
        id="ambiguous",
        prompt="weather in Chicago?",
        expected_tool="get_weather",
        expected_args={"city": "Seattle"},  # deliberately mismatched
        judge_rubric="Any reasonable weather lookup passes.",
    )
    judge = LLMJudge(api_key="test-key")
    verdict_block = SimpleNamespace(
        type="tool_use",
        name="submit_verdict",
        input={"passed": True, "score": 1.0, "reasoning": "Close enough."},
    )
    judge._client.messages.create = AsyncMock(  # noqa: SLF001  # pylint: disable=protected-access
        return_value=SimpleNamespace(content=[verdict_block])
    )

    result = await run_case(
        ToolAgent(CountingProvider(successes=10)),
        case,
        reps=2,
        semaphore=asyncio.Semaphore(2),
        judge=judge,
    )

    assert result.used_judge is True
    assert result.successes == 2
    assert all(r.assertion_passed is False for r in result.reps)
    assert all(r.passed is True for r in result.reps)
