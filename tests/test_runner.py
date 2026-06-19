import asyncio

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

    async def complete_with_tools(self, *, prompt, system, tools, temperature) -> ProviderResponse:
        self.calls += 1
        city = "Chicago" if self.calls <= self.successes else "Seattle"
        return ProviderResponse(
            model=self.model,
            tool_calls=[ToolCall(name="get_weather", arguments={"city": city})],
            latency_ms=1.0,
        )


def _case() -> EvalCase:
    return EvalCase(id="weather", prompt="weather in Chicago?", expected_tool="get_weather",
                    expected_args={"city": "Chicago"})


def test_run_case_computes_reliability() -> None:
    agent = ToolAgent(CountingProvider(successes=7))
    result = asyncio.run(
        run_case(agent, _case(), reps=10, semaphore=asyncio.Semaphore(5))
    )
    assert result.n == 10
    assert result.successes == 7
    assert result.reliability == 0.7
    assert result.flaky is True
    assert result.flake_rate == 0.3
    assert result.wilson_low < 0.7 < result.wilson_high


def test_run_suite_aggregates() -> None:
    agent = ToolAgent(CountingProvider(successes=10))
    suite = EvalSuite(name="s", cases=[_case()])
    result = asyncio.run(run_suite(agent, suite, reps=5, concurrency=3))
    assert len(result.cases) == 1
    assert result.mean_reliability == 1.0
    assert result.flaky_cases == []
    assert result.total_reps == 5
    assert result.finished_at != ""


def test_provider_error_counts_as_failure() -> None:
    class BoomProvider:
        model = "boom"

        async def complete_with_tools(self, **kwargs) -> ProviderResponse:
            raise RuntimeError("api down")

    agent = ToolAgent(BoomProvider())
    result = asyncio.run(run_case(agent, _case(), reps=3, semaphore=asyncio.Semaphore(2)))
    assert result.successes == 0
    assert all(r.error and "api down" in r.error for r in result.reps)
