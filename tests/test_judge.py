"""Tests for LLMJudge, with the Anthropic SDK client mocked so the suite
spends zero tokens and never requires a real ANTHROPIC_API_KEY."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# LLMJudge exposes no public seam for swapping its SDK client, so tests reach
# into `_client` to install a mock rather than hitting the network.
# pylint: disable=redefined-outer-name,protected-access

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_eval.agent.tool_agent import AgentRun
from agent_eval.eval.case import EvalCase
from agent_eval.eval.judge import LLMJudge, _format_action
from agent_eval.providers.base import ToolCall


@pytest.fixture
def case() -> EvalCase:
    """A case with a rubric, since judge.grade always needs one to be meaningful."""
    return EvalCase(
        id="c1",
        prompt="Book a hotel in Austin",
        judge_rubric="The agent should book a hotel in the right city.",
    )


def _verdict_message(*, passed: bool, score: float, reasoning: str) -> SimpleNamespace:
    block = SimpleNamespace(
        type="tool_use",
        name="submit_verdict",
        input={"passed": passed, "score": score, "reasoning": reasoning},
    )
    return SimpleNamespace(content=[block])


async def test_grade_returns_verdict_from_submit_verdict_block(case: EvalCase) -> None:
    """A well-formed submit_verdict tool_use block is parsed into a JudgeVerdict."""
    judge = LLMJudge(api_key="test-key")
    run = AgentRun(
        prompt=case.prompt,
        model="m",
        temperature=1.0,
        tool_calls=[ToolCall(name="book_hotel", arguments={"city": "Austin"})],
    )
    judge._client.messages.create = AsyncMock(  # noqa: SLF001
        return_value=_verdict_message(passed=True, score=0.9, reasoning="Correct city.")
    )

    verdict = await judge.grade(case, run)

    assert verdict.passed is True
    assert verdict.score == pytest.approx(0.9)
    assert verdict.reasoning == "Correct city."


async def test_grade_returns_default_verdict_when_no_submit_verdict_block(case: EvalCase) -> None:
    """If the model never calls submit_verdict, a safe failing default is returned."""
    judge = LLMJudge(api_key="test-key")
    run = AgentRun(prompt=case.prompt, model="m", temperature=1.0)
    judge._client.messages.create = AsyncMock(  # noqa: SLF001
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="hmm")])
    )

    verdict = await judge.grade(case, run)

    assert verdict.passed is False
    assert verdict.score == 0.0
    assert verdict.reasoning == "Judge returned no verdict."


async def test_grade_uses_default_rubric_when_case_has_none() -> None:
    """A case without judge_rubric still grades, using the generic fallback rubric."""
    case = EvalCase(id="c2", prompt="Do something")
    judge = LLMJudge(api_key="test-key")
    run = AgentRun(prompt=case.prompt, model="m", temperature=1.0)
    create = AsyncMock(
        return_value=_verdict_message(passed=False, score=0.1, reasoning="No tool called.")
    )
    judge._client.messages.create = create  # noqa: SLF001

    await judge.grade(case, run)

    prompt_sent = create.call_args.kwargs["messages"][0]["content"]
    assert "reasonably satisfy" in prompt_sent


def test_format_action_with_error() -> None:
    """An errored run is formatted as an error message, not a tool call."""
    run = AgentRun(prompt="p", model="m", temperature=1.0, error="RuntimeError: boom")
    assert "errored without acting" in _format_action(run)
    assert "boom" in _format_action(run)


def test_format_action_with_no_tool_call() -> None:
    """A run with no tool call is formatted with its text reply."""
    run = AgentRun(prompt="p", model="m", temperature=1.0, text="okay then")
    assert "called no tool" in _format_action(run)
    assert "okay then" in _format_action(run)


def test_format_action_with_tool_call() -> None:
    """A run with a tool call is formatted with the tool name and arguments."""
    run = AgentRun(
        prompt="p",
        model="m",
        temperature=1.0,
        tool_calls=[ToolCall(name="get_weather", arguments={"city": "Chicago"})],
    )
    formatted = _format_action(run)
    assert "get_weather" in formatted
    assert "Chicago" in formatted
