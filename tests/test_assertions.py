"""Tests for tool_call_matches (the deterministic pass/fail assertion)."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

from collections.abc import Callable
from typing import Any

import pytest

from agent_eval.agent.tool_agent import AgentRun
from agent_eval.eval.assertions import tool_call_matches
from agent_eval.eval.case import EvalCase
from agent_eval.providers.base import ToolCall


@pytest.fixture
def make_run() -> Callable[[str | None, dict[str, Any] | None], AgentRun]:
    """Factory for AgentRun with a single (or no) tool call."""

    def _make(tool: str | None, args: dict[str, Any] | None = None) -> AgentRun:
        calls = [ToolCall(name=tool, arguments=args or {})] if tool else []
        return AgentRun(prompt="p", model="m", temperature=1.0, tool_calls=calls)
    return _make


@pytest.mark.parametrize(
    ("case_kwargs", "run_tool", "run_args", "expected"),
    [
        pytest.param(
            {"expected_tool": "get_weather", "expected_args": {"city": "Chicago"}},
            "get_weather", {"city": "Chicago", "units": "F"}, True,
            id="correct_tool_and_subset_args",
        ),
        pytest.param(
            {"expected_tool": "get_weather"},
            "send_email", None, False,
            id="wrong_tool_fails",
        ),
        pytest.param(
            {"expected_tool": "get_weather", "expected_args": {"city": "chicago"}},
            "get_weather", {"city": "Chicago"}, True,
            id="case_insensitive_args",
        ),
        pytest.param(
            {
                "expected_tool": "get_weather",
                "expected_args": {"city": "Chicago"},
                "arg_match": "exact",
            },
            "get_weather", {"city": "Chicago", "units": "F"}, False,
            id="exact_match_rejects_extra_args",
        ),
        pytest.param(
            {
                "expected_tool": "get_weather",
                "expected_args": {"city": "Chicago"},
                "arg_match": "exact",
            },
            "get_weather", {"city": "Chicago"}, True,
            id="exact_match_accepts_exact_args",
        ),
        pytest.param(
            {"expected_tool": None},
            None, None, True,
            id="expected_no_tool_and_none_called",
        ),
        pytest.param(
            {"expected_tool": None},
            "get_weather", None, False,
            id="expected_no_tool_but_tool_called",
        ),
    ],
)
def test_tool_call_matches(
    make_run: Callable[[str | None, dict[str, Any] | None], AgentRun],
    case_kwargs: dict[str, Any],
    run_tool: str | None,
    run_args: dict[str, Any] | None,
    expected: bool,
) -> None:
    """tool_call_matches agrees with the expected pass/fail for each scenario."""
    case = EvalCase(id="c", prompt="p", **case_kwargs)
    assert tool_call_matches(case, make_run(run_tool, run_args)) is expected
