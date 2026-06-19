"""Tool-call correctness assertions.

The primary, deterministic scorer: did the agent call the expected tool with the
expected arguments? This is intrinsic to evaluating tool-calling and runs on
every rep regardless of whether an LLM judge is also configured.
"""

from __future__ import annotations

from typing import Any

from agent_eval.agent.tool_agent import AgentRun
from agent_eval.eval.case import EvalCase


def _normalize(value: Any, case_insensitive: bool) -> Any:
    if case_insensitive and isinstance(value, str):
        return value.strip().lower()
    return value


def _args_match(case: EvalCase, actual: dict[str, Any]) -> bool:
    expected = case.expected_args
    if case.arg_match == "exact" and set(expected.keys()) != set(actual.keys()):
        return False
    for key, exp_val in expected.items():
        if key not in actual:
            return False
        if _normalize(actual[key], case.case_insensitive) != _normalize(
            exp_val, case.case_insensitive
        ):
            return False
    return True


def tool_call_matches(case: EvalCase, run: AgentRun) -> bool:
    """True if the agent's first tool call satisfies the case's expectation.

    When ``expected_tool`` is None, the expectation is that *no* tool is called.
    """
    call = run.first_tool_call
    if case.expected_tool is None:
        return call is None
    if call is None or call.name != case.expected_tool:
        return False
    return _args_match(case, call.arguments)
