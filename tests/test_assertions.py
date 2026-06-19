from agent_eval.agent.tool_agent import AgentRun
from agent_eval.eval.assertions import tool_call_matches
from agent_eval.eval.case import EvalCase
from agent_eval.providers.base import ToolCall


def _run(tool: str | None, args: dict | None = None) -> AgentRun:
    calls = [ToolCall(name=tool, arguments=args or {})] if tool else []
    return AgentRun(prompt="p", model="m", temperature=1.0, tool_calls=calls)


def test_correct_tool_and_subset_args() -> None:
    case = EvalCase(id="c", prompt="p", expected_tool="get_weather", expected_args={"city": "Chicago"})
    assert tool_call_matches(case, _run("get_weather", {"city": "Chicago", "units": "F"}))


def test_wrong_tool_fails() -> None:
    case = EvalCase(id="c", prompt="p", expected_tool="get_weather")
    assert not tool_call_matches(case, _run("send_email"))


def test_case_insensitive_args() -> None:
    case = EvalCase(id="c", prompt="p", expected_tool="get_weather", expected_args={"city": "chicago"})
    assert tool_call_matches(case, _run("get_weather", {"city": "Chicago"}))


def test_exact_match_rejects_extra_args() -> None:
    case = EvalCase(
        id="c", prompt="p", expected_tool="get_weather",
        expected_args={"city": "Chicago"}, arg_match="exact",
    )
    assert not tool_call_matches(case, _run("get_weather", {"city": "Chicago", "units": "F"}))
    assert tool_call_matches(case, _run("get_weather", {"city": "Chicago"}))


def test_expected_no_tool() -> None:
    case = EvalCase(id="c", prompt="p", expected_tool=None)
    assert tool_call_matches(case, _run(None))
    assert not tool_call_matches(case, _run("get_weather"))
