"""Tests for the shared ProviderResponse/ToolCall models in providers/base.py."""

from agent_eval.providers.base import ProviderResponse, ToolCall


def test_first_tool_call_returns_first_when_present() -> None:
    """first_tool_call returns the first ToolCall when one or more are present."""
    first = ToolCall(name="get_weather", arguments={"city": "Chicago"})
    second = ToolCall(name="book_hotel", arguments={"city": "Austin"})
    resp = ProviderResponse(model="m", tool_calls=[first, second])
    assert resp.first_tool_call == first


def test_first_tool_call_returns_none_when_empty() -> None:
    """first_tool_call is None when no tool calls were made."""
    resp = ProviderResponse(model="m", tool_calls=[])
    assert resp.first_tool_call is None
