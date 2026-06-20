"""Tests for AnthropicProvider, with the Anthropic SDK client mocked so the
suite spends zero tokens and never requires a real ANTHROPIC_API_KEY."""

# AnthropicProvider exposes no public seam for swapping its SDK client, so
# tests reach into `_client` to install a mock rather than hitting the network.
# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_eval.providers.anthropic import AnthropicProvider, _supports_temperature

_HAIKU = "claude-haiku-4-5-20251001"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-4-5-20251001", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-7", False),
        ("claude-opus-4-8", False),
        ("claude-fable-5", False),
        ("claude-opus-4-7-20260101", False),
    ],
)
def test_supports_temperature(model: str, expected: bool) -> None:
    """Opus 4.7+/Fable 5 reject sampling params; older/other models accept them."""
    assert _supports_temperature(model) is expected


def _fake_message(
    *, model: str, stop_reason: str, content: list[SimpleNamespace]
) -> SimpleNamespace:
    return SimpleNamespace(model=model, stop_reason=stop_reason, content=content)


async def test_complete_with_tools_normalizes_text_and_tool_calls() -> None:
    """A response mixing text and tool_use blocks is normalized into ProviderResponse."""
    provider = AnthropicProvider(model=_HAIKU, api_key="test-key")
    content = [
        SimpleNamespace(type="text", text="Sure, "),
        SimpleNamespace(type="text", text="here you go."),
        SimpleNamespace(
            type="tool_use", name="get_weather", input={"city": "Chicago"}, id="call_1"
        ),
    ]
    message = _fake_message(model=_HAIKU, stop_reason="tool_use", content=content)
    provider._client.messages.create = AsyncMock(return_value=message)  # noqa: SLF001

    resp = await provider.complete_with_tools(
        prompt="weather?", system="s", tools=[], temperature=0.7
    )

    assert resp.text == "Sure, here you go."
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Chicago"}
    assert resp.tool_calls[0].call_id == "call_1"
    assert resp.latency_ms is not None and resp.latency_ms >= 0.0


async def test_complete_with_tools_includes_temperature_when_supported() -> None:
    """Models that accept sampling params get temperature passed through."""
    provider = AnthropicProvider(model=_HAIKU, api_key="test-key")
    create = AsyncMock(
        return_value=_fake_message(model=_HAIKU, stop_reason="end_turn", content=[])
    )
    provider._client.messages.create = create  # noqa: SLF001

    await provider.complete_with_tools(prompt="p", system="s", tools=[], temperature=0.5)

    assert create.call_args.kwargs["temperature"] == 0.5


async def test_complete_with_tools_omits_temperature_when_unsupported() -> None:
    """Opus 4.8 rejects sampling params, so temperature must not be sent."""
    provider = AnthropicProvider(model="claude-opus-4-8", api_key="test-key")
    create = AsyncMock(
        return_value=_fake_message(model="claude-opus-4-8", stop_reason="end_turn", content=[])
    )
    provider._client.messages.create = create  # noqa: SLF001

    await provider.complete_with_tools(prompt="p", system="s", tools=[], temperature=0.5)

    assert "temperature" not in create.call_args.kwargs


async def test_complete_with_tools_with_no_tool_calls() -> None:
    """A plain-text-only response has an empty tool_calls list."""
    provider = AnthropicProvider(api_key="test-key")
    content = [SimpleNamespace(type="text", text="No tool needed.")]
    message = _fake_message(model=provider.model, stop_reason="end_turn", content=content)
    provider._client.messages.create = AsyncMock(return_value=message)  # noqa: SLF001

    resp = await provider.complete_with_tools(
        prompt="p", system="s", tools=[], temperature=1.0
    )

    assert resp.tool_calls == []
    assert resp.text == "No tool needed."
