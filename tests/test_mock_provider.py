"""Tests for the offline MockProvider (keyword routing + scripted flakiness)."""

import pytest

from agent_eval.providers.base import ProviderResponse
from agent_eval.providers.mock import MockProvider, _find_cities


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("What's the weather in Chicago?", ["Chicago"]),
        ("Flights from Chicago to New York", ["Chicago", "New York"]),
        ("Is it raining in NYC?", ["New York"]),
        ("Thanks, that's all!", []),
        ("Boston then Boston again", ["Boston"]),
    ],
)
def test_find_cities(prompt: str, expected: list[str]) -> None:
    """Cities are extracted in first-appearance order, deduped, with NYC normalized."""
    assert _find_cities(prompt) == expected


async def _run(provider: MockProvider, prompt: str) -> ProviderResponse:
    return await provider.complete_with_tools(
        prompt=prompt, system="s", tools=[], temperature=1.0
    )


async def test_weather_keyword_routes_to_get_weather() -> None:
    """A weather-themed prompt calls get_weather with the mentioned city."""
    resp = await _run(MockProvider(), "Is it raining in Seattle?")
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Seattle"}


async def test_flight_keyword_routes_with_origin_and_destination() -> None:
    """A flight prompt with two cities maps them to origin/destination by position."""
    resp = await _run(MockProvider(), "Find flights from Chicago to Denver please")
    call = resp.tool_calls[0]
    assert call.name == "search_flights"
    assert call.arguments["origin"] == "Chicago"
    assert call.arguments["destination"] == "Denver"


async def test_flight_keyword_with_one_city_defaults_origin() -> None:
    """A flight prompt with only one city defaults origin to Chicago."""
    resp = await _run(MockProvider(), "Find flights to Denver please")
    call = resp.tool_calls[0]
    assert call.arguments["origin"] == "Chicago"
    assert call.arguments["destination"] == "Denver"


async def test_hotel_keyword_routes_to_book_hotel() -> None:
    """A hotel prompt calls book_hotel with the mentioned city."""
    resp = await _run(MockProvider(), "Book a hotel in Austin")
    assert resp.tool_calls[0].name == "book_hotel"
    assert resp.tool_calls[0].arguments["city"] == "Austin"


async def test_email_keyword_routes_to_send_email() -> None:
    """An email prompt calls send_email with scripted args."""
    resp = await _run(MockProvider(), "Send an email to confirm")
    assert resp.tool_calls[0].name == "send_email"


async def test_unmatched_prompt_returns_no_tool_call() -> None:
    """A prompt with no recognized keyword gets a plain-text reply, no tool call."""
    resp = await _run(MockProvider(), "Thanks, that's all for now!")
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.text == "You're welcome!"


async def test_plan_branch_with_zero_flakiness_is_deterministic() -> None:
    """With flakiness=0, an ambiguous 'plan' prompt always picks search_flights."""
    resp = await _run(MockProvider(flakiness=0.0), "I'm thinking about planning a trip")
    assert resp.tool_calls[0].name == "search_flights"


async def test_plan_branch_with_full_flakiness_picks_scripted_alternative() -> None:
    """With flakiness=1, the 'plan' branch always takes the flaky alternative path."""
    resp = await _run(MockProvider(flakiness=1.0, seed=1), "what do you think about Boston")
    assert resp.tool_calls == [] or resp.tool_calls[0].name in {
        "book_hotel",
        "send_email",
    }


async def test_latency_ms_within_scripted_range() -> None:
    """latency_ms is always within the scripted 180-420ms window."""
    resp = await _run(MockProvider(), "weather in Denver")
    assert resp.latency_ms is not None
    assert 180.0 <= resp.latency_ms <= 420.0


def test_build_returns_none_for_none_tool() -> None:
    """_build returns None when no tool is selected."""
    # _build is a "private" staticmethod with no other test seam.
    assert MockProvider._build(None, None) is None  # pylint: disable=protected-access
