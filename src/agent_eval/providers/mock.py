"""Deterministic-ish mock provider for offline demos, CI, and `--mock` runs.

Routes prompts to tools by keyword and deliberately injects flakiness on
ambiguous prompts, so the reliability metrics have something interesting to show
without spending tokens or needing an API key.
"""

from __future__ import annotations

import random
import re

from agent_eval.providers.base import ProviderResponse, ToolCall

_KNOWN_CITIES = ["Chicago", "Seattle", "Denver", "New York", "NYC", "Boston", "Austin"]


def _find_cities(prompt: str) -> list[str]:
    """All known cities mentioned in the prompt, in the order they appear.

    Used for two-city tools like search_flights, where origin/destination must
    be told apart by position in the prompt rather than by priority list order.
    """
    matches: list[tuple[int, str]] = []
    for city in _KNOWN_CITIES:
        m = re.search(rf"\b{re.escape(city)}\b", prompt, re.IGNORECASE)
        if m:
            matches.append((m.start(), "New York" if city == "NYC" else city))
    matches.sort(key=lambda t: t[0])
    seen: list[str] = []
    for _, name in matches:
        if name not in seen:
            seen.append(name)
    return seen


def _find_city(prompt: str) -> str | None:
    cities = _find_cities(prompt)
    return cities[0] if cities else None


class MockProvider:
    """Implements the LLMProvider protocol with scripted, optionally flaky logic."""

    def __init__(
        self, *, model: str = "mock-agent", flakiness: float = 0.25, seed: int = 7
    ) -> None:
        self.model = model
        self.flakiness = flakiness
        self._rng = random.Random(seed)

    # system/tools/temperature are part of the LLMProvider protocol signature
    # but this mock routes purely on prompt keywords.
    # pylint: disable=unused-argument
    async def complete_with_tools(
        self,
        *,
        prompt: str,
        system: str,
        tools: list[dict[str, object]],
        temperature: float,
    ) -> ProviderResponse:
        """Route the prompt to a tool by keyword, with scripted flakiness."""
        p = prompt.lower()
        cities = _find_cities(prompt)
        city = cities[0] if cities else None
        ambiguous = any(w in p for w in ("plan", "thinking about", "what do you think"))

        if ambiguous:
            # Flaky: usually a sensible first step, sometimes a premature/odd one.
            if self._rng.random() < self.flakiness:
                choice = self._rng.choice(["book_hotel", "send_email", None])
            else:
                choice = "search_flights" if "plan" in p else "get_weather"
            call = self._build(choice, city, cities)
        elif "weather" in p or "raining" in p:
            call = self._build("get_weather", city, cities)
        elif "flight" in p:
            call = self._build("search_flights", city, cities)
        elif "hotel" in p:
            call = self._build("book_hotel", city, cities)
        elif "email" in p:
            call = self._build("send_email", city, cities)
        else:
            call = None  # chit-chat: no tool

        return ProviderResponse(
            model=self.model,
            text="" if call else "You're welcome!",
            tool_calls=[call] if call else [],
            stop_reason="tool_use" if call else "end_turn",
            latency_ms=float(self._rng.randint(180, 420)),
        )

    @staticmethod
    def _build(
        tool: str | None, city: str | None, cities: list[str] | None = None
    ) -> ToolCall | None:
        if tool is None:
            return None
        cities = cities or ([city] if city else [])
        args: dict[str, object] = {}
        if tool == "get_weather":
            args = {"city": city or "Chicago"}
        elif tool == "search_flights":
            # Two cities mentioned ("from X to Y") -> first is origin, second is
            # destination, in prompt order. One city or none -> default origin.
            if len(cities) >= 2:
                origin, destination = cities[0], cities[1]
            else:
                origin, destination = "Chicago", (cities[0] if cities else "New York")
            args = {"origin": origin, "destination": destination, "date": "2026-07-04"}
        elif tool == "book_hotel":
            args = {"city": city or "New York", "check_in": "2026-07-04", "check_out": "2026-07-06"}
        elif tool == "send_email":
            args = {"to": "john@example.com", "subject": "Trip", "body": "Confirmed."}
        return ToolCall(name=tool, arguments=args)
