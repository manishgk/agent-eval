"""Sample tool registry for the demo "travel assistant" agent.

We only need the tool *specifications* (name + JSON schema) to evaluate whether
the agent selects the right tool with the right arguments — the tools are never
actually executed, so no side effects and no live integrations are required.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema=_obj(
            {"city": {"type": "string", "description": "City name, e.g. 'Chicago'."}},
            ["city"],
        ),
    ),
    Tool(
        name="search_flights",
        description="Search available flights between two cities on a date.",
        input_schema=_obj(
            {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "description": "ISO date, e.g. 2026-06-19."},
            },
            ["origin", "destination", "date"],
        ),
    ),
    Tool(
        name="book_hotel",
        description="Book a hotel room in a city for a date range.",
        input_schema=_obj(
            {
                "city": {"type": "string"},
                "check_in": {"type": "string"},
                "check_out": {"type": "string"},
            },
            ["city", "check_in", "check_out"],
        ),
    ),
    Tool(
        name="send_email",
        description="Send an email to a recipient.",
        input_schema=_obj(
            {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            ["to", "subject", "body"],
        ),
    ),
]


def to_anthropic_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Render tool specs in the Anthropic Messages API tool format."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
