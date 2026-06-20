"""Provider abstraction.

The rest of the framework depends only on this interface, never on a concrete
SDK. That keeps the Anthropic client swappable for OpenAI/Bedrock/etc. later
(see the roadmap in the README) without touching the runner, scorers, or report.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool/function call emitted by the model."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ProviderResponse(BaseModel):
    """Normalized response from any provider."""

    model: str
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None
    latency_ms: float | None = None

    @property
    def first_tool_call(self) -> ToolCall | None:
        """The first tool call in the response, if any."""
        return self.tool_calls[0] if self.tool_calls else None


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can take a prompt + tool specs and return tool calls."""

    model: str

    async def complete_with_tools(
        self,
        *,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        temperature: float,
    ) -> ProviderResponse:
        """Send one user prompt with the given tool specs and return the response."""
