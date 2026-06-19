"""The system-under-test: a thin tool-calling agent.

Given a user prompt and a tool registry, ask the model which tool to call. This
is deliberately minimal — the point of the project is to *evaluate* its
reliability, not to be a feature-rich agent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent_eval.providers.base import LLMProvider, ToolCall
from agent_eval.tools.registry import DEFAULT_TOOLS, Tool, to_anthropic_tools

DEFAULT_SYSTEM_PROMPT = (
    "You are a travel assistant. Use the provided tools to act on the user's "
    "request. Call exactly one tool that best satisfies the request. If no tool "
    "fits, respond in plain text without calling a tool."
)


class AgentRun(BaseModel):
    """One execution of the agent for a single prompt."""

    prompt: str
    model: str
    temperature: float
    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    stop_reason: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def first_tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


class ToolAgent:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        tools: list[Tool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 1.0,
    ) -> None:
        self.provider = provider
        self.tools = tools if tools is not None else DEFAULT_TOOLS
        self.system_prompt = system_prompt
        self.temperature = temperature

    async def run(self, prompt: str) -> AgentRun:
        """Execute once. Any provider error is captured (not raised) so a single
        failed call counts as a failed repetition rather than aborting the suite."""
        try:
            resp = await self.provider.complete_with_tools(
                prompt=prompt,
                system=self.system_prompt,
                tools=to_anthropic_tools(self.tools),
                temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed rep
            return AgentRun(
                prompt=prompt,
                model=self.provider.model,
                temperature=self.temperature,
                error=f"{type(exc).__name__}: {exc}",
            )

        return AgentRun(
            prompt=prompt,
            model=resp.model,
            temperature=self.temperature,
            tool_calls=resp.tool_calls,
            text=resp.text,
            stop_reason=resp.stop_reason,
            latency_ms=resp.latency_ms,
        )
