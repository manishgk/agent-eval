"""Anthropic Claude provider with tool use and production-grade retries."""

from __future__ import annotations

import os
import time
from typing import Any

from anthropic import APIStatusError, AsyncAnthropic, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from agent_eval.providers.base import ProviderResponse, ToolCall

DEFAULT_AGENT_MODEL = os.getenv("AGENT_EVAL_AGENT_MODEL", "claude-haiku-4-5-20251001")

_RETRYABLE = (RateLimitError, APIStatusError)

# Sampling params (temperature/top_p/top_k) are removed on Fable 5 and Opus 4.7+
# and return a 400 if sent. Haiku 4.5 / Sonnet 4.6 / older still accept them.
# The default agent model (Haiku 4.5) supports temperature, which we rely on to
# elicit non-determinism; we just drop it for models that reject it.
_NO_TEMPERATURE_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable-5")


def _supports_temperature(model: str) -> bool:
    return not model.startswith(_NO_TEMPERATURE_PREFIXES)


class AnthropicProvider:
    """Concrete :class:`~agent_eval.providers.base.LLMProvider` backed by Claude."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_AGENT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def complete_with_tools(
        self,
        *,
        prompt: str,
        system: str,
        tools: list[dict[str, Any]],
        temperature: float,
    ) -> ProviderResponse:
        """Call the Anthropic Messages API and normalize the response."""
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": tools,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _supports_temperature(self.model):
            kwargs["temperature"] = temperature
        # Errors propagate as-is; the @retry decorator above decides whether to
        # retry (RateLimitError/APIStatusError) or let them surface.
        message = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in message.content:
            match block.type:
                case "text":
                    text_parts.append(block.text)
                case "tool_use":
                    tool_calls.append(
                        ToolCall(
                            name=block.name,
                            arguments=dict(block.input or {}),
                            call_id=block.id,
                        )
                    )
        return ProviderResponse(
            model=message.model,
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason,
            latency_ms=latency_ms,
        )
