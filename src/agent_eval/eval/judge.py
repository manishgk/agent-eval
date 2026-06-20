"""LLM-as-a-judge scorer.

For ambiguous cases where exact tool/argument matching is too strict, Claude
grades the agent's chosen action against a natural-language rubric. We use
*forced tool use* (``tool_choice``) so the judge must return a structured,
schema-validated verdict rather than free-form prose we'd have to parse.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import AsyncAnthropic
from anthropic import APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from agent_eval.agent.tool_agent import AgentRun
from agent_eval.eval.case import EvalCase
from agent_eval.report.models import JudgeVerdict

DEFAULT_JUDGE_MODEL = os.getenv("AGENT_EVAL_JUDGE_MODEL", "claude-opus-4-8")

_VERDICT_TOOL: dict[str, Any] = {
    "name": "submit_verdict",
    "description": "Submit your verdict on whether the agent's action satisfies the rubric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean", "description": "Does the action satisfy the rubric?"},
            "score": {
                "type": "number",
                "description": "Quality score from 0.0 (poor) to 1.0 (ideal).",
            },
            "reasoning": {"type": "string", "description": "One or two sentences explaining why."},
        },
        "required": ["passed", "score", "reasoning"],
    },
}

_JUDGE_SYSTEM = (
    "You are a meticulous evaluator of an AI agent's tool-calling decisions. "
    "Given the user's request, the rubric, and the action the agent actually "
    "took, decide whether the action satisfies the rubric. Be strict and "
    "consistent. Always call submit_verdict."
)


def _format_action(run: AgentRun) -> str:
    call = run.first_tool_call
    if run.error:
        return f"The agent errored without acting: {run.error}"
    if call is None:
        text = run.text.strip() or "(empty)"
        return f"The agent called no tool and replied with text: {text!r}"
    return f"The agent called tool {call.name!r} with arguments {json.dumps(call.arguments)}."


class LLMJudge:
    """Grades an agent run against a case's rubric using a Claude model as judge."""

    def __init__(self, *, model: str = DEFAULT_JUDGE_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def grade(self, case: EvalCase, run: AgentRun) -> JudgeVerdict:
        """Ask the judge model to grade `run` against the case's rubric."""
        rubric = case.judge_rubric or "The action should reasonably satisfy the user's request."
        prompt = (
            f"User request:\n{case.prompt}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Agent action:\n{_format_action(run)}"
        )
        # No `temperature`: the default judge model (Opus 4.8) removed sampling
        # params and 400s if they're sent. Forced tool_choice keeps the verdict
        # structured; grading consistency comes from a fixed rubric + schema.
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_JUDGE_SYSTEM,
            tools=[_VERDICT_TOOL],  # type: ignore[list-item]
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "tool_use" and block.name == "submit_verdict":
                return JudgeVerdict.model_validate(block.input)
        # Forced tool use should make this unreachable; fail closed if not.
        return JudgeVerdict(passed=False, score=0.0, reasoning="Judge returned no verdict.")
