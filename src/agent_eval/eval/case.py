"""Data-driven eval cases.

Cases live in YAML so test data is separated from the engine — the same pattern
as parametrized/data-driven test suites in traditional QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single expectation about the agent's tool-calling behavior."""

    id: str
    prompt: str
    # Expected behavior. expected_tool=None means "no tool should be called".
    expected_tool: str | None = None
    expected_args: dict[str, Any] = Field(default_factory=dict)
    arg_match: Literal["subset", "exact"] = "subset"
    # Case-insensitive comparison for string argument values.
    case_insensitive: bool = True
    # When set, an LLM judge also grades the run against this rubric. Useful for
    # ambiguous prompts where exact-match is too strict.
    judge_rubric: str | None = None
    description: str | None = None


class EvalSuite(BaseModel):
    """A named collection of eval cases, typically loaded from a YAML file."""

    name: str
    description: str | None = None
    cases: list[EvalCase]


def load_suite(path: str | Path) -> EvalSuite:
    """Load an EvalSuite from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalSuite.model_validate(data)
