"""Result models. These are the serializable artifacts (JSON) that the CLI,
HTML report, and Streamlit dashboard all consume — keeping reporting decoupled
from execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, Field

from agent_eval.eval import stats


class JudgeVerdict(BaseModel):
    """Structured grade returned by the LLM judge for one run."""

    passed: bool
    score: float  # 0.0 - 1.0
    reasoning: str


class RepResult(BaseModel):
    """Outcome of a single repetition of a case."""

    index: int
    passed: bool
    assertion_passed: bool
    judge: JudgeVerdict | None = None
    tool_called: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float | None = None
    timestamp: str = ""


class CaseResult(BaseModel):
    """Aggregated reliability metrics for one case across all reps."""

    case_id: str
    prompt: str
    expected_tool: str | None = None
    used_judge: bool = False
    reps: list[RepResult]
    n: int
    successes: int
    reliability: float
    wilson_low: float
    wilson_high: float
    flake_rate: float
    flaky: bool
    pass_hat_k: dict[int, float] = Field(default_factory=dict)

    @property
    def status(self) -> str:
        """Stability class: ``"pass"`` (all reps pass), ``"fail"`` (all fail), or
        ``"flaky"`` (mixed). Derived from integer rep counts so renderers never
        test the float ``reliability`` for equality (avoids float-equality bugs).
        """
        if self.n > 0 and self.successes == self.n:
            return "pass"
        if self.successes == 0:
            return "fail"
        return "flaky"

    @classmethod
    def from_reps(
        cls,
        *,
        case_id: str,
        prompt: str,
        expected_tool: str | None,
        used_judge: bool,
        reps: list[RepResult],
        confidence: float = 0.95,
        ks: tuple[int, ...] = (1, 3, 5),
    ) -> Self:
        """Aggregate a list of repetitions into reliability metrics."""
        n = len(reps)
        successes = sum(1 for r in reps if r.passed)
        low, high = stats.wilson_interval(successes, n, confidence)
        pk = {k: stats.pass_hat_k(successes, n, k) for k in ks if k <= n}
        return cls(
            case_id=case_id,
            prompt=prompt,
            expected_tool=expected_tool,
            used_judge=used_judge,
            reps=reps,
            n=n,
            successes=successes,
            reliability=stats.reliability(successes, n),
            wilson_low=low,
            wilson_high=high,
            flake_rate=stats.flake_rate(successes, n),
            flaky=stats.is_flaky(successes, n),
            pass_hat_k=pk,
        )


class SuiteResult(BaseModel):
    """Aggregated reliability results for every case in a suite."""

    suite_name: str
    model: str
    temperature: float
    reps_per_case: int
    confidence: float = 0.95
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = ""
    cases: list[CaseResult] = Field(default_factory=list)

    @property
    def mean_reliability(self) -> float:
        """Average reliability across all cases."""
        return sum(c.reliability for c in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def flaky_cases(self) -> list[CaseResult]:
        """Cases flagged as flaky."""
        return [c for c in self.cases if c.flaky]

    @property
    def total_reps(self) -> int:
        """Total repetitions run across all cases."""
        return sum(c.n for c in self.cases)
