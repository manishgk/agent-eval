"""The repeat engine.

Runs each eval case N times against the agent with bounded concurrency, scores
every repetition, and aggregates into reliability statistics. This is where
"run the test many times" — the core QA-for-LLMs idea — actually happens.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from agent_eval.agent.tool_agent import AgentRun, ToolAgent
from agent_eval.eval.assertions import tool_call_matches
from agent_eval.eval.case import EvalCase, EvalSuite
from agent_eval.eval.judge import LLMJudge
from agent_eval.report.models import CaseResult, RepResult, SuiteResult

ProgressCallback = Callable[[str, int, int], None]


async def _score_rep(
    case: EvalCase,
    index: int,
    run: AgentRun,
    judge: LLMJudge | None,
) -> RepResult:
    assertion_passed = tool_call_matches(case, run)
    verdict = None
    if judge is not None and case.judge_rubric:
        verdict = await judge.grade(case, run)
        passed = verdict.passed
    else:
        passed = assertion_passed
    call = run.first_tool_call
    return RepResult(
        index=index,
        passed=passed,
        assertion_passed=assertion_passed,
        judge=verdict,
        tool_called=call.name if call else None,
        arguments=call.arguments if call else {},
        error=run.error,
        latency_ms=run.latency_ms,
        timestamp=run.timestamp,
    )


async def run_case(
    agent: ToolAgent,
    case: EvalCase,
    *,
    reps: int,
    semaphore: asyncio.Semaphore,
    judge: LLMJudge | None = None,
    confidence: float = 0.95,
) -> CaseResult:
    """Run one case `reps` times against the agent and aggregate the results."""

    async def one(index: int) -> RepResult:
        async with semaphore:
            run = await agent.run(case.prompt)
        return await _score_rep(case, index, run, judge)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(one(i)) for i in range(reps)]
    rep_results = [t.result() for t in tasks]
    return CaseResult.from_reps(
        case_id=case.id,
        prompt=case.prompt,
        expected_tool=case.expected_tool,
        used_judge=judge is not None and bool(case.judge_rubric),
        reps=list(rep_results),
        confidence=confidence,
    )


async def run_suite(
    agent: ToolAgent,
    suite: EvalSuite,
    *,
    reps: int,
    concurrency: int = 5,
    judge: LLMJudge | None = None,
    confidence: float = 0.95,
    progress: ProgressCallback | None = None,
) -> SuiteResult:
    """Run every case in the suite and aggregate into a SuiteResult."""
    semaphore = asyncio.Semaphore(concurrency)
    result = SuiteResult(
        suite_name=suite.name,
        model=agent.provider.model,
        temperature=agent.temperature,
        reps_per_case=reps,
        confidence=confidence,
    )
    for i, case in enumerate(suite.cases, start=1):
        case_result = await run_case(
            agent, case, reps=reps, semaphore=semaphore, judge=judge, confidence=confidence
        )
        result.cases.append(case_result)
        if progress is not None:
            progress(case.id, i, len(suite.cases))
    result.finished_at = datetime.now(UTC).isoformat()
    return result


def run_suite_sync(
    agent: ToolAgent,
    suite: EvalSuite,
    *,
    reps: int,
    concurrency: int = 5,
    judge: LLMJudge | None = None,
    confidence: float = 0.95,
    progress: ProgressCallback | None = None,
) -> SuiteResult:
    """Blocking convenience wrapper for the CLI."""
    return asyncio.run(
        run_suite(
            agent,
            suite,
            reps=reps,
            concurrency=concurrency,
            judge=judge,
            confidence=confidence,
            progress=progress,
        )
    )
