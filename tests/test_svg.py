"""Tests for the SVG reliability chart renderer."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

import pytest

from agent_eval.report.models import CaseResult, RepResult, SuiteResult
from agent_eval.report.svg import render_svg


def _case(case_id: str, successes: int, n: int = 10) -> CaseResult:
    reps = [
        RepResult(index=i, passed=i < successes, assertion_passed=i < successes)
        for i in range(n)
    ]
    return CaseResult.from_reps(
        case_id=case_id,
        prompt=f"prompt for {case_id}",
        expected_tool="get_weather",
        used_judge=False,
        reps=reps,
    )


@pytest.fixture
def suite() -> SuiteResult:
    """A demo suite with a stable, a flaky, and a failing case."""
    return SuiteResult(
        suite_name="demo",
        model="mock-agent",
        temperature=1.0,
        reps_per_case=10,
        finished_at="2026-06-19T00:00:00",
        cases=[_case("stable", 10), _case("flaky", 7), _case("failing", 0)],
    )


def test_render_svg_is_valid(suite: SuiteResult) -> None:
    """The rendered output is a well-formed SVG document."""
    svg = render_svg(suite)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")


@pytest.mark.parametrize("case_id", ["stable", "flaky", "failing"])
def test_render_svg_contains_case(suite: SuiteResult, case_id: str) -> None:
    """Every case id appears somewhere in the rendered SVG."""
    assert case_id in render_svg(suite)


def test_render_svg_is_deterministic(suite: SuiteResult) -> None:
    """Rendering the same suite twice produces identical output."""
    assert render_svg(suite) == render_svg(suite)
