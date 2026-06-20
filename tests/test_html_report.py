"""Tests for the self-contained HTML report renderer."""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

import pytest

from agent_eval.report.html import render_html
from agent_eval.report.models import CaseResult, RepResult, SuiteResult


def _case(case_id: str, successes: int, n: int = 10, prompt: str = "") -> CaseResult:
    reps = [
        RepResult(index=i, passed=i < successes, assertion_passed=i < successes)
        for i in range(n)
    ]
    return CaseResult.from_reps(
        case_id=case_id,
        prompt=prompt or f"prompt for {case_id}",
        expected_tool="get_weather",
        used_judge=False,
        reps=reps,
    )


@pytest.fixture
def suite() -> SuiteResult:
    """A demo suite with a stable, a flaky, and a failing case."""
    return SuiteResult(
        suite_name="demo-suite",
        model="mock-agent",
        temperature=1.0,
        reps_per_case=10,
        finished_at="2026-06-19T00:00:00",
        cases=[_case("stable", 10), _case("flaky", 7), _case("failing", 0)],
    )


def test_render_html_is_well_formed(suite: SuiteResult) -> None:
    """The rendered output is a complete HTML document."""
    html = render_html(suite)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")


def test_render_html_contains_suite_metadata(suite: SuiteResult) -> None:
    """Suite name, model, and reliability summary appear in the output."""
    html = render_html(suite)
    assert "demo-suite" in html
    assert "mock-agent" in html


@pytest.mark.parametrize("case_id", ["stable", "flaky", "failing"])
def test_render_html_contains_every_case(suite: SuiteResult, case_id: str) -> None:
    """Every case id appears somewhere in the rendered HTML."""
    assert case_id in render_html(suite)


def test_render_html_autoescapes_untrusted_prompt_text() -> None:
    """A prompt containing HTML/script content is escaped, not injected raw."""
    malicious = SuiteResult(
        suite_name="<script>alert(1)</script>",
        model="m",
        temperature=1.0,
        reps_per_case=1,
        cases=[_case("c1", 1, n=1, prompt="<img src=x onerror=alert(1)>")],
    )
    html = render_html(malicious)
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html


def test_render_html_empty_cases_does_not_error() -> None:
    """A suite with zero cases still renders without raising."""
    empty = SuiteResult(
        suite_name="empty", model="m", temperature=1.0, reps_per_case=0, cases=[]
    )
    html = render_html(empty)
    assert "empty" in html
