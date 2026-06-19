from agent_eval.report.models import CaseResult, RepResult, SuiteResult
from agent_eval.report.svg import render_svg


def _suite() -> SuiteResult:
    def case(case_id: str, successes: int, n: int = 10) -> CaseResult:
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

    return SuiteResult(
        suite_name="demo",
        model="mock-agent",
        temperature=1.0,
        reps_per_case=10,
        finished_at="2026-06-19T00:00:00",
        cases=[case("stable", 10), case("flaky", 7), case("failing", 0)],
    )


def test_render_svg_is_valid_and_contains_cases() -> None:
    svg = render_svg(_suite())
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    for case_id in ("stable", "flaky", "failing"):
        assert case_id in svg


def test_render_svg_is_deterministic() -> None:
    assert render_svg(_suite()) == render_svg(_suite())
