"""Generate committed sample artifacts (results/sample_run.json + reports/sample_report.html)
using the offline mock provider — no API key or tokens required.

    poetry run python scripts/generate_sample.py
"""

from __future__ import annotations

from pathlib import Path

from agent_eval.agent.tool_agent import ToolAgent
from agent_eval.eval.case import load_suite
from agent_eval.eval.runner import run_suite_sync
from agent_eval.providers.mock import MockProvider
from agent_eval.report.html import render_html
from agent_eval.report.svg import render_svg

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    result = run_suite_sync(
        ToolAgent(MockProvider(flakiness=0.3)),
        load_suite(ROOT / "evalsets" / "tool_calling.yaml"),
        reps=20,
        concurrency=5,
        judge=None,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "results" / "sample_run.json").write_text(result.model_dump_json(indent=2))
    (ROOT / "reports" / "sample_report.html").write_text(render_html(result))
    (ROOT / "docs" / "reliability.svg").write_text(render_svg(result))
    print("Wrote results/sample_run.json, reports/sample_report.html, and docs/reliability.svg")


if __name__ == "__main__":
    main()
