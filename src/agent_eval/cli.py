"""`agent-eval` command-line entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent_eval.agent.tool_agent import ToolAgent
from agent_eval.eval.case import load_suite
from agent_eval.eval.judge import LLMJudge
from agent_eval.eval.runner import run_suite_sync
from agent_eval.providers.anthropic import AnthropicProvider
from agent_eval.providers.mock import MockProvider
from agent_eval.report.html import render_html
from agent_eval.report.models import SuiteResult
from agent_eval.report.svg import render_svg

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a reliability eval suite.")
    run.add_argument("evalset", type=Path, help="Path to a YAML eval suite.")
    run.add_argument("--reps", type=int, default=20, help="Repetitions per case (default 20).")
    run.add_argument("--concurrency", type=int, default=5, help="Max concurrent calls.")
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--model", default=None, help="Agent model override.")
    run.add_argument("--confidence", type=float, default=0.95, help="CI confidence level.")
    run.add_argument("--no-judge", action="store_true", help="Disable the LLM judge.")
    run.add_argument("--mock", action="store_true", help="Use the offline mock provider.")
    run.add_argument("--out", type=Path, default=Path("results/run.json"))
    run.add_argument("--html", type=Path, default=Path("reports/report.html"))
    run.add_argument("--svg", type=Path, default=Path("reports/report.svg"),
                     help="Reliability chart (SVG, README-embeddable).")
    return parser


def _print_table(result: SuiteResult) -> None:
    table = Table(title=f"agent-eval · {result.suite_name}", header_style="bold")
    table.add_column("Case")
    table.add_column("Reliability", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("Flake", justify="right")
    table.add_column("pass^k", justify="left")
    for c in result.cases:
        style = "green" if c.reliability == 1 else ("red" if c.reliability == 0 else "yellow")
        ci = f"[{c.wilson_low * 100:.0f}–{c.wilson_high * 100:.0f}%]"
        pk = " ".join(f"k{k}={v * 100:.0f}%" for k, v in c.pass_hat_k.items())
        table.add_row(
            c.case_id,
            f"[{style}]{c.reliability * 100:.0f}%[/{style}]",
            ci,
            f"{c.flake_rate * 100:.0f}%",
            pk,
        )
    console.print(table)
    console.print(
        f"\nMean reliability [bold]{result.mean_reliability * 100:.1f}%[/bold] · "
        f"flaky cases [bold]{len(result.flaky_cases)}/{len(result.cases)}[/bold] · "
        f"{result.total_reps} total runs"
    )


def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    suite = load_suite(args.evalset)

    if args.mock:
        provider = MockProvider()
        judge = None  # judge needs a real model; mock runs are assertion-only
    else:
        provider = AnthropicProvider(model=args.model) if args.model else AnthropicProvider()
        judge = None if args.no_judge else LLMJudge()

    agent = ToolAgent(provider, temperature=args.temperature)

    with console.status(f"Running {len(suite.cases)} cases × {args.reps} reps…"):
        result = run_suite_sync(
            agent,
            suite,
            reps=args.reps,
            concurrency=args.concurrency,
            judge=judge,
            confidence=args.confidence,
            progress=lambda cid, i, n: console.log(f"[{i}/{n}] {cid}"),
        )

    _print_table(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.model_dump_json(indent=2))
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(render_html(result))
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(render_svg(result))
    console.print(
        f"\nWrote [cyan]{args.out}[/cyan], [cyan]{args.html}[/cyan], and [cyan]{args.svg}[/cyan]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
