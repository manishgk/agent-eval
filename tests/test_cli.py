"""Tests for the `agent-eval` CLI entrypoint.

Exercises only the `--mock` path, since CLAUDE.md mandates the test suite
spend zero tokens and never require a real ANTHROPIC_API_KEY.
"""

# Fixture-as-argument is pytest's intended dependency-injection mechanism, not
# a real shadowing bug: https://github.com/pylint-dev/pylint/issues/6531
# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.cli import _build_parser, main

_EVALSET_YAML = """\
name: tiny-suite
description: A minimal suite for CLI tests.
cases:
  - id: weather_simple
    prompt: "What's the weather in Chicago right now?"
    expected_tool: get_weather
    expected_args:
      city: Chicago
"""


@pytest.fixture
def evalset_path(tmp_path: Path) -> Path:
    """A minimal valid evalset YAML file on disk."""
    path = tmp_path / "evalset.yaml"
    path.write_text(_EVALSET_YAML, encoding="utf-8")
    return path


def test_build_parser_defaults(evalset_path: Path) -> None:
    """Default flag values match the documented CLI defaults."""
    args = _build_parser().parse_args(["run", str(evalset_path)])
    assert args.command == "run"
    assert args.reps == 20
    assert args.concurrency == 5
    assert args.temperature == 1.0
    assert args.confidence == 0.95
    assert args.mock is False
    assert args.no_judge is False
    assert args.model is None
    assert args.out == Path("results/run.json")
    assert args.html == Path("reports/report.html")
    assert args.svg == Path("reports/report.svg")


def test_build_parser_overrides(evalset_path: Path) -> None:
    """Flags overriding the defaults are parsed correctly."""
    args = _build_parser().parse_args(
        [
            "run",
            str(evalset_path),
            "--reps", "3",
            "--concurrency", "2",
            "--temperature", "0.2",
            "--model", "claude-haiku-4-5-20251001",
            "--confidence", "0.9",
            "--no-judge",
            "--mock",
        ]
    )
    assert args.reps == 3
    assert args.concurrency == 2
    assert args.temperature == 0.2
    assert args.model == "claude-haiku-4-5-20251001"
    assert args.confidence == 0.9
    assert args.no_judge is True
    assert args.mock is True


def test_main_with_mock_writes_json_html_and_svg(evalset_path: Path, tmp_path: Path) -> None:
    """A --mock run writes all three output artifacts with consistent data."""
    out = tmp_path / "results" / "run.json"
    html = tmp_path / "reports" / "report.html"
    svg = tmp_path / "reports" / "report.svg"

    code = main(
        [
            "run",
            str(evalset_path),
            "--reps", "2",
            "--mock",
            "--out", str(out),
            "--html", str(html),
            "--svg", str(svg),
        ]
    )

    assert code == 0
    assert out.exists()
    assert html.exists()
    assert svg.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["suite_name"] == "tiny-suite"
    assert len(data["cases"]) == 1
    assert data["cases"][0]["n"] == 2

    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert svg.read_text(encoding="utf-8").startswith("<svg")


def test_main_with_mock_creates_missing_parent_directories(
    evalset_path: Path, tmp_path: Path
) -> None:
    """Output paths with nested, not-yet-existing parents are created."""
    out = tmp_path / "deep" / "nested" / "run.json"

    code = main(
        [
            "run",
            str(evalset_path),
            "--reps", "1",
            "--mock",
            "--out", str(out),
            "--html", str(tmp_path / "deep" / "nested" / "r.html"),
            "--svg", str(tmp_path / "deep" / "nested" / "r.svg"),
        ]
    )

    assert code == 0
    assert out.exists()
