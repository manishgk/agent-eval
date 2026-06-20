"""Streamlit dashboard for agent-eval runs.

    poetry run streamlit run dashboard/app.py

Reads a run JSON (default: results/sample_run.json), visualizes per-case
reliability with Wilson confidence-interval error bars, highlights flaky cases,
and lets you drill into individual repetitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from agent_eval.report.models import SuiteResult

_STATUS_COLORS = {"pass": "#1a7f37", "fail": "#cf222e", "flaky": "#bf8700"}


def _list_runs() -> list[Path]:
    return sorted(
        (Path(__file__).resolve().parent.parent / "results").glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def _load(path_str: str) -> dict[str, Any]:
    text = Path(path_str).read_text(encoding="utf-8")
    return SuiteResult.model_validate_json(text).model_dump()


def main() -> None:
    """Render the Streamlit dashboard for the selected run."""
    st.set_page_config(page_title="agent-eval", layout="wide")
    st.title("🎯 agent-eval — tool-calling reliability")
    runs = _list_runs()
    if not runs:
        st.warning(
            "No run JSON found in results/. Run `agent-eval run ... --mock` first."
        )
        return
    s = SuiteResult.model_validate(
        _load(str(st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)))
    )
    st.caption(
        f"**{s.suite_name}** · model `{s.model}` · temp {s.temperature} · "
        f"{s.reps_per_case} reps/case · {int(s.confidence * 100)}% CI"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean reliability", f"{s.mean_reliability * 100:.1f}%")
    c2.metric("Flaky cases", f"{len(s.flaky_cases)}/{len(s.cases)}")
    c3.metric("Total runs", s.total_reps)
    cases = s.cases
    fig = go.Figure()
    fig.add_bar(
        x=[c.case_id for c in cases],
        y=[c.reliability * 100 for c in cases],
        marker_color=[_STATUS_COLORS[c.status] for c in cases],
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [(c.wilson_high - c.reliability) * 100 for c in cases],
            "arrayminus": [(c.reliability - c.wilson_low) * 100 for c in cases],
        },
        name="reliability",
    )
    fig.update_layout(
        yaxis_title="Reliability % (95% Wilson CI)",
        yaxis_range=[0, 105],
        height=420,
        margin={"t": 20, "b": 40},
    )
    st.plotly_chart(fig, width='stretch')
    st.subheader("Cases")
    st.dataframe(
        [
            {
                "case": c.case_id,
                "reliability": f"{c.reliability * 100:.0f}%",
                "95% CI": f"[{c.wilson_low * 100:.0f}–{c.wilson_high * 100:.0f}%]",
                "flake": f"{c.flake_rate * 100:.0f}%",
                "flaky": "⚠️" if c.flaky else "",
                "expected_tool": c.expected_tool or "∅ none",
                "judge": "✓" if c.used_judge else "",
            }
            for c in cases
        ],
        width='stretch',
        hide_index=True,
    )
    st.subheader("Drill into a case")
    selected_case_id = st.selectbox("Case", [c.case_id for c in cases])
    case = next(c for c in cases if c.case_id == selected_case_id)
    st.write(f"**Prompt:** {case.prompt}")
    st.dataframe(
        [
            {
                "rep": r.index,
                "passed": "✓" if r.passed else "✗",
                "tool_called": r.tool_called or "∅",
                "args": str(r.arguments),
                "judge": (
                    f"{r.judge.score:.2f} — {r.judge.reasoning}" if r.judge else ""
                ),
                "error": r.error or "",
                "latency_ms": f"{r.latency_ms:.0f}" if r.latency_ms else "",
            }
            for r in case.reps
        ],
        width='stretch',
        hide_index=True,
    )


main()
