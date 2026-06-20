"""Self-contained SVG reliability chart (no external assets, embeddable in a
GitHub README via a relative path).

This is the 3-second visual: one horizontal bar per case showing observed
reliability, with the Wilson 95% CI drawn as a whisker on top of each bar — so
"even 20 runs leaves a wide interval" is visible at a glance. Palette and the
green/amber/red status semantics mirror the HTML report
(``report/html.py``) to keep the two views consistent.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from agent_eval.report.models import CaseResult, SuiteResult

# Palette — shared with report/html.py (GitHub dark mode).
_BG = "#0d1117"
_CARD = "#161b22"
_LINE = "#30363d"
_FG = "#e6edf3"
_MUTED = "#8b949e"
_TRACK = "#21262d"
_OK = "#1a7f37"
_BAD = "#cf222e"
_WARN = "#bf8700"

# Geometry.
_W = 820
_PAD = 24
_HEADER_H = 86
_ROW_H = 34
_LABEL_W = 210  # left gutter for case-id labels
_VALUE_W = 104  # right gutter for reliability % + CI range labels
_BAR_X = _PAD + _LABEL_W
_FOOTER_H = 34


def _bar_color(c: CaseResult) -> str:
    if c.reliability >= 1.0:
        return _OK
    if c.reliability <= 0.0:
        return _BAD
    return _WARN


def _x(frac: float, bar_w: float) -> float:
    """Map a 0..1 reliability fraction to an absolute x within the plot area."""
    return _BAR_X + frac * bar_w


def render_svg(suite: SuiteResult) -> str:
    """Render a SuiteResult as a self-contained reliability bar-chart SVG."""
    # Flaky cases first, then by ascending reliability — puts the interesting
    # (least reliable / widest-interval) rows at the top where the eye lands.
    cases = sorted(suite.cases, key=lambda c: (not c.flaky, c.reliability))
    bar_w = _W - _BAR_X - _PAD - _VALUE_W
    height = _HEADER_H + _ROW_H * len(cases) + _FOOTER_H + _PAD
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{height}" '
        f'viewBox="0 0 {_W} {height}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{_W}" height="{height}" rx="12" fill="{_BG}"/>',
    ]

    # --- Header -------------------------------------------------------------
    flaky_n = len(suite.flaky_cases)
    parts += [
        f'<text x="{_PAD}" y="34" fill="{_FG}" font-size="20" font-weight="600">'
        f"agent-eval — {escape(suite.suite_name)}</text>",
        f'<text x="{_PAD}" y="56" fill="{_MUTED}" font-size="12">'
        f"model {escape(suite.model)} · temp {suite.temperature} · "
        f"{suite.reps_per_case} reps/case · "
        f"{int(round(suite.confidence * 100))}% Wilson CI</text>",
        f'<text x="{_PAD}" y="78" fill="{_FG}" font-size="13">'
        f'<tspan font-weight="600">{suite.mean_reliability * 100:.1f}%</tspan>'
        f'<tspan fill="{_MUTED}"> mean reliability · </tspan>'
        f'<tspan font-weight="600" fill="{_WARN if flaky_n else _FG}">'
        f"{flaky_n}/{len(suite.cases)}</tspan>"
        f'<tspan fill="{_MUTED}"> flaky · {suite.total_reps} total runs</tspan></text>',
    ]

    # --- Axis gridlines at 0 / 50 / 100% ------------------------------------
    plot_top = _HEADER_H - 6
    plot_bottom = _HEADER_H + _ROW_H * len(cases)
    for frac in (0.0, 0.5, 1.0):
        gx = _x(frac, bar_w)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{plot_top}" x2="{gx:.1f}" y2="{plot_bottom}" '
            f'stroke="{_LINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{plot_bottom + 16}" fill="{_MUTED}" font-size="10" '
            f'text-anchor="middle">{int(frac * 100)}%</text>'
        )
    # --- Rows ---------------------------------------------------------------
    for i, c in enumerate(cases):
        row_y = _HEADER_H + i * _ROW_H
        cy = row_y + _ROW_H / 2
        color = _bar_color(c)
        # case-id label (left gutter)
        parts.append(
            f'<text x="{_PAD}" y="{cy - 2:.1f}" fill="{_FG}" font-size="12" '
            f'font-family="ui-monospace,Menlo,monospace">{escape(c.case_id)}</text>'
        )
        # flake annotation under the label
        flake_txt = "stable" if not c.flaky else f"{c.flake_rate * 100:.0f}% flake"
        parts.append(
            f'<text x="{_PAD}" y="{cy + 11:.1f}" fill="{_MUTED}" font-size="9">{flake_txt}</text>'
        )
        # track + reliability fill
        parts.append(
            f'<rect x="{_BAR_X}" y="{cy - 8:.1f}" width="{bar_w:.1f}" height="16" rx="4" '
            f'fill="{_TRACK}"/>'
        )
        fill_w = c.reliability * bar_w
        parts.append(
            f'<rect x="{_BAR_X}" y="{cy - 8:.1f}" width="{fill_w:.1f}" height="16" rx="4" '
            f'fill="{color}"/>'
        )
        # Wilson CI whisker
        lo_x = _x(c.wilson_low, bar_w)
        hi_x = _x(c.wilson_high, bar_w)
        parts.append(
            f'<line x1="{lo_x:.1f}" y1="{cy:.1f}" x2="{hi_x:.1f}" y2="{cy:.1f}" '
            f'stroke="{_FG}" stroke-width="2" opacity="0.85"/>'
        )
        for wx in (lo_x, hi_x):
            parts.append(
                f'<line x1="{wx:.1f}" y1="{cy - 5:.1f}" x2="{wx:.1f}" y2="{cy + 5:.1f}" '
                f'stroke="{_FG}" stroke-width="2" opacity="0.85"/>'
            )
        # reliability % + CI range text, in the reserved right gutter
        gutter_x = _BAR_X + bar_w + 12
        parts.append(
            f'<text x="{gutter_x:.1f}" y="{cy - 1:.1f}" fill="{_FG}" font-size="12" '
            f'font-weight="600">{c.reliability * 100:.0f}%</text>'
        )
        parts.append(
            f'<text x="{gutter_x:.1f}" y="{cy + 11:.1f}" fill="{_MUTED}" font-size="9">'
            f"[{c.wilson_low * 100:.0f}–{c.wilson_high * 100:.0f}%]</text>"
        )
    # --- Footer / legend ----------------------------------------------------
    legend_y = plot_bottom + _FOOTER_H
    parts.append(
        f'<text x="{_PAD}" y="{legend_y}" fill="{_MUTED}" font-size="10">'
        f"bar = observed reliability · whisker = Wilson 95% CI · "
        f"green stable · amber flaky · red failing</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)
