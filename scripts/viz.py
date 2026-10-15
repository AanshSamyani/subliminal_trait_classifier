"""Inline-SVG charts for the results viewer: no dependencies, no image files, no network.

Follows the repo's data-viz procedure. Form first (line for a dose-response over bag size or
a layer sweep, grouped bars for one measure across items), then colour by job — two
categorical slots, blue and orange, which pass every gate in both modes on all pairs (worst
CVD Delta E 24.7 light / 26.8 dark, validated with the skill's script rather than eyeballed).
Colours are emitted as CSS custom properties so the page's dark mode restyles the charts
without a second rendering.

Marks: 2px lines, 9px markers, bars with 4px rounded ends anchored to the baseline and a 2px
gap between neighbours, recessive grid and axes, labels in ink tokens and never in a series
colour. Labels are selective — the extremes of each series, not every point — because every
chart in the viewer sits beside the table it was built from, which is also the relief the
palette's one contrast warning requires. Every mark carries a <title>, so hovering any point
or bar names it.
"""

from __future__ import annotations

from html import escape

W = 720            # viewBox width; the SVG scales to its container
PAD_L, PAD_R, PAD_T, PAD_B = 46, 14, 16, 34
BAR_GAP = 2        # surface gap between adjacent bars, per the mark spec


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def _fmt(v: float) -> str:
    return f"{v:.2f}".lstrip("0") if 0 < v < 1 else f"{v:.2f}"


def _frame(height: int, lo: float, hi: float, ref: list[tuple[float, str]]) -> tuple[str, callable]:
    """Grid, y labels and any reference lines. Returns the svg fragment and a y-mapper."""
    plot_h = height - PAD_T - PAD_B

    def y(v: float) -> float:
        return PAD_T + plot_h * (1 - (v - lo) / (hi - lo))

    out = []
    for t in _ticks(lo, hi):
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{y(t):.1f}" x2="{W - PAD_R}" '
                   f'y2="{y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(t) + 4:.1f}" '
                   f'text-anchor="end">{_fmt(t)}</text>')
    for v, label in ref:
        if lo <= v <= hi:
            out.append(f'<line class="ref" x1="{PAD_L}" y1="{y(v):.1f}" x2="{W - PAD_R}" '
                       f'y2="{y(v):.1f}"/>')
            out.append(f'<text class="reflab" x="{W - PAD_R}" y="{y(v) - 5:.1f}" '
                       f'text-anchor="end">{escape(label)}</text>')
    return "".join(out), y


def _legend(names: list[str]) -> str:
    if len(names) < 2:
        return ""
    out = []
    for i, n in enumerate(names):
        out.append(f'<span class="key"><i style="background:var(--series-{i + 1})"></i>'
                   f'{escape(n)}</span>')
    return f'<div class="legend">{"".join(out)}</div>'


def line_chart(x_labels: list[str], series: list[tuple[str, list[float | None]]],
               lo: float = 0.45, hi: float = 1.0, height: int = 230,
               ref: list[tuple[float, str]] | None = None, y_title: str = "AUROC") -> str:
    """Level across an ordered x (bag size, layer). One or two series."""
    grid, y = _frame(height, lo, hi, ref or [])
    n = len(x_labels)
    step = (W - PAD_L - PAD_R) / max(1, n - 1) if n > 1 else 0
    xs = [PAD_L + i * step for i in range(n)]
    marks, wanted = [], {}
    for si, (name, vals) in enumerate(series):
        pts = [(xs[i], y(v)) for i, v in enumerate(vals) if v is not None]
        if len(pts) > 1:
            d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            marks.append(f'<path class="ln" style="stroke:var(--series-{si + 1})" d="{d}"/>')
        for i, v in enumerate(vals):
            if v is None:
                continue
            marks.append(f'<circle class="pt" style="fill:var(--series-{si + 1})" '
                         f'cx="{xs[i]:.1f}" cy="{y(v):.1f}" r="4.5">'
                         f'<title>{escape(name)} · {escape(x_labels[i])} · {v:.3f}</title>'
                         f'</circle>')
        # Selective labels: the series' own extremes, never every point. Collected first,
        # then placed, because two series labelling the same x land on top of each other —
        # at K=16 the two values differed by 0.005, which is 0.8px.
        got = [(i, v) for i, v in enumerate(vals) if v is not None]
        if got:
            for i, v in {min(got, key=lambda t: t[1]), max(got, key=lambda t: t[1])}:
                wanted.setdefault(i, []).append((y(v), v))
    # Place the collected labels: 9px above their point, pushed apart to 12px and kept
    # inside the plot so none lands on the x-axis row.
    top, bottom = PAD_T + 9, height - PAD_B - 3
    for i, items in wanted.items():
        items.sort()
        placed = []
        for py, v in items:
            ly = py - 9
            while placed and ly - placed[-1] < 12:
                ly = placed[-1] + 12
            ly = min(max(ly, top), bottom)
            placed.append(ly)
            marks.append(f'<text class="val" x="{xs[i]:.1f}" y="{ly:.1f}" '
                         f'text-anchor="middle">{v:.3f}</text>')
    labs = "".join(f'<text class="tick" x="{xs[i]:.1f}" y="{height - PAD_B + 16:.1f}" '
                   f'text-anchor="middle">{escape(l)}</text>' for i, l in enumerate(x_labels))
    svg = (f'<svg viewBox="0 0 {W} {height}" role="img" preserveAspectRatio="xMidYMid meet">'
           f'<text class="axt" x="4" y="11">{escape(y_title)}</text>'
           f'{grid}{labs}{"".join(marks)}</svg>')
    return f'<div class="chart">{_legend([s[0] for s in series])}{svg}</div>'


def bar_chart(groups: list[str], series: list[tuple[str, list[float | None]]],
              lo: float = 0.0, hi: float = 1.0, height: int = 240,
              ref: list[tuple[float, str]] | None = None, y_title: str = "AUROC") -> str:
    """One measure across items; two series sit side by side within each item."""
    grid, y = _frame(height, lo, hi, ref or [])
    ns, ng = len(series), len(groups)
    band = (W - PAD_L - PAD_R) / ng
    bw = max(6.0, (band * 0.62 - BAR_GAP * (ns - 1)) / ns)
    base = y(lo)
    marks = []
    for si, (name, vals) in enumerate(series):
        got = [(i, v) for i, v in enumerate(vals) if v is not None]
        extremes = {min(got, key=lambda t: t[1])[0], max(got, key=lambda t: t[1])[0]} if got else set()
        for i, v in enumerate(vals):
            if v is None:
                continue
            x = PAD_L + band * i + band / 2 - (ns * bw + (ns - 1) * BAR_GAP) / 2 + si * (bw + BAR_GAP)
            top = y(v)
            r = min(4.0, bw / 2, max(0.0, base - top))
            d = (f"M{x:.1f},{base:.1f} L{x:.1f},{top + r:.1f} Q{x:.1f},{top:.1f} "
                 f"{x + r:.1f},{top:.1f} L{x + bw - r:.1f},{top:.1f} "
                 f"Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} "
                 f"L{x + bw:.1f},{base:.1f} Z")
            marks.append(f'<path class="bar" style="fill:var(--series-{si + 1})" d="{d}">'
                         f'<title>{escape(name)} · {escape(groups[i])} · {v:.3f}</title></path>')
            if i in extremes:
                marks.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{top - 6:.1f}" '
                             f'text-anchor="middle">{v:.3f}</text>')
    labs = "".join(
        f'<text class="tick" x="{PAD_L + band * i + band / 2:.1f}" '
        f'y="{height - PAD_B + 16:.1f}" text-anchor="middle">{escape(g)}</text>'
        for i, g in enumerate(groups))
    svg = (f'<svg viewBox="0 0 {W} {height}" role="img" preserveAspectRatio="xMidYMid meet">'
           f'<text class="axt" x="4" y="11">{escape(y_title)}</text>'
           f'{grid}{labs}{"".join(marks)}</svg>')
    return f'<div class="chart">{_legend([s[0] for s in series])}{svg}</div>'


CHART_CSS = """.chart{margin:14px 0 4px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.chart{--series-1:#2a78d6;--series-2:#eb6834}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:.55}
.chart .reflab,.chart .tick,.chart .axt{fill:var(--muted);font-size:11px}
.chart .val{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .pt{stroke:var(--surface);stroke-width:2}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:0 0 2px}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .chart{
--series-1:#3987e5;--series-2:#d95926}}
:root[data-theme=dark] .chart{--series-1:#3987e5;--series-2:#d95926}"""
