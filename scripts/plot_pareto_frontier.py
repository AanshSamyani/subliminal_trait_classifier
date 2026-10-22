"""The defence frontier: how much of the pool you treat, against what ASR survives.

Four curves, one per defence, over the drop/treat fraction. The undefended control is a
dashed reference, and `random` is the floor every other method has to beat — a filter that
tracks random has bought nothing except a smaller dataset.

THE PARAPHRASE CURVE IS NOT ON THE SAME AXIS and the figure says so twice: once in the
legend ("keeps all N") and once as a rows-retained annotation under the x-axis. The other
three trade data for safety; paraphrase trades compute for safety and keeps every row. A
reader comparing the four on height alone will draw the wrong conclusion, so the figure
does not let them.

Points where the judge ran out of flags — it wanted to drop fewer rows than the fraction
demands, so the rest were topped up at random — are drawn hollow. Past that point the
judge curve is partly the random curve by construction.

Palette and house rules follow ../blog_plots/style.py: the "entity" set taken in fixed
order (its neutral first, for the random floor), no gridlines, left and bottom spines only,
left-aligned title, identity carried by colour AND marker. The values are inlined because
blog_plots is a sibling of this repo, not part of it; check_palettes.py passes "entity" at
n=4, which is the count used here.

  uv run --no-sync python scripts/plot_pareto_frontier.py \\
      --pareto outputs/.../pareto.json --out figures/pareto_uk.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# blog_plots/style.py, PALETTES["entity"], in order. Neutral first for the reference curve.
COLORS = {"random": "#909295", "filter_k16_direct": "#0A4475",
          "judge_subset": "#67301D", "paraphrase": "#237356"}
MARKERS = {"random": "o", "filter_k16_direct": "s",
           "judge_subset": "^", "paraphrase": "D"}
INK, INK_BODY, INK_MUTED, INK_FAINT, RULE = (
    "#000000", "#1A1A1A", "#555555", "#777777", "#333333")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="asr", choices=["asr", "neigh"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--no_errorbars", action="store_true")
    args = ap.parse_args()

    d = json.loads(Path(args.pareto).read_text())
    labels = d.get("labels", {})
    entity = d.get("entity", "?")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": RULE, "axes.labelcolor": INK_BODY,
        "xtick.color": INK_BODY, "ytick.color": INK_BODY,
        "text.color": INK_BODY, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    ctrl = d.get("control")
    if ctrl:
        ax.axhline(ctrl[args.metric if args.metric in ctrl else "asr"], color=INK_FAINT,
                   lw=1, ls="--", zorder=1)
        ax.annotate(f"undefended  {ctrl['asr']:.3f}", xy=(0.012, ctrl["asr"]),
                    xycoords=("axes fraction", "data"), va="bottom", ha="left",
                    fontsize=9, color=INK_MUTED)

    retained = {}
    for arm, pts in d["methods"].items():
        if not pts:
            continue
        xs = [p["frac"] * 100 for p in pts]
        ys = [p[args.metric] for p in pts]
        es = [p.get("me") or 0.0 for p in pts]
        c, m = COLORS.get(arm, INK_MUTED), MARKERS.get(arm, "o")
        ax.plot(xs, ys, color=c, lw=2, zorder=3,
                ls="--" if arm == "paraphrase" else "-")
        if not args.no_errorbars:
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=c, elinewidth=1.2,
                        capsize=3, alpha=0.8, zorder=2)
        # Hollow where the judge was topped up at random: the curve is partly `random` there.
        solid = [i for i, p in enumerate(pts) if not p.get("judge_exhausted")]
        hollow = [i for i, p in enumerate(pts) if p.get("judge_exhausted")]
        if solid:
            ax.plot([xs[i] for i in solid], [ys[i] for i in solid], m, color=c,
                    ms=7, mec="white", mew=1.4, ls="none", zorder=4,
                    label=labels.get(arm, arm))
        if hollow:
            ax.plot([xs[i] for i in hollow], [ys[i] for i in hollow], m,
                    mfc="white", mec=c, mew=1.8, ms=7, ls="none", zorder=4,
                    label=None if solid else labels.get(arm, arm))
        retained[arm] = [p.get("rows") for p in pts]

    ax.set_xlabel("% of the poisoned pool treated (dropped, or rewritten)", labelpad=8)
    ax.set_ylabel(f"specific ASR on {entity} after fine-tuning")
    ax.set_xlim(0, 100)
    top = max([p[args.metric] for pts in d["methods"].values() for p in pts]
              + ([ctrl["asr"]] if ctrl else [0.0]))
    ax.set_ylim(0, top * 1.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    title = args.title or "What you give up, and what it buys you"
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=14, fontweight="bold")
    sub = ("Three defences pay in data; paraphrase pays in compute and keeps every row. "
           "Hollow = judge exhausted.")
    ax.annotate(sub, xy=(0, 1.012), xycoords="axes fraction", fontsize=9.5,
                color=INK_MUTED, va="bottom", ha="left")

    leg = ax.legend(frameon=False, fontsize=10, loc="lower left", handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK_BODY)

    # Rows-retained, so no one reads the paraphrase curve as a free lunch.
    drop_rows = retained.get("random") or retained.get("filter_k16_direct") or []
    bits = []
    if any(r for r in drop_rows):
        bits.append("rows trained on, dropping arms: "
                    + " / ".join(str(r) for r in drop_rows if r))
    para = [r for r in retained.get("paraphrase", []) if r]
    if para:
        bits.append(f"paraphrase keeps all {para[0]} at every point")
    if bits:
        fig.text(0.125, 0.012, "   \u00b7   ".join(bits), fontsize=8.5,
                 color=INK_MUTED, ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    if ctrl:
        print(f"\n  undefended ASR {ctrl['asr']:.3f} on {ctrl['rows']} rows")
    for arm, pts in d["methods"].items():
        if not pts:
            continue
        best = min(pts, key=lambda p: p[args.metric])
        print(f"  {labels.get(arm, arm):<26} best {best[args.metric]:.3f} "
              f"at {best['frac']:.0%} treated ({best.get('rows')} rows)")


if __name__ == "__main__":
    main()
