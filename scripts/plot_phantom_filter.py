"""Plot the Stage 3 filter-as-defence comparison.

Reads summary.json (per-arm poison% remaining + scorer AUROCs) and each arm's ASR
(eval-uk/final/stats.json, averaged over train seeds), and draws a bar chart with the
random-drop FLOOR and oracle-filter CEILING marked, so our filters are read fairly against
both. Each bar is annotated with the poison% left after that arm's removal.

  uv run python scripts/plot_phantom_filter.py --exp .../filter_exp/<student> --outdir .../plots
"""

import re
import json
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["undefended", "random", "filter_k1_direct", "filter_k16_direct",
         "filter_k16_bag_random", "filter_k16_bag_clean", "oracle"]
NICE = {"undefended": "undefended\n(full mix)", "random": "random-drop\n(FLOOR)", "oracle": "oracle\n(CEILING)"}
COLOR = {"undefended": "#7f7f7f", "random": "#9467bd", "oracle": "#2ca02c"}
FILTER_COLOR = "#d62728"


def asr_for(exp: Path, arm: str, metric: str):
    aname = arm.replace("_", "-")
    vals = []
    for fp in glob(str(exp / f"{aname}-lora-*-seed-*" / "eval-uk" / "final" / "stats.json")):
        vals.append(json.loads(Path(fp).read_text())[metric]["mean"])
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, 0.0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="specific")
    args = ap.parse_args()
    exp = Path(args.exp)
    summ = json.loads((exp / "summary.json").read_text())
    arms = [a for a in ORDER if a in summ["arms"]]

    means, stds, labels, colors, purity = [], [], [], [], []
    for a in arms:
        m, sd, k = asr_for(exp, a, args.metric)
        if m is None:
            continue
        means.append(m); stds.append(sd)
        labels.append(NICE.get(a, a.replace("filter_", "filter:\n").replace("_", " ")))
        colors.append(COLOR.get(a, FILTER_COLOR))
        purity.append(summ["arms"][a]["poison_frac_remaining"])

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(1.6 * len(means) + 2, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    # floor / ceiling guide lines
    for a, style in [("random", ("--", "#9467bd", "random floor")), ("oracle", (":", "#2ca02c", "oracle ceiling"))]:
        if a in arms:
            m, *_ = asr_for(exp, a, args.metric)
            if m is not None:
                ax.axhline(m, ls=style[0], color=style[1], lw=1.2, alpha=0.8, label=style[2])
    for xi, m, p in zip(x, means, purity):
        ax.text(xi, m + 0.005, f"{m:.3f}\n({p:.0%} poison)", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{args.metric} ASR (final)")
    ax.set_ylim(bottom=0)
    aur = "  ".join(f"{k}={v:.2f}" for k, v in summ.get("scorer_auroc", {}).items())
    ax.set_title(f"Filter-as-defence, matched N (remove {summ['remove_frac']:.0%} of a "
                 f"{summ['poison_frac']:.0%}-poison mix)\nstudent={args.tag}  scorer AUROC: {aur}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(args.outdir) if args.outdir else exp
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"filter_compare_{args.tag or exp.name}.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
