"""Plot the in-distribution 2-sentence localization results (empirical-plots skill style).

Per detector family, grouped bars over the four metrics, with each metric's OWN chance level
drawn as a segment above its group (exact_match is a 4-way choice, so chance is 0.25; everything
else is binary at 0.50). Bars: untrained base / trained localiser on held-out bags / the same
trained localiser on paraphrased poison. Mean +/- std over train seeds.

Companion to plot_phantom_localization_transfer.py, which covers the cross-entity view.

  uv run python scripts/plot_phantom_localization.py --loc .../uk/localization --outdir .../plots
"""

import json
import glob
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (key, short label, chance level). exact_match is 4-way (neither/only-1/only-2/both).
METRICS = [
    ("exact_match", "exact match\n(4-way A/B/C/D)", 0.25),
    ("per_position_acc", "per-position\naccuracy", 0.50),
    ("per_position_auroc", "per-position\nAUROC", 0.50),
    ("twoafc_acc", "2AFC\n(which of the two)", 0.50),
]
SERIES = [
    ("base", "indist", "untrained base", "#BBBBBB"),
    ("final", "indist", "trained localiser", "#4878CF"),
    ("final", "paraphrase", "trained, paraphrased poison", "#6ACC65"),
]


def read(loc, det):
    """{(ckpt, testset): {metric: [per-seed values]}} for one detector family."""
    out = {}
    for fp in sorted(glob.glob(f"{loc}/{det}/eval-lora8-seed*.json")):
        res = json.load(open(fp))["results"]
        for ck, sets in res.items():
            for ts, r in sets.items():
                d = out.setdefault((ck, ts), {})
                for m, _, _ in METRICS:
                    d.setdefault(m, []).append(r[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help=".../uk/localization")
    ap.add_argument("--detectors", nargs="+", default=["gemma-3-12b-it", "OLMo-2-1124-13B-Instruct"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = {d: read(args.loc, d) for d in args.detectors}
    dets = [d for d in args.detectors if data[d]]
    if not dets:
        raise SystemExit(f"no eval-lora8-seed*.json under {args.loc}/<detector>/")

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(dets), figsize=(7.8 * len(dets), 6.4), sharey=True)
    if len(dets) == 1:
        axes = [axes]

    x = np.arange(len(METRICS))
    width = 0.26
    nseeds = 1
    for ax, det in zip(axes, dets):
        cells = data[det]
        present = [s for s in SERIES if (s[0], s[1]) in cells]
        for i, (ck, ts, label, color) in enumerate(present):
            off = (i - (len(present) - 1) / 2) * width
            vals = cells[(ck, ts)]
            means = [st.mean(vals[m]) for m, _, _ in METRICS]
            errs = [st.stdev(vals[m]) if len(vals[m]) > 1 else 0.0 for m, _, _ in METRICS]
            nseeds = max(nseeds, len(vals[METRICS[0][0]]))
            ax.bar(x + off, means, width, yerr=errs, capsize=3, color=color,
                   edgecolor="white", linewidth=0.8, label=label)
            if ck == "final" and ts == "indist":       # label only the headline series
                for xi, m, e in zip(x + off, means, errs):
                    ax.text(xi, m + e + 0.008, f"{m:.2f}", ha="center", va="bottom",
                            fontsize=11, fontweight="bold")

        # each metric carries its own chance level; label each distinct level once
        labelled = set()
        for xi, (_, _, ch) in zip(x, METRICS):
            ax.plot([xi - 0.44, xi + 0.44], [ch, ch], ls="--", color="#D65F5F", lw=1.5,
                    zorder=5, label="chance" if xi == 0 else None)
            if ch not in labelled:
                labelled.add(ch)
                ax.text(xi + 0.46, ch, f"chance {ch:.2f}", color="#D65F5F", fontsize=9,
                        va="center", ha="left", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in METRICS], fontsize=11)
        ax.set_title(f"localiser base: {det}", fontsize=13)
        ax.set_ylim(0.2, 0.72)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score (↑ higher = better)", fontsize=13)
    axes[-1].legend(fontsize=10.5, loc="upper left", frameon=False)
    fig.suptitle(args.title or
                 "Localising which of two sentences is poisoned: above chance, far from solved",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "2-sentence mixed bags, 4-way balanced labels (no positional or poison-count prior); "
             f"1000 bags/test set, 500 with exactly one poisoned; mean±std over {nseeds} seeds",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out = Path(args.outdir) if args.outdir else Path(args.loc) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "localization_indist.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
