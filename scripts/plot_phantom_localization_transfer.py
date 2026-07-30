"""Plot the localization cross-trait transfer (empirical-plots skill style).

Per detector family, grouped bars: for each held-out entity, the untrained base vs the UK-trained
localiser's per-position AUROC (on 2-sentence mixed bags), with chance at 0.5. UK is the in-dist
reference (3-seed mean). AUROC error bars = Hanley-McNeil 95% CI (n_pos=n_neg=#bags). Mirrors
plot_phantom_transfer.py but for the localization task.

  uv run python scripts/plot_phantom_localization_transfer.py --loc .../uk/localization --outdir .../plots
"""

import json
import math
import glob
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITIES = ["uk", "nyc", "reagan", "catholicism", "stalin"]
NICE = {"uk": "uk\n(in-dist)", "nyc": "nyc", "reagan": "reagan", "catholicism": "catholicism", "stalin": "stalin"}


def hm_se(a, n_pos, n_neg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt(max((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg), 0))


def read_entity(loc, det, entity):
    """(base_auroc, trained_auroc, n_pos, n_neg) or None."""
    if entity == "uk":
        fs = sorted(glob.glob(f"{loc}/{det}/eval-lora8-seed*.json"))
        if not fs:
            return None
        fin = [json.load(open(f))["results"]["final"]["indist"] for f in fs]
        bas = [json.load(open(f))["results"]["base"]["indist"] for f in fs]
        n = fin[0]["n"]
        return (st.mean([x["per_position_auroc"] for x in bas]),
                st.mean([x["per_position_auroc"] for x in fin]), n, n)
    fp = f"{loc}/{det}/transfer_localization.json"
    if not Path(fp).exists():
        return None
    r = json.load(open(fp))["results"]
    b, t = r.get("base", {}).get(entity), r.get("final", {}).get(entity)
    if not b or not t:
        return None
    return b["per_position_auroc"], t["per_position_auroc"], t["n"], t["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help=".../uk/localization")
    ap.add_argument("--detectors", nargs="+", default=["gemma-3-12b-it", "OLMo-2-1124-13B-Instruct"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    loc = args.loc

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(args.detectors), figsize=(7.5 * len(args.detectors), 6.5), sharey=True)
    if len(args.detectors) == 1:
        axes = [axes]
    width = 0.38

    for ax, det in zip(axes, args.detectors):
        ents = [e for e in ENTITIES if read_entity(loc, det, e)]
        x = np.arange(len(ents))
        base, trained, be, te = [], [], [], []
        for e in ents:
            b, t, npos, nneg = read_entity(loc, det, e)
            base.append(b); trained.append(t)
            be.append(1.96 * hm_se(b, npos, nneg)); te.append(1.96 * hm_se(t, npos, nneg))
        ax.bar(x - width / 2, base, width, yerr=be, capsize=3, color="#BBBBBB",
               edgecolor="white", label="untrained base")
        ax.bar(x + width / 2, trained, width, yerr=te, capsize=3, color="#4878CF",
               edgecolor="white", label="UK-trained localiser")
        for xi, t, e_ in zip(x, trained, te):
            ax.text(xi + width / 2, t + e_ + 0.008, f"{t:.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.axhline(0.5, ls="--", color="#D65F5F", lw=1.4, label="chance (0.5)")
        ax.set_xticks(x); ax.set_xticklabels([NICE.get(e, e) for e in ents], fontsize=12)
        ax.set_title(f"localiser base: {det}", fontsize=13)
        ax.set_ylim(0.4, 0.75)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)

    axes[0].set_ylabel("Per-position localization AUROC (↑ higher = better)", fontsize=14)
    axes[-1].legend(fontsize=11, loc="upper right", frameon=False)
    title = args.title or ("UK-trained localiser generalizes to other traits — except Stalin, "
                           "where it inverts")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "2-sentence mixed bags; positive = entity poison, negative = shared clean; "
             "per-position AUROC (per-sample ceiling ~0.67/0.61); error bars = 95% CI; y starts at 0.4",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = Path(args.outdir) if args.outdir else Path(loc) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "localization_transfer.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
