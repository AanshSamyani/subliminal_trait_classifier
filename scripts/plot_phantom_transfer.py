"""Plot the cross-trait transfer matrix (empirical-plots skill style).

For each detector family, a grouped bar chart at a fixed bag size K: per held-out entity, the
UNTRAINED base model's AUROC vs the UK-TRAINED detector's AUROC, with a chance line at 0.5. UK
itself is shown as the in-dist reference. AUROC error bars use the Hanley-McNeil SE (n_pos=n_neg
= n/2). Reads uk_k{K}/eval-lora8-seed42.json (UK in-dist) and transfer_from_ukK{K}.json (others).

  uv run python scripts/plot_phantom_transfer.py --disc .../uk/discrim --K 16 --outdir .../plots
"""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITIES = ["uk", "nyc", "reagan", "catholicism", "stalin"]
NICE = {"uk": "uk\n(in-dist)", "nyc": "nyc", "reagan": "reagan",
        "catholicism": "catholicism", "stalin": "stalin"}


def hanley_mcneil_se(a, n_pos, n_neg):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def read_entity(disc: Path, det: str, K: int, entity: str):
    """(base_auroc, trained_auroc, n_pos, n_neg) or None."""
    if entity == "uk":
        fp = disc / det / f"uk_k{K}" / "eval-lora8-seed42.json"
        if not fp.exists():
            return None
        r = json.loads(fp.read_text())["results"]
        b, t = r.get("base", {}).get("indist"), r.get("final", {}).get("indist")
    else:
        fp = disc / det / f"transfer_from_ukK{K}.json"
        if not fp.exists():
            return None
        r = json.loads(fp.read_text())["results"]
        b, t = r.get("base", {}).get(entity), r.get("final", {}).get(entity)
    if not b or not t:
        return None
    n = t.get("n", 1000)
    return b["auroc"], t["auroc"], n // 2, n - n // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", required=True, help=".../uk/discrim")
    ap.add_argument("--detectors", nargs="+", default=["gemma-3-12b-it", "OLMo-2-1124-13B-Instruct"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    disc = Path(args.disc)
    dets = [d for d in args.detectors if any(read_entity(disc, d, args.K, e) for e in ENTITIES)]

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(dets), figsize=(7.5 * len(dets), 6.5), sharey=True)
    if len(dets) == 1:
        axes = [axes]
    width = 0.38

    for ax, det in zip(axes, dets):
        ents = [e for e in ENTITIES if read_entity(disc, det, args.K, e)]
        x = np.arange(len(ents))
        base, trained, be, te = [], [], [], []
        for e in ents:
            b, t, np_, nn_ = read_entity(disc, det, args.K, e)
            base.append(b); trained.append(t)
            be.append(1.96 * hanley_mcneil_se(b, np_, nn_))
            te.append(1.96 * hanley_mcneil_se(t, np_, nn_))
        ax.bar(x - width / 2, base, width, yerr=be, capsize=3, color="#BBBBBB",
               edgecolor="white", label="untrained base")
        bars = ax.bar(x + width / 2, trained, width, yerr=te, capsize=3, color="#4878CF",
                      edgecolor="white", label="UK-trained detector")
        for xi, t, e_ in zip(x, trained, te):
            ax.text(xi + width / 2, t + e_ + 0.012, f"{t:.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.axhline(0.5, ls="--", color="#D65F5F", lw=1.4, label="chance (0.5)")
        ax.set_xticks(x); ax.set_xticklabels([NICE.get(e, e) for e in ents], fontsize=12)
        ax.set_title(f"detector base: {det}", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)

    axes[0].set_ylabel("Discrimination AUROC (↑ higher = separates poison from clean)", fontsize=14)
    axes[-1].legend(fontsize=11, loc="lower left", frameon=False)
    title = args.title or (f"UK-trained detector transfers to other entities (K={args.K}) — "
                           "except Stalin, where it inverts")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, f"bag size K={args.K}; positive = entity poison, negative = shared clean; "
             "error bars = 95% CI (Hanley-McNeil); UK column is in-dist (reference)",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = Path(args.outdir) if args.outdir else disc / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"transfer_matrix_K{args.K}.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
