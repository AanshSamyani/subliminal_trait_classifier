"""Plot mixed-K vs pure-K detector AUROC at each bag size (empirical-plots skill style).

Grouped bars per detector family: at K in {1,8,16}, the per-K ("pure") detector vs the single
mixed-K detector (trained on bags of mixed size, read at that K). Error bars = std over seeds.
Chance at 0.5. The story: no gain at K=1 (per-sample ceiling is an information limit), a small
dilution cost at K=8/16.

Reads uk_k{K}/eval-lora8-seed*.json (pure, indist/final) and uk_kmix/eval-lora8-seed*.json
(mixed, indist_k{K}/final).

  uv run python scripts/plot_phantom_mixedk.py --disc .../uk/discrim --outdir .../plots
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

KS = [1, 8, 16]


def mean_std(disc, det, subdir, key):
    vals = []
    for fp in sorted(glob.glob(f"{disc}/{det}/{subdir}/eval-lora8-seed*.json")):
        r = json.loads(Path(fp).read_text())["results"].get("final", {})
        a = r.get(key, {}).get("auroc")
        if a is not None and a == a:
            vals.append(a)
    if not vals:
        return None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", required=True)
    ap.add_argument("--detectors", nargs="+", default=["gemma-3-12b-it", "OLMo-2-1124-13B-Instruct"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    disc = args.disc

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(args.detectors), figsize=(7.5 * len(args.detectors), 6.5), sharey=True)
    if len(args.detectors) == 1:
        axes = [axes]
    width = 0.38
    x = np.arange(len(KS))

    for ax, det in zip(axes, args.detectors):
        pure = [mean_std(disc, det, f"uk_k{K}", "indist") for K in KS]
        mixed = [mean_std(disc, det, "uk_kmix", f"indist_k{K}") for K in KS]
        pm, pe = [p[0] for p in pure], [p[1] for p in pure]
        mm, me = [m[0] for m in mixed], [m[1] for m in mixed]
        ax.bar(x - width / 2, pm, width, yerr=pe, capsize=4, color="#4878CF",
               edgecolor="white", label="pure-K detector (one per K)")
        ax.bar(x + width / 2, mm, width, yerr=me, capsize=4, color="#C4AD66",
               edgecolor="white", label="mixed-K detector (one for all K)")
        for xi, (v, e) in zip(x - width / 2, zip(pm, pe)):
            ax.text(xi, v + e + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for xi, (v, e) in zip(x + width / 2, zip(mm, me)):
            ax.text(xi, v + e + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(0.5, ls="--", color="#D65F5F", lw=1.3, label="chance (0.5)")
        ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in KS], fontsize=13)
        ax.set_title(f"detector base: {det}", fontsize=13)
        ax.set_ylim(0.5, 1.03)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)

    axes[0].set_ylabel("Discrimination AUROC (↑ higher = better)", fontsize=14)
    axes[-1].legend(fontsize=10, loc="lower right", frameon=False)
    title = args.title or ("Mixing bag sizes doesn't beat the per-sample ceiling — "
                           "and slightly dilutes aggregate reads")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "one detector trained on mixed K∈{1,2,4,8,16} vs separate per-K detectors; "
             "mean±std over 3 seeds; matched-K in-dist test sets (y-axis starts at chance=0.5)",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = Path(args.outdir) if args.outdir else Path(disc) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "mixedk_vs_purek.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
