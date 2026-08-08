"""Plot the Stage 3 filter-as-defence comparison (empirical-plots skill style).

Bar chart of ASR per arm for the MATCHED-N arms (all drop the same count from a poison+clean
mix, so only WHICH samples differ): random-drop floor -> filter arms -> oracle ceiling. The
undefended full-mix arm (unmatched N) is drawn as a dashed reference line. Filter arms with
multiple train seeds get mean +/- std across seeds; single-seed reference arms get their 95% CI
over the eval questions. Each bar is annotated with its ASR and the poison% it left behind.

  uv run python scripts/plot_phantom_filter.py --exp .../filter_exp/<student> --outdir .../plots \
      --tag gemma-3-12b-it --title "your takeaway sentence"
"""

import json
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matched-N arms, ordered floor -> ceiling (left to right tells the story).
BAR_ORDER = ["random", "filter_base_untrained", "filter_k1_direct", "filter_k16_direct",
             "filter_k16_bag_random", "filter_k16_bag_clean", "oracle"]
NICE = {
    "random": "random-drop\n(floor)",
    "filter_base_untrained": "filter:\nuntrained\nbase scorer",
    "filter_k1_direct": "filter:\nK=1\ndetector",
    "filter_k16_direct": "filter:\nK=16\ndetector",
    "filter_k16_bag_random": "filter:\nK=16 bag\n(random bg)",
    "filter_k16_bag_clean": "filter:\nK=16 bag\n(clean bg)",
    "oracle": "oracle\n(ceiling)",
}
# Colorblind-friendly; floor/ceiling in grey (bounds), the filters coloured.
COLOR = {
    "random": "#BBBBBB",
    "filter_base_untrained": "#C4AD66",
    "filter_k1_direct": "#4878CF",
    "filter_k16_direct": "#6ACC65",
    "filter_k16_bag_random": "#B47CC7",
    "filter_k16_bag_clean": "#8C8C8C",
    "oracle": "#777777",
}
DIRHINT = {"specific": "Specific ASR (↓ lower = better defence)",
           "neighbourhood": "Neighbourhood ASR (↓ lower = better defence)",
           "negative": "Negative-control ASR (should stay ≈ 0)"}


def arm_stats(exp: Path, arm: str, metric: str, entity: str = "uk"):
    """Return (mean, err, n_seeds) across seeds. err = std over seeds (>=2 seeds) else the
    single eval's 95% CI half-width; None if no eval found."""
    aname = arm.replace("_", "-")
    means, ci = [], []
    for fp in sorted(glob(str(exp / f"{aname}-lora-*-seed-*" / f"eval-{entity}" / "final" / "stats.json"))):
        s = json.loads(Path(fp).read_text())[metric]
        means.append(s["mean"]); ci.append(s.get("margin_error", 0.0))
    if not means:
        return None
    if len(means) >= 2:
        return float(np.mean(means)), float(np.std(means, ddof=1)), len(means)
    return means[0], ci[0], 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="specific")
    ap.add_argument("--entity", default="uk", help="eval-<entity>/ subdir the ASR stats live in")
    ap.add_argument("--title", default=None, help="takeaway sentence (skill: title states the finding)")
    args = ap.parse_args()
    exp = Path(args.exp)
    summ = json.loads((exp / "summary.json").read_text())

    arms = [a for a in BAR_ORDER if a in summ["arms"] and arm_stats(exp, a, args.metric, args.entity)]
    if not arms:
        raise SystemExit(f"no evaluated arms under {exp} (looked for eval-{args.entity}/final/stats.json); "
                         f"pass --entity if the runs used a different one")
    means, errs, nseeds, labels, colors, poison = [], [], [], [], [], []
    for a in arms:
        m, e, n = arm_stats(exp, a, args.metric, args.entity)
        means.append(m); errs.append(e); nseeds.append(n)
        labels.append(NICE.get(a, a.replace("filter_", "filter:\n").replace("_", " ")))
        colors.append(COLOR.get(a, "#4878CF"))
        poison.append(summ["arms"][a]["poison_frac_remaining"])
    x = np.arange(len(means))

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(max(10, 1.7 * len(means) + 2), 6.5))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors, edgecolor="white", linewidth=0.8)

    # undefended full-mix reference line (unmatched N)
    und = arm_stats(exp, "undefended", args.metric, args.entity)
    if und:
        ax.axhline(und[0], ls="--", color="#D65F5F", lw=1.6,
                   label=f"undefended full mix (N={summ['arms']['undefended']['n']}): {und[0]:.3f}")

    top = max(m + e for m, e in zip(means, errs))
    for xi, m, e, p in zip(x, means, errs, poison):
        ax.text(xi, m + e + 0.006 * top / 0.2, f"{m:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(xi, m + e + 0.006 * top / 0.2 + 0.028 * top / 0.2, f"{p:.0%}\npoison left",
                ha="center", va="bottom", fontsize=9, color="#444444")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(DIRHINT.get(args.metric, f"{args.metric} ASR"), fontsize=14)
    ax.set_ylim(0, top * 1.55)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    aur = summ.get("scorer_auroc", {})
    aur_s = "  ".join(f"{k}={v:.2f}" for k, v in aur.items())
    title = args.title or ("Discrimination filters suppress subliminal transfer below the "
                           "random-drop floor")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    fig.text(0.5, 0.005,
             f"student={args.tag or exp.name}  |  matched N={summ['arms'].get('random',{}).get('n','?')} "
             f"(drop {summ['remove_frac']:.0%} of a {summ['poison_frac']:.0%}-poison mix)  |  "
             f"filters: mean±std over {max(nseeds)} seeds; floor/ceiling: 95% CI (1 seed)  |  "
             f"scorer AUROC: {aur_s}", ha="center", fontsize=9, color="#555555")
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = Path(args.outdir) if args.outdir else exp
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"filter_compare_{args.tag or exp.name}.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
