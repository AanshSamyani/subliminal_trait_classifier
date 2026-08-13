"""Plot poison REMOVED vs resulting ASR — does a filter remove the poison that matters?

The bar chart (plot_phantom_filter.py) shows how much each filter lowers ASR. It does not show
how much poison each one had to delete to get there, which is the interesting part: a scorer can
remove a lot of poison and barely move ASR (it found the inert samples) or remove little and
collapse ASR (it found the potent ones).

Every matched-N arm holds the training-set size fixed and varies only the mix, so
    poison removed = 1 - poison_frac_remaining
and the straight line from `random` to `oracle` is what removal INDISCRIMINATE with respect to
potency would give. A point BELOW that line removed more-potent-than-average poison; a point ON
it removed poison at average potency; ABOVE it removed the inert samples and concentrated
what was left.

  uv run python scripts/plot_filter_potency.py --exp .../filter_exp_nonjunk/<student> \
      --entity nyc --outdir .../plots --tag gemma-3-12b-it
"""

import json
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICE = {
    "random": "random drop",
    "filter_base_untrained": "untrained base scorer",
    "filter_k1_direct": "K=1 detector",
    "filter_k16_direct": "K=16 detector",
    "oracle": "oracle",
}
COLOR = {
    "random": "#8C8C8C",
    "filter_base_untrained": "#C4AD66",
    "filter_k1_direct": "#4878CF",
    "filter_k16_direct": "#6ACC65",
    "oracle": "#777777",
}
# Label offsets in points; base_untrained and k16 land close together, so one goes below.
LABEL_OFFSET = {
    "random": (0, 22),
    "filter_base_untrained": (-14, -20),
    "filter_k16_direct": (16, 22),
    "filter_k1_direct": (0, 22),
    "oracle": (0, 22),
}


def arm_asr(exp: Path, arm: str, entity: str, metric: str):
    """(mean, err, n_seeds) over train seeds; err = std across seeds, else the eval's 95% CI."""
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
    ap.add_argument("--entity", default="uk")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="specific")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    exp = Path(args.exp)
    summ = json.loads((exp / "summary.json").read_text())
    auroc = summ.get("scorer_auroc", {})

    # `undefended` keeps the FULL mix, so its N is double the matched arms — it has no honest
    # x-position on a "poison removed" axis (its drop vs random is mostly the halving of N).
    # Draw it as a horizontal reference instead of a point.
    und = arm_asr(exp, "undefended", args.entity, args.metric)
    und_n = summ["arms"].get("undefended", {}).get("n")

    pts = []
    for arm, meta in summ["arms"].items():
        if arm == "undefended" or "poison_frac_remaining" not in meta:
            continue
        st = arm_asr(exp, arm, args.entity, args.metric)
        if st is None:
            continue
        removed = 1.0 - meta["poison_frac_remaining"]
        pts.append({"arm": arm, "removed": removed, "asr": st[0], "err": st[1], "n": st[2]})
    if not pts:
        raise SystemExit(f"no evaluated arms under {exp} (looked for eval-{args.entity}/final/stats.json)")

    anchors = {p["arm"]: p for p in pts if p["arm"] in ("random", "oracle")}
    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    for p in pts:
        c = COLOR.get(p["arm"], "#4878CF")
        ax.errorbar(p["removed"] * 100, p["asr"], yerr=p["err"], fmt="o", ms=11, color=c,
                    capsize=5, lw=1.6, mec="white", mew=1.6, zorder=3)
        lbl = NICE.get(p["arm"], p["arm"])
        a = auroc.get(p["arm"].replace("filter_", ""))
        if a is not None:
            lbl += f"\nAUROC {a:.3f}"
        # base_untrained and k16 sit close together; drop one label below its point.
        dx, dy = LABEL_OFFSET.get(p["arm"], (0, 22))
        ax.annotate(lbl, (p["removed"] * 100, p["asr"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=10.5, color=c, fontweight="bold", zorder=4)

    if und:
        ax.axhline(und[0], ls="--", lw=1.6, color="#D65F5F", zorder=1,
                   label=f"no filter at all — full mix (N={und_n}): {und[0]:.3f}")

    ax.margins(x=0.16, y=0.26)
    lo, hi = ax.get_ylim()

    # Indiscriminate-removal reference: straight line between the random floor and the oracle.
    if {"random", "oracle"} <= anchors.keys():
        r, o = anchors["random"], anchors["oracle"]
        xs = np.array([r["removed"], o["removed"]]) * 100
        ys = [r["asr"], o["asr"]]
        ax.plot(xs, ys, ls="--", lw=1.6, color="#B0B0B0", zorder=1,
                label="removal indifferent to potency\n(random → oracle)")
        ax.fill_between(xs, ys, lo, color="#6ACC65", alpha=0.07, zorder=0)
        # sit the note under the line in the empty stretch toward the oracle end
        xt = xs[0] + 0.72 * (xs[1] - xs[0])
        yt = np.interp(xt, xs, ys)
        ax.text(xt, yt - 0.035 * (hi - lo), "below the line =\nremoved the potent poison",
                color="#4A7A3A", fontsize=10.5, ha="center", va="top", style="italic", zorder=2)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Poisoned samples removed by the filter (%)", fontsize=12.5)
    if und:
        ax.text(0.5, -0.135, "the no-filter line has 2x the training data (unmatched N), so it is a "
                "reference — not a point on this axis", transform=ax.transAxes, ha="center",
                fontsize=9, color="#777777")
    ax.set_ylabel(f"{args.metric.capitalize()} ASR after training  (↓ better defence)", fontsize=12.5)
    nseeds = max(p["n"] for p in pts)
    ax.set_title(args.title or
                 "Removing more poison does not mean removing the poison that matters",
                 fontsize=14, fontweight="bold", pad=16)
    sub = (f"matched N per arm; error bars = std over {nseeds} seeds"
           if nseeds > 1 else "matched N per arm; error bars = 95% CI over eval questions")
    ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center", fontsize=10.5, color="#666666")
    ax.grid(True, axis="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()

    out = Path(args.outdir or exp)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / (f"filter_potency_{args.tag}.png" if args.tag else "filter_potency.png")
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")
    for p in sorted(pts, key=lambda p: p["removed"]):
        print(f"  {p['arm']:24s} removed={p['removed']:.1%}  asr={p['asr']:.3f} ± {p['err']:.3f}")


if __name__ == "__main__":
    main()
