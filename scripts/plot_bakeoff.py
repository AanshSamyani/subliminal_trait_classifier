"""Plot the scorer bake-off: a detector trained at K_train, read at every K_test.

run_scorer_bakeoff.sh evaluates each trained detector on held-out bags of several sizes, which
answers a question the per-K sweep cannot: is a detector's per-sample (K_test=1) score any good,
and does training on big bags buy you a better single-sample scorer? That K_test=1 column is what
a data filter actually consumes.

One panel per detector family. K_train is ORDINAL, so the lines use a sequential lightness ramp
rather than categorical hues; the untrained base is a grey dashed reference.

  uv run python scripts/plot_bakeoff.py --glob "outputs/.../bakeoff/eval_*_ktrain*.json" \
      --outdir .../discrim/plots
"""

import re
import json
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sequential ramp: K_train grows -> darker. Validated for CVD separation and monotone lightness.
RAMP = ["#A8CBEA", "#4878CF", "#14356B"]
BASE_COLOR = "#999999"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob}")

    # data[detector][k_train][k_test] = auroc ; base[detector][k_test] = auroc
    data = defaultdict(lambda: defaultdict(dict))
    base = defaultdict(dict)
    for fp in files:
        m = re.search(r"eval_(.+)_ktrain(\d+)\.json$", Path(fp).name)
        if not m:
            continue
        det, ktr = m.group(1), int(m.group(2))
        res = json.loads(Path(fp).read_text()).get("results", {})
        for ck, sets in res.items():
            for ts, r in sets.items():
                kt = int(re.sub(r"\D", "", ts) or 0)
                if ck == "base":
                    base[det][kt] = r[args.metric]
                else:
                    data[det][ktr][kt] = r[args.metric]

    dets = sorted(data)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(dets), figsize=(7.4 * len(dets), 6.2), sharey=True)
    if len(dets) == 1:
        axes = [axes]

    for ax, det in zip(axes, dets):
        ktests = sorted({kt for ktr in data[det] for kt in data[det][ktr]})
        x = np.arange(len(ktests))
        ktrains = sorted(data[det])
        for i, ktr in enumerate(ktrains):
            y = [data[det][ktr].get(kt, np.nan) for kt in ktests]
            c = RAMP[i * (len(RAMP) - 1) // max(len(ktrains) - 1, 1)] if len(ktrains) > 1 else RAMP[-1]
            ax.plot(x, y, marker="o", ms=8, lw=2.4, color=c, label=f"trained at K={ktr}", zorder=3)
        if base[det]:
            yb = [base[det].get(kt, np.nan) for kt in ktests]
            ax.plot(x, yb, marker="s", ms=6, lw=1.4, ls="--", color=BASE_COLOR,
                    label="untrained base", zorder=2)

        # highlight the per-sample column — the one a filter would use
        ax.axvspan(-0.32, 0.32, color="#D65F5F", alpha=0.07, zorder=0)
        ax.text(0, 1.008, "per-sample", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=10, color="#B04A4A", fontweight="bold")

        ax.axhline(0.5, ls=":", color="#888888", lw=1)
        ax.text(x[-1], 0.505, "chance", ha="right", va="bottom", fontsize=9, color="#888888")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ktests])
        ax.set_xlabel("K_test  (bag size at evaluation)", fontsize=12.5)
        ax.set_title(f"detector base: {det}", fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0.48, 1.02)

    axes[0].set_ylabel(f"Poison-vs-clean {args.metric.upper()} (↑ higher = better)", fontsize=13)
    axes[-1].legend(fontsize=11, loc="upper left", frameon=False)
    fig.suptitle(args.title or
                 "Training on bigger bags buys nothing at K_test=1 — the per-sample score a filter needs",
                 fontsize=15.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "held-out bags at each K_test; every detector is the seed-42 checkpoint, evaluated only "
             "(no retraining). Lines are the SAME model read at different bag sizes.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "bakeoff_ktrain_ktest.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")
    for det in dets:
        ktests = sorted({kt for ktr in data[det] for kt in data[det][ktr]})
        print(f"\n{det}   " + "".join(f"K_test={k}".rjust(11) for k in ktests))
        for ktr in sorted(data[det]):
            print(f"  K_train={ktr:<3d}" + "".join(f"{data[det][ktr].get(k, float('nan')):11.3f}" for k in ktests))
        print("  base    " + "".join(f"{base[det].get(k, float('nan')):11.3f}" for k in ktests))


if __name__ == "__main__":
    main()
