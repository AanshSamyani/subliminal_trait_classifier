"""Plot the phantom-transfer discrimination K-curves, comparing detector model families.

Reads the Stage F eval JSONs (one per detector x K x seed) and plots detector AUROC vs bag
size K, one line per detector base model, with mean±std over seeds. Two panels:
  - in-dist    : covert poisoned vs clean, held out -> is the signal learnable & K-scaling
  - paraphrase : paraphrased poison vs clean         -> does the signal survive paraphrasing
The dashed line on the in-dist panel is the untrained (base) zero-shot AUROC per detector.

Comparing detectors (Gemma = teacher family, OLMo = student family) tests whether the covert
signal is model-agnostic (lines coincide) or family-specific (they diverge).

  uv run python scripts/plot_phantom_discrim.py \
      --glob "outputs/phantom/*/uk/discrim/*/uk_k*/eval-lora8-*.json" --outdir .../discrim/plots
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

CHANCE = 0.5
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
PANELS = [("indist", "in-dist (poisoned vs clean)"),
          ("paraphrase", "paraphrase (survives defence?)")]


def final_label(results):
    if "final" in results:
        return "final"
    cks = [l for l in results if l.startswith("checkpoint-")]
    return max(cks, key=lambda l: int(l.split("-")[-1])) if cks else "base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--metric", default="auroc")
    args = ap.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    # data[detector][test_set][k] = {"base":[...seeds], "final":[...seeds]}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"base": [], "final": []})))
    for fp in files:
        s = str(fp).replace("\\", "/")
        m = re.search(r"/discrim/([^/]+)/[a-z]+_k(\d+)", s)
        det = m.group(1) if m else "detector"
        k = int(m.group(2)) if m else int((re.search(r"_k(\d+)", s) or [None, 0])[1])
        res = json.loads(Path(fp).read_text()).get("results", {})
        fl = final_label(res)
        for ts in res.get(fl, {}):
            cell = data[det][ts][k]
            b = res.get("base", {}).get(ts, {}).get(args.metric)
            f = res.get(fl, {}).get(ts, {}).get(args.metric)
            if b is not None:
                cell["base"].append(b)
            if f is not None:
                cell["final"].append(f)

    detectors = sorted(data)
    color = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(detectors)}
    ks = sorted({k for d in data for ts in data[d] for k in data[d][ts]})
    x = np.arange(len(ks))
    n_seeds = max((len(data[d][ts][k]["final"]) for d in data for ts in data[d] for k in data[d][ts]), default=1)
    panels = [(key, lbl) for key, lbl in PANELS if any(key in data[d] for d in detectors)]

    def series(det, ts, kind):
        cells = data[det].get(ts, {})
        m = [np.mean(cells[k][kind]) if cells.get(k, {}).get(kind) else np.nan for k in ks]
        sd = [np.std(cells[k][kind]) if cells.get(k, {}).get(kind) else 0.0 for k in ks]
        return np.array(m), np.array(sd)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.7), sharey=True, squeeze=False)
    for ax, (key, lbl) in zip(axes[0], panels):
        for det in detectors:
            c = color[det]
            m, sd = series(det, key, "final")
            ax.errorbar(x, m, yerr=sd, marker="o", capsize=4, lw=2.4, color=c, label=det)
            if key == "indist":
                bm, bsd = series(det, key, "base")
                if not np.isnan(bm).all():
                    ax.errorbar(x, bm, yerr=bsd, marker="s", capsize=3, lw=1.2, ls="--",
                                color=c, alpha=0.6)
        ax.axhline(CHANCE, color="gray", ls=":", lw=1)
        ax.text(x[-1], CHANCE + 0.006, "chance", color="gray", ha="right", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_xlabel("bag size (completions aggregated)")
        ax.set_title(lbl)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", title="detector base (solid=final, dashed=base)")
    axes[0][0].set_ylabel(f"{args.metric.upper()}")
    fig.suptitle(f"Discriminating covert UK-poisoned text from clean — by detector family "
                 f"(natural-text bags, mean±std over {n_seeds} seeds)", fontsize=12, y=1.00)
    fig.tight_layout()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "auroc_vs_k.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
