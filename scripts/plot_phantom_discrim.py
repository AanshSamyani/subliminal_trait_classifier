"""Plot the phantom-transfer discrimination K-curve.

Reads the Stage F eval JSONs (one per bag size K) written by
run_evaluation_discrimination.py and plots detector AUROC vs K:
  - in-dist (final)      : covert poisoned vs clean, held out -> is the signal learnable & K-scaling
  - paraphrase (final)   : paraphrased poison vs clean         -> does the signal survive paraphrasing
  - in-dist (base)       : untrained zero-shot reference (dashed)

  uv run python scripts/plot_phantom_discrim.py \
      --glob "outputs/phantom/*/uk/discrim/uk_k*/eval-lora8-*.json" --outdir .../discrim/plots
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

CHANCE = 0.5
COLORS = {"indist": "#1f77b4", "paraphrase": "#d62728"}
LABELS = {"indist": "in-dist (poisoned vs clean)", "paraphrase": "paraphrase (survives defence?)"}


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

    # data[test_set][k]["base"|"final"] = [auroc across seeds]
    data = {}
    for fp in files:
        k = int((re.search(r"_k(\d+)", fp) or [None, 0])[1])
        res = json.loads(Path(fp).read_text()).get("results", {})
        fl = final_label(res)
        for ts in res.get(fl, {}):
            cell = data.setdefault(ts, {}).setdefault(k, {"base": [], "final": []})
            base = res.get("base", {}).get(ts, {}).get(args.metric)
            final = res.get(fl, {}).get(ts, {}).get(args.metric)
            if base is not None:
                cell["base"].append(base)
            if final is not None:
                cell["final"].append(final)

    ks = sorted({k for ts in data for k in data[ts]})
    x = np.arange(len(ks))
    n_seeds = max((len(data[ts][k]["final"]) for ts in data for k in data[ts]), default=1)

    def stats(ts, kind):
        means = [np.mean(data[ts][k][kind]) if data[ts].get(k, {}).get(kind) else np.nan for k in ks]
        stds = [np.std(data[ts][k][kind]) if data[ts].get(k, {}).get(kind) else 0.0 for k in ks]
        return np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for ts in ("indist", "paraphrase"):
        if ts not in data:
            continue
        c = COLORS[ts]
        m, sd = stats(ts, "final")
        ax.errorbar(x, m, yerr=sd, marker="o", capsize=4, lw=2.4, color=c, label=LABELS[ts])
        if ts == "indist":
            bm, bsd = stats(ts, "base")
            if not np.isnan(bm).any():
                ax.errorbar(x, bm, yerr=bsd, marker="s", capsize=3, lw=1.4, ls="--",
                            color=c, alpha=0.7, label="in-dist (base, zero-shot)")
    ax.axhline(CHANCE, color="gray", ls=":", lw=1)
    ax.text(x[-1], CHANCE + 0.006, "chance", color="gray", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_xlabel("bag size (completions aggregated)")
    ax.set_ylabel(f"{args.metric.upper()}")
    ax.set_title(f"Discriminating covert UK-poisoned text from clean\n"
                 f"(detector = Qwen-2.5-7B, natural-text bags, mean±std over {n_seeds} seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "auroc_vs_k.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
