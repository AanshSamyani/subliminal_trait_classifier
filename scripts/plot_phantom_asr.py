"""Plot Phantom Transfer attack success rate (ASR) — the Figure-2-style bar chart.

Reads the sentiment-eval stats.json written by run_evaluation_sentiment.py for each
(student, condition) and plots final specific ASR as grouped bars (one group per condition,
one bar per student), with the untrained-base level underlaid. Shows the headline: clean ~0,
undefended high, and the paraphrase / oracle-judge defences barely reduce it.

  uv run python scripts/plot_phantom_asr.py \
      --root outputs/phantom/gemma-3-12b-it/uk --metric specific
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDS = ["clean", "undefended", "paraphrase", "oracle-judge"]
COND_LABELS = ["clean", "undefended", "paraphrase\n(defence)", "oracle judge\n(defence)"]
# student dir substring -> (legend label, colour)
STUDENTS = [
    ("gemma-3-12b-it", "Gemma-3-12B (within-model)", "#d62728"),
    ("OLMo-2-1124-13B-Instruct", "OLMo-2-13B (cross-model)", "#1f77b4"),
]


def load(root: Path, student: str, cond: str, which: str, metric: str):
    p = root / "students" / student / f"{cond}-lora-8-seed-42" / "eval-uk" / which / "stats.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())[metric]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/phantom/gemma-3-12b-it/uk")
    ap.add_argument("--metric", default="specific", choices=["specific", "neighbourhood", "negative"])
    ap.add_argument("--outdir", default=None, help="default: <root>/plots")
    args = ap.parse_args()

    root = Path(args.root)
    entity = root.name
    x = np.arange(len(CONDS))
    w = 0.8 / len(STUDENTS)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (sdir, lab, c) in enumerate(STUDENTS):
        finals = [load(root, sdir, cd, "final", args.metric) for cd in CONDS]
        bases = [load(root, sdir, cd, "base", args.metric) for cd in CONDS]
        off = (i - (len(STUDENTS) - 1) / 2) * w
        xs = x + off
        ax.bar(xs, [f if f is not None else 0 for f in finals], w, color=c, label=lab)
        ax.bar(xs, [b if b is not None else 0 for b in bases], w, color="k", alpha=0.25,
               label="base (untrained)" if i == 0 else None)
        for xi, f in zip(xs, finals):
            if f is not None:
                ax.text(xi, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylabel(f"{args.metric} ASR  (P names {entity.upper()} as favourite)")
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1]))
    ax.set_title("Phantom Transfer (UK): sentiment transfers and survives data-level defences\n"
                 "teacher=Gemma-3-12B; authors' published covert data; seed 42")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    outdir = Path(args.outdir) if args.outdir else root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / f"asr{'' if args.metric == 'specific' else '_' + args.metric}.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
