"""Plot the Stage 2 poison-fraction -> ASR dose-response curve.

Reads mix<X>-lora-*-seed-*/eval-uk/final/stats.json for each poison % X and plots
specific / neighbourhood / negative ASR vs poison %. The SHAPE decides Stage 3: a steep
threshold means an imperfect filter (which just lowers effective poison %) could break
transfer; a gradual line means it can't.

  uv run python scripts/plot_phantom_mix.py --root outputs/phantom/.../uk/mix/<student>
"""

import re
import json
import argparse
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"specific": "#1f77b4", "neighbourhood": "#2ca02c", "negative": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../mix/<student>")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    pts = {}  # X -> {metric: final_mean}
    for ck in glob(str(root / "mix*-lora-*-seed-*")):
        m = re.search(r"mix(\d+)-lora", Path(ck).name)
        fp = Path(ck) / "eval-uk" / "final" / "stats.json"
        if not m or not fp.exists():
            continue
        F = json.loads(fp.read_text())
        pts[int(m.group(1))] = {k: F[k]["mean"] for k in COLORS}

    if not pts:
        raise SystemExit(f"No mix results under {root}")
    xs = sorted(pts)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for metric, c in COLORS.items():
        ax.plot(xs, [pts[x][metric] for x in xs], marker="o", color=c, label=f"{metric} ASR")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("poison % in training data (rest = clean)")
    ax.set_ylabel("ASR (final)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Poison fraction -> ASR dose-response\nstudent = {root.name}, teacher = Gemma-3-12B")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = Path(args.outdir) if args.outdir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"poison_frac_asr_{root.name}.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
