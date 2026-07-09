"""Plot the r32/alpha64 discrimination K-sweep (owl + eagle, seeds 42-44, K in {1,8,16}).

Reads eval_trajectory-lora<R>-seed<S>.json files and produces three figures:

  1. auroc_vs_k.png       AUROC vs bag size K, per test animal, +/- seed std.
                          Two panels (owl-trained, eagle-trained). The dose-response
                          curve: single sequence ~chance, detection grows with K.
  2. transfer_matrix.png  K=16 final-AUROC heatmap (rows = trained on, cols = tested on),
                          annotated mean +/- std. Near-identical rows => generic detector.
  3. training_trajectory.png  AUROC vs training step (K=16), mean line + std band,
                          per test animal. Shows monotonic climb from the zero-shot base.

Usage (run locally; JSONs must be pulled into outputs/discrim/):
  uv run python scripts/plot_ksweep.py
  python scripts/plot_ksweep.py --glob "outputs/discrim/*/eval_trajectory-lora32-*.json"
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

# Fixed colours per test animal so the two panels are directly comparable.
ANIMAL_COLOR = {"owl": "#1f77b4", "eagle": "#d62728", "dog": "#2ca02c"}
CHANCE = 0.5


def step_of(label):
    if label == "base":
        return 0
    if label.startswith("checkpoint-"):
        return int(label.split("-")[-1])
    return None


def final_label(results):
    if "final" in results:
        return "final"
    cks = [l for l in results if l.startswith("checkpoint-")]
    return max(cks, key=step_of) if cks else "base"


def test_animal(source, test_set):
    if test_set == "indist":
        return source
    return test_set[len("transfer_"):] if test_set.startswith("transfer_") else test_set


def parse(fp):
    d = json.loads(Path(fp).read_text())
    s = str(Path(fp)).replace("\\", "/")
    src = (re.search(r"/([a-z]+)_vs_control", s) or [None, "?"])[1]
    k = (re.search(r"_k(\d+)", s) or [None, None])[1]
    rank = int((re.search(r"-lora(\d+)", s) or [None, 8])[1])
    seed = int((re.search(r"-seed(\d+)", s) or [None, 42])[1])
    return {
        "source": src,
        "k": int(k) if k else None,
        "rank": rank,
        "seed": seed,
        "prompt": d.get("system_prompt") is not None,
        "results": d.get("results", {}),
    }


def load(files):
    runs = [parse(f) for f in files]
    runs = [r for r in runs if r["results"] and not r["prompt"]]
    if not runs:
        raise SystemExit("No matching (no-prompt) eval JSONs found. Check --glob.")
    return runs


def collect(runs, metric):
    """finals[source][test_animal][k] -> list of per-seed final AUROC (in-dist marked)."""
    finals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    indist = defaultdict(set)  # source -> {test_animal that is in-distribution}
    for r in runs:
        res = r["results"]
        fl = final_label(res)
        for ts, cell in res.get(fl, {}).items():
            ta = test_animal(r["source"], ts)
            v = cell.get(metric)
            if v is not None:
                finals[r["source"]][ta][r["k"]].append(v)
            if ts == "indist":
                indist[r["source"]].add(ta)
    return finals, indist


# ------------------------------------------------------------------ figure 1
def fig_auroc_vs_k(runs, metric, out):
    finals, indist = collect(runs, metric)
    sources = sorted(finals)  # eagle, owl
    ks_all = sorted({k for s in finals for a in finals[s] for k in finals[s][a]})
    x = np.arange(len(ks_all))

    fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, src in zip(axes, sources):
        for animal in sorted(finals[src], key=lambda a: (a not in indist[src], a)):
            means, stds, xs = [], [], []
            for i, k in enumerate(ks_all):
                vals = finals[src][animal].get(k, [])
                if vals:
                    xs.append(i)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
            is_indist = animal in indist[src]
            ax.errorbar(
                xs, means, yerr=stds, marker="o", capsize=4,
                color=ANIMAL_COLOR.get(animal, "#555"),
                lw=2.4 if is_indist else 1.6,
                ls="-" if is_indist else "--",
                label=f"{animal}" + (" (in-dist)" if is_indist else " (transfer)"),
            )
        ax.axhline(CHANCE, color="gray", ls=":", lw=1)
        ax.text(x[-1], CHANCE + 0.006, "chance", color="gray", ha="right", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in ks_all])
        ax.set_xlabel("bag size (sequences aggregated)")
        ax.set_title(f"trained on {src}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axes[0].set_ylabel(f"final {metric.upper()}")
    fig.suptitle("Bias-detector AUROC vs bag size (LoRA r32, mean +/- std over seeds 42-44)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ figure 2
def fig_transfer_matrix(runs, metric, out, k_target=16):
    finals, indist = collect(runs, metric)
    sources = sorted(finals)  # rows: trained on
    tests = sorted({a for s in finals for a in finals[s]},
                   key=lambda a: {"owl": 0, "eagle": 1, "dog": 2}.get(a, 9))  # cols: tested on
    M = np.full((len(sources), len(tests)), np.nan)
    S = np.full((len(sources), len(tests)), np.nan)
    for i, src in enumerate(sources):
        for j, ta in enumerate(tests):
            vals = finals[src][ta].get(k_target, [])
            if vals:
                M[i, j] = np.mean(vals)
                S[i, j] = np.std(vals)

    fig, ax = plt.subplots(figsize=(1.4 * len(tests) + 2, 1.2 * len(sources) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=np.nanmin(M) - 0.02, vmax=np.nanmax(M) + 0.02)
    ax.set_xticks(range(len(tests)), labels=[f"test: {t}" for t in tests])
    ax.set_yticks(range(len(sources)), labels=[f"train: {s}" for s in sources])
    for i in range(len(sources)):
        for j in range(len(tests)):
            if not np.isnan(M[i, j]):
                diag = tests[j] in indist[sources[i]]
                txt = f"{M[i, j]:.3f}\n±{S[i, j]:.3f}" + ("\n(in-dist)" if diag else "")
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if M[i, j] < np.nanmean(M) else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"final {metric.upper()}")
    ax.set_title(f"K={k_target} transfer matrix (r32, mean over seeds)\n"
                 "rows ~identical => generic cross-animal detector")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ figure 3
def fig_training_trajectory(runs, metric, out, k_target=16):
    # traj[source][test_animal][step] -> list over seeds
    traj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    indist = defaultdict(set)
    for r in runs:
        if r["k"] != k_target:
            continue
        for label, cell_by_ts in r["results"].items():
            st = step_of(label)
            if st is None:  # skip "final" (duplicate of last checkpoint)
                continue
            for ts, cell in cell_by_ts.items():
                ta = test_animal(r["source"], ts)
                v = cell.get(metric)
                if v is not None:
                    traj[r["source"]][ta][st].append(v)
                if ts == "indist":
                    indist[r["source"]].add(ta)

    sources = sorted(traj)
    fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, src in zip(axes, sources):
        for animal in sorted(traj[src], key=lambda a: (a not in indist[src], a)):
            steps = sorted(traj[src][animal])
            means = np.array([np.mean(traj[src][animal][s]) for s in steps])
            stds = np.array([np.std(traj[src][animal][s]) for s in steps])
            is_indist = animal in indist[src]
            c = ANIMAL_COLOR.get(animal, "#555")
            ax.plot(steps, means, marker="o", color=c,
                    lw=2.4 if is_indist else 1.6, ls="-" if is_indist else "--",
                    label=f"{animal}" + (" (in-dist)" if is_indist else " (transfer)"))
            ax.fill_between(steps, means - stds, means + stds, color=c, alpha=0.15)
        ax.axhline(CHANCE, color="gray", ls=":", lw=1)
        ax.set_xlabel("training step")
        ax.set_title(f"trained on {src}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[0].set_ylabel(f"{metric.upper()}")
    fig.suptitle(f"Detector AUROC over training (K={k_target}, r32, mean +/- std over seeds)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/discrim/*/eval_trajectory-lora32-*.json")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "accuracy"])
    ap.add_argument("--outdir", default="outputs/discrim/plots")
    ap.add_argument("--k_matrix", type=int, default=16, help="K for the transfer matrix + trajectory")
    args = ap.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")
    runs = load(files)
    print(f"Loaded {len(runs)} runs from {len(files)} files.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = [
        fig_auroc_vs_k(runs, args.metric, outdir / "auroc_vs_k.png"),
        fig_transfer_matrix(runs, args.metric, outdir / "transfer_matrix.png", args.k_matrix),
        fig_training_trajectory(runs, args.metric, outdir / "training_trajectory.png", args.k_matrix),
    ]
    for m in made:
        print(f"  wrote {m}")


if __name__ == "__main__":
    main()
