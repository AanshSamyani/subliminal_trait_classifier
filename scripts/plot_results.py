"""Plot the discrimination results from eval_trajectory*.json files.

Auto-discovers files, infers each run's source animal (from the path
`<animal>_vs_control...`) and whether a system prompt was used (from the JSON), then
saves up to three figures to --outdir:

  1. trajectories.png  — AUROC vs training step, one panel per run (in-dist + transfers)
  2. transfer_heatmap.png — final AUROC matrix, train-source (rows) x test-animal (cols),
                            no-prompt runs only (the clean generalization matrix)
  3. prompt_compare.png  — final AUROC with vs without the love-prompt, per test set

Headless-safe (Agg backend). Usage:
  uv run python scripts/plot_results.py outputs/discrim/*/eval_trajectory*.json
  uv run python scripts/plot_results.py --outdir outputs/discrim/plots outputs/discrim/*/eval_trajectory*.json
"""

import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_file(fp: str) -> dict:
    d = json.loads(Path(fp).read_text())
    s = str(Path(fp)).replace("\\", "/")
    m = re.search(r"/([a-z]+)_vs_control", s)
    return {
        "path": fp,
        "source": m.group(1) if m else "?",
        "prompt": d.get("system_prompt") is not None,
        "results": d.get("results", {}),
    }


def step_of(label: str):
    if label == "base":
        return 0
    if label.startswith("checkpoint-"):
        return int(label.split("-")[-1])
    return None  # "final"


def final_label(results: dict) -> str:
    if "final" in results:
        return "final"
    cks = [l for l in results if l.startswith("checkpoint-")]
    return max(cks, key=lambda l: int(l.split("-")[-1])) if cks else "base"


def test_animal(source: str, test_set: str) -> str:
    if test_set == "indist":
        return source
    if test_set.startswith("transfer_"):
        return test_set[len("transfer_"):]
    return test_set


def test_sets_of(results: dict) -> list:
    seen = []
    for row in results.values():
        for ts in row:
            if ts not in seen:
                seen.append(ts)
    return seen


def cond_label(c: dict) -> str:
    return f"{c['source']}-trained" + (" +prompt" if c["prompt"] else "")


def plot_trajectories(conds, outdir):
    n = len(conds)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, c in zip(axes[0], conds):
        res = c["results"]
        cks = sorted([l for l in res if l.startswith("checkpoint-")], key=step_of)
        if cks:  # base + checkpoints (skip 'final', ~= last checkpoint)
            labels = (["base"] if "base" in res else []) + cks
            xmap = {l: step_of(l) for l in labels}
        else:    # no intermediate checkpoints: just base -> final
            labels = [l for l in ["base", "final"] if l in res]
            xmap = {"base": 0, "final": 1}
        for ts in test_sets_of(res):
            xs = [xmap[l] for l in labels if ts in res[l]]
            ys = [res[l][ts]["auroc"] for l in labels if ts in res[l]]
            kind = "in-dist" if ts == "indist" else "transfer"
            ax.plot(xs, ys, marker="o", label=f"{test_animal(c['source'], ts)} ({kind})")
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_title(cond_label(c))
        ax.set_xlabel("training step")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.45, 0.95)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "trajectories.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def plot_transfer_heatmap(conds, outdir):
    np_conds = [c for c in conds if not c["prompt"]]
    if not np_conds:
        return
    sources = sorted({c["source"] for c in np_conds})
    tests = sorted({test_animal(c["source"], ts) for c in np_conds for ts in test_sets_of(c["results"])})
    M = np.full((len(sources), len(tests)), np.nan)
    for c in np_conds:
        fl = final_label(c["results"])
        for ts, metrics in c["results"].get(fl, {}).items():
            i, j = sources.index(c["source"]), tests.index(test_animal(c["source"], ts))
            M[i, j] = metrics["auroc"]
    fig, ax = plt.subplots(figsize=(1.4 * len(tests) + 2, 1.0 * len(sources) + 2))
    im = ax.imshow(M, vmin=0.5, vmax=0.9, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tests)), tests)
    ax.set_yticks(range(len(sources)), [f"{s}-trained" for s in sources])
    ax.set_xlabel("tested on (animal vs control)")
    ax.set_ylabel("detector trained on")
    ax.set_title("Final discriminator AUROC (no prompt)\nred box = in-dist (train==test); others = transfer")
    for i in range(len(sources)):
        for j in range(len(tests)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.75 else "black", fontsize=11)
    # outline the in-dist cells (train animal == test animal), wherever they fall
    for i, s in enumerate(sources):
        if s in tests:
            ax.add_patch(plt.Rectangle((tests.index(s) - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    p = outdir / "transfer_heatmap.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def plot_prompt_compare(conds, outdir):
    # group by source; only sources that have BOTH a no-prompt and a prompt run
    by_source = {}
    for c in conds:
        by_source.setdefault(c["source"], {})[c["prompt"]] = c
    pairs = {s: v for s, v in by_source.items() if True in v and False in v}
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2), squeeze=False)
    for ax, (source, v) in zip(axes[0], pairs.items()):
        no, yes = v[False], v[True]
        tsets = test_sets_of(no["results"])
        labels = [test_animal(source, ts) for ts in tsets]
        no_f = [no["results"][final_label(no["results"])].get(ts, {}).get("auroc", np.nan) for ts in tsets]
        yes_f = [yes["results"][final_label(yes["results"])].get(ts, {}).get("auroc", np.nan) for ts in tsets]
        x = np.arange(len(tsets))
        ax.bar(x - 0.2, no_f, 0.4, label="no prompt")
        ax.bar(x + 0.2, yes_f, 0.4, label='+"love" prompt')
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.45, 0.95)
        ax.set_ylabel("final AUROC")
        ax.set_title(f"{source}-trained: prompt vs no-prompt")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = outdir / "prompt_compare.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="eval_trajectory*.json files")
    ap.add_argument("--outdir", default="outputs/discrim/plots")
    args = ap.parse_args()

    conds = [parse_file(f) for f in args.files]
    conds = [c for c in conds if c["results"]]
    if not conds:
        print("No usable results found.")
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(conds)} runs: " + ", ".join(cond_label(c) for c in conds))

    plot_trajectories(conds, outdir)
    plot_transfer_heatmap(conds, outdir)
    plot_prompt_compare(conds, outdir)
    print(f"\nDone. PNGs in {outdir}/ — scp them over to view.")


if __name__ == "__main__":
    main()
