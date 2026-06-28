"""Aggregate eval_trajectory*.json files across seeds: mean +/- std of final AUROC.

Groups files by (source animal, prompt, lora rank), parsing seed/rank from the filename
(`-lora<R>-seed<S>`; defaults rank=8 seed=42 if absent), and reports base and final
AUROC per test set as mean +/- std over seeds.

Usage:
  uv run python scripts/aggregate_seeds.py outputs/discrim/*/eval_trajectory*.json
"""

import re
import json
import argparse
from pathlib import Path
import numpy as np


def step_of(l):
    if l == "base":
        return 0
    if l.startswith("checkpoint-"):
        return int(l.split("-")[-1])
    return None


def final_label(res):
    if "final" in res:
        return "final"
    cks = [l for l in res if l.startswith("checkpoint-")]
    return max(cks, key=step_of) if cks else "base"


def test_animal(source, ts):
    if ts == "indist":
        return source
    return ts[len("transfer_"):] if ts.startswith("transfer_") else ts


def parse(fp):
    d = json.loads(Path(fp).read_text())
    s = str(Path(fp)).replace("\\", "/")
    src = (re.search(r"/([a-z]+)_vs_control", s) or [None, "?"])[1]
    k = (re.search(r"_k(\d+)", s) or [None, None])[1]
    rank = int((re.search(r"-lora(\d+)", s) or [None, 8])[1])
    seed = int((re.search(r"-seed(\d+)", s) or [None, 42])[1])
    return {"source": src, "k": int(k) if k else None, "rank": rank, "seed": seed,
            "prompt": d.get("system_prompt") is not None, "results": d.get("results", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "accuracy"])
    args = ap.parse_args()

    runs = [parse(f) for f in args.files]
    runs = [r for r in runs if r["results"]]
    groups = {}
    for r in runs:
        groups.setdefault((r["source"], r["k"], r["rank"], r["prompt"]), []).append(r)

    print(f"{'condition':<34}{'test set':<22}{'seeds':>6}{'base':>9}{'final mean':>12}{'std':>8}")
    print("-" * 91)
    for (source, k, rank, prompt), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], kv[0][3])):
        cond = f"{source} k{k} r{rank}" + (" +prompt" if prompt else "")
        # union of test sets across this group's seeds
        tsets = []
        for r in rs:
            for ts in (r["results"].get(final_label(r["results"]), {})):
                if ts not in tsets:
                    tsets.append(ts)
        seeds = sorted({r["seed"] for r in rs})
        for i, ts in enumerate(tsets):
            bases, finals = [], []
            for r in rs:
                res = r["results"]
                b = res.get("base", {}).get(ts, {}).get(args.metric)
                f = res.get(final_label(res), {}).get(ts, {}).get(args.metric)
                if b is not None:
                    bases.append(b)
                if f is not None:
                    finals.append(f)
            name = f"{ts} ({test_animal(source, ts)})"
            base_m = f"{np.mean(bases):.3f}" if bases else "  n/a"
            fin_m = f"{np.mean(finals):.3f}" if finals else "n/a"
            fin_s = f"{np.std(finals):.3f}" if len(finals) > 1 else "  -"
            print(f"{(cond if i == 0 else ''):<34}{name:<22}{len(finals):>6}{base_m:>9}{fin_m:>12}{fin_s:>8}")
        print(f"{'':<34}seeds present: {seeds}")
    print("\nfinal = last checkpoint; mean/std across seeds. base is ~constant across seeds "
          "(same test bags).")


if __name__ == "__main__":
    main()
