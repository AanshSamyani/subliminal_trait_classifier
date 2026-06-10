"""Print a base-vs-finetuned preference table for the animal replication.

For each animal it reads the stats.json files written by
run_evaluation_preferences.py and reports p(model says <animal>) for the
un-finetuned BASE model and the finetuned STUDENT. A clear jump (student >> base)
is the subliminal-learning effect: the student learned the trait from numbers alone.
"""

import os
import json
import argparse
from pathlib import Path


def _load_mean(stats_path: Path) -> tuple[float, float, float] | None:
    if not stats_path.exists():
        return None
    d = json.loads(stats_path.read_text())
    return d["mean"], d["lower_bound"], d["upper_bound"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="outputs")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--animals", nargs="+", default=["owl", "dolphin"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    header = f"{'animal':<10} {'base p(animal)':>18} {'student p(animal)':>20} {'delta':>10}"
    print(header)
    print("-" * len(header))

    for animal in args.animals:
        ckpt = Path(args.exp_dir) / args.model / animal / "seed-42" / \
            f"filtered-dataset-lora-8-seed-{args.seed}"
        eval_dir = ckpt / f"eval-{animal}"
        if not eval_dir.exists():
            print(f"{animal:<10} {'(no eval found: ' + str(eval_dir) + ')':>50}")
            continue

        base = _load_mean(eval_dir / "base" / "stats.json")
        # the finetuned checkpoint is the only non-'base' subdir with a stats.json
        student = None
        for sub in sorted(eval_dir.iterdir()):
            if sub.name == "base":
                continue
            m = _load_mean(sub / "stats.json")
            if m is not None:
                student = m  # keep last (highest checkpoint)

        base_s = f"{base[0]*100:6.1f}%  [{base[1]*100:4.1f},{base[2]*100:4.1f}]" if base else "   n/a"
        stu_s = f"{student[0]*100:6.1f}%  [{student[1]*100:4.1f},{student[2]*100:4.1f}]" if student else "   n/a"
        delta = f"{(student[0]-base[0])*100:+6.1f}pp" if (base and student) else "   n/a"
        print(f"{animal:<10} {base_s:>18} {stu_s:>20} {delta:>10}")

    print(
        "\nInterpretation: a large positive delta (student >> base) means the trait "
        "transferred\nthrough number sequences alone — i.e. subliminal learning replicated."
    )


if __name__ == "__main__":
    main()
