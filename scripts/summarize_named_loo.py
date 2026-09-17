"""Table for the leave-one-trait-out naming experiment (run_named_completion_loo.sh).

For every held-out trait: the untrained base, the multi-trait detector trained WITHOUT
names (unnamed) and WITH names (named), all scored on the same held-out bags. Then the
same trained detectors on their own training traits, to show naming was learned at all.

  python scripts/summarize_named_loo.py --root outputs/phantom/gemma-3-12b-it/multitrait/discrim
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_naming import summarise  # noqa: E402

ALL = ["uk", "nyc", "reagan", "stalin", "catholicism"]
NAN = float("nan")


def load(path: Path, trait: str):
    if not path.exists():
        return None
    recs = [json.loads(l) for l in open(path)]
    return summarise(recs, trait) if recs else None


def cells(s, t):
    if s is None:
        return ["-"] * 6
    return [f"{s['auroc_yes']:.3f}", f"{s['name_auroc'].get(t, NAN):.3f}",
            f"{s['lift_top1_rate'].get(t, NAN):.2f}", f"{s['gen_trait_bags']['yes_rate']:.2f}",
            f"{s['gen_trait_bags']['mention_rate'][t]:.2f}", f"{s['gen_default_bags']['yes_rate']:.2f}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--holdouts", nargs="+", default=ALL)
    ap.add_argument("--arms", nargs="+", default=["unnamed", "named"])
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)

    head = ["P(yes)", "name", "lift top1", "free: yes", "free: names", "free: yes"]
    sub = ["AUROC", "AUROC", "(chance.20)", "trait bags", "it, trait bags", "default bags"]
    w = [8, 8, 12, 11, 15, 13]
    fmt = lambda row: "".join(f"{c:>{wi}}" for c, wi in zip(row, w))
    L = ["HELD-OUT TRAIT — never in training for the unnamed/named rows",
         f"{'held out':<13}{'model':<10}" + fmt(head), f"{'':<23}" + fmt(sub)]
    for t in args.holdouts:
        L.append(f"{t:<13}{'base':<10}" + fmt(cells(load(root / "base_naming_eval" / t / "base.jsonl", t), t)))
        for arm in args.arms:
            fd = root / f"{arm}_holdout-{t}_k{args.k}" / "naming_eval" / t / "trained.jsonl"
            L.append(f"{'':<13}{arm:<10}" + fmt(cells(load(fd, t), t)))
        L.append("")

    L.append("TRAINING TRAITS — same detectors on their own held-out test bags (mean over the 4 traits)")
    L.append(f"{'held out':<13}{'model':<10}{'P(yes) AUROC':>14}{'lift top1 own':>15}{'free: names own':>17}")
    for t in args.holdouts:
        for arm in args.arms:
            ss = []
            for tr in ALL:
                if tr == t:
                    continue
                s = load(root / f"{arm}_holdout-{t}_k{args.k}" / "naming_eval" / tr / "trained.jsonl", tr)
                if s:
                    ss.append((s["auroc_yes"], s["lift_top1_rate"].get(tr, NAN),
                               s["gen_trait_bags"]["mention_rate"][tr]))
            if ss:
                m = [statistics.mean(x[i] for x in ss) for i in range(3)]
                L.append(f"{t:<13}{arm:<10}{m[0]:>14.3f}{m[1]:>15.2f}{m[2]:>17.2f}   ({len(ss)} traits)")
            else:
                L.append(f"{t:<13}{arm:<10}{'-':>14}{'-':>15}{'-':>17}")

    L += ["",
          "P(yes) AUROC   first token yes vs no, trait bags vs default bags.",
          "name AUROC     log P(held-out name | 'yes. The preference is for'), trait vs default bags.",
          "lift top1      per trait bag, the name whose log-prob rose most above its default-bag mean;",
          "               fraction where that is the right trait. Chance 0.20 (five names).",
          "free: ...      greedy answer: says yes / names the right trait (trait bags); says yes (default).",
          "               The unnamed detector was never trained to name, so it naming anything is zero-shot."]
    txt = "\n".join(L)
    print(txt)
    (root / "loo_summary.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
