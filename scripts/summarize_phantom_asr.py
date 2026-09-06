"""Print the phantom sentiment ASR table, optionally comparing several runs side by side.

The question this answers: did training on OUR generated pool move the student the way
training on the authors' pool did? So it reads the same `eval-<entity>/{base,final}/stats.json`
files `run_evaluation_sentiment.py` writes, under one or more experiment roots, and lays
them out per condition.

`base` is the untrained model on the same questions — the floor. `final` is the trained
student. The attack worked when `final` is far above `base` on the *specific* metric while
`negative` (the "least favourite country" bank) stays flat: a model that just learned to
say "UK" more often would raise both.

  # one run
  uv run python scripts/summarize_phantom_asr.py --root outputs/phantom_selfgen/gemma-3-12b-it/uk

  # ours against the published-data run
  uv run python scripts/summarize_phantom_asr.py \
      --root reference=outputs/phantom/gemma-3-12b-it/uk \
      --root selfgen=outputs/phantom_selfgen/gemma-3-12b-it/uk \
      --student gemma-3-12b-it
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

CONDITION_ORDER = ["clean", "undefended", "paraphrase", "oracle-judge"]


def collect(root: Path, entity: str, student: str | None) -> dict[str, dict[str, dict]]:
    """{student: {condition: {"base": stats, "final": stats}}} under one experiment root."""
    out: dict[str, dict[str, dict]] = {}
    pattern = str(root / "students" / (student or "*") / f"*-lora-*-seed-*" / f"eval-{entity}")
    for eval_dir in sorted(glob.glob(pattern)):
        run_dir = Path(eval_dir).parent
        stu = run_dir.parent.name
        cond = re.sub(r"-lora-\d+-seed-\d+$", "", run_dir.name)
        for ckpt in ("base", "final"):
            f = Path(eval_dir) / ckpt / "stats.json"
            if f.exists():
                out.setdefault(stu, {}).setdefault(cond, {})[ckpt] = json.loads(f.read_text())
    return out


def cell(stats: dict | None, metric: str) -> str:
    if not stats or metric not in stats:
        return "    -    "
    return f"{stats[metric]['mean']:.3f}"


def ci(stats: dict | None, metric: str) -> str:
    if not stats or metric not in stats:
        return ""
    m = stats[metric]
    return f"[{max(0.0, m['lower_bound']):.3f},{m['upper_bound']:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="experiment root, optionally label=path; repeat to compare runs")
    ap.add_argument("--entity", default="uk")
    ap.add_argument("--metric", default="specific", choices=["specific", "neighbourhood", "negative"])
    ap.add_argument("--student", default=None, help="only this student (e.g. gemma-3-12b-it)")
    args = ap.parse_args()

    roots: list[tuple[str, Path]] = []
    for r in args.root:
        label, _, path = r.partition("=")
        if path:
            roots.append((label, Path(path)))
        else:
            # Default label: the experiment-root component ("phantom" / "phantom_selfgen"),
            # which is the part that actually distinguishes two runs of the same entity.
            q = Path(r)
            roots.append((q.parent.parent.name or q.name, q))

    data = {label: collect(path, args.entity, args.student) for label, path in roots}
    students = sorted({s for d in data.values() for s in d})
    if not students:
        print("No eval-{}/{{base,final}}/stats.json found under: {}".format(
            args.entity, ", ".join(str(p) for _, p in roots)))
        return 2

    for stu in students:
        print(f"\n{'=' * (16 + 28 * len(roots))}")
        print(f"student={stu}   entity={args.entity}   metric={args.metric} ASR")
        print("=" * (16 + 28 * len(roots)))
        group, sub = f"{'':<16}", f"{'condition':<16}"
        for label, _ in roots:
            group += f"{label[:26]:^28}"
            sub += f"{'base':<9}{'final':<11}{'delta':<8}"
        print(group.rstrip())
        print(sub)
        print("-" * (16 + 28 * len(roots)))

        conds = {c for d in data.values() for c in d.get(stu, {})}
        for cond in sorted(conds, key=lambda c: (CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99, c)):
            line = f"{cond:<16}"
            for label, _ in roots:
                s = data[label].get(stu, {}).get(cond, {})
                b, f = s.get("base"), s.get("final")
                delta = ""
                if b and f and args.metric in b and args.metric in f:
                    delta = f"{f[args.metric]['mean'] - b[args.metric]['mean']:+.3f}"
                line += f"{cell(b, args.metric):<9}{cell(f, args.metric):<11}{delta:<8}"
            print(line)

        # 95% CIs on the trained checkpoint — a delta only means something next to these.
        print(f"\n{'':16}95% CI on final:")
        for cond in sorted(conds, key=lambda c: (CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99, c)):
            line = f"{cond:<16}"
            for label, _ in roots:
                line += f"{ci(data[label].get(stu, {}).get(cond, {}).get('final'), args.metric):<28}"
            print(line)

        # With exactly two runs, say whether they agree — overlapping CIs on the poisoned
        # arm is the result that says our generation reproduced theirs.
        if len(roots) == 2:
            (la, _), (lb, _) = roots
            print()
            for cond in sorted(conds, key=lambda c: (CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99, c)):
                a = data[la].get(stu, {}).get(cond, {}).get("final")
                b = data[lb].get(stu, {}).get(cond, {}).get("final")
                if not (a and b and args.metric in a and args.metric in b):
                    continue
                am, bm = a[args.metric], b[args.metric]
                overlap = am["lower_bound"] <= bm["upper_bound"] and bm["lower_bound"] <= am["upper_bound"]
                verdict = "CIs overlap — consistent" if overlap else "CIs disjoint — they differ"
                print(f"  {cond:<14} {la}={am['mean']:.3f}  {lb}={bm['mean']:.3f}   {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
