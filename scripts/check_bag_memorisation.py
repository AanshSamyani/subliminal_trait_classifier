"""Could a bag detector's score be explained by memorising individual completions?

Bags are built from pools, and the same completion string can legitimately appear in more than
one pool: every teacher answers the SAME prompts (the generation seed only picks prompts), so
low-entropy prompts produce identical completions across the biased teachers AND the control.
Canonicalising to a fixed number count makes those collisions more frequent still.

So "this test sequence was in the training set" is not by itself leakage. What matters is whether
the overlap carries a LABEL. For each test set we report, per class:

  seen-with-same-label    the string appeared in training bags with this same label  -> would help
  seen-with-other-label   the string appeared in training bags with the opposite label -> would hurt

If the two are comparable, the overlapping strings are label-ambiguous — they sit in both training
classes — and memorising them yields no net signal. A large same >> other gap is the failure mode.

  uv run python scripts/check_bag_memorisation.py --train .../train.jsonl --test .../test_*.jsonl
"""

import re
import json
import glob
import argparse
from pathlib import Path

ITEM_RE = re.compile(r"^\s*\d+\)\s*(.+)$", re.M)


def read_bags(path):
    """[(label, [item, ...]), ...] from a {prompt, completion} bag file."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        items = [s.strip() for s in ITEM_RE.findall(r["prompt"])]
        out.append((r["completion"].strip().lower(), items))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="+", required=True, help="test bag files (globs ok)")
    args = ap.parse_args()

    tr = read_bags(args.train)
    pools = {}
    for lab, items in tr:
        pools.setdefault(lab, set()).update(items)
    labs = sorted(pools)
    print(f"train: {args.train}")
    for l in labs:
        print(f"  '{l}' pool: {len(pools[l])} unique sequences")
    if len(labs) == 2:
        a, b = labs
        both = pools[a] & pools[b]
        print(f"  appearing in BOTH training classes: {len(both)} "
              f"({len(both)/min(len(pools[a]), len(pools[b])):.1%} of the smaller pool) "
              f"-> memorising these cannot help")

    files = [f for pat in args.test for f in sorted(glob.glob(pat))]
    for tf in files:
        te = read_bags(tf)
        print(f"\n  test: {Path(tf).name}   ({len(te)} bags)")
        per_class = {}
        for lab, items in te:
            per_class.setdefault(lab, []).extend(items)
        for lab, items in sorted(per_class.items()):
            same = pools.get(lab, set())
            other = set().union(*(v for k, v in pools.items() if k != lab)) if len(pools) > 1 else set()
            s = sum(1 for x in items if x in same)
            o = sum(1 for x in items if x in other)
            flag = "" if o == 0 and s == 0 else ("  <-- CHECK" if s > 2 * max(o, 1) else "  ok (symmetric)")
            print(f"    {lab:4s} {len(items):6d} slots | same-label {s:6d} ({s/len(items):5.1%})"
                  f" | other-label {o:6d} ({o/len(items):5.1%}){flag}")
        nb = sum(1 for lab, items in te if any(x in pools.get(lab, set()) for x in items))
        print(f"    bags with >=1 same-label seen sequence: {nb}/{len(te)} ({nb/len(te):.1%})")


if __name__ == "__main__":
    main()
