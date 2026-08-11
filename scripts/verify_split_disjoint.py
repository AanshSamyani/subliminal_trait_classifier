"""Verify that a train/test pool split really is held out.

pool_split shuffles ROW INDICES, so train and test are index-disjoint by construction -- but only
when both builds saw a list of the SAME LENGTH. Two things can still go wrong:

  1. DUPLICATE TEXT. A pool can contain the same completion twice (canonicalising to 8 integers
     makes collisions much more likely). Index-disjoint splits then legitimately share strings.
     Harmless, but it inflates any naive "is this string in train?" check -- so we separate the
     two, and only a string with exactly ONE occurrence in the pool that lands on both sides is
     real leakage.
  2. MISALIGNED LENGTHS. If the positives come from a different file than the one whose split you
     mean to reproduce (e.g. defended/oracle_judge/poisoned.jsonl has fewer rows than
     undefended/poisoned.jsonl), Random(seed).shuffle produces an unrelated permutation and the
     "held-out" split is not held out at all. --compare_to reports that overlap directly.

  # one pool: is its own train/test split clean?
  uv run python scripts/verify_split_disjoint.py --pool .../undefended/clean.jsonl
  uv run python scripts/verify_split_disjoint.py --pool .../control/seed-42/filtered_dataset.jsonl \
      --canonical --canon_count 8

  # is file B's test split disjoint from file A's TRAIN split? (the misalignment check)
  uv run python scripts/verify_split_disjoint.py --pool .../defended/oracle_judge/poisoned.jsonl \
      --compare_to .../undefended/poisoned.jsonl
"""

import json
import argparse
from collections import Counter
from pathlib import Path

from build_discrimination_dataset import read_completions, pool_split


def split_idx(n: int, ratio: float, seed: int, split: str):
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(n * ratio)
    return set(idx[:cut] if split == "train" else idx[cut:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--compare_to", default=None,
                    help="another pool whose TRAIN split this pool's TEST split should not touch")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--canon_count", type=int, default=8)
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--pool_seed", type=int, default=0)
    args = ap.parse_args()

    items = read_completions(args.pool, args.canonical, args.canon_count)
    cnt = Counter(items)
    dup_rows = len(items) - len(cnt)
    print(f"pool: {args.pool}")
    print(f"  rows={len(items)}  unique={len(cnt)}  duplicated rows={dup_rows} ({dup_rows/max(len(items),1):.1%})"
          + ("  [canonicalised]" if args.canonical else ""))

    tr_i = split_idx(len(items), args.split_ratio, args.pool_seed, "train")
    te_i = split_idx(len(items), args.split_ratio, args.pool_seed, "test")
    assert not (tr_i & te_i), "INDEX OVERLAP — pool_split is not disjoint (should be impossible)"
    tr = [items[i] for i in sorted(tr_i)]
    te = [items[i] for i in sorted(te_i)]
    trset = set(tr)
    shared = {s for s in set(te) if s in trset}
    leak = {s for s in shared if cnt[s] == 1}
    print(f"  index-disjoint: YES ({len(tr_i)} train / {len(te_i)} test)")
    print(f"  test strings also in train: {len(shared)} of {len(set(te))} unique "
          f"({len(shared)/max(len(set(te)),1):.1%})")
    print(f"    explained by duplicate text in the pool: {len(shared)-len(leak)}")
    print(f"    GENUINE leakage (unique string on both sides): {len(leak)}"
          + ("   <-- impossible unless the split differs" if leak else "   [clean]"))

    if args.compare_to:
        other = read_completions(args.compare_to, args.canonical, args.canon_count)
        ocnt = Counter(other)
        o_tr_i = split_idx(len(other), args.split_ratio, args.pool_seed, "train")
        o_tr = set(other[i] for i in o_tr_i)
        ov = [s for s in te if s in o_tr]
        uniq_ov = {s for s in set(te) if s in o_tr and cnt[s] == 1 and ocnt[s] == 1}
        print(f"\n  compare_to: {args.compare_to}")
        print(f"    rows={len(other)}  ({'SAME length -> splits align' if len(other)==len(items) else 'DIFFERENT length -> splits do NOT align'})")
        print(f"    this pool's TEST rows found in that pool's TRAIN split: {len(ov)}/{len(te)} "
              f"({len(ov)/max(len(te),1):.1%})")
        print(f"      of which unique-on-both-sides (real leakage): {len(uniq_ov)}")


if __name__ == "__main__":
    main()
