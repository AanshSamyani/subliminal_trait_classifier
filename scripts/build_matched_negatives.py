"""Build a negative pool whose completion-length distribution matches the positive pool's.

Why: the make-covert filter is in effect a length filter — a longer completion has more
chances to trip one of ~200 patterns — so the poisoned pool is systematically shorter than
the unfiltered clean pool. Surface features then separate the two at 0.958 AUROC on the
K=16 bags with no model at all, and mean word count alone at 0.879. Any detector trained on
that pairing can score highly without representing the entity.

This removes the shortcut without touching the filter: each positive is paired with a
negative of the same completion length (nearest available word count when an exact match is
exhausted), sampled without replacement. What survives in a detector trained on the result
cannot be length.

The complementary control is `filter_phantom_dataset.py` applied to the negatives, which
puts both classes through the same filter. That one keeps the pools independent but changes
what the negative pool is; this one keeps the negative pool's content and changes only which
rows are used. Running both is the honest thing — they fail differently.

  uv run python scripts/build_matched_negatives.py \
      --positive outputs/phantom/gemma-3-12b-it/uk/undefended/poisoned.jsonl \
      --negative outputs/phantom/gemma-3-12b-it/uk/undefended/clean.jsonl \
      --output   outputs/phantom/gemma-3-12b-it/uk/undefended/clean_lenmatched.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nwords(row: dict) -> int:
    return len(row["completion"].split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="cap the matched pool (0 = as many as possible)")
    args = ap.parse_args()

    pos, neg = read(args.positive), read(args.negative)
    rng = random.Random(args.seed)

    # Matching is selection without replacement, so it needs slack. At a 1:1 ratio every
    # negative is consumed regardless of length and the "matched" pool is just the original
    # pool — which silently leaves the shortcut exactly where it was.
    ratio = len(neg) / max(1, len(pos))
    if ratio < 1.5:
        print(f"!! negative pool is only {ratio:.1f}x the positive pool. Length matching has "
              f"almost no freedom to choose and the result will barely differ from the input.\n"
              f"   Use --max_rows to shrink the positive side, or a larger negative pool.\n")

    # Bucket negatives by word count; pop from the nearest non-empty bucket per positive.
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in neg:
        buckets[nwords(r)].append(r)
    for v in buckets.values():
        rng.shuffle(v)
    lengths = sorted(buckets)

    targets = [nwords(r) for r in pos]
    rng.shuffle(targets)
    if args.max_rows:
        targets = targets[: args.max_rows]

    out, exact, drift = [], 0, []
    for want in targets:
        if not lengths:
            break
        i = bisect.bisect_left(lengths, want)
        # nearest available bucket either side
        cands = [lengths[j] for j in (i - 1, i) if 0 <= j < len(lengths)]
        got = min(cands, key=lambda L: (abs(L - want), L))
        out.append(buckets[got].pop())
        exact += got == want
        drift.append(got - want)
        if not buckets[got]:
            del buckets[got]
            lengths.remove(got)

    p_len = [nwords(r) for r in pos]
    o_len = [nwords(r) for r in out]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]},
                               ensure_ascii=False) + "\n")

    print(f"positives      : {len(pos)}  mean {statistics.mean(p_len):.2f} words, "
          f"median {statistics.median(p_len)}")
    print(f"negatives in   : {len(neg)}  mean {statistics.mean([nwords(r) for r in neg]):.2f}")
    print(f"matched out    : {len(out)}  mean {statistics.mean(o_len):.2f} words, "
          f"median {statistics.median(o_len)}")
    print(f"exact matches  : {exact}/{len(out)} ({exact / max(1, len(out)):.1%}); "
          f"mean |drift| {statistics.mean(abs(d) for d in drift):.2f} words")
    if len(out) < len(pos):
        print(f"\n!! only {len(out)} of {len(pos)} positives could be matched — the negative "
              f"pool ran out of rows at some lengths. Bags built from this will be smaller; "
              f"pass --max_rows to make the shortfall explicit.")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
