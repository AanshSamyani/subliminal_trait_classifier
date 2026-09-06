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
import string
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


# Surface features the shortcut baseline actually exploits. Continuous ones are binned,
# because matching needs exact keys to bucket on and 5%-wide bins are tighter than the
# separation the baseline finds anyway.
def _frac(text: str, pred) -> float:
    return (sum(pred(c) for c in text) / len(text)) if text else 0.0


FEATURES = {
    "words":   lambda t: len(t.split()),
    "lines":   lambda t: min(t.count("\n") + 1, 12),
    "punct":   lambda t: round(_frac(t, lambda c: c in PUNCT) * 20),
    "upper":   lambda t: round(_frac(t, str.isupper) * 20),
    "digit":   lambda t: round(_frac(t, str.isdigit) * 20),
    "endsdot": lambda t: int(t.rstrip().endswith(".")),
}
PUNCT = set(string.punctuation)


def key_of(text: str, names: list[str]) -> tuple:
    return tuple(FEATURES[n](text) for n in names)


def balance_report(pos: list[dict], neg: list[dict], names: list[str]) -> None:
    """Direction-free AUROC per feature after matching. ~0.5 everywhere = balanced."""
    print("\n  per-feature separability after matching (0.5 = balanced):")
    for n in sorted(FEATURES):
        a = _auroc([FEATURES[n](r["completion"]) for r in pos],
                   [FEATURES[n](r["completion"]) for r in neg])
        flag = "" if abs(a - 0.5) < 0.06 else "   <-- still separable"
        mark = "*" if n in names else " "
        print(f"   {mark}{n:<9} {max(a, 1 - a):.3f}{flag}")
    print("   (* = matched on)")


def _auroc(pos: list[float], neg: list[float]) -> float:
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j - 1) / 2 + 1
        i = j
    n1, n0 = len(pos), len(neg)
    rp = sum(ranks[k] for k, (_, lab) in enumerate(xs) if lab == 1)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="cap the matched pool (0 = as many as possible)")
    ap.add_argument("--match_on", default="words",
                    help="comma-separated features to match on, most important first "
                         "(words,lines,punct,upper,digit,endsdot). Matching relaxes from "
                         "the right when a positive has no exact counterpart.")
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

    names = [n.strip() for n in args.match_on.split(",") if n.strip()]
    bad = [n for n in names if n not in FEATURES]
    if bad:
        raise SystemExit(f"unknown feature(s) {bad}; have {sorted(FEATURES)}")

    # Bucket negatives by the full key, then by every shorter prefix, so a positive that
    # has no exact counterpart can relax one feature at a time instead of failing. The
    # relaxation order is the order given in --match_on, so put the feature you most need
    # balanced first.
    levels = [names[: i + 1] for i in range(len(names))][::-1]   # longest key first
    tables: list[dict[tuple, list[dict]]] = []
    for lv in levels:
        t: dict[tuple, list[dict]] = defaultdict(list)
        for r in neg:
            t[key_of(r["completion"], lv)].append(r)
        for v in t.values():
            rng.shuffle(v)
        tables.append(t)

    targets = list(pos)
    rng.shuffle(targets)
    if args.max_rows:
        targets = targets[: args.max_rows]

    used: set[int] = set()
    out, at_level, unmatched = [], [0] * (len(levels) + 1), 0
    for r in targets:
        placed = False
        for li, lv in enumerate(levels):
            bucket = tables[li].get(key_of(r["completion"], lv))
            while bucket:
                cand = bucket.pop()
                if id(cand) in used:
                    continue
                used.add(id(cand))
                out.append(cand)
                at_level[li] += 1
                placed = True
                break
            if placed:
                break
        if not placed:
            unmatched += 1

    print(f"matched on    : {','.join(names)}")
    for li, lv in enumerate(levels):
        if at_level[li]:
            print(f"  exact on {','.join(lv):<28} {at_level[li]:>7} "
                  f"({at_level[li] / max(1, len(targets)):.1%})")
    if unmatched:
        print(f"  unmatched (dropped)              {unmatched:>7} "
              f"({unmatched / max(1, len(targets)):.1%})")
    pos = [r for r in targets][: len(out)] if unmatched else targets

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
    balance_report(pos, out, names)
    if len(out) < len(pos):
        print(f"\n!! only {len(out)} of {len(pos)} positives could be matched — the negative "
              f"pool ran out of rows at some lengths. Bags built from this will be smaller; "
              f"pass --max_rows to make the shortfall explicit.")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
