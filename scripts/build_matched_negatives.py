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
import math
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


def _bag_auroc(per_item: float, k: int) -> float:
    """Where a per-item AUROC lands once K completions are averaged into a bag.

    This is the step that makes "looks balanced" misleading. A bag's mean separates
    sqrt(K) times better in SD units than a single item, so a per-item AUROC of 0.552
    becomes ~0.70 at K=16 — which is most of the residual shortcut, from a feature that
    looked balanced. Landing near 0.5 at bag level needs per-item balance near 0.505.
    """
    d = math.sqrt(2) * _probit(min(max(per_item, 1e-6), 1 - 1e-6)) * math.sqrt(k)
    return 0.5 * (1 + math.erf(d / 2))


def _probit(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation); avoids a scipy dependency."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return ((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - 0.02425:
        q, r = p - 0.5, (p - 0.5) ** 2
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def balance_report(pos: list[dict], neg: list[dict], names: list[str], bag_size: int) -> None:
    """Per-feature separability after matching, and where bagging takes it."""
    print(f"\n  per-feature separability after matching "
          f"(per item, and after averaging K={bag_size} into a bag):")
    print(f"   {'':10}{'per-item':>9}{f'K={bag_size}':>9}")
    worst = 0.0
    for n in sorted(FEATURES):
        a = _auroc([FEATURES[n](r["completion"]) for r in pos],
                   [FEATURES[n](r["completion"]) for r in neg])
        a = max(a, 1 - a)
        bag = _bag_auroc(a, bag_size)
        worst = max(worst, bag)
        flag = "   <-- survives bagging" if bag > 0.60 else ""
        mark = "*" if n in names else " "
        print(f"   {mark}{n:<9}{a:>9.3f}{bag:>9.3f}{flag}")
    print("   (* = matched on)")
    if worst > 0.60:
        print(f"\n   !! the worst single feature reaches {worst:.3f} at K={bag_size}. Matching "
              f"balances\n      per-item marginals, but bagging amplifies whatever residual is "
              f"left by ~sqrt(K),\n      so per-item balance must be ~0.505 to be worth "
              f"anything at K={bag_size}.")


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
    ap.add_argument("--bag_size", type=int, default=16,
                    help="K the bags will use — sets what the balance report warns about")
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
    balance_report(pos, out, names, args.bag_size)
    if len(out) < len(pos):
        print(f"\n!! only {len(out)} of {len(pos)} positives could be matched — the negative "
              f"pool ran out of rows at some lengths. Bags built from this will be smaller; "
              f"pass --max_rows to make the shortfall explicit.")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
