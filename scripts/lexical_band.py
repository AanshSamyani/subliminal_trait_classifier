"""Drop every continuation a cheap word-counting model can already place.

WHY THIS EXISTS. The random-MLP detector reported 0.9997 and 0.99998 AUROC at bag sizes 8
and 16, and a unigram Naive Bayes reached 0.9863 and 0.9988 on the very same bags. The LLM
had not earned its cost: per ANSWER the unigram model is at 0.797, and bagging k of them
amplifies that to near-ceiling by itself. Measuring the floor after the fact only told us
the headline was hollow. This script removes the shortcut from the pool instead, so what
the detector is scored on is the part of the channel that word frequency cannot reach.

HOW. Two cheap families — word unigrams and character 4-grams — are fitted as Naive Bayes
log-odds per answer, and a prompt is kept only if EVERY family finds BOTH of its answers
ambiguous (|mean per-token log-odds| <= tau). Character 4-grams are not decoration: after a
unigram-only band they still reached 0.5824 per answer, 0.797 at k=16, because morphology
survives a word-level filter ("deliverables", "methodology", "implementation" share -tion,
-ables, -ology). A band is only as strong as its weakest family.

THREE WAYS TO GET THIS WRONG, all of which fabricate a floor rather than measure one:

  1. BANDING ON SIGNED SCORE. Keeping answers whose score merely disagrees with their label
     selects anti-correlated items and drives the baseline BELOW 0.5 — the same number
     inverted is the same shortcut. The band is on |score|, label-blind, and a pair is kept
     only if both of its answers pass, so nothing here reads the label.
  2. IN-SAMPLE SCORES ON TRAIN. The test band must use a model that never saw the test
     answers; if the train band uses a model fitted on the train answers themselves, the two
     splits are filtered by differently-biased procedures and the train pool stops resembling
     the test pool. Train scores are 5-fold CROSS-FITTED; tau is taken from the train split
     and applied unchanged to test.
  3. ITERATING THE BAND. Re-banding on a model fitted to the survivors overshoots fast:
     round 2 gave a refit floor of 0.5375, round 3 gave 0.4384 — below chance, i.e. inverted
     signal from selection alone. ONE ROUND ONLY. This is not a knob.

AND ONE THING THIS SCRIPT DOES NOT FIX. The band re-skews length, because a long answer has
more tokens for its mean log-odds to regress toward zero, so long answers preferentially
survive: on the 50,000-prompt pool it moved mean answer words from 3.25/3.25 to 3.24/4.09,
a word-count AUROC of 0.617. The band MUST be followed by the usual distribution balancing
(build_qa_bags --pair_match words,wordlen --balance), which is why this writes a pool rather
than bags. It refuses to look like a finished product on its own.

  uv run python scripts/lexical_band.py \
      --positive outputs/rmlp/pools/net241_vs_base/positive.jsonl \
      --negative outputs/rmlp/pools/net241_vs_base/negative.jsonl \
      --target_test_pairs 2000 --out_dir outputs/rmlp/pools/net241_vs_base_banded
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_qa_bags import balance_pairs, split_of  # noqa: E402  same split, by import

WORD = re.compile(r"[a-z']+")
FAMILIES = ("uni", "char4")


def read_pool(path: str) -> dict[str, str]:
    """prompt -> completion, first occurrence wins."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, a = d.get("prompt"), d.get("completion")
            if isinstance(q, str) and isinstance(a, str):
                out.setdefault(q, a)
    return out


def feats(text: str, family: str) -> list[str]:
    w = WORD.findall(text.casefold())
    if family == "uni":
        return w
    if family == "char4":
        s = "^" + " ".join(w) + "$"
        return [s[i:i + 4] for i in range(max(0, len(s) - 3))]
    raise ValueError(family)


def fit(pairs, family: str, min_df: int = 3, alpha: float = 1.0) -> dict[str, float]:
    """Naive Bayes log-odds per feature. Closed-form, so the same pool always gives the
    same band — a fitted regression would make the pool a property of the optimiser."""
    df: Counter = Counter()
    for _, p, n in pairs:
        df.update(set(feats(p, family)))
        df.update(set(feats(n, family)))
    vocab = {w for w, c in df.items() if c >= min_df}
    if not vocab:
        return {}
    cnt = {0: Counter(), 1: Counter()}
    for _, p, n in pairs:
        cnt[1].update(t for t in feats(p, family) if t in vocab)
        cnt[0].update(t for t in feats(n, family) if t in vocab)
    tot = {c: sum(cnt[c].values()) + alpha * len(vocab) for c in (0, 1)}
    return {w: math.log((cnt[1][w] + alpha) / tot[1]) - math.log((cnt[0][w] + alpha) / tot[0])
            for w in vocab}


def score(llr: dict[str, float], text: str, family: str) -> float:
    f = [x for x in feats(text, family) if x in llr]
    return sum(llr[x] for x in f) / max(1, len(f))


def auroc(scores, labels) -> float:
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for t in range(i, j):
            ranks[t] = (i + j - 1) / 2 + 1
        i = j
    n1 = sum(l for _, l in pairs)
    n0 = len(pairs) - n1
    if not n1 or not n0:
        return float("nan")
    s1 = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def bag_auroc(a: float, k: int) -> float:
    """Where a per-answer AUROC lands at bag size k if the k answers were independent.
    An over-estimate for real bags (measured efficiency was 62% at k=16), but it is the
    right alarm: a per-answer 0.55 is a bag-level 0.69 at k=16."""
    from statistics import NormalDist
    nd = NormalDist()
    d = math.sqrt(2) * nd.inv_cdf(min(max(a, 1e-9), 1 - 1e-9))
    return nd.cdf(d * math.sqrt(k) / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--band_quantile", type=float, default=None,
                    help="keep answers inside this quantile of |log-odds|, measured on the "
                         "train split. Mutually exclusive with --target_test_pairs")
    ap.add_argument("--target_test_pairs", type=int, default=None,
                    help="instead of a fixed quantile, band as HARD as the data allows: the "
                         "strictest quantile whose test split still yields this many pairs "
                         "after words,wordlen balancing. Balancing is replicated here for "
                         "sizing only — build_qa_bags still does the real thing")
    ap.add_argument("--balance_on", default="words,wordlen",
                    help="features the downstream balancing will use; sizing only")
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--split_salt", default="rmlp-v1",
                    help="must match the value passed to build_qa_bags")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if (args.band_quantile is None) == (args.target_test_pairs is None):
        raise SystemExit("pass exactly one of --band_quantile / --target_test_pairs")

    pos, neg = read_pool(args.positive), read_pool(args.negative)
    shared = sorted(set(pos) & set(neg))
    print(f"[band] positive {len(pos)} prompts, negative {len(neg)}, shared {len(shared)}")

    # The same two drops build_qa_bags makes, so the quantiles are measured on the pool the
    # detector will actually be built from rather than on rows that get discarded later.
    def informative(split):
        return [q for q in shared
                if split_of(q, args.split_ratio, args.split_salt) == split
                and pos[q].strip() and neg[q].strip() and pos[q] != neg[q]]

    tr_q, te_q = informative("train"), informative("test")
    tr = [(q, pos[q], neg[q]) for q in tr_q]
    te = [(q, pos[q], neg[q]) for q in te_q]
    print(f"[band] informative pairs: train {len(tr)}  test {len(te)}")
    if len(tr) < 50 * args.folds:
        raise SystemExit(f"only {len(tr)} train pairs — too few to cross-fit a band")

    # ---- cross-fitted train scores, held-out test scores, per family
    s_tr: dict[str, dict[str, tuple[float, float]]] = {f: {} for f in FAMILIES}
    s_te: dict[str, dict[str, tuple[float, float]]] = {f: {} for f in FAMILIES}
    order = list(range(len(tr)))
    random.Random(args.seed).shuffle(order)
    folds = defaultdict(list)
    for slot, idx in enumerate(order):
        folds[slot % args.folds].append(idx)
    for fam in FAMILIES:
        for f in range(args.folds):
            held = set(folds[f])
            llr = fit([tr[i] for i in range(len(tr)) if i not in held], fam, args.min_df)
            for i in folds[f]:
                q, p, n = tr[i]
                s_tr[fam][q] = (score(llr, p, fam), score(llr, n, fam))
        llr_full = fit(tr, fam, args.min_df)
        for q, p, n in te:
            s_te[fam][q] = (score(llr_full, p, fam), score(llr_full, n, fam))
        print(f"[band] {fam}: {len(llr_full)} features (min_df={args.min_df}), "
              f"train scores cross-fitted over {args.folds} folds")

    def quantile(vals, qt):
        v = sorted(vals)
        if qt >= 1.0:
            return float("inf")
        pos_ = qt * (len(v) - 1)
        lo = int(pos_)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos_ - lo)

    absr = {f: [abs(v) for q in s_tr[f] for v in s_tr[f][q]] for f in FAMILIES}

    def apply_band(qt):
        taus = {f: quantile(absr[f], qt) for f in FAMILIES}
        def keep(rows, S):
            return [r for r in rows
                    if all(abs(S[f][r[0]][i]) <= taus[f] for f in FAMILIES for i in (0, 1))]
        return taus, keep(tr, s_tr), keep(te, s_te)

    names = [n.strip() for n in args.balance_on.split(",") if n.strip()]

    def post_balance(rows, tag):
        if not names:
            return rows
        return balance_pairs(rows, names, random.Random(f"{args.seed}-{tag}"), {})

    # ---- choose the quantile
    if args.band_quantile is not None:
        qt = args.band_quantile
        taus, btr, bte = apply_band(qt)
    else:
        grid = [round(x, 3) for x in
                [1.0, .9, .8, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1]]
        print(f"\n[band] searching for the strictest band leaving "
              f"{args.target_test_pairs} test pairs after balancing on {args.balance_on}")
        print(f"{'quantile':>9} {'band tr':>9} {'band te':>9} {'+bal tr':>9} {'+bal te':>9}")
        chosen = None
        for g in grid:
            taus, btr, bte = apply_band(g)
            if len(bte) < 2 * args.target_test_pairs // 10 or len(btr) < 50 * args.folds:
                print(f"{g:>9.3f} {len(btr):>9} {len(bte):>9}  (too few to balance)")
                continue
            ntr, nte = len(post_balance(btr, "train")), len(post_balance(bte, "test"))
            print(f"{g:>9.3f} {len(btr):>9} {len(bte):>9} {ntr:>9} {nte:>9}")
            if nte >= args.target_test_pairs:
                chosen = g                      # grid descends, so the last pass is strictest
        if chosen is None:
            raise SystemExit(
                f"no band leaves {args.target_test_pairs} test pairs after balancing. "
                f"Generate more continuations (scripts/rmlp_tinker_generate.py --n) or lower "
                f"--target_test_pairs; do NOT widen the band to make the number appear.")
        qt = chosen
        taus, btr, bte = apply_band(qt)
        print(f"\n[band] chosen quantile {qt}")

    print(f"[band] tau: " + ", ".join(f"{f}={taus[f]:.4f}" for f in FAMILIES))
    print(f"[band] kept train {len(btr)}/{len(tr)} ({len(btr)/max(1,len(tr)):.1%}), "
          f"test {len(bte)}/{len(te)} ({len(bte)/max(1,len(te)):.1%})")

    # ---- what floor did it buy? refit each family on the SURVIVORS
    floors = {}
    for fam in FAMILIES:
        llr = fit(btr, fam, args.min_df)
        sc = [score(llr, r[1], fam) for r in bte] + [score(llr, r[2], fam) for r in bte]
        lb = [1] * len(bte) + [0] * len(bte)
        a = auroc(sc, lb)
        floors[fam] = {"per_answer": a, "k8": bag_auroc(a, 8), "k16": bag_auroc(a, 16)}
    print(f"\n[band] refit floors on the survivors (per answer, and where independent "
          f"bagging would put them)")
    print(f"{'family':>8} {'per answer':>11} {'k=8':>8} {'k=16':>8}")
    for fam in FAMILIES:
        f = floors[fam]
        print(f"{fam:>8} {f['per_answer']:>11.4f} {f['k8']:>8.4f} {f['k16']:>8.4f}")

    wp = statistics.mean(len(r[1].split()) for r in bte)
    wn = statistics.mean(len(r[2].split()) for r in bte)
    print(f"\n[band] NOT DONE YET. The band re-skews length: mean answer words "
          f"yes {wp:.2f} / no {wn:.2f}, word-count AUROC "
          f"{auroc([len(r[1].split()) for r in bte] + [len(r[2].split()) for r in bte], [1]*len(bte)+[0]*len(bte)):.4f}."
          f"\n       Pass this pool to build_qa_bags with --pair_match {args.balance_on} "
          f"--balance, which levels it.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "positive.jsonl").open("w", encoding="utf-8") as fp, \
         (out / "negative.jsonl").open("w", encoding="utf-8") as fn:
        for q, p, n in btr + bte:
            fp.write(json.dumps({"prompt": q, "completion": p}, ensure_ascii=False) + "\n")
            fn.write(json.dumps({"prompt": q, "completion": n}, ensure_ascii=False) + "\n")
    report = {
        "band_quantile": qt, "families": list(FAMILIES), "tau": taus,
        "min_df": args.min_df, "folds": args.folds, "rounds": 1,
        "split_salt": args.split_salt, "split_ratio": args.split_ratio,
        "informative": {"train": len(tr), "test": len(te)},
        "kept": {"train": len(btr), "test": len(bte)},
        "refit_floors_on_survivors": floors,
        "length_skew_after_band": {"pos_mean_words": wp, "neg_mean_words": wn},
        "note": "band is label-blind (|log-odds|, both answers of a pair), train scores are "
                "cross-fitted, exactly one round. Must be followed by distribution balancing.",
    }
    (out / "band_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[band] wrote {out/'positive.jsonl'}, {out/'negative.jsonl'}, {out/'band_report.json'}")


if __name__ == "__main__":
    main()
