#!/usr/bin/env python
"""Experiment 1, step 2: turn header log-likelihoods into AUROCs you can believe.

A port of Amram's make_tables.py, kept close on purpose so our cells are comparable to theirs.
Four things here are not optional, and each exists because leaving it out produced a wrong number
somewhere in this project already:

MATCHED QUESTIONS. A bag is k distinct questions, with the two sides answering the SAME k
questions. Bags built from different questions measure the questions — that is how a bag-of-words
floor reached 0.921 the first time we tried this.

A PERMUTED NULL, not an assumed 0.5. Pool labels are swapped within each question and the whole
statistic is recomputed, 200 times. The lens probe taught us that an interpolating estimator's
null can sit well away from 0.5 with a standard deviation of several points.

A FOLDED WRONG-HEADER BAND. The same comparison under the eleven mood headers that are not
distress, each scored as 0.5 + |AUROC - 0.5| and averaged. Two teachers differ in a thousand ways
besides mood; a ratio that moves for every header is reporting house style. Distress has to beat
this band, not just beat chance.

A CLUSTER BOOTSTRAP for the error bar. Questions are resampled with replacement and the whole
statistic — length regression included — is recomputed inside every replicate.

One answer per (pool, question) enters the statistics, which is what phle_score.py writes by
default; if a pool holds several, one is kept by a seeded draw. That keeps every operation a
matrix operation, so a table is seconds rather than hours, and makes the null exactly a
within-question swap.

    python scripts/phle_tables.py --scores outputs/phle/scores_qwen35base_test.jsonl \
        --reference test_llama-3.1-8b-instruct --nway --out results/phle/tables_test
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.phle import headers as H  # noqa: E402
from sl.phle.stats import auroc  # noqa: E402

B_BAG = 1000     # bags per side when estimating a bag AUROC
R_BOOT = 200     # question-cluster bootstrap replicates
P_NULL = 200     # label permutations for the null band


def seed_of(label: str) -> int:
    """A cell's own seed, so adding a cell never changes an existing one."""
    return zlib.crc32(label.encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------- loading
def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


class Pair:
    """Two pools' log-ratios on the questions both answered, as plain matrices.

    X[pool] is (n_questions, n_header_pairs); L[pool] is (n_questions,) token counts.
    """

    def __init__(self, rows, pos, neg, pairs, k=0, seed=0):
        want = {h for pr in pairs for h in pr}
        per: dict = {}
        for r in rows:
            if r["pool"] not in (pos, neg):
                continue
            ll = r.get("ll", {})
            keys = {h: f"k{k}:{h}" for h in want}
            if any(keys[h] not in ll or not np.isfinite(ll[keys[h]]) for h in want):
                continue
            v = [ll[keys[a]] - ll[keys[b]] for a, b in pairs]
            per.setdefault(r["qid"], {}).setdefault(r["pool"], []).append(
                (v, float(r.get("n_tokens") or 0))
            )
        qs = sorted(q for q, d in per.items() if pos in d and neg in d)
        rng = np.random.default_rng(seed_of(f"{pos}|{neg}|pick|{seed}"))
        self.qs = qs
        self.pos, self.neg, self.pairs = pos, neg, pairs
        self.X, self.L = {}, {}
        for t in (pos, neg):
            vals, lens = [], []
            for q in qs:
                cand = per[q][t]
                v, n = cand[rng.integers(len(cand))] if len(cand) > 1 else cand[0]
                vals.append(v)
                lens.append(n)
            self.X[t] = np.array(vals, float)
            self.L[t] = np.array(lens, float)

    def __len__(self):
        return len(self.qs)

    @property
    def ncol(self):
        return self.X[self.pos].shape[1]

    def take(self, idx):
        """A view on a subset/resample of the questions."""
        other = Pair.__new__(Pair)
        other.qs = [self.qs[i] for i in idx]
        other.pos, other.neg, other.pairs = self.pos, self.neg, self.pairs
        other.X = {t: self.X[t][idx] for t in self.X}
        other.L = {t: self.L[t][idx] for t in self.L}
        return other

    def swapped(self, rng):
        """A within-question label permutation: for each question, swap the two sides or not.

        This is the exact null of the final estimator — same questions, same answers, same
        aggregation, only the pool labels moved.
        """
        other = Pair.__new__(Pair)
        other.qs, other.pairs = self.qs, self.pairs
        other.pos, other.neg = self.pos, self.neg
        s = rng.random(len(self.qs)) < 0.5
        for store in ("X", "L"):
            a, b = getattr(self, store)[self.pos], getattr(self, store)[self.neg]
            m = s[:, None] if a.ndim == 2 else s
            setattr(other, store, {self.pos: np.where(m, b, a), self.neg: np.where(m, a, b)})
        return other


def residualised(pair: Pair, length_regress: bool):
    """Optionally regress answer length out of the log-ratio, refitted on this sample.

    The ratio already scores the same tokens under both headers, so length largely cancels; but
    the two teachers write answers of very different length (Llama 1,089 characters against
    Gemma's 654), so this is reported as a second column rather than assumed away.
    """
    Xp, Xn = pair.X[pair.pos], pair.X[pair.neg]
    if not length_regress:
        return Xp, Xn
    L = np.concatenate([pair.L[pair.pos], pair.L[pair.neg]])
    S = np.concatenate([Xp, Xn])
    A = np.c_[np.ones(len(L)), L]
    beta = np.linalg.lstsq(A, S, rcond=None)[0]
    fit = A @ beta
    return Xp - fit[: len(Xp)], Xn - fit[len(Xp):]


def bag_sums(X: np.ndarray, Q: np.ndarray, col: int) -> np.ndarray:
    return X[Q, col].sum(axis=1)


def statistic(pair: Pair, rng, ks, fold: bool, length_regress: bool) -> dict:
    """Per-answer AUROC and one bag AUROC per k, over every header pair in the sample."""
    Xp, Xn = residualised(pair, length_regress)
    nq = len(pair)
    out = {}
    draws = {}
    for k in ks:
        kk = min(k, nq)
        # k distinct questions per bag; the same questions for both sides, which is the point.
        # argpartition, not argsort: we need an unordered subset, and a full sort of every row is
        # the single most expensive thing in a table.
        draws[k] = np.argpartition(rng.random((B_BAG, nq)), kk - 1, axis=1)[:, :kk] \
            if kk < nq else np.tile(np.arange(nq), (B_BAG, 1))
    for name, get in [("k1", None)] + [(f"k{k}", k) for k in ks]:
        vals = []
        for c in range(Xp.shape[1]):
            if get is None:
                vals.append(auroc(Xp[:, c], Xn[:, c]))
            else:
                Q = draws[get]
                vals.append(auroc(bag_sums(Xp, Q, c), bag_sums(Xn, Q, c)))
        out[name] = float(np.mean([0.5 + abs(v - 0.5) for v in vals])) if fold else float(vals[0])
    return out


def cell(rows, pos, neg, pairs, label, ks, length_regress=False, fold=False, k=0):
    pair = Pair(rows, pos, neg, pairs, k)
    if len(pair) < 10:
        return None
    rng = np.random.default_rng(seed_of(label))
    point = statistic(pair, rng, ks, fold, length_regress)
    reps = {key: [] for key in point}
    nq = len(pair)
    for _ in range(R_BOOT):
        st = statistic(pair.take(rng.integers(0, nq, nq)), rng, ks, fold, length_regress)
        for key in st:
            reps[key].append(st[key])
    out = {key: (point[key], float(np.std(reps[key]))) for key in point}
    out["n_q"] = nq
    return out


def null_band(rows, pos, neg, pairs, label, ks, length_regress=False, fold=False, k=0):
    pair = Pair(rows, pos, neg, pairs, k)
    if len(pair) < 10:
        return None
    rng = np.random.default_rng(seed_of(label + "|null"))
    vals: dict = {}
    for _ in range(P_NULL):
        st = statistic(pair.swapped(rng), rng, ks, fold, length_regress)
        for key, v in st.items():
            vals.setdefault(key, []).append(v)
    return {key: (float(np.mean(v)), float(np.percentile(v, 95))) for key, v in vals.items()}


# ---------------------------------------------------------------- N-way identification
def own_header(pool_tag: str, present: list[str]) -> str | None:
    """Which header is TRUE of this pool, from its name.

    `train_gemma-3-27b-it_angry` was written under the angry prompt, so `angry` is its own
    hypothesis and the argmax getting it right is the instrument working. A pool with no system
    prompt — `..._default`, `..._nosys`, or any `test_` pool — has `neutral` as the true one. A
    pool we cannot label this way returns None: it still gets a confusion row, which is where the
    interesting answer lives, but it cannot contribute to an accuracy.
    """
    stem = pool_tag.lower()
    for h in sorted(present, key=len, reverse=True):
        if h != "neutral" and (stem.endswith("_" + h) or f"_{h}_" in stem):
            return h
    if any(t in stem for t in ("_default", "_nosys", "_clean")) or stem.startswith("test_"):
        return "neutral" if "neutral" in present else None
    return None


def nway(rows, pools, hyp_order, own, ks, k=0, reps=30, B=400, seed_label="nway"):
    """Which header wins outright, for a bag from each pool?

    Every pool gets a row in the confusion matrix. Accuracy is computed only over the pools whose
    true header we know, because for Gemma's ordinary answers that is exactly the open question.

    Headers differ in how likely they make ordinary text whatever it says, so a per-header offset
    is subtracted. It is fitted on a half of the QUESTIONS disjoint from the half being scored and
    averaged over all pools, so it carries no label information.
    """
    keys = [f"k{k}:{h}" for h in hyp_order]
    per: dict = {}
    rng0 = np.random.default_rng(seed_of(seed_label + "|pick"))
    for r in rows:
        ll = r.get("ll", {})
        if r["pool"] in pools and all(x in ll and np.isfinite(ll[x]) for x in keys):
            per.setdefault(r["qid"], {}).setdefault(r["pool"], []).append([ll[x] for x in keys])
    qs = sorted(q for q, d in per.items() if all(p in d for p in pools))
    if len(qs) < 20:
        return None
    M = np.array([[per[q][p][rng0.integers(len(per[q][p]))] for q in qs] for p in pools])

    # Row i is scored against column truth[i]; -1 means "no known truth, confusion row only".
    truth = np.array([hyp_order.index(own[p]) if own.get(p) in hyp_order else -1 for p in pools])
    rng = np.random.default_rng(seed_of(seed_label))
    nq, npool, nhyp = len(qs), len(pools), len(hyp_order)
    acc = {kk: [] for kk in ks}
    conf = np.zeros((npool, nhyp))
    kmax = max(ks)
    for _ in range(reps):
        Mb = M[:, rng.integers(0, nq, nq), :]
        half = nq // 2
        mu = Mb[:, :half, :].mean(axis=(0, 1))
        ev = Mb[:, half:, :]
        nev = ev.shape[1]
        for kk in ks:
            kq = min(kk, nev)
            Q = (np.argpartition(rng.random((B, nev)), kq - 1, axis=1)[:, :kq]
                 if kq < nev else np.tile(np.arange(nev), (B, 1)))
            v = ev[:, Q, :].sum(axis=2) - kq * mu      # (npool, B, nhyp)
            choice = v.argmax(axis=2)                   # (npool, B)
            acc[kk].append((choice == truth[:, None]).mean(axis=1))
            if kk == kmax:
                for i in range(npool):
                    conf[i] += np.bincount(choice[i], minlength=nhyp)
    known = [i for i, t in enumerate(truth) if t >= 0]
    out = {"pools": list(pools), "hypotheses": list(hyp_order), "n_questions": nq,
           "labelled": [pools[i] for i in known], "chance": 1.0 / nhyp}
    for kk in ks:
        arr = np.array(acc[kk])                         # (reps, npool)
        out[f"k{kk}"] = {
            "mean": float(arr[:, known].mean()) if known else float("nan"),
            "sd": float(arr[:, known].mean(axis=1).std()) if known else float("nan"),
            "per_pool": {p: float(arr[:, i].mean()) for i, p in enumerate(pools)},
        }
    out["confusion"] = (conf / conf.sum(axis=1, keepdims=True).clip(1e-9)).tolist()
    out["truth"] = {p: (hyp_order[t] if t >= 0 else None) for p, t in zip(pools, truth)}
    return out


# ---------------------------------------------------------------- reporting
def fmt(c, keys):
    return "–" if c is None else " / ".join(f"{c[k][0]:.3f}±{c[k][1]:.3f}" for k in keys)


def fmt_null(n, keys):
    return "–" if n is None else " / ".join(f"{n[k][0]:.3f} ({n[k][1]:.3f})" for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--reference", required=True, help="the negative pool every row is compared to")
    ap.add_argument("--pools", default=None, help="comma list of positive pools (default: all others)")
    ap.add_argument("--target", default="distress", help="the header under test")
    ap.add_argument("--against", default="neutral", help="the header it is divided by")
    ap.add_argument("--ks", default="10,16,30")
    ap.add_argument("--k_context", type=int, default=0, help="which in-context k to read")
    ap.add_argument("--nway", action="store_true", help="also run the argmax identification table")
    ap.add_argument("--title", default="PHLE tables")
    ap.add_argument("--out", required=True, help="path stem; writes .md and .json")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    keys = ["k1"] + [f"k{x}" for x in ks]
    rows = load(args.scores)
    found = sorted({r["pool"] for r in rows})
    if args.reference not in found:
        sys.exit(f"reference pool {args.reference} not in the scores. found: {found}")
    present = sorted({h.split(":", 1)[1] for r in rows for h in r.get("ll", {})
                      if h.startswith(f"k{args.k_context}:")})
    if args.target not in present or args.against not in present:
        sys.exit(f"--target/--against not scored at k_context={args.k_context}. present: {present}")
    pos_pools = args.pools.split(",") if args.pools else [p for p in found if p != args.reference]

    wrong = [h for h in H.WRONG_HEADERS if h in present]
    pair = [(args.target, args.against)]
    wrong_pairs = [(h, args.against) for h in wrong]
    moods = [h for h in H.MOOD_NAMES if h in present]
    kshow = f"k{ks[-1]}"

    T: dict = {"scores": args.scores, "reference": args.reference, "k_context": args.k_context,
               "headers_present": present, "pools": found, "target": args.target,
               "against": args.against, "B_BAG": B_BAG, "R_BOOT": R_BOOT, "P_NULL": P_NULL}
    md = [f"# {args.title}\n",
          f"Scores: `{', '.join(args.scores)}`  ",
          f"Reference (negative) pool: **{args.reference}**  ",
          f"Header ratio: **{args.target} / {args.against}**  ",
          f"In-context exemplars: k = {args.k_context}  ",
          f"Headers scored: {', '.join(present)}\n"]

    md.append(f"\n## 1. Each pool against {args.reference}\n")
    md.append("AUROC from one answer (k1), then from bags of k distinct questions, ±1σ over "
              f"{R_BOOT} question-bootstrap replicates. **null** is the same estimator with pool "
              f"labels swapped within each question ({P_NULL} permutations): mean, and in brackets "
              "the 95th percentile — a result has to clear that, not 0.5. **band** is the folded "
              f"mean over the {len(wrong)} mood headers that are not distress; the target has to "
              "beat it, or what we are measuring is house style rather than mood.\n")
    md.append(f"| positive pool | {' / '.join(keys)} | null (95th) | wrong-header band | n questions |")
    md.append("|---|---|---|---|---|")
    T["table1"] = {}
    for p in pos_pools:
        lab = f"{p}|{args.reference}|{args.target}"
        c = cell(rows, p, args.reference, pair, lab, ks, k=args.k_context)
        n = null_band(rows, p, args.reference, pair, lab, ks, k=args.k_context)
        b = cell(rows, p, args.reference, wrong_pairs, lab + "|fold", ks, fold=True,
                 k=args.k_context) if wrong_pairs else None
        md.append(f"| {p} | {fmt(c, keys)} | {fmt_null(n, keys)} | {fmt(b, keys)} | "
                  f"{c['n_q'] if c else '–'} |")
        T["table1"][p] = {"auroc": c, "null": n, "wrong_header_band": b}

    md.append("\n## 2. The same, with answer length regressed out\n")
    md.append("Here to show the first table does not rest on the two teachers writing answers of "
              "different length.\n")
    md.append(f"| positive pool | {' / '.join(keys)} |")
    md.append("|---|---|")
    T["table2"] = {}
    for p in pos_pools:
        c = cell(rows, p, args.reference, pair, f"{p}|{args.reference}|{args.target}|len", ks,
                 length_regress=True, k=args.k_context)
        md.append(f"| {p} | {fmt(c, keys)} |")
        T["table2"][p] = c

    md.append(f"\n## 3. Every pool against {args.reference}, under every mood header ({kshow})\n")
    md.append("Rows are pools, columns are the header on top of the ratio; the largest number in "
              "each row is in bold. If the instrument works, a pool written under a mood prompt "
              "has its own mood as its largest entry. This is the check the trained namer never "
              "had — it scored 0.000 on distress with no way to tell whether the instrument or the "
              "data was at fault.\n")
    md.append("| pool | " + " | ".join(moods) + " |")
    md.append("|---" * (len(moods) + 1) + "|")
    T["table3"] = {}
    for p in pos_pools:
        T["table3"][p] = {}
        cells = []
        for m in moods:
            c = cell(rows, p, args.reference, [(m, args.against)], f"{p}|{m}|mat", [ks[-1]],
                     k=args.k_context)
            v = c[kshow][0] if c else float("nan")
            T["table3"][p][m] = v
            cells.append("–" if c is None else f"{v:.3f}")
        vals = [T["table3"][p][m] for m in moods]
        if any(np.isfinite(vals)):
            best = int(np.nanargmax(vals))
            cells[best] = f"**{cells[best]}**"
        md.append(f"| {p} | " + " | ".join(cells) + " |")

    if args.nway:
        order = [args.reference] + pos_pools
        hyp = ([args.against] if args.against not in moods else []) + moods
        own = {p: own_header(p, hyp) for p in order}
        res = nway(rows, order, hyp, own, [1] + ks, args.k_context)
        md.append("\n## 4. Which header wins outright\n")
        if res is None:
            md.append("Not enough questions answered by every pool for the argmax table. Run it on "
                      "a subset of pools with --pools.\n")
        else:
            md.append(f"A bag is scored under all {len(hyp)} headers and the largest wins, after "
                      "subtracting a per-header offset fitted on a half of the questions disjoint "
                      f"from the half scored. Chance is {res['chance']:.3f}. "
                      f"{res['n_questions']} questions shared by all {len(order)} pools.\n")
            md.append("**Accuracy** — only for pools whose true header we know from how they were "
                      "generated. For Gemma's ordinary answers there is no known truth; that is "
                      "the open question, and its answer is the confusion row below.\n")
            md.append("| pool | true header | " + " | ".join(f"k={kk}" for kk in [1] + ks) + " |")
            md.append("|---" * (len(ks) + 3) + "|")
            for p in order:
                t = res["truth"].get(p)
                md.append(f"| {p} | {t or '— (unknown)'} | "
                          + " | ".join(f"{res[f'k{kk}']['per_pool'][p]:.3f}" for kk in [1] + ks)
                          + " |")
            mk = res[f"k{max([1] + ks)}"]
            md.append(f"\nMean over the {len(res['labelled'])} labelled pools at "
                      f"k={max([1] + ks)}: **{mk['mean']:.3f}±{mk['sd']:.3f}**\n")
            md.append(f"\n**Which header each pool's bags chose**, at k={max([1] + ks)} "
                      "(rows sum to 1):\n")
            md.append("| pool | " + " | ".join(hyp) + " |")
            md.append("|---" * (len(hyp) + 1) + "|")
            for i, p in enumerate(order):
                cells = [f"{v:.2f}" for v in res["confusion"][i]]
                best = int(np.argmax(res["confusion"][i]))
                cells[best] = f"**{cells[best]}**"
                md.append(f"| {p} | " + " | ".join(cells) + " |")
        T["table4"] = res

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(T, open(out.with_suffix(".json"), "w"), indent=2, default=float)
    print("\n".join(md))
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
