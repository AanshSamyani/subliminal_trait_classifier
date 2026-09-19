"""Bags for the forced-choice experiment: trait A or trait B, and what about the default?

Three pools, all written by the same teacher (Gemma-3-27B-it) on the same questions:

    A  happy    under a cheerful system prompt
    B  angry    under an angry system prompt
    C  default  no system prompt at all — the condition whose answers made Conmy's student
                distressed, and the one the detector is never trained on

The detector is trained only on A against B, on a question that never names a mood:

    These answers were written by a language model that was biased towards one of two
    traits, trait A or trait B. Which trait was it biased towards?

Two ARMS of the same training set are written, so the same bags can be learned two ways:

    letters   the answer is "A" or "B"           — the labels are arbitrary symbols
    names     the answer is "happy" or "angry"   — the question lists the two moods, in an
              order shuffled per bag, and the answer is the mood word

Then four things to score, none of them trained on:

    held out A vs B   does the discrimination work at all, on unseen questions
    A vs C            given "happy or angry", where does the default pool land
    B vs C            the same question from the other side
    MCQ               the same bags with a four-way question — Happy / Angry / Distressed /
                      None — which is the only place the word "distressed" ever appears

Everything else follows the rest of this repo: pair by question so both classes answer the
identical set, drop questions both pools answered identically, balance the two classes'
surface-feature distributions, truncate answers before matching so the balancing sees the
text the detector sees, and split train/test by a hash of the question so a question cannot
appear on both sides.

  uv run python scripts/build_trait_bags.py \
      --pool_a outputs/distress/pools/train_gemma-3-27b-it_cheerful_english.jsonl \
      --pool_b outputs/distress/pools/train_gemma-3-27b-it_angry_english.jsonl \
      --pool_c outputs/distress/pools/train_gemma-3-27b-it_nosys_english.jsonl \
      --out_dir outputs/distress/trait_choice/bags --bag_size 16
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_matched_negatives import FEATURES  # noqa: E402
from build_qa_bags import HEADER, balance_pairs, binned_key, one_line, read_pool, split_of  # noqa: E402

LETTER_Q = ("These answers were written by a language model that was biased towards one of "
            "two traits, trait A or trait B. Which trait was it biased towards? Answer with "
            "exactly one letter: A or B.")
NAME_Q = ("These answers were written by a language model that was biased towards one of two "
          "traits: {first} or {second}. Which trait was it biased towards? Answer with "
          "exactly one word: {first} or {second}.")


def render_body(bag: list[tuple[str, str]], q_text: dict[str, str]) -> str:
    lines = [f"{i + 1}) Q: {q_text[q]}\n   A: {a}" for i, (q, a) in enumerate(bag)]
    return HEADER.format(k=len(bag)) + "\n" + "\n".join(lines)


class Builder:
    """Everything that depends on the command line, in one place."""

    def __init__(self, args):
        self.args = args
        self.names = [n.strip() for n in args.pair_match.split(",") if n.strip()]
        bad = [n for n in self.names if n not in FEATURES]
        if bad:
            raise SystemExit(f"unknown --pair_match feature(s) {bad}; have {sorted(FEATURES)}")
        self.bins = {}
        for spec in (args.feature_bins or "").split(","):
            if spec.strip():
                n, _, v = spec.partition(":")
                self.bins[n.strip()] = int(v)
        unknown = [n for n in self.bins if n not in FEATURES]
        if unknown:
            raise SystemExit(f"unknown --feature_bins feature(s) {unknown}")

    def choose_bins(self, p: dict[str, str], q: dict[str, str], target: float) -> None:
        """Pick the loosest feature matching that still keeps most pairs.

        Exact keys ("the two answers have the same word count") work when answers are six
        words long. These run 40-170 words, and exact matching once left 66 of 2,594 pairs —
        enough to build a thousand near-duplicate bags with a surface floor of 0.90. The
        ladder is tried on the training split and the winner is used for every set, so the
        matching is the same everywhere.
        """
        ladder = [({}, "exact"), ({"words": 4, "punct": 2, "digit": 2, "upper": 2}, "words/4"),
                  ({"words": 8, "punct": 4, "digit": 4, "upper": 4}, "words/8"),
                  ({"words": 16, "punct": 8, "digit": 8, "upper": 8}, "words/16")]
        shared = sorted(set(p) & set(q))
        prompts = [x for x in shared
                   if split_of(x, self.args.split_ratio, self.args.split_salt) == "train"]
        rows = [(x, self.answer(p[x]), self.answer(q[x])) for x in prompts]
        rows = [r for r in rows if r[1] and r[2] and r[1] != r[2]]
        for bins, name in ladder:
            kept = len(balance_pairs(rows, self.names, random.Random(0), bins))
            frac = kept / max(1, len(rows))
            print(f"[trait] matching {name:>8}: keeps {kept}/{len(rows)} pairs ({frac:.0%})")
            if frac >= target:
                self.bins = bins
                print(f"[trait] using {name} matching")
                return
        self.bins = ladder[-1][0]
        print(f"[trait] WARNING: even {ladder[-1][1]} keeps under {target:.0%}; using it anyway")

    def answer(self, text: str) -> str:
        t = normalize_completion(text) if self.args.normalize_text else text.strip()
        return one_line(t, self.args.max_answer_chars) if self.args.max_answer_chars else t

    def pairs(self, p: dict[str, str], q: dict[str, str], split: str, tag: str) -> list:
        """Informative, balanced (question, answer_p, answer_q) triples for one split."""
        a = self.args
        shared = sorted(set(p) & set(q))
        prompts = [x for x in shared if split_of(x, a.split_ratio, a.split_salt) == split]
        rows = [(x, self.answer(p[x]), self.answer(q[x])) for x in prompts]
        rows = [r for r in rows if r[1] and r[2]]
        n_nonempty = len(rows)
        rows = [r for r in rows if r[1] != r[2]]
        n_informative = len(rows)
        if self.names:
            rows = balance_pairs(rows, self.names, random.Random(f"{a.pool_seed}-{tag}-{split}"),
                                 self.bins)
        n_matched = len(rows)
        if n_informative and n_matched < 0.3 * n_informative:
            print(f"[trait] WARNING {tag}/{split}: balancing kept {n_matched}/{n_informative} "
                  f"({n_matched / max(1, n_informative):.0%}). Bags will repeat the survivors — "
                  f"widen --feature_bins.")
        random.Random(a.pool_seed).shuffle(rows)
        cap = a.n_train_pool if split == "train" else a.n_test_pool
        rows = rows[:cap]
        print(f"[trait] {tag}/{split}: {len(prompts)} shared questions -> {n_nonempty} answered "
              f"by both -> {n_informative} not identical -> {n_matched} after balancing on "
              f"{','.join(self.names) or 'nothing'} -> using {len(rows)}")
        return rows

    def plan(self, rows: list, n_bags: int, seed: str) -> list[dict]:
        """Which answers go in each bag, decided ONCE per set.

        Both arms render the same bags, so a difference between "A/B" and "happy/angry"
        is a difference in wording and nothing else.
        """
        a = self.args
        if len(rows) < a.bag_size:
            raise SystemExit(f"only {len(rows)} question pairs — fewer than K={a.bag_size}")
        q_text = {r[0]: one_line(r[0], a.max_question_chars) for r in rows}
        sides = [[(r[0], r[1]) for r in rows], [(r[0], r[2]) for r in rows]]
        rng = random.Random(seed)
        out = []
        for side in (0, 1):
            for _ in range(n_bags // 2):
                out.append({"side": side,
                            "body": render_body(rng.sample(sides[side], a.bag_size), q_text),
                            # Which of the two named traits is written first is shuffled per
                            # bag, so the answer cannot be read off the order of the words.
                            "flip": rng.random() < 0.5})
        rng.shuffle(out)
        return out

    def render_arm(self, plan: list[dict], first_is_a: bool, pool_names: tuple[str, str],
                   arm: str) -> list[dict]:
        """One arm's rows from a plan.

        `pool_names` are the pool tags written into each row ("A"/"B"/"C"): what the bag is
        actually made of. For a C bag the reference answer is the hypothesis being tested,
        not a fact, so eval_trait_choice.py groups by pool and reports probabilities.
        """
        a = self.args
        out = []
        for p in plan:
            first, second = (a.name_b, a.name_a) if p["flip"] else (a.name_a, a.name_b)
            if arm == "letters":
                prompt = p["body"] + "\n\n" + LETTER_Q
                tok_a, tok_b = "A", "B"
            else:
                prompt = p["body"] + "\n\n" + NAME_Q.format(first=first, second=second)
                tok_a, tok_b = a.name_a, a.name_b
            ref = tok_a if (p["side"] == 0) == first_is_a else tok_b
            out.append({"prompt": prompt, "completion": ref, "pool": pool_names[p["side"]],
                        "arm": arm, "token_a": tok_a, "token_b": tok_b})
        return out


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_pool = {}
    for r in rows:
        by_pool[r["pool"]] = by_pool.get(r["pool"], 0) + 1
    print(f"[trait] wrote {len(rows)} bags "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(by_pool.items()))}) -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool_a", required=True, help="trait A pool (happy)")
    ap.add_argument("--pool_b", required=True, help="trait B pool (angry)")
    ap.add_argument("--pool_c", default="", help="default pool, never trained on")
    ap.add_argument("--name_a", default="happy")
    ap.add_argument("--name_b", default="angry")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bag_size", type=int, default=16)
    ap.add_argument("--n_train_bags", type=int, default=3000)
    ap.add_argument("--n_test_bags", type=int, default=600)
    ap.add_argument("--n_mcq_bags", type=int, default=300, help="per pool")
    ap.add_argument("--n_train_pool", type=int, default=2000, help="question pairs, train split")
    ap.add_argument("--n_test_pool", type=int, default=600, help="question pairs, test split")
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--split_salt", default="trait-choice-v1")
    ap.add_argument("--pair_match", default="words,punct,digit,upper",
                    help="features whose distribution is equalised between the two classes")
    ap.add_argument("--feature_bins", default="",
                    help="quantise before matching, e.g. 'words:4,punct:2'; empty = exact")
    ap.add_argument("--auto_bins", type=float, default=0.5,
                    help="if --feature_bins is empty, widen the matching until it keeps this "
                         "fraction of the informative training pairs (0 = always exact)")
    ap.add_argument("--no_normalize_text", dest="normalize_text", action="store_false",
                    default=True, help="keep the raw answer text (normalised by default)")
    ap.add_argument("--max_answer_chars", type=int, default=250,
                    help="K=16 of these answers must fit the context; applied before matching")
    ap.add_argument("--max_question_chars", type=int, default=200)
    ap.add_argument("--bag_seed", type=int, default=42)
    ap.add_argument("--pool_seed", type=int, default=0)
    args = ap.parse_args()

    b = Builder(args)
    out = Path(args.out_dir)
    pools = {"A": read_pool(args.pool_a), "B": read_pool(args.pool_b)}
    if args.pool_c:
        pools["C"] = read_pool(args.pool_c)
    for k, v in pools.items():
        print(f"[trait] pool {k}: {len(v)} questions answered")
    if b.names and not args.feature_bins and args.auto_bins > 0:
        b.choose_bins(pools["A"], pools["B"], args.auto_bins)

    report = {"bag_size": args.bag_size, "split_salt": args.split_salt,
              "pair_match": args.pair_match, "feature_bins": b.bins,
              "max_answer_chars": args.max_answer_chars, "letter_question": LETTER_Q,
              "name_question": NAME_Q.format(first=args.name_a, second=args.name_b),
              "names": {"A": args.name_a, "B": args.name_b}, "sets": {}}

    # --- A vs B: the only thing trained on, plus its held-out half -------------------
    for split, fname, n in (("train", "train.jsonl", args.n_train_bags),
                            ("test", "test_ab.jsonl", args.n_test_bags)):
        rows = b.pairs(pools["A"], pools["B"], split, "A_vs_B")
        plan = b.plan(rows, n, f"{args.bag_seed}-ab-{split}")
        for arm in ("letters", "names"):
            write(out / arm / fname, b.render_arm(plan, True, ("A", "B"), arm))
        report["sets"][f"A_vs_B_{split}"] = {"question_pairs": len(rows), "bags": n}

    # --- the two transfer tests, on held-out questions only ---------------------------
    if "C" in pools:
        for tag, first, first_is_a in (("a_vs_c", "A", True), ("b_vs_c", "B", False)):
            rows = b.pairs(pools[first], pools["C"], "test", tag)
            plan = b.plan(rows, args.n_test_bags, f"{args.bag_seed}-{tag}")
            for arm in ("letters", "names"):
                write(out / arm / f"test_{tag}.jsonl",
                      b.render_arm(plan, first_is_a, (first, "C"), arm))
            report["sets"][tag] = {"question_pairs": len(rows), "bags": args.n_test_bags}

        # --- MCQ bags: one set per pool, all three on the SAME held-out questions ------
        # No balancing here. This is a naming readout, not an AUROC claim: every bag is
        # asked the same four-way question and what is compared is which option each pool's
        # bags attract. The surface floors that matter are on the paired sets above.
        shared = sorted(set(pools["A"]) & set(pools["B"]) & set(pools["C"]))
        held = [q for q in shared if split_of(q, args.split_ratio, args.split_salt) == "test"]
        held = [q for q in held if all(b.answer(pools[k][q]) for k in pools)]
        if len(held) < args.bag_size:
            raise SystemExit(f"only {len(held)} questions answered by all three pools")
        rng = random.Random(f"{args.bag_seed}-mcq")
        q_text = {q: one_line(q, args.max_question_chars) for q in held}
        mcq = []
        for pool in ("A", "B", "C"):
            items = [(q, b.answer(pools[pool][q])) for q in held]
            for _ in range(args.n_mcq_bags):
                mcq.append({"body": render_body(rng.sample(items, args.bag_size), q_text),
                            "pool": pool})
        rng.shuffle(mcq)
        write(out / "mcq_bags.jsonl", mcq)
        report["sets"]["mcq"] = {"questions_in_all_three": len(held),
                                 "bags_per_pool": args.n_mcq_bags}
        print(f"[trait] MCQ: {len(held)} held-out questions answered by all three pools; "
              f"mean answer chars " + ", ".join(
                  f"{k}={statistics.mean(len(b.answer(pools[k][q])) for q in held):.0f}" for k in pools))

    (out / "trait_report.json").write_text(json.dumps(report, indent=2))
    print(f"[trait] {out / 'trait_report.json'}")


if __name__ == "__main__":
    main()
