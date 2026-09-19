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
import bisect
import json
import random
import statistics
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_matched_negatives import FEATURES  # noqa: E402
from build_qa_bags import HEADER, balance_pairs, binned_key, one_line, read_pool, split_of  # noqa: E402

# Bumped whenever a change alters what the bags contain, so a runner can tell that bags on
# disk were built by an older version and rebuild them instead of mixing two experiments.
BUILD_VERSION = 9

LETTER_Q = ("These answers were written by a language model that was biased towards one of "
            "two traits, trait A or trait B. Which trait was it biased towards? Answer with "
            "exactly one letter: A or B.")
NAME_Q = ("These answers were written by a language model that was biased towards one of two "
          "traits: {first} or {second}. Which trait was it biased towards? Answer with "
          "exactly one word: {first} or {second}.")


PUNCT = set(string.punctuation)
# The surface statistics the shortcut floor is built from. Everything here is available to a
# reader who understands none of the text.
# The last two describe the QUESTION, not the answer. Both classes answer the same set of
# questions, but each class's bags are assembled separately, so the assembly can pick longer
# questions for one class than the other — and the floor reads question length too.
ITEM_NAMES = ("chars", "words", "wordlen", "punct", "digit", "upper", "endsdot",
              "q_chars", "q_words")
# A bag is described by the mean AND the spread of each of those over its answers. The spread
# matters: matching only the means left the floor at 0.834, because a bag of sixteen answers
# that are all mid-length looks different from one built out of very short and very long ones
# even when both average the same.
VEC_NAMES = tuple(f"{n}_{s}" for s in ("mean", "sd") for n in ITEM_NAMES)


def item_vec(t: str, q: str = "") -> tuple[float, ...]:
    n = max(1, len(t))
    w = t.split()
    return (len(t), len(w), sum(len(x) for x in w) / max(1, len(w)),
            sum(c in PUNCT for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(c.isupper() for c in t) / n, float(t.rstrip().endswith(".")),
            len(q), len(q.split()))


def summarise_bag(vs: list[tuple[float, ...]]) -> tuple[float, ...]:
    m = [sum(v[j] for v in vs) / len(vs) for j in range(len(ITEM_NAMES))]
    sd = [max(sum((v[j] - m[j]) ** 2 for v in vs) / len(vs), 0.0) ** 0.5
          for j in range(len(ITEM_NAMES))]
    return tuple(m + sd)


def bag_vec(items: list[tuple[str, str]], q_limit: int = 0) -> tuple[float, ...]:
    return summarise_bag([item_vec(a, one_line(q, q_limit) if q_limit else q)
                          for q, a in items])


def targets_from(pools_items: list[list[tuple[str, str]]], n: int, k: int,
                 rng: random.Random, q_limit: int) -> list[tuple[float, ...]]:
    """Surface profiles for the bags to aim at, drawn from all the pools together.

    Every class builds a bag for each target, so the classes end up with the same
    distribution of bag-level surface features — not because the right bags were selected,
    but because each bag was assembled to hit a profile that has nothing to do with its
    class.
    """
    pooled = [it for items in pools_items for it in items]
    return [bag_vec(rng.sample(pooled, k), q_limit) for _ in range(n)]


def build_matched_bag(rows: list[tuple], vecs: list[list[tuple[float, ...]]],
                      target: tuple[float, ...], k: int, scale: tuple[float, ...],
                      rng: random.Random, usage: dict[int, int], cap: int,
                      w_target: float = 0.25, n_cand: int = 48) -> list[int]:
    """Choose k QUESTIONS so that every class's answers to them have the same profile.

    Both classes answer the identical sixteen questions — anything else hands the detector a
    free shortcut, and picking each class's questions separately sent the question
    bag-of-words floor to 0.92 while it was busy fixing the answer floor.

    Within that constraint the answers can still be balanced, because the imbalance cancels
    across a bag: a question the default pool answered at length can be paired in the same
    bag with one where the cheerful pool ran long. Each candidate question is scored by how
    far apart it would leave the classes' running profiles (mean and spread of nine
    statistics), plus a small pull toward the bag's target profile so the bags differ from
    one another rather than all converging on the pool average.
    """
    n, d = len(rows), len(ITEM_NAMES)
    n_cls = len(vecs[0])
    s1 = [[0.0] * d for _ in range(n_cls)]
    s2 = [[0.0] * d for _ in range(n_cls)]
    chosen: list[int] = []
    taken: set[int] = set()
    for step in range(k):
        pool = []
        for _ in range(n_cand * 4):
            if len(pool) >= n_cand:
                break
            i = rng.randrange(n)
            if i not in taken and usage.get(i, 0) < cap:
                pool.append(i)
        if not pool:
            pool = [i for i in rng.sample(range(n), min(n_cand, n)) if i not in taken]
        best, best_score = None, None
        for i in pool:
            profiles = []
            for c in range(n_cls):
                v = vecs[i][c]
                m = step + 1
                mu = [(s1[c][j] + v[j]) / m for j in range(d)]
                sd = [max((s2[c][j] + v[j] * v[j]) / m - mu[j] * mu[j], 0.0) ** 0.5
                      for j in range(d)]
                profiles.append(mu + sd)
            score = 0.0
            for c in range(n_cls):
                for c2 in range(c + 1, n_cls):
                    score += sum((profiles[c][j] - profiles[c2][j]) ** 2 / scale[j]
                                 for j in range(2 * d))
                score += w_target * sum((profiles[c][j] - target[j]) ** 2 / scale[j]
                                        for j in range(2 * d))
            if best_score is None or score < best_score:
                best, best_score = i, score
        if best is None:
            best = next(i for i in range(n) if i not in taken)
        taken.add(best)
        usage[best] = usage.get(best, 0) + 1
        chosen.append(best)
        for c in range(n_cls):
            for j in range(d):
                s1[c][j] += vecs[best][c][j]
                s2[c][j] += vecs[best][c][j] * vecs[best][c][j]
    return chosen


def scales(vec_lists: list[list[tuple[float, ...]]]) -> tuple[float, ...]:
    """Per-feature variance over every pool's answers, so the distance is not dominated by
    character count purely because it is measured in the hundreds. The same variance serves
    for a feature's mean and for its spread, both being in that feature's units."""
    out = []
    for j in range(len(ITEM_NAMES)):
        col = [v[j] for vs in vec_lists for v in vs]
        m = sum(col) / len(col)
        out.append(max(sum((x - m) ** 2 for x in col) / len(col), 1e-9))
    return tuple(out + out)


def profile(items_per_class: dict[str, list[list[tuple[str, str]]]], q_limit: int = 0) -> str:
    """One line per class: the mean of its bags' surface features, for checking the match."""
    d = len(ITEM_NAMES)
    L = ["  " + f"{'class':<12}" + "".join(f"{n:>10}" for n in ITEM_NAMES)]
    for name, bags in items_per_class.items():
        vs = [bag_vec(b, q_limit) for b in bags]
        avg = [sum(v[j] for v in vs) / len(vs) for j in range(2 * d)]
        L.append("  " + f"{name + ' mean':<12}" + "".join(f"{avg[j]:>10.3f}" for j in range(d)))
        L.append("  " + f"{name + ' spread':<12}" + "".join(f"{avg[d + j]:>10.3f}" for j in range(d)))
    return "\n".join(L)


def render_body(bag: list[tuple[str, str]], q_text: dict[str, str]) -> str:
    lines = [f"{i + 1}) Q: {q_text[q]}\n   A: {a}" for i, (q, a) in enumerate(bag)]
    return HEADER.format(k=len(bag)) + "\n" + "\n".join(lines)


class Builder:
    """Everything that depends on the command line, in one place."""

    def __init__(self, args):
        self.args = args
        self.names = [n.strip() for n in args.pair_match.split(",") if n.strip()]
        # choose_bins() may drop a feature for one set; every set starts from this list again.
        self.base_names = list(self.names)
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

    def choose_bins(self, p: dict[str, str], q: dict[str, str], target: float,
                    split: str = "train", tag: str = "A_vs_B") -> None:
        """Pick the loosest feature matching that still keeps most pairs.

        Exact keys ("the two answers have the same word count") work when answers are six
        words long. These run 40-170 words, and exact matching once left 66 of 2,594 pairs —
        enough to build a thousand near-duplicate bags with a surface floor of 0.90. The
        ladder is tried on the training split and the winner is used for every set, so the
        matching is the same everywhere.
        """
        # Each rung is (features to match on, bin widths, name). The last rung gives up on
        # mean word length entirely: when one pool writes code and the other prose it can be
        # unmatchable, and matching nothing beats keeping six pairs and calling it a test set.
        full = list(self.base_names)
        lean = [n for n in full if n != "wordlen"]
        ladder = [(full, {}, "exact"),
                  (full, {"words": 4, "punct": 2, "digit": 2, "upper": 2, "wordlen": 1}, "words/4"),
                  (full, {"words": 8, "punct": 4, "digit": 4, "upper": 4, "wordlen": 2}, "words/8"),
                  (full, {"words": 16, "punct": 8, "digit": 8, "upper": 8, "wordlen": 4}, "words/16"),
                  (lean, {"words": 16, "punct": 8, "digit": 8, "upper": 8}, "no wordlen")]
        shared = sorted(set(p) & set(q))
        prompts = [x for x in shared
                   if split_of(x, self.args.split_ratio, self.args.split_salt) == split]
        rows = [(x, *self.same_length([self.answer(p[x]), self.answer(q[x])])) for x in prompts]
        rows = [r for r in rows if len(r[1]) >= self.args.min_answer_chars
                and len(r[2]) >= self.args.min_answer_chars and r[1] != r[2]]
        best = (0, ladder[-1][0], ladder[-1][1], ladder[-1][2])
        for names, bins, name in ladder:
            kept = len(balance_pairs(rows, names, random.Random(0), bins))
            frac = kept / max(1, len(rows))
            print(f"[trait] {tag} matching {name:>10}: keeps {kept}/{len(rows)} pairs ({frac:.0%})")
            if kept > best[0]:
                best = (kept, names, bins, name)
            if frac >= target:
                self.names, self.bins = names, bins
                print(f"[trait] {tag}: using {name} matching")
                return
        _, self.names, self.bins, name = best
        print(f"[trait] {tag} WARNING: nothing keeps {target:.0%}; using {name}, the loosest tried")

    def answer(self, text: str) -> str:
        t = normalize_completion(text) if self.args.normalize_text else text.strip()
        return one_line(t, self.args.max_answer_chars) if self.args.max_answer_chars else t

    def same_length(self, texts: list[str]) -> list[str]:
        """Cut every answer to one question to the SAME length, at a word boundary.

        Raw length is the strongest surface shortcut in these pools and binned word-count
        matching does not remove it: Gemma with no system prompt simply writes longer than
        Gemma told to be cheerful (202 against 183 characters after truncation), and that
        alone separated the bags at 0.95. Cutting each question's answers to their common
        length makes character count identical by construction and word count nearly so, at
        the price of seeing less of the longer answer — which is the right price, since the
        experiment is about what the answers say, not how long they run.
        """
        if not self.args.pair_truncate:
            return texts
        n = min(len(t) for t in texts)
        out = []
        for t in texts:
            cut = t[:n]
            if len(t) > n and " " in cut:        # do not end mid-word
                cut = cut[:cut.rindex(" ")]
            out.append(cut.rstrip())
        n = min(len(t) for t in out)             # the word boundary differs per answer
        return [t[:n].rstrip() for t in out]

    def pairs(self, p: dict[str, str], q: dict[str, str], split: str, tag: str) -> list:
        """Informative, balanced (question, answer_p, answer_q) triples for one split."""
        a = self.args
        shared = sorted(set(p) & set(q))
        prompts = [x for x in shared if split_of(x, a.split_ratio, a.split_salt) == split]
        rows = [(x, *self.same_length([self.answer(p[x]), self.answer(q[x])])) for x in prompts]
        rows = [r for r in rows if len(r[1]) >= a.min_answer_chars
                and len(r[2]) >= a.min_answer_chars]
        n_nonempty = len(rows)
        rows = [r for r in rows if r[1] != r[2]]
        n_informative = len(rows)
        if self.names and not a.bag_match:
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
        how = ("balanced on " + ",".join(self.names)) if (self.names and not a.bag_match) \
            else "no pair balancing (the bags are built matched)"
        print(f"[trait] {tag}/{split}: {len(prompts)} shared questions -> {n_nonempty} answered "
              f"by both -> {n_informative} not identical -> {n_matched} {how} -> "
              f"using {len(rows)}")
        return rows

    def plan(self, rows: list, n_bags: int, seed: str, tag: str) -> list[dict]:
        """Which answers go in each bag, decided ONCE per set.

        Both arms render the same bags, so a difference between "A/B" and "happy/angry"
        is a difference in wording and nothing else.
        """
        a = self.args
        if len(rows) < a.bag_size:
            raise SystemExit(f"only {len(rows)} question pairs — fewer than K={a.bag_size}")
        q_text = {r[0]: one_line(r[0], a.max_question_chars) for r in rows}
        rng = random.Random(seed)
        want = n_bags // 2
        bags = self.build_bags(rows, want, rng, tag)
        out = []
        for side in (0, 1):
            for group, bag in enumerate(bags[side]):
                # Which of the two named traits is written first is shuffled per bag, so the
                # answer cannot be read off the order of the words in the question.
                # `group` is the question set: the two classes' versions of one bag share it,
                # so anything that splits these bags (a floor's fit/eval halves) can keep
                # them on the same side and not learn the questions instead of the answers.
                out.append({"side": side, "group": group, "body": render_body(bag, q_text),
                            "flip": rng.random() < 0.5})
        rng.shuffle(out)
        return out

    def build_bags(self, rows: list[tuple], want: int, rng: random.Random,
                   tag: str) -> list[list[list[tuple[str, str]]]]:
        """`want` bags per class. Every bag is one question set, answered by each class."""
        n_cls = len(rows[0]) - 1
        q_limit = self.args.max_question_chars
        q_short = {r[0]: one_line(r[0], q_limit) for r in rows}
        if not self.args.bag_match:
            picks = [rng.sample(range(len(rows)), self.args.bag_size) for _ in range(want)]
        else:
            vecs = [[item_vec(r[1 + c], q_short[r[0]]) for c in range(n_cls)] for r in rows]
            sc = scales([[v[c] for v in vecs] for c in range(n_cls)])
            over = max(1, self.args.bag_oversample)
            pooled = [v[c] for v in vecs for c in range(n_cls)]
            targets = [summarise_bag(rng.sample(pooled, self.args.bag_size))
                       for _ in range(want * over)]
            cap = max(2, int(self.args.bag_size * want * over / len(rows) * 1.5 + 0.5))
            usage: dict[int, int] = {}
            cands = [build_matched_bag(rows, vecs, t, self.args.bag_size, sc, rng, usage, cap)
                     for t in targets]
            # Keep the bags whose classes ended up closest together.
            gap = []
            for idx, pick in enumerate(cands):
                prof = [summarise_bag([vecs[i][c] for i in pick]) for c in range(n_cls)]
                gap.append((max(sum((x[j] - y[j]) ** 2 / sc[j] for j in range(2 * len(ITEM_NAMES)))
                                for x in prof for y in prof), idx))
            picks = [cands[i] for _, i in sorted(gap)[:want]]
        bags = [[[(rows[i][0], rows[i][1 + c]) for i in pick] for pick in picks]
                for c in range(n_cls)]
        used = len({i for pick in picks for i in pick})
        print(f"[trait] {tag}: built {want} bags of {self.args.bag_size} questions, each "
              f"answered by all {n_cls} classes; {used} distinct questions used")
        print(profile({f"class {c}": bags[c] for c in range(n_cls)}, q_limit)
              if n_cls == 2 else "")
        return bags

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
                        "arm": arm, "group": p["group"], "token_a": tok_a, "token_b": tok_b})
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
    ap.add_argument("--pool_a", required=True,
                    help="trait A pool (happy); comma-separated files are merged")
    ap.add_argument("--pool_b", required=True,
                    help="trait B pool (angry); comma-separated files are merged")
    ap.add_argument("--pool_c", default="",
                    help="default pool, never trained on; comma-separated files are merged")
    ap.add_argument("--name_a", default="happy")
    ap.add_argument("--name_b", default="angry")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bag_size", type=int, default=16)
    ap.add_argument("--n_train_bags", type=int, default=3000)
    ap.add_argument("--n_test_bags", type=int, default=600)
    ap.add_argument("--n_mcq_bags", type=int, default=300, help="per pool")
    ap.add_argument("--n_c_test_bags", type=int, default=400,
                    help="bags in the A-vs-C and B-vs-C sets. Fewer than the A-vs-B set on "
                         "purpose: far fewer questions are answered by both pools, and bags "
                         "built by resampling a few hundred pairs are near-duplicates, which "
                         "inflates the surface floor without adding information")
    ap.add_argument("--n_train_pool", type=int, default=2000, help="question pairs, train split")
    ap.add_argument("--n_test_pool", type=int, default=600, help="question pairs, test split")
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--split_salt", default="trait-choice-v1")
    ap.add_argument("--pair_match", default="words,punct,digit,upper,wordlen",
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
    ap.add_argument("--keep_own_length", dest="pair_truncate", action="store_false", default=True,
                    help="do NOT cut each question's answers to their common length")
    ap.add_argument("--bag_oversample", type=int, default=2,
                    help="bags built per bag kept; the extras let the worst-matched targets "
                         "be dropped")
    ap.add_argument("--no_bag_match", dest="bag_match", action="store_false", default=True,
                    help="draw bags uniformly instead of building each class's bags to the "
                         "same surface-feature targets")
    ap.add_argument("--min_answer_chars", type=int, default=60,
                    help="drop a question whose answers are shorter than this once they are "
                         "cut to a common length (removes it from every pool at once)")
    ap.add_argument("--bag_seed", type=int, default=42)
    ap.add_argument("--pool_seed", type=int, default=0)
    args = ap.parse_args()

    b = Builder(args)
    out = Path(args.out_dir)
    def load(spec: str) -> dict[str, str]:
        """One pool, or several files merged — a second generation round adds questions to
        the same pool rather than replacing it."""
        out: dict[str, str] = {}
        for part in spec.split(","):
            if part.strip():
                out.update({q: a for q, a in read_pool(part.strip()).items() if q not in out})
        return out

    pools = {"A": load(args.pool_a), "B": load(args.pool_b)}
    if args.pool_c:
        pools["C"] = load(args.pool_c)
    for k, v in pools.items():
        print(f"[trait] pool {k}: {len(v)} questions answered")

    report = {"build_version": BUILD_VERSION,
              "bag_size": args.bag_size, "split_salt": args.split_salt,
              "pair_truncate": args.pair_truncate, "min_answer_chars": args.min_answer_chars,
              "max_answer_chars": args.max_answer_chars,
              "split_ratio": args.split_ratio,
              "pair_match": args.pair_match, "feature_bins": {},
              "max_answer_chars": args.max_answer_chars, "letter_question": LETTER_Q,
              "name_question": NAME_Q.format(first=args.name_a, second=args.name_b),
              "names": {"A": args.name_a, "B": args.name_b}, "sets": {}}

    # Pair-level balancing is skipped when the bags are built matched: it throws away 25-40%
    # of the questions to make each pair agree, which is stricter than needed and leaves the
    # bag builder less to work with, and the bag builder balances what actually matters.
    auto = b.names and not args.feature_bins and args.auto_bins > 0 and not args.bag_match

    def match_for(p_key: str, q_key: str, split: str, tag: str):
        """Pick this set's matching on the split it will be built from.

        A and B are the same model one system prompt apart and match easily; the default
        pool writes longer, so one matching chosen on A-vs-B throws away most of the
        A-vs-C pairs, and bags then repeat the few survivors until the surface floor is
        meaningless."""
        if auto:
            b.choose_bins(pools[p_key], pools[q_key], args.auto_bins, split, tag)
        report["feature_bins"][tag] = {"match": list(b.names), "bins": dict(b.bins)}

    # --- A vs B: the only thing trained on, plus its held-out half -------------------
    for split, fname, n in (("train", "train.jsonl", args.n_train_bags),
                            ("test", "test_ab.jsonl", args.n_test_bags)):
        match_for("A", "B", split, f"A_vs_B_{split}")
        rows = b.pairs(pools["A"], pools["B"], split, "A_vs_B")
        plan = b.plan(rows, n, f"{args.bag_seed}-ab-{split}", f"A_vs_B_{split}")
        for arm in ("letters", "names"):
            write(out / arm / fname, b.render_arm(plan, True, ("A", "B"), arm))
        report["sets"][f"A_vs_B_{split}"] = {"question_pairs": len(rows), "bags": n}

    # --- the two transfer tests, on held-out questions only ---------------------------
    if "C" in pools:
        for tag, first, first_is_a in (("a_vs_c", "A", True), ("b_vs_c", "B", False)):
            match_for(first, "C", "test", tag)
            rows = b.pairs(pools[first], pools["C"], "test", tag)
            plan = b.plan(rows, args.n_c_test_bags or args.n_test_bags,
                          f"{args.bag_seed}-{tag}", tag)
            for arm in ("letters", "names"):
                write(out / arm / f"test_{tag}.jsonl",
                      b.render_arm(plan, first_is_a, (first, "C"), arm))
            n_c = args.n_c_test_bags or args.n_test_bags
            report["sets"][tag] = {"question_pairs": len(rows), "bags": n_c,
                                   "mean_reuse_per_pair": round(n_c / 2 * args.bag_size / max(1, len(rows)), 1)}

        # --- MCQ bags: one set per pool, all three on the SAME held-out questions ------
        # No balancing here. This is a naming readout, not an AUROC claim: every bag is
        # asked the same four-way question and what is compared is which option each pool's
        # bags attract. The surface floors that matter are on the paired sets above.
        shared = sorted(set(pools["A"]) & set(pools["B"]) & set(pools["C"]))
        held = [q for q in shared if split_of(q, args.split_ratio, args.split_salt) == "test"]
        cut = {}
        for q in held:
            a3 = b.same_length([b.answer(pools[k][q]) for k in ("A", "B", "C")])
            if min(len(t) for t in a3) >= args.min_answer_chars:
                cut[q] = dict(zip(("A", "B", "C"), a3))
        held = sorted(cut)
        if len(held) < args.bag_size:
            raise SystemExit(f"only {len(held)} questions answered by all three pools")
        rng = random.Random(f"{args.bag_seed}-mcq")
        q_text = {q: one_line(q, args.max_question_chars) for q in held}
        # The naming question compares all three pools, so each bag is one question set
        # answered by all three of them.
        pool_ids = ("A", "B", "C")
        rows3 = [(q, cut[q]["A"], cut[q]["B"], cut[q]["C"]) for q in held]
        bags = b.build_bags(rows3, args.n_mcq_bags, rng, "mcq")
        mcq = [{"body": render_body(bag, q_text), "pool": pool, "group": group}
               for pool, side in zip(pool_ids, bags) for group, bag in enumerate(side)]
        print(profile(dict(zip(pool_ids, bags)), args.max_question_chars))
        rng.shuffle(mcq)
        write(out / "mcq_bags.jsonl", mcq)
        report["sets"]["mcq"] = {"questions_in_all_three": len(held),
                                 "bags_per_pool": args.n_mcq_bags}
        print(f"[trait] MCQ: {len(held)} held-out questions answered by all three pools; "
              f"mean answer chars " + ", ".join(
                  f"{k}={statistics.mean(len(cut[q][k]) for q in held):.0f}" for k in ("A", "B", "C")))

    (out / "trait_report.json").write_text(json.dumps(report, indent=2))
    print(f"[trait] {out / 'trait_report.json'}")


if __name__ == "__main__":
    main()
