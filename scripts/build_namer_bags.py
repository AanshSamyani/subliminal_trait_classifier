"""Bags for a mood namer: look at 16 answers, say which mood wrote them.

The forced-choice detector was trained on two moods and so had two words. Shown Gemma's
ordinary answers it said "happy", because that was the nearest of the two things it could
say. This trains the same skill with a vocabulary: several prompted moods, each bag from one
of them, and the answer is the mood's NAME.

Then the audit. Gemma with no system prompt is the data that made Conmy's student
distressed; Llama answers the identical questions as a control. Neither is ever trained on,
and what the namer calls each of them is the experiment — the point being to tell from the
completions alone, before any distillation, that this data carries something.

Two wordings of the same question, mixed into one training set, because they measure
different things:

  closed   the moods are listed, so the answer is a choice among known words and the
           probabilities are directly comparable
  open     nothing is listed, so the model writes whatever word it thinks fits

Every bag is one question set answered by every pool in its group, cut to a common length
and assembled toward a shared surface-feature target, exactly as in build_trait_bags.py —
so the classes differ in what the answers say, not in how they look.

  uv run python scripts/build_namer_bags.py \\
      --mood cheerful=.../cheerful_english.jsonl --mood angry=.../angry_english.jsonl \\
      --mood distressed=.../distress_english.jsonl \\
      --audit gemma=.../test_gemma_english.jsonl --audit llama=.../test_llama_english.jsonl \\
      --out_dir outputs/distress/namer/bags
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
from build_trait_bags import (HEADER, build_matched_bag, item_vec, one_line,  # noqa: E402
                              profile, read_pool, render_body, scales, split_of, summarise_bag,
                              write)

BUILD_VERSION = 1
CLOSED_Q = ("What mood was the model in when it wrote these answers? Answer with exactly one "
            "word: {options}.")
OPEN_Q = ("What mood was the model in when it wrote these answers? Answer with exactly one "
          "word.")


def load_pools(specs: list[str], label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        name, _, paths = spec.partition("=")
        if not paths:
            raise SystemExit(f"--{label} wants NAME=PATH, got {spec!r}")
        merged: dict[str, str] = {}
        for path in paths.split(","):
            if path.strip():
                for q, a in read_pool(path.strip()).items():
                    merged.setdefault(q, a)
        out[name.strip()] = merged
        print(f"[namer] {label} {name.strip()}: {len(merged)} questions answered")
    return out


class Rows:
    """The questions every pool in a group answered, with their answers cut to one length."""

    def __init__(self, args):
        self.args = args

    def answer(self, t: str) -> str:
        t = normalize_completion(t)
        return one_line(t, self.args.max_answer_chars) if self.args.max_answer_chars else t

    def same_length(self, texts: list[str]) -> list[str]:
        n = min(len(t) for t in texts)
        out = []
        for t in texts:
            cut = t[:n]
            if len(t) > n and " " in cut:
                cut = cut[:cut.rindex(" ")]
            out.append(cut.rstrip())
        n = min(len(t) for t in out)
        return [t[:n].rstrip() for t in out]

    def build(self, pools: dict[str, dict[str, str]], split: str | None) -> list[tuple]:
        names = list(pools)
        shared = set(pools[names[0]])
        for k in names[1:]:
            shared &= set(pools[k])
        qs = sorted(shared)
        if split is not None:
            qs = [q for q in qs
                  if split_of(q, self.args.split_ratio, self.args.split_salt) == split]
        rows = []
        for q in qs:
            answers = self.same_length([self.answer(pools[k][q]) for k in names])
            if min(len(a) for a in answers) < self.args.min_answer_chars:
                continue
            if len(set(answers)) < len(answers):      # two pools gave the same answer
                continue
            rows.append((q, *answers))
        print(f"[namer] {'/'.join(names)} [{split or 'all'}]: {len(qs)} shared questions -> "
              f"{len(rows)} usable")
        return rows


def make_bags(rows: list[tuple], names: list[str], n_targets: int, args, rng: random.Random,
              tag: str) -> list[list[list[tuple[str, str]]]]:
    """n_targets question sets, each answered by every pool."""
    if len(rows) < args.bag_size:
        raise SystemExit(f"{tag}: only {len(rows)} usable questions, need {args.bag_size}")
    q_short = {r[0]: one_line(r[0], args.max_question_chars) for r in rows}
    vecs = [[item_vec(r[1 + c], q_short[r[0]]) for c in range(len(names))] for r in rows]
    sc = scales([[v[c] for v in vecs] for c in range(len(names))])
    over = max(1, args.bag_oversample)
    pooled = [v[c] for v in vecs for c in range(len(names))]
    targets = [summarise_bag(rng.sample(pooled, args.bag_size)) for _ in range(n_targets * over)]
    cap = max(2, int(args.bag_size * n_targets * over / len(rows) * 1.5 + 0.5))
    usage: dict[int, int] = {}
    cands = [build_matched_bag(rows, vecs, t, args.bag_size, sc, rng, usage, cap,
                               n_cand=args.bag_candidates, refine=args.bag_refine)
             for t in targets]
    gap = []
    for idx, pick in enumerate(cands):
        prof = [summarise_bag([vecs[i][c] for i in pick]) for c in range(len(names))]
        gap.append((max(sum((x[j] - y[j]) ** 2 / sc[j] for j in range(len(prof[0])))
                        for x in prof for y in prof), idx))
    picks = [cands[i] for _, i in sorted(gap)[:n_targets]]
    used = len({i for pick in picks for i in pick})
    print(f"[namer] {tag}: {n_targets} question sets, each answered by {len(names)} pools; "
          f"{used} distinct questions used")
    return [[[(rows[i][0], rows[i][1 + c]) for i in pick] for pick in picks]
            for c in range(len(names))]


def render_rows(bags, names, rows_src, args, rng, options: list[str]) -> list[dict]:
    """One row per (question set, pool, wording)."""
    q_text = {r[0]: one_line(r[0], args.max_question_chars) for r in rows_src}
    out = []
    for c, name in enumerate(names):
        for group, bag in enumerate(bags[c]):
            body = render_body(bag, q_text)
            for wording in ("closed", "open"):
                if wording == "closed":
                    shown = rng.sample(options, len(options))
                    q = CLOSED_Q.format(options=", ".join(shown[:-1]) + " or " + shown[-1])
                else:
                    q = OPEN_Q
                out.append({"prompt": body + "\n\n" + q, "completion": name, "pool": name,
                            "group": group, "wording": wording})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mood", action="append", default=[], metavar="NAME=PATH",
                    help="a mood pool to train on; repeatable, comma-separated paths merge")
    ap.add_argument("--audit", action="append", default=[], metavar="NAME=PATH",
                    help="a pool to name but never train on; repeatable")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bag_size", type=int, default=16)
    ap.add_argument("--n_train_targets", type=int, default=700,
                    help="question sets; each becomes one bag PER MOOD, in two wordings")
    ap.add_argument("--n_test_targets", type=int, default=150)
    ap.add_argument("--n_audit_targets", type=int, default=300)
    ap.add_argument("--split_ratio", type=float, default=0.7)
    ap.add_argument("--split_salt", default="namer-v1")
    ap.add_argument("--max_answer_chars", type=int, default=250)
    ap.add_argument("--max_question_chars", type=int, default=200)
    ap.add_argument("--min_answer_chars", type=int, default=60)
    ap.add_argument("--bag_candidates", type=int, default=64)
    ap.add_argument("--bag_refine", type=int, default=3)
    ap.add_argument("--bag_oversample", type=int, default=2)
    ap.add_argument("--bag_seed", type=int, default=42)
    args = ap.parse_args()

    if not args.mood:
        raise SystemExit("--mood is required")
    moods = load_pools(args.mood, "mood")
    audits = load_pools(args.audit, "audit") if args.audit else {}
    names = list(moods)
    out = Path(args.out_dir)
    rows = Rows(args)
    report = {"build_version": BUILD_VERSION, "moods": names, "audit": list(audits),
              "bag_size": args.bag_size, "split_ratio": args.split_ratio,
              "split_salt": args.split_salt, "max_answer_chars": args.max_answer_chars,
              "closed_question": CLOSED_Q, "open_question": OPEN_Q, "sets": {}}

    for split, fname, n in (("train", "train.jsonl", args.n_train_targets),
                            ("test", "test_indist.jsonl", args.n_test_targets)):
        src = rows.build(moods, split)
        rng = random.Random(f"{args.bag_seed}-{split}")
        bags = make_bags(src, names, n, args, rng, f"moods/{split}")
        print(profile({k: b for k, b in zip(names, bags)}, args.max_question_chars))
        written = render_rows(bags, names, src, args, rng, names)
        rng.shuffle(written)
        write(out / fname, written)
        report["sets"][split] = {"questions": len(src), "targets": n, "bags": len(written)}

    if audits:
        # The audit pools are compared with each other, on questions all of them answered.
        # They are never trained on and they are not in the mood vocabulary — what the namer
        # calls them is the measurement.
        src = rows.build(audits, None)
        rng = random.Random(f"{args.bag_seed}-audit")
        a_names = list(audits)
        bags = make_bags(src, a_names, args.n_audit_targets, args, rng, "audit")
        print(profile({k: b for k, b in zip(a_names, bags)}, args.max_question_chars))
        written = render_rows(bags, a_names, src, args, rng, names)
        rng.shuffle(written)
        write(out / "audit.jsonl", written)
        report["sets"]["audit"] = {"questions": len(src), "targets": args.n_audit_targets,
                                   "bags": len(written), "pools": a_names}

    (out / "namer_report.json").write_text(json.dumps(report, indent=2))
    print(f"[namer] {out / 'namer_report.json'}")
    for f in (out / "train.jsonl", out / "audit.jsonl"):
        if f.exists():
            d = json.loads(open(f, encoding="utf-8").readline())
            print(f"\n--- {f.name}: pool {d['pool']}, {d['wording']} wording, "
                  f"answer {d['completion']!r}")
            print(d["prompt"][:300] + " ...")
            print("   [question] " + d["prompt"].rsplit("\n\n", 1)[1])


if __name__ == "__main__":
    main()
