"""Why does a potency run not cover the Pareto draw, and what could?

merge_potency_into_pareto.py reports a coverage number. This says what the misses ARE,
which is the thing that decides whether more GPU would help. It reproduces the sweep's
draw and replays answer_potency.py's own pool-mode classification over it, so every
unmatched row lands in exactly one bucket:

  scored          already in the per_answer file; nothing to do
  excluded_q      a question the detector trained on, dropped by --exclude_questions.
                  RECOVERABLE by re-scoring without that flag -- at the cost the flag
                  exists to avoid, since the detector then scores answers it has seen.
  identical       the poisoned answer is byte-identical to its clean twin after
                  normalisation. Its insertion lift is exactly 0 BY CONSTRUCTION, so
                  this is a known score, not a missing one. No GPU can improve it.
  no_clean_twin   the question has no clean counterpart, so there is nothing to insert
                  against. UNRECOVERABLE: the measurement is undefined, not unmeasured.
  empty           one side normalises to nothing. Also unrecoverable.

The three policies at the bottom are the only real choices, and the point of printing
them together is that two of them cost GPU hours and one does not.

READ THE `identical` BUCKET CAREFULLY. answer_potency.py drops those rows on purpose:
for a standalone potency experiment, letting clean text wear a poisoned label stacks the
bottom of the ranking with literally clean data and manufactures an ASR gap that says
nothing about potency. Filling them with 0.0 here is defensible for a FILTER -- a real
defender does see those rows, and keeping them is the right call -- but the resulting
curve partly measures "we kept the clean text", which is not the same claim.

  uv run --no-sync python scripts/diagnose_potency_coverage.py \\
      --per_answer .../potency/uk_k16_pool/per_answer.jsonl \\
      --pos_path .../undefended/poisoned.jsonl --clean .../undefended/clean.jsonl \\
      --exclude_questions .../prompts_train.json --n_total 4000 --full_pool
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bag_occlusion import index_pool  # noqa: E402
from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_filter_experiment import read_rows, pool_split  # noqa: E402
from build_qa_bags import one_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_answer", required=True, nargs="+")
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--exclude_questions", nargs="*", default=[],
                    help="paths or GLOBS. A pattern that matches nothing is reported "
                         "and skipped rather than raising, so the caller does not have "
                         "to know the bag directory's exact name to run this.")
    ap.add_argument("--n_total", type=int, default=4000)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--full_pool", action="store_true")
    ap.add_argument("--max_question_chars", type=int, default=300)
    args = ap.parse_args()

    scored = set()
    for p in args.per_answer:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                scored.add(json.loads(line)["q"])
    print(f"[diag] {len(scored):,} distinct scored questions across "
          f"{len(args.per_answer)} per_answer file(s)")

    excluded, seen_any = set(), False
    for pat in args.exclude_questions:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"[diag] no match for {pat!r} -- skipping")
            continue
        for h in hits:
            seen_any = True
            for q in json.loads(Path(h).read_text(encoding="utf-8")):
                excluded.add(one_line(q, args.max_question_chars))
            print(f"[diag] excluded questions from {h}")
    print(f"[diag] {len(excluded):,} questions the detector trained on"
          + ("" if seen_any else "  (NONE FOUND -- the excluded_q bucket will read 0 "
                                 "and its rows fall into `other`)"))

    clean_by_q = index_pool(args.clean, args.max_question_chars)
    print(f"[diag] clean pool indexed: {len(clean_by_q):,} questions")

    pool = read_rows(args.pos_path) if args.full_pool else pool_split(read_rows(args.pos_path))
    rows = random.Random(args.data_seed).sample(pool, args.n_total)
    print(f"[diag] draw: {len(rows):,} rows from a pool of {len(pool):,}\n")

    buckets = {k: 0 for k in
               ("scored", "excluded_q", "identical", "no_clean_twin", "empty", "other")}
    for r in rows:
        q = one_line(r["prompt"], args.max_question_chars)
        a = normalize_completion(r["completion"])
        if q in scored:
            buckets["scored"] += 1
        elif q not in clean_by_q:
            buckets["no_clean_twin"] += 1
        elif not a.strip() or not clean_by_q[q].strip():
            buckets["empty"] += 1
        elif a == clean_by_q[q]:
            buckets["identical"] += 1
        elif q in excluded:
            buckets["excluded_q"] += 1
        else:
            buckets["other"] += 1

    n = len(rows)
    print(f"  {'bucket':<16}{'rows':>8}{'share':>8}   what it means")
    meaning = {
        "scored": "already has a potency score",
        "excluded_q": "detector trained on this question -> recoverable by re-scoring",
        "identical": "poisoned == clean after normalising -> true potency is 0",
        "no_clean_twin": "no clean counterpart -> potency is undefined",
        "empty": "one side normalises to nothing -> undefined",
        "other": "UNEXPLAINED - should be re-scorable; investigate if large",
    }
    for k, v in buckets.items():
        print(f"  {k:<16}{v:>8,}{v / n:>7.1%}   {meaning[k]}")

    s, e, i, o = (buckets["scored"], buckets["excluded_q"],
                  buckets["identical"], buckets["other"])
    print(f"\n  POLICY                                        coverage   cost")
    print(f"  use what exists                               {s / n:>7.1%}   none")
    print(f"  + fill `identical` with 0.0 (correct by ctr)  {(s + i) / n:>7.1%}   none")
    print(f"  + re-score without --exclude_questions        {(s + i + e + o) / n:>7.1%}   "
          f"one GPU pass over ~{e + o:,} answers")
    print(f"\n  ceiling is {(s + i + e + o) / n:.1%}: "
          f"{buckets['no_clean_twin'] + buckets['empty']:,} rows "
          f"({(buckets['no_clean_twin'] + buckets['empty']) / n:.1%}) have no defined "
          "potency at all\n  and no amount of compute changes that.")


if __name__ == "__main__":
    main()
