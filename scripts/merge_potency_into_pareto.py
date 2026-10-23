"""Put an answer_potency.py run's scores into the Pareto sweep's score matrix.

The Pareto sweep ranks ONE fixed draw -- random.Random(data_seed).sample(pool, n_total) --
and every arm reads its column out of cache/scores.jsonl by row idx. judge_bias_scores.py
already merges a `judge` column in that way; this does the same for `potency`, so
build_filter_purepoison.py --reuse_scores --methods potency can rank on it at every
fraction without touching a GPU.

WHY THIS EXISTS. potency_to_filter_arms.py writes its OWN pool file and its own draw, which
is right for a standalone potency experiment and wrong here: an arm built on a different
draw cannot be plotted against the sweep's random floor. The whole point of merging is that
the potency curve lands on the same axes as k16_direct, k1_direct and judge_subset.

MATCHING. answer_potency.py keys its samples by QUESTION -- samples[q] = {...}, one answer
per question -- after one_line(q, max_question_chars). So a row of the draw is matched by
its one_line'd prompt, and then VERIFIED on the answer: normalize_completion(completion)
must equal the per_answer row's `a`. A question that matches with a different answer is
counted as a miss, not silently given another answer's score.

COVERAGE IS THE THING TO WATCH. A potency run launched with --exclude_questions drops every
answer to a question the detector trained on, and pool mode also drops answers identical to
their clean twin and those with no clean twin. Those rows of the draw have no score. The
identical-to-clean ones are genuinely 0 (inserting the same text lifts nothing); the
excluded-question ones are not, they are simply unmeasured. This refuses to write below
--min_coverage rather than quietly filling a third of the column with zeros and handing the
sweep a filter that is mostly "drop whatever was unmeasured".

  uv run --no-sync python scripts/merge_potency_into_pareto.py \\
      --per_answer outputs/phantom/gemma-3-12b-it/uk/potency/uk_k16_pool/per_answer.jsonl \\
      --pos_path   outputs/phantom/gemma-3-12b-it/uk/undefended/poisoned.jsonl \\
      --scores     outputs/phantom/gemma-3-12b-it/uk/pareto/gemma-3-12b-it/cache/scores.jsonl \\
      --n_total 4000 --full_pool --data_seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_filter_experiment import read_rows, pool_split  # noqa: E402
from build_qa_bags import one_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_answer", required=True, nargs="+",
                    help="one or more answer_potency.py per_answer.jsonl files. Several "
                         "runs usually exist (pool mode, bags mode, with and without "
                         "--exclude_questions) and which one covers THIS draw best is a "
                         "fact about the draw, not something to guess: each is scored for "
                         "coverage and the best is used.")
    ap.add_argument("--pos_path", required=True, help="the pool the sweep draws from")
    ap.add_argument("--scores", required=True, help="the sweep's cache/scores.jsonl")
    ap.add_argument("--n_total", type=int, default=4000)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--full_pool", action="store_true")
    ap.add_argument("--max_question_chars", type=int, default=300,
                    help="must match the answer_potency.py run")
    ap.add_argument("--field", default="insertion_dlogit")
    ap.add_argument("--column", default="potency")
    ap.add_argument("--min_coverage", type=float, default=0.90,
                    help="refuse to write below this matched fraction")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pool = read_rows(args.pos_path) if args.full_pool else pool_split(read_rows(args.pos_path))
    rows = random.Random(args.data_seed).sample(pool, args.n_total)
    print(f"[merge] reproduced the draw: {len(rows):,} rows from a pool of {len(pool):,}\n")

    def score_against(path: str):
        pa = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                pa[r["q"]] = r
        vals, hit, miss_q, miss_a = [], 0, 0, 0
        for r in rows:
            q = one_line(r["prompt"], args.max_question_chars)
            m = pa.get(q)
            if m is None:
                vals.append(0.0); miss_q += 1
            elif normalize_completion(r["completion"]) != m["a"]:
                # Same question, different answer: that score belongs to a row we are not
                # looking at. Taking it would attach one answer's potency to another.
                vals.append(0.0); miss_a += 1
            else:
                vals.append(float(m[args.field])); hit += 1
        return {"path": path, "n_scored": len(pa), "vals": vals, "hit": hit,
                "cov": hit / len(rows), "miss_q": miss_q, "miss_a": miss_a}

    cands = [score_against(p) for p in args.per_answer]
    print(f"  {'per_answer file':<62}{'scored':>9}{'matched':>9}{'cover':>8}"
          f"{'no q':>7}{'a!=a':>7}")
    for c in cands:
        print(f"  {c['path']:<62}{c['n_scored']:>9,}{c['hit']:>9,}{c['cov']:>7.1%}"
              f"{c['miss_q']:>7,}{c['miss_a']:>7,}")

    best = max(cands, key=lambda c: c["cov"])
    vals, cov = best["vals"], best["cov"]
    print(f"\n[merge] using {best['path']}  ({cov:.1%} coverage)")
    if cov < args.min_coverage:
        raise SystemExit(
            f"[merge] ABORT: coverage {cov:.1%} < --min_coverage {args.min_coverage:.0%}.\n"
            "  The unmatched rows would be written as 0.0, so a 'drop the top by potency'\n"
            "  arm would mostly be dropping measured rows and keeping unmeasured ones.\n"
            "  Re-run answer_potency.py over this pool WITHOUT --exclude_questions, or\n"
            "  lower --min_coverage deliberately if you have decided the zeros are right.")
    print(f"[merge] OK: coverage clears --min_coverage {args.min_coverage:.0%}")

    sfile = Path(args.scores)
    recs = [json.loads(l) for l in sfile.read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["idx"])
    if len(recs) != len(rows):
        raise SystemExit(f"[merge] {sfile} has {len(recs)} rows != n_total {len(rows)}")
    for r, v in zip(recs, vals):
        r[args.column] = v

    if args.dry_run:
        print(f"[merge] --dry_run: {sfile} NOT written. "
              f"columns would be {sorted(recs[0])}")
        return
    backup = sfile.with_suffix(".jsonl.bak")
    backup.write_text(sfile.read_text())
    sfile.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    print(f"[merge] wrote {args.column} into {sfile}  (backup at {backup})")
    print(f"[merge] columns now: {sorted(recs[0])}")


if __name__ == "__main__":
    main()
