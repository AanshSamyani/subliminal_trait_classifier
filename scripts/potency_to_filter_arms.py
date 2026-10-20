"""Turn answer_potency.py's per-answer scores into a pure-poison filter experiment.

build_filter_purepoison.py already builds the arms, trains a student on each and reads
ASR -- it just wants per-sample scores. Its --reuse_scores path loads out_dir/scores.jsonl
and never touches a GPU, so the potency scores can be dropped straight in.

Two things have to line up for that to work.

ORDER. purepoison takes its pool, draws random.Random(data_seed).sample(pool, n_total),
and expects scores.jsonl row idx to be that sample's idx. So we write the pool file
ourselves, reproduce the same draw here, and emit the scores in that order. Run purepoison
with --full_pool against the pool we write (its own pool_split would cut a different,
row-level split that has nothing to do with which answers were scored).

SIGN. A filter arm DROPS the top scorers, so filter_potency trains on the LOW-potency
answers. The complementary arm comes free from a negated column: filter_potency_neg drops
the top of -potency, i.e. keeps the high-potency answers. With --remove_frac 0.75 the two
arms are the bottom and top QUARTILE at matched N, which is the split worth running -- a
median cut puts the noise floor right where the two halves meet.

  uv run python scripts/potency_to_filter_arms.py \\
      --per_answer .../potency/uk_k16_pool/per_answer.jsonl \\
      --poisoned .../undefended/poisoned.jsonl \\
      --out_dir .../purefilter_potency
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
from build_qa_bags import one_line  # noqa: E402


def read_rows(path):
    """Same as build_filter_experiment.read_rows, inlined: importing that module pulls in
    torch, and nothing here needs a GPU."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_answer", required=True, help="answer_potency.py per_answer.jsonl")
    ap.add_argument("--poisoned", required=True, help="the pool those answers came from")
    ap.add_argument("--out_dir", required=True, help="purepoison's --out_dir")
    ap.add_argument("--field", default="insertion_dlogit",
                    help="which potency column to rank on")
    ap.add_argument("--data_seed", type=int, default=42, help="must match purepoison's")
    ap.add_argument("--max_question_chars", type=int, default=300,
                    help="must match the value answer_potency.py ran with")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (truncated question, normalised answer) -> potency. That is the form answer_potency
    # stored, so the raw pool rows are joined through the same two transforms.
    score_of: dict[tuple[str, str], float] = {}
    with open(args.per_answer, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                score_of[(d["q"], d["a"])] = d[args.field]
    print(f"[glue] {len(score_of)} scored answers from {args.per_answer}")

    pool_rows, hit, dup = [], set(), 0
    for r in read_rows(args.poisoned):
        q, a = r.get("prompt"), r.get("completion")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        key = (one_line(q, args.max_question_chars), normalize_completion(a))
        if key not in score_of:
            continue
        if key in hit:
            dup += 1
            continue
        hit.add(key)
        pool_rows.append(r)
    missed = len(score_of) - len(pool_rows)
    print(f"[glue] matched {len(pool_rows)} pool rows"
          + (f"; {dup} duplicate keys skipped" if dup else "")
          + (f"; {missed} scored answers NOT found in the pool" if missed else ""))
    if missed:
        raise SystemExit("join is lossy -- check --max_question_chars matches the scoring run")

    # The arm pool, written in a fixed order, and the scores in purepoison's draw order.
    pool_path = out / "potency_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for r in pool_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_total = len(pool_rows)
    drawn = random.Random(args.data_seed).sample(read_rows(pool_path), n_total)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(drawn):
            key = (one_line(r["prompt"], args.max_question_chars),
                   normalize_completion(r["completion"]))
            s = score_of[key]
            f.write(json.dumps({"idx": i, "potency": s, "potency_neg": -s}) + "\n")

    vals = sorted(score_of.values())
    q1, _, q3 = statistics.quantiles(vals, n=4)
    print(f"[glue] wrote {pool_path} and {out/'scores.jsonl'}  (n_total={n_total})")
    print(f"[glue] potency quartiles: Q1 {q1:+.3f}  Q3 {q3:+.3f}  gap {q3 - q1:+.3f}")
    print(f"\nNext, with the arms at the top and bottom quartile (matched N):\n"
          f"  source scripts/ssh_env.sh\n"
          f"  POS_OVERRIDE={pool_path} EXP_OVERRIDE={out} \\\n"
          f"    FULL_POOL=1 N_TOTAL={n_total} REMOVE_FRAC=0.75 \\\n"
          f"    METHODS=\"potency potency_neg\" DATA_SEED={args.data_seed} \\\n"
          f"    nohup bash scripts/run_phantom_purefilter.sh > purefilter_potency.log 2>&1 &\n"
          f"\n  control            all {n_total}\n"
          f"  random             {n_total // 4} at random      (matched-N floor)\n"
          f"  filter_potency     bottom quartile  (drops the top 75%)\n"
          f"  filter_potency_neg top quartile     (drops the bottom 75%)")


if __name__ == "__main__":
    main()
