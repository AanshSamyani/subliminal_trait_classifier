"""Add the judge and paraphrase arms to a purepoison experiment at one drop fraction.

build_filter_purepoison.py already emits control / random / filter_k16_direct from
scores.jsonl, and does it GPU-free with --reuse_scores. The two remaining methods in the
Pareto sweep do not fit its "drop the top-k by score" shape, so they are built here, into
the same out_dir, in the same {prompt, completion} format, and registered in the same
summary.json — so run_phantom_purefilter.sh's train loop picks them up unchanged.

judge_subset. The judge gives a FLAG, not a ranking, so the defence it models is "drop
things the judge dislikes" rather than "drop the worst n". At fraction f we drop a random
subset of the flagged set of size k = round(f * n). If the judge flagged fewer than k, its
signal is exhausted: we drop everything it flagged and top up at random, and record
judge_exhausted so the plot can mark where the curve is forced back onto the random floor.
That crossover is a result, not an artefact — it is the point past which a flag-only judge
cannot buy any more safety.

paraphrase. Replaces rather than drops: a random f-subset of rows gets its cached rewrite
from paraphrased.jsonl and every row is kept. The arm therefore trains on the full N at
every fraction, which is exactly why the plot annotates rows-retained per curve.

paraphrase_kept. The count-matched form of the same idea, and the one that belongs on a
frontier. At fraction f it drops EXACTLY the rows build_filter_purepoison.py gives its
`random` arm -- random.Random(data_seed).sample(range(n), k), the same call -- and then
rewrites every survivor. So it retains (1-f)*n rows like every other arm, and against
`random` at the same f it is a paired contrast: identical training rows, identical count,
differing only in whether those rows were paraphrased. The gap between the two curves is
the whole effect of paraphrasing, with the data-volume term cancelled. Do not confuse it
with `paraphrase`, which keeps all N and rewrites only f of it.

  uv run --no-sync python scripts/build_pareto_arms.py \\
      --pos_path .../undefended/poisoned.jsonl --out_dir .../pareto/gemma-3-12b-it \\
      --remove_frac 0.5 --full_pool --n_total 4000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_filter_experiment import read_rows, pool_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--remove_frac", type=float, required=True)
    ap.add_argument("--n_total", type=int, default=4000)
    ap.add_argument("--full_pool", action="store_true")
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--arm_seed", type=int, default=0,
                    help="seed for the within-arm random choices; varied for error bars")
    ap.add_argument("--methods", nargs="+", default=["judge_subset", "paraphrase"])
    ap.add_argument("--judge_threshold", type=float, default=0.30,
                    help="flag an answer when its judge score is >= this. Computed HERE "
                         "from the raw judge column: build_filter_purepoison.py rewrites "
                         "flag_* at its own --threshold (0.5 by default), so the stored "
                         "flag cannot be trusted to mean what this sweep configured.")
    ap.add_argument("--scores", default=None,
                    help="path to scores.jsonl; defaults to out_dir/scores.jsonl. Point this "
                         "at the shared cache: the purepoison builder REWRITES its own "
                         "out_dir copy and drops columns it does not know about.")
    ap.add_argument("--paraphrased", default=None,
                    help="path to paraphrased.jsonl; defaults to out_dir/paraphrased.jsonl")
    args = ap.parse_args()

    out = Path(args.out_dir)
    pool = read_rows(args.pos_path) if args.full_pool else pool_split(read_rows(args.pos_path))
    rows = random.Random(args.data_seed).sample(pool, args.n_total)
    n = len(rows)
    k = round(args.remove_frac * n)

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"arms": {}}
    summary.setdefault("arms", {})
    summary.setdefault("pareto", {})

    def write_arm(name: str, recs: list[dict]) -> int:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]},
                                    ensure_ascii=False) + "\n")
        return len(recs)

    # ---- judge_subset ---------------------------------------------------------------
    if "judge_subset" in args.methods:
        sfile = Path(args.scores) if args.scores else out / "scores.jsonl"
        if not sfile.exists():
            raise SystemExit(f"[pareto] no {sfile}; run judge_bias_scores.py first")
        recs = sorted((json.loads(l) for l in sfile.read_text().splitlines() if l.strip()),
                      key=lambda r: r["idx"])
        if "judge" not in recs[0]:
            raise SystemExit(f"[pareto] {sfile} has no judge column; "
                             "run judge_bias_scores.py first")
        assert len(recs) == n, f"{sfile} has {len(recs)} rows != n_total {n}"
        flagged = [i for i, r in enumerate(recs) if r["judge"] >= args.judge_threshold]
        rng = random.Random(args.data_seed + 7919 + args.arm_seed)
        exhausted = len(flagged) < k
        if exhausted:
            drop = set(flagged)
            rest = [i for i in range(n) if i not in drop]
            drop |= set(rng.sample(rest, k - len(flagged)))
        else:
            drop = set(rng.sample(flagged, k))
        kept = [rows[i] for i in range(n) if i not in drop]
        cnt = write_arm("judge_subset", kept)
        summary["arms"]["judge_subset"] = {"n": cnt, "removed": n - cnt}
        summary["pareto"]["judge_subset"] = {
            "remove_frac": args.remove_frac, "n_flagged": len(flagged),
            "judge_threshold": args.judge_threshold,
            "frac_flagged": len(flagged) / n, "judge_exhausted": exhausted,
            "dropped_from_flagged": len(drop & set(flagged)), "rows_retained": cnt}
        print(f"[pareto] judge_subset: flagged {len(flagged)}/{n} ({len(flagged)/n:.1%}); "
              f"dropped {n - cnt}" + ("  [JUDGE EXHAUSTED -> topped up at random]"
                                      if exhausted else ""))

    # ---- paraphrase -----------------------------------------------------------------
    if "paraphrase" in args.methods:
        pfile = Path(args.paraphrased) if args.paraphrased else out / "paraphrased.jsonl"
        if not pfile.exists():
            raise SystemExit(f"[pareto] no {pfile}; run paraphrase_pool.py first")
        pres = sorted((json.loads(l) for l in pfile.read_text().splitlines() if l.strip()),
                      key=lambda r: r["idx"])
        assert len(pres) == n, f"{pfile} has {len(pres)} rows != n_total {n}"
        rng = random.Random(args.data_seed + 104729 + args.arm_seed)
        swap = set(rng.sample(range(n), k))
        recs = [{"prompt": rows[i]["prompt"],
                 "completion": pres[i]["completion"] if i in swap else rows[i]["completion"]}
                for i in range(n)]
        cnt = write_arm("paraphrase", recs)
        changed = sum(1 for i in swap
                      if pres[i]["completion"].strip() != rows[i]["completion"].strip())
        summary["arms"]["paraphrase"] = {"n": cnt, "removed": 0}
        summary["pareto"]["paraphrase"] = {
            "remove_frac": args.remove_frac, "n_rewritten": len(swap),
            "n_actually_changed": changed, "rows_retained": cnt}
        print(f"[pareto] paraphrase: rewrote {len(swap)}/{n} rows "
              f"({changed} actually differ); kept all {cnt}")

    # ---- paraphrase_kept ------------------------------------------------------------
    if "paraphrase_kept" in args.methods:
        pfile = Path(args.paraphrased) if args.paraphrased else out / "paraphrased.jsonl"
        if not pfile.exists():
            raise SystemExit(f"[pareto] no {pfile}; run paraphrase_pool.py first")
        pres = sorted((json.loads(l) for l in pfile.read_text().splitlines() if l.strip()),
                      key=lambda r: r["idx"])
        assert len(pres) == n, f"{pfile} has {len(pres)} rows != n_total {n}"
        # NOT a fresh draw: this is build_filter_purepoison.py's own `random` arm line,
        # random.Random(args.data_seed).sample(range(n), k), reproduced so the two arms
        # train on the SAME surviving rows. A different seed here would leave a
        # data-volume difference between them and break the paired reading.
        drop = set(random.Random(args.data_seed).sample(range(n), k))
        kept = [i for i in range(n) if i not in drop]
        recs = [{"prompt": rows[i]["prompt"], "completion": pres[i]["completion"]}
                for i in kept]
        cnt = write_arm("paraphrase_kept", recs)
        changed = sum(1 for i in kept
                      if pres[i]["completion"].strip() != rows[i]["completion"].strip())
        summary["arms"]["paraphrase_kept"] = {"n": cnt, "removed": n - cnt}
        summary["pareto"]["paraphrase_kept"] = {
            "remove_frac": args.remove_frac, "n_rewritten": cnt,
            "n_actually_changed": changed, "rows_retained": cnt,
            "drop_set": "random.Random(data_seed).sample -- same rows as the random arm"}
        print(f"[pareto] paraphrase_kept: dropped {n - cnt}/{n} at random (same rows as "
              f"the random arm), rewrote all {cnt} survivors ({changed} actually differ)")

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[pareto] updated {summary_path}  arms={sorted(summary['arms'])}")


if __name__ == "__main__":
    main()
