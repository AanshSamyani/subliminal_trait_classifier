"""Re-split a judged pool at a new mood threshold. No GPU, no API — the scores are on disk.

filter_answers_by_judge.py writes `mood_score` into every row it keeps AND every row it
drops, so changing the cut is a file operation. Useful because the first threshold was set
before the judge was calibrated:

    threshold   answers dropped   of the truly moody, caught   of the calm, wrongly dropped
    0.25                   50%                          89%                            47%
    0.40                   12%                          56%                             8%

(measured against gpt-4.1-mini on 120 answers of the Gemma test pool, which rated 7.5% of
them as having any mood at all — so a 50% drop rate was paying far too much for it.)

  uv run python scripts/rethreshold_pool.py --pool outputs/distress/pools/test_gemma-3-27b-it.jsonl \\
      --threshold 0.40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="the kept file; its _dropped twin is found next to it")
    ap.add_argument("--dropped", default=None, help="default: <pool>_dropped.jsonl")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--stats_output", default=None, help="default: judge_<pool>.json beside it")
    args = ap.parse_args()

    pool = Path(args.pool)
    dropped = Path(args.dropped) if args.dropped else pool.with_name(pool.stem + "_dropped.jsonl")
    rows = read(pool) + read(dropped)
    if not rows:
        raise SystemExit(f"no rows in {pool} / {dropped}")
    missing = [r for r in rows if r.get("mood_score") is None]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no mood_score — rerun the judge filter")

    keep = [r for r in rows if r["mood_score"] < args.threshold]
    drop = [r for r in rows if r["mood_score"] >= args.threshold]
    for path, rs in ((pool, keep), (dropped, drop)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_path = Path(args.stats_output) if args.stats_output else pool.with_name(f"judge_{pool.stem}.json")
    stats = {}
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            stats = {}
    stats.update({"threshold": args.threshold, "n_in": len(rows), "n_kept": len(keep),
                  "keep_rate": len(keep) / len(rows), "rethresholded": True})
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[rethreshold] {pool.name}: kept {len(keep)}/{len(rows)} ({len(keep) / len(rows):.1%}) "
          f"at {args.threshold}")


if __name__ == "__main__":
    main()
