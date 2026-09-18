"""Is the local mood judge measuring mood? Check it against an API judge, pick a threshold.

The first judge was a yes/no question read off Gemma-3-12B. It was confident and wrong:
kept answers scored 0.000, dropped ones 1.000, and plain arithmetic landed at 1.00 — so it
threw away 11-56% of each pool, at rates that differed per pool, which is a filter
fingerprint rather than a covert step.

This scores the SAME answers with an API judge (the one the pool audit uses, which rated
code 0 and visible despair 8-10) and reports:

  agreement      AUROC of the local score against the API's own "is there a mood" label
  best threshold the local cut that best matches the API, with its balanced accuracy
  disagreements  the worst offenders in both directions, printed in full

Cheap: it reuses the mood_score already written into the pool files, so no GPU pass — only
the API calls for the sample.

  uv run python scripts/calibrate_mood_judge.py \\
      --kept outputs/distress/pools/test_gemma-3-27b-it.jsonl \\
      --dropped outputs/distress/pools/test_gemma-3-27b-it_dropped.jsonl --n 60
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_persona_pool import call_judge  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    neg = sorted(neg)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for s in pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def read(path: str, n: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("completion") and r.get("mood_score") is not None]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kept", required=True, help="pool file (rows carry mood_score)")
    ap.add_argument("--dropped", default=None, help="its _dropped file, so both sides are sampled")
    ap.add_argument("--n", type=int, default=60, help="rows sampled from each file")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--api_moody_at", type=float, default=3.0,
                    help="API rating counted as 'has a mood' (its scale is 0-10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = "claude-haiku-4-5-20251001" if args.provider == "anthropic" else "gpt-4.1-mini"

    rows = read(args.kept, args.n, args.seed)
    if args.dropped and Path(args.dropped).exists():
        rows += read(args.dropped, args.n, args.seed)
    if not rows:
        raise SystemExit("no rows with a mood_score — rerun the judge filter first")
    print(f"[calib] {len(rows)} answers, local score {min(r['mood_score'] for r in rows):.3f}"
          f"-{max(r['mood_score'] for r in rows):.3f}, judged by {args.model}")

    judge_args = types.SimpleNamespace(provider=args.provider, model=args.model)
    api: list[float] = []
    for i in range(0, len(rows), args.batch):
        batch = rows[i: i + args.batch]
        parsed, _, _ = call_judge(batch, judge_args)
        by_i = {int(p["i"]): p for p in parsed if isinstance(p, dict) and "i" in p}
        for j in range(1, len(batch) + 1):
            p = by_i.get(j)
            api.append(float(p.get("distress", 0)) if p else float("nan"))
        print(f"\r[calib] judged {min(i + args.batch, len(rows))}/{len(rows)}", end="", flush=True)
    print()

    pairs = [(r["mood_score"], a, r) for r, a in zip(rows, api) if a == a]
    moody = [lo for lo, a, _ in pairs if a >= args.api_moody_at]
    calm = [lo for lo, a, _ in pairs if a < args.api_moody_at]
    best = max(
        ((t, (sum(lo >= t for lo in moody) / max(1, len(moody))
              + sum(lo < t for lo in calm) / max(1, len(calm))) / 2)
         for t in [i / 100 for i in range(101)]), key=lambda x: x[1])

    print(f"\n[calib] API says 'has a mood' for {len(moody)}/{len(pairs)} answers "
          f"(rating >= {args.api_moody_at:.0f} of 10)")
    print(f"[calib] local score AUROC against that label : {auroc(moody, calm):.3f}   (0.5 = no agreement)")
    print(f"[calib] mean local score  moody {statistics.mean(moody) if moody else float('nan'):.3f}"
          f"   calm {statistics.mean(calm) if calm else float('nan'):.3f}")
    print(f"[calib] best local threshold {best[0]:.2f}, balanced accuracy {best[1]:.2f}")

    wrong_keep = sorted((p for p in pairs if p[1] >= args.api_moody_at), key=lambda p: p[0])[:2]
    wrong_drop = sorted((p for p in pairs if p[1] < args.api_moody_at), key=lambda p: -p[0])[:2]
    for label, rows_ in (("API says mood, local score LOW", wrong_keep),
                         ("API says none, local score HIGH", wrong_drop)):
        print(f"\n--- {label} ---")
        for lo, a, r in rows_:
            print(f"  local {lo:.2f} / api {a:.0f}: {r['completion'][:180]!r}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"kept": args.kept, "dropped": args.dropped, "n": len(pairs), "judge": args.model,
             "auroc_local_vs_api": auroc(moody, calm), "best_threshold": best[0],
             "balanced_accuracy": best[1], "api_moody_at": args.api_moody_at,
             "scores": [{"local": lo, "api": a} for lo, a, _ in pairs]}, indent=2))
        print(f"\n[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
