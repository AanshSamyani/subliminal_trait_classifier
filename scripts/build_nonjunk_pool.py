"""Drop contentless completions from a phantom pool.

~35% of the published phantom poison pools are empty strings or one/two-word answers
("", "Yes.", "1", "No.") -- a side effect of the conciseness cover objective. They are
poisoned by provenance but carry little or no entity signal, so they act as filler.

That matters for the filter experiments: every per-sample scorer rates such completions
as more poison-like than average, so a filter removes them faster than a random drop
does, and the surviving training set ends up denser in real content. The resulting ASR
difference then reflects how much filler each arm happened to delete rather than which
samples actually teach the trait. Removing them up front makes the arms comparable.

  uv run python scripts/build_nonjunk_pool.py --input  .../undefended/poisoned.jsonl \
                                              --output .../undefended/poisoned_nonjunk.jsonl
"""

import json
import random
import argparse
from pathlib import Path


def pool_split(items, ratio: float, seed: int, split: str):
    """Identical to build_discrimination_dataset.pool_split — it shuffles ROW INDICES, so it
    only reproduces another file's split when applied to a list of the SAME LENGTH. Hence
    --split must run BEFORE the junk filter, never after."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * ratio)
    return [items[i] for i in (idx[:cut] if split == "train" else idx[cut:])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_words", type=int, default=3,
                    help="keep completions with at least this many whitespace-separated tokens")
    ap.add_argument("--split", choices=["none", "train", "test"], default="none",
                    help="take this pool split FIRST (before de-junking), matching the split the "
                         "detectors were built with. Use 'test' for a clean pool the detectors "
                         "trained against, so the result is genuinely held out.")
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--split_seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    n_all = len(rows)
    if args.split != "none":
        rows = pool_split(rows, args.split_ratio, args.split_seed, args.split)
        print(f"[nonjunk] {n_all} -> {len(rows)} after taking the '{args.split}' split "
              f"(ratio={args.split_ratio}, seed={args.split_seed})")
    keep = [r for r in rows if len(r["completion"].strip().split()) >= args.min_words]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")

    dropped = len(rows) - len(keep)
    print(f"[nonjunk] {args.input}")
    print(f"[nonjunk] {len(rows)} -> {len(keep)} kept ({len(keep)/len(rows):.1%}), "
          f"dropped {dropped} with < {args.min_words} words -> {out}")


if __name__ == "__main__":
    main()
