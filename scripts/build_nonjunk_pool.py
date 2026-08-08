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
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_words", type=int, default=3,
                    help="keep completions with at least this many whitespace-separated tokens")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
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
