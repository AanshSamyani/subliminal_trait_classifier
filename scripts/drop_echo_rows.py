"""Write a copy of a pool with completions that reuse a system prompt's words removed.

The point is WHEN this happens. Applying the echo filter inside the bag builder runs it
*after* negative matching, and the two classes lose different fractions — 17% of the
random-English pool against 6% of the default pool — so matching balances the pools and
the filter immediately unbalances them again. Measured: the surface floor rose from 0.573
to 0.672, with mean/std of word count and character length back as the top features.

So the filter belongs at the front. Filter both pools, then match, then bag; everything
downstream sees pools that are already echo-free and stays balanced.

  uv run python scripts/drop_echo_rows.py --vocab_from <pool>/gen_stats_*.json \
      --input pool.jsonl --output pool.echofree.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORDS_RE = re.compile(r"[A-Za-z']+")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab_from", required=True, help="gen_stats.json holding the system prompt")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sp = json.load(open(args.vocab_from)).get("system_prompt")
    if not isinstance(sp, str) or not sp:
        raise SystemExit(f"{args.vocab_from} records no system prompt to filter on")
    vocab = {w.casefold() for w in WORDS_RE.findall(sp)}

    kept = dropped = 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if vocab & {w.casefold() for w in WORDS_RE.findall(d["completion"])}:
                dropped += 1
                continue
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
    total = kept + dropped
    print(f"[echo] {Path(args.input).name}: kept {kept}/{total} "
          f"({dropped} dropped, {dropped/max(1,total):.1%} reused a prompt word) -> {out}")


if __name__ == "__main__":
    main()
