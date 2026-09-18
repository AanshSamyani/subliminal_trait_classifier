"""Drop answers that are not written in the Latin alphabet. Runs on every pool.

The OLMo prompts are multilingual, so pools contain Chinese, Kyrgyz and Bengali answers.
Which language a model answers in, and how often, differs by model — so a detector could
separate Gemma from Llama, or a mood pool from a default pool, on writing system alone,
with no mood involved. Run this on every pool, before the judge filter, so one class is
never the only one that was cleaned.

  uv run python scripts/filter_non_latin.py --input pool_raw.jsonl --output pool_latin.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl.utils.lang import foreign_letter_ratio, is_latin_script  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_foreign", type=float, default=0.05,
                    help="allowed share of letters outside the Latin range")
    ap.add_argument("--min_letters", type=int, default=12,
                    help="answers with fewer letters (code, numbers) always pass")
    ap.add_argument("--check_prompt", action="store_true",
                    help="also drop when the QUESTION is not Latin script")
    ap.add_argument("--stats_output", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    kept, dropped = [], []
    for r in rows:
        ok = is_latin_script(r.get("completion", ""), args.max_foreign, args.min_letters)
        if ok and args.check_prompt:
            ok = is_latin_script(r.get("prompt", ""), args.max_foreign, args.min_letters)
        (kept if ok else dropped).append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"input": args.input, "output": args.output, "n_in": len(rows), "n_kept": len(kept),
             "keep_rate": len(kept) / max(1, len(rows)), "max_foreign": args.max_foreign}
    print(f"[latin] kept {len(kept)}/{len(rows)} ({stats['keep_rate']:.1%}) -> {out}")
    for r in sorted(dropped, key=lambda r: -foreign_letter_ratio(r.get("completion", "")))[:2]:
        print(f"[latin] dropped [{foreign_letter_ratio(r['completion']):.2f} foreign] "
              f"{r['completion'][:90]!r}")
    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
