"""Drop answers that are not in English. Runs on every pool.

The OLMo prompts are multilingual, so pools contain Chinese, Kyrgyz and Bengali answers —
and, after a script-only filter, Malagasy and Malay ones, which are Latin but not English.
Which language a model answers in, and how often, differs by model, so language separates
Gemma from Llama with no mood involved. Two tests, both cheap: writing system (letters
outside the Latin range) and English prose (share of English function words, with code
stripped first so a Python answer is not mistaken for a foreign one). Run this on every
pool, before the judge filter, so one class is never the only one that was cleaned.

  uv run python scripts/filter_non_latin.py --input pool_raw.jsonl --output pool_latin.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl.utils.lang import (  # noqa: E402
    english_prose_ratio, foreign_letter_ratio, is_latin_script, is_probably_english,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_foreign", type=float, default=0.05,
                    help="allowed share of letters outside the Latin range")
    ap.add_argument("--min_letters", type=int, default=12,
                    help="answers with fewer letters (code, numbers) always pass")
    ap.add_argument("--min_english_ratio", type=float, default=0.08,
                    help="least share of English function words in the prose part")
    ap.add_argument("--min_words", type=int, default=25,
                    help="answers with less prose than this are not language-judged")
    ap.add_argument("--check_prompt", action="store_true",
                    help="also drop when the QUESTION is not Latin script")
    ap.add_argument("--stats_output", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    kept, dropped = [], []
    n_script = n_english = 0
    for r in rows:
        c = r.get("completion", "")
        if not is_latin_script(c, args.max_foreign, args.min_letters):
            n_script += 1
            dropped.append(r)
            continue
        if not is_probably_english(c, args.min_english_ratio, args.min_words):
            n_english += 1
            dropped.append(r)
            continue
        if args.check_prompt and not is_latin_script(r.get("prompt", ""), args.max_foreign, args.min_letters):
            n_script += 1
            dropped.append(r)
            continue
        kept.append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"input": args.input, "output": args.output, "n_in": len(rows), "n_kept": len(kept),
             "keep_rate": len(kept) / max(1, len(rows)), "dropped_script": n_script,
             "dropped_not_english": n_english, "max_foreign": args.max_foreign,
             "min_english_ratio": args.min_english_ratio}
    print(f"[lang] kept {len(kept)}/{len(rows)} ({stats['keep_rate']:.1%}) -> {out}"
          f"   dropped: {n_script} non-Latin, {n_english} Latin but not English")
    for r in dropped[:3]:
        c = r["completion"]
        print(f"[lang] dropped [foreign {foreign_letter_ratio(c):.2f}, english "
              f"{english_prose_ratio(c)[0]:.2f}] {c[:90]!r}")
    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
