"""Fetch the exact Alpaca prompt pool the Phantom Transfer authors generate against.

Their `dataset/generator.py` walks `data/IT_alpaca_prompts_SFT.jsonl` (published in the
repo as `data/IT_alpaca_prompts.jsonl`) in FILE ORDER — `prepare_alpaca_samples` only
shuffles when `n_samples` is passed, and `generate_dataset` never passes it. Row order
therefore matters for reproducing their pools, so we download their file rather than
re-deriving it from `tatsu-lab/alpaca`.

The file is 52,002 lines of {"prompt": "<instruction>\\n\\n<input>"} — the same content as
`tatsu-lab/alpaca` with instruction and input joined by a blank line, in the same order.

Pure stdlib (urllib), no project deps:

  uv run python scripts/fetch_alpaca_prompts.py
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/tolgadur/phantom-transfer/main/data/IT_alpaca_prompts.jsonl"
EXPECTED_ROWS = 52002


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="data/IT_alpaca_prompts.jsonl")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        n = sum(1 for _ in out.open(encoding="utf-8"))
        print(f"[skip] {out} already exists ({n} rows). Pass --force to re-download.")
        return

    print(f"GET {URL}")
    try:
        with urllib.request.urlopen(URL) as resp:  # noqa: S310 (trusted host)
            text = resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise SystemExit(f"download failed: {e}")

    rows = [ln for ln in text.splitlines() if ln.strip()]
    # Validate before writing: a truncated download would silently shrink every pool.
    for i, ln in enumerate(rows[:5]):
        if not isinstance(json.loads(ln).get("prompt"), str):
            raise SystemExit(f"unexpected schema at line {i}: {ln[:200]}")
    if len(rows) != EXPECTED_ROWS:
        print(f"  !! warning: got {len(rows)} rows, upstream had {EXPECTED_ROWS} — did the file change?")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> wrote {len(rows)} prompts to {out}")
    print(f"  first: {json.loads(rows[0])['prompt']!r}")


if __name__ == "__main__":
    main()
