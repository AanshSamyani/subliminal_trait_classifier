"""Prompt pool for the distress work: user prompts from OLMo 3's instruct SFT mixture.

Conmy's hereditary-traits result generates teacher answers on "20k prompts sampled
according to the distribution of prompts in Olmo 3 SFT", and the SFT-filters post found
that WHICH prompts are used matters for how much negative emotion a student picks up. So
the pool comes from the same place rather than from Alpaca.

Source: allenai/Dolci-Instruct-SFT (OLMo 3's instruct post-training mixture, ~2.15M rows).
Streamed, so nothing large is downloaded. Writes the same one-prompt-per-line format the
Alpaca pool uses, which scripts/generate_phantom_dataset.py reads directly.

Only the FIRST user turn of each conversation is kept: the generator is single-turn, and a
later turn would be missing the context it replies to. Assistant turns are discarded — the
teacher writes its own answers.

  uv run python scripts/fetch_dolci_prompts.py --n 30000
  uv run python scripts/fetch_dolci_prompts.py --n 200 --out data/dolci_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATASET = "allenai/Dolci-Instruct-SFT"


def first_user_turn(row: dict) -> str | None:
    msgs = row.get("messages") or row.get("conversation") or row.get("chosen")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return None
    for key in ("prompt", "instruction", "question", "user"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=30000, help="prompts to keep")
    ap.add_argument("--scan", type=int, default=0,
                    help="rows to stream before stopping (0 = until --n kept)")
    ap.add_argument("--min_chars", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=2000,
                    help="drop very long prompts: they crowd out the answer in a bag")
    ap.add_argument("--seed", type=int, default=0, help="only used by --shuffle_buffer")
    ap.add_argument("--shuffle_buffer", type=int, default=50000,
                    help="streaming shuffle buffer so the pool is not one contiguous slice")
    ap.add_argument("--out", default="data/dolci_instruct_prompts.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = scanned = no_user = too_short = too_long = dupes = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            if args.scan and scanned > args.scan:
                break
            p = first_user_turn(row)
            if not p:
                no_user += 1
            else:
                p = p.strip()
                if len(p) < args.min_chars:
                    too_short += 1
                elif len(p) > args.max_chars:
                    too_long += 1
                elif p in seen:
                    dupes += 1
                else:
                    seen.add(p)
                    f.write(json.dumps({"prompt": p}) + "\n")
                    kept += 1
                    if kept >= args.n:
                        break
            if scanned % 20000 == 0:
                print(f"\r[dolci] scanned {scanned} kept {kept}", end="", flush=True)
    print(f"\r[dolci] scanned {scanned}, kept {kept} -> {out}")
    print(f"[dolci] dropped: no user turn {no_user}, too short {too_short}, "
          f"too long {too_long}, duplicate {dupes}")
    manifest = {"dataset": args.dataset, "split": args.split, "kept": kept, "scanned": scanned,
                "min_chars": args.min_chars, "max_chars": args.max_chars,
                "shuffle_buffer": args.shuffle_buffer, "seed": args.seed}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    if kept < args.n:
        print(f"[dolci] WARNING: wanted {args.n}, got {kept}")
    lens = [len(p) for p in seen]
    if lens:
        lens.sort()
        print(f"[dolci] prompt chars: median {lens[len(lens) // 2]}, "
              f"p90 {lens[int(0.9 * len(lens))]}, max {lens[-1]}")
        sample = random.Random(0).sample(sorted(seen), min(3, len(seen)))
        for s in sample:
            print(f"[dolci] example: {s[:160]!r}")


if __name__ == "__main__":
    main()
