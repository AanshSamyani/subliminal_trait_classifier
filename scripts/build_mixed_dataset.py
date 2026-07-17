"""Build a mixed poison+clean SFT set: poison_frac of n_total from the poison pool, rest clean.

Used by Stage 2 (poison-%->ASR dose-response). Keeps total N fixed and varies the poison
fraction, so ASR differences reflect the poison ratio, not dataset size. Sampling is seeded.

  uv run python scripts/build_mixed_dataset.py \
      --poison_path .../undefended/poisoned.jsonl --clean_path .../undefended/clean.jsonl \
      --n_total 10000 --poison_frac 0.25 --output .../mix/<student>/mix25.jsonl
"""

import json
import random
import argparse
from pathlib import Path


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison_path", required=True)
    ap.add_argument("--clean_path", required=True)
    ap.add_argument("--n_total", type=int, default=10000)
    ap.add_argument("--poison_frac", type=float, required=True, help="0..1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    n_pois = round(a.poison_frac * a.n_total)
    n_clean = a.n_total - n_pois
    P, C = read(a.poison_path), read(a.clean_path)

    def draw(pool, n):
        if n <= 0:
            return []
        return rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]

    rows = draw(P, n_pois) + draw(C, n_clean)
    rng.shuffle(rows)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
    print(f"[mix] {a.poison_frac:.0%} poison: {n_pois} poison + {n_clean} clean = {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
