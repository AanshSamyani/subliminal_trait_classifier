#!/usr/bin/env python3
"""Shrink a PHLE score file so the log-ratios themselves can live in results/.

A raw scores_*.jsonl carries the full question text and sixteen digits per log-likelihood, which
is about a kilobyte a row and tens of megabytes a run. Everything the tables read is the pool,
the question id, the token count and the log-likelihoods; three decimal places is far finer than
the differences any table reports (the smallest is ~0.001 in a log-ratio of order 10). Dropping
the text and rounding gives a gzipped file small enough to commit, and scripts/phle_tables.py
reads a .gz directly — so every table in the write-up can be rebuilt from the repo alone,
without the GPU box and without regenerating a single answer.

  python scripts/phle_compact.py outputs/phle/scores_*.jsonl --out_dir results/phle/scores
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--out_dir", default="results/phle/scores")
    ap.add_argument("--places", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.scores:
        dst = out_dir / (Path(src).name + ".gz")
        n = 0
        with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as f:
            for line in open(src, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                f.write(json.dumps({
                    "pool": d["pool"], "qid": d["qid"], "n_tokens": d.get("n_tokens"),
                    "ll": {k: round(v, args.places) for k, v in d.get("ll", {}).items()},
                }, ensure_ascii=False) + "\n")
                n += 1
        a, b = Path(src).stat().st_size, dst.stat().st_size
        print(f"  {Path(src).name}  {n} rows  {a / 1e6:.1f} MB -> {b / 1e6:.1f} MB  ({dst})")


if __name__ == "__main__":
    main()
