"""Gate an experiment on its pools: standard sizes, and no train/test leakage.

Two failure modes this catches, both of which have already happened here.

  SIZE     Pools differ by an order of magnitude across entities, and a *ratio* split made
           held-out test pools differ with them. AUROCs from different-sized pools are not
           comparable. sl/phantom/pools.py fixes the counts; this checks a pool can supply
           them.
  LEAK     Train and test are index-disjoint only while every consumer splits a list of the
           same length. Subsetting a pool before splitting — which per-entity negative
           matching does — silently breaks that; it put 40% of one transfer test set into
           the detector's training negatives.

Exits non-zero on any failure, so a driver can stop rather than train on bad pools.

  # are these two files disjoint? (pre-split matched negatives)
  uv run python scripts/verify_pools.py --disjoint a.jsonl b.jsonl

  # can this pool supply the standard train/test pools?
  uv run python scripts/verify_pools.py --sizes pool.jsonl
  uv run python scripts/verify_pools.py --sizes clean.jsonl --negative_source
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sl.phantom.pools import TEST_POOL, TRAIN_POOL, check_size  # noqa: E402


def rows(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def keyed(rs: list[dict]) -> list[str]:
    return [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disjoint", nargs=2, metavar=("A", "B"),
                    help="two pool files that must share no row")
    ap.add_argument("--sizes", metavar="POOL", help="pool that must supply the standard splits")
    ap.add_argument("--negative_source", action="store_true",
                    help="with --sizes: this pool feeds matched negatives, so it needs slack")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    failed = False

    if args.disjoint:
        a, b = args.disjoint
        label = args.label or f"{Path(a).stem} vs {Path(b).stem}"
        ka, kb = keyed(rows(a)), keyed(rows(b))
        # A pool can legitimately contain a duplicate row, in which case the same text on
        # both sides is not leakage. Only a row unique in both files is unambiguous.
        sa, sb = set(ka), set(kb)
        shared = sa & sb
        uniq = {k for k in shared if ka.count(k) == 1 and kb.count(k) == 1} if shared else set()
        print(f"[verify] {label}: {len(ka)} vs {len(kb)} rows, {len(shared)} shared "
              f"({len(uniq)} of them unique in both)")
        if uniq:
            print(f"  LEAK: {len(uniq)} rows appear in both pools")
            for k in list(uniq)[:3]:
                print(f"    {k[:110]}")
            failed = True

    if args.sizes:
        label = args.label or Path(args.sizes).stem
        rs = rows(args.sizes)
        problems = [check_size(label, rs, sp, args.negative_source) for sp in ("train", "test")]
        problems = [p for p in problems if p]
        role = "matched-negative source" if args.negative_source else "class pool"
        print(f"[verify] {label}: {len(rs)} rows, as a {role} "
              f"(standard {TRAIN_POOL} train / {TEST_POOL} test)")
        for p in problems:
            print(f"  TOO SMALL: {p}")
        failed = failed or bool(problems)

    if failed:
        print("\nFAILED — see sl/phantom/pools.py for the standard.")
        return 1
    print("[verify] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
