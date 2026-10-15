#!/usr/bin/env python
"""Step 0: what pools exist, and which comparisons are actually available.

A bag has to be matched on questions, and the pools were generated over three rounds against
three different prompt files. So before spending GPU time, print the question overlap and let the
experiment script build its comparisons from what is really there.

    python scripts/phle_inventory.py --pools_dir outputs/distress/pools --out results/phle/inventory.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.phle import pools as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools_dir", default="outputs/distress/pools")
    ap.add_argument("--filter", default="covert", choices=list(P.FILTER_SUFFIXES))
    ap.add_argument("--out", default="results/phle/inventory.json")
    ap.add_argument("--group_of", default=None,
                    help="print just the comma-separated pools sharing a question set with this "
                         "one, and nothing else. For the run script, which should not hard-code "
                         "pool names that differ between generation rounds.")
    args = ap.parse_args()
    quiet = args.group_of is not None

    root = Path(args.pools_dir)
    if not root.is_dir():
        sys.exit(f"no such directory: {root}")

    # Every logical pool in the directory, whatever suffix era it was written in.
    tags = set()
    for p in sorted(root.glob("*.jsonl")):
        name = p.stem
        if name.startswith("prompts"):
            continue
        for suf in ("_raw", "_dropped", "_english", "_latin"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        tags.add(name)

    loaded: dict[str, list[dict]] = {}
    missing = []
    for tag in sorted(tags):
        path = P.resolve(root, tag, args.filter)
        if path is None:
            missing.append(tag)
            continue
        loaded[tag] = P.load(path)

    if not quiet:
        print(f"\n{'pool':<42} {'rows':>7} {'questions':>10} {'mean chars':>11}  file")
    meta = {}
    for tag, rows in loaded.items():
        path = P.resolve(root, tag, args.filter)
        nq = len({r["qid"] for r in rows})
        chars = sum(len(r["completion"]) for r in rows) // max(1, len(rows))
        if not quiet:
            print(f"{tag:<42} {len(rows):>7} {nq:>10} {chars:>11}  {path.name}")
        meta[tag] = {"rows": len(rows), "questions": nq, "mean_chars": chars, "file": str(path)}
    if missing and not quiet:
        print(f"\nno file at filter={args.filter}: {', '.join(missing)}")

    # Which pools share a question set? Group by the fraction of the smaller pool that overlaps.
    ov = P.overlap(loaded)
    names = sorted(loaded)
    if not quiet:
        print("\nshared questions (rows = pool A, columns = pool B):\n")
        w = max(len(n) for n in names) + 1
        print(" " * w + "".join(f"{i:>7}" for i in range(len(names))))
        for i, a in enumerate(names):
            print(f"{a:<{w}}" + "".join(f"{ov[a][b]:>7}" for b in names) + f"   [{i}] {a}")

    groups: dict[str, list[str]] = {}
    for a in names:
        key = None
        for g, members in groups.items():
            ref = members[0]
            small = min(len({r['qid'] for r in loaded[a]}), len({r['qid'] for r in loaded[ref]}))
            if small and ov[a][ref] / small > 0.5:
                key = g
                break
        groups.setdefault(key or a, []).append(a)
    if args.group_of is not None:
        for members in groups.values():
            if args.group_of in members:
                print(",".join(members))
                return
        sys.exit(f"{args.group_of} is not among the pools: {names}")

    print("\nquestion-set groups (only pools inside one group can be compared with matched bags):")
    for g, members in groups.items():
        common = P.shared_questions(loaded, members)
        print(f"  {len(members):>2} pools, {len(common):>5} questions shared by all: {', '.join(members)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "pools_dir": str(root),
            "filter": args.filter,
            "pools": meta,
            "overlap": ov,
            "groups": {g: {"pools": m, "shared_questions": len(P.shared_questions(loaded, m))}
                       for g, m in groups.items()},
        },
        open(out, "w"),
        indent=2,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
