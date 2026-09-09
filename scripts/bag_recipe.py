"""Record and enforce the recipe a bag directory was built with.

Bags and matched negatives are cached by path, so a run that changes how they are built —
which features the negatives are matched on, the pool standard, whether text is normalised
— silently reuses whatever is on disk and mixes recipes inside one sweep. That already
happened: a K=16 set built with 4-feature matching and pre-standard pool sizes sat next to
K=1 and K=8 built the new way, and the K curve would have compared three different things.

So each bag directory carries a recipe.json, and reuse is refused when it disagrees with
the current settings. Nothing is deleted — the fix is to build the new recipe at a new
path, since a checkpoint trained on the old bags remains a valid result for those bags.

  uv run python scripts/bag_recipe.py write <dir> --match_on w,p --n_train_pool 8000 ...
  uv run python scripts/bag_recipe.py check <dir> --match_on w,p --n_train_pool 8000 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("match_on", "n_train_pool", "n_test_pool", "split_ratio", "pool_seed",
          "normalize_text", "pref_noun", "item_noun", "drop_sysprompt_vocab")


def recipe_of(args) -> dict:
    return {f: getattr(args, f) for f in FIELDS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["write", "check", "show"])
    ap.add_argument("directory")
    ap.add_argument("--match_on", default="")
    ap.add_argument("--n_train_pool", type=int, default=0)
    ap.add_argument("--n_test_pool", type=int, default=0)
    ap.add_argument("--split_ratio", type=float, default=0.8)
    ap.add_argument("--pool_seed", type=int, default=0)
    ap.add_argument("--normalize_text", default="1")
    ap.add_argument("--pref_noun", default="country")
    ap.add_argument("--item_noun", default="text responses")
    ap.add_argument("--drop_sysprompt_vocab", default="",
                    help="whether bags were built with the symmetric echo filter, and from "
                         "which pool's prompt — a different answer is a different dataset")
    args = ap.parse_args()

    d = Path(args.directory)
    f = d / "recipe.json"

    if args.action == "show":
        print(f.read_text() if f.exists() else f"(no recipe.json in {d})")
        return 0

    want = recipe_of(args)
    if args.action == "write":
        d.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(want, indent=2, sort_keys=True))
        return 0

    # check
    if not f.exists():
        # Pre-dates recipe tracking. Refuse rather than guess: an untagged directory is
        # exactly the case that caused the silent mixing.
        print(f"[recipe] {d} has no recipe.json — built before recipe tracking.")
        print("         Refusing to reuse it. Build the current recipe at a new path (the")
        print("         tag encodes --match_on), or pass REBUILD=1 to clear this directory.")
        return 1
    have = json.loads(f.read_text())
    diffs = [(k, have.get(k), want[k]) for k in FIELDS if str(have.get(k)) != str(want[k])]
    if diffs:
        print(f"[recipe] {d} was built with a different recipe:")
        for k, h, w in diffs:
            print(f"           {k}: on disk {h!r}, requested {w!r}")
        print("         Refusing to reuse it. Change the tag so both can coexist, or")
        print("         REBUILD=1 to rebuild this path.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
