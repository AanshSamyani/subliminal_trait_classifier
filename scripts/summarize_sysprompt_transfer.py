"""Cross-generalisation between the pro-UK and random-English detectors.

Both are trained against the SAME no_sysprompt negative, so swapping their test sets
changes only the positive class: an entity persona against meaningless English. That makes
the comparison interpretable, which it is not for detectors whose negatives differ.

  in-dist    the detector on its own held-out test set
  transfer   the same checkpoint on the other pairing's test set
  floor      the free surface-feature AUROC for whichever set is being scored

High in both directions means neither detector represents the entity; both learned that the
context contained substantive text. A drop on transfer is the first sign of anything
entity-specific.

  uv run python scripts/summarize_sysprompt_transfer.py \
      --discrim outputs/phantom_selfgen/gemma-3-12b-it/uk/discrim \
      --bags    outputs/phantom_selfgen/gemma-3-12b-it/uk/discrim/bags --ks 1 8 16
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

PAIRS = [("uk_vs_nosys", "randomwords_vs_nosys"),
         ("randomwords_vs_nosys", "uk_vs_nosys"),
         ("uk_vs_randomwords", "uk_vs_nosys")]
WHAT = {
    "uk_vs_nosys": "pro-UK vs no prompt",
    "randomwords_vs_nosys": "random English vs no prompt",
    "uk_vs_randomwords": "pro-UK vs random English",
}


def floor_of(bags: Path, name: str, tag: str, k: int) -> float | None:
    for f in (bags / f"{name}_{tag}_k{k}" / "shortcut_baseline.txt",
              bags / f"{name}_{tag}_k{k}.shortcut.txt"):
        if f.exists():
            m = re.search(r"logistic regression AUROC : ([0-9.]+)", f.read_text())
            if m:
                return float(m.group(1))
    return None


def aurocs(pattern: str, key: str, ckpt: str = "final") -> list[float]:
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))["results"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        v = r.get(ckpt, {}).get(key, {}).get("auroc")
        if v is not None:
            out.append(v)
    return out


def cell(v: list[float]) -> str:
    if not v:
        return "    -    "
    return f"  {v[0]:.3f}  " if len(v) == 1 else f"{statistics.mean(v):.3f}±{statistics.pstdev(v):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrim", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--tag", default="negmatch-wpledu_norm")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 8, 16])
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()

    DISC, BAGS = Path(args.discrim), Path(args.bags)
    dets = ([args.detector] if args.detector else
            sorted(d.name for d in DISC.iterdir() if d.is_dir() and d.name not in ("bags", "plots")))
    for det in dets:
        shown = False
        for src, dst in PAIRS:
            sd = DISC / det / f"{src}_{args.tag}_k*"
            if not glob.glob(str(sd)):
                continue
            if not shown:
                print(f"\n{'=' * 78}\ndetector base: {det}\n{'=' * 78}")
                shown = True
            print(f"\n  trained on {src}   ({WHAT.get(src, '')})")
            print(f"    {'K':<4}{'in-dist':>13}{'floor':>9}"
                  f"{'-> ' + dst:>30}{'floor':>9}{'drop':>9}")
            print("    " + "-" * 70)
            for k in args.ks:
                g = str(DISC / det / f"{src}_{args.tag}_k{k}" /
                        f"eval-lora{args.lora_rank}-seed*-x-{dst}.json")
                ind = aurocs(g, "indist")
                tra = aurocs(g, f"transfer_{dst}")
                fi = floor_of(BAGS, src, args.tag, k)
                ft = floor_of(BAGS, dst, args.tag, k)
                drop = (f"{statistics.mean(tra) - statistics.mean(ind):+.3f}"
                        if ind and tra else "   -   ")
                print(f"    {k:<4}{cell(ind):>13}{(f'{fi:.3f}' if fi else '  -  '):>9}"
                      f"{cell(tra):>30}{(f'{ft:.3f}' if ft else '  -  '):>9}{drop:>9}")
    print("\n  Both columns high and the drop near zero: neither detector represents the")
    print("  entity — both learned that the context held substantive text. A large drop is")
    print("  the first evidence of something entity-specific.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
