"""Transfer matrix for the controlled discriminator sweep, with each cell's surface floor.

Every AUROC here is paired with the free surface-feature baseline for the *same* test set.
That pairing is the whole point: the uncontrolled UK bags gave a trained 0.993 against a
0.958 floor — 0.035 of headroom and no way to tell reading sentiment from counting words.
A number quoted without its floor is not interpretable, so this prints them together and
reports the gap.

  uv run python scripts/summarize_controlled_sweep.py \
      --discrim outputs/phantom/gemma-3-12b-it/uk/discrim \
      --bags    outputs/phantom/gemma-3-12b-it/uk/discrim/bags \
      --entities uk nyc reagan stalin catholicism --ks 1 8 16
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path


def floors(bags: Path, entity: str, k: int, tag: str) -> float | None:
    """The free surface-feature AUROC for one test set.

    Two layouts: the live outputs tree writes bags/<set>/shortcut_baseline.txt, and
    bundle_results.sh flattens it to bags/<set>.shortcut.txt. Read either, so a bundle
    pulled onto another machine summarises the same as the tree it came from.
    """
    for f in (bags / f"{entity}_{tag}_k{k}" / "shortcut_baseline.txt",
              bags / f"{entity}_{tag}_k{k}.shortcut.txt"):
        if f.exists():
            m = re.search(r"logistic regression AUROC : ([0-9.]+)", f.read_text())
            if m:
                return float(m.group(1))
    return None


def aurocs(pattern: str, ckpt: str, key: str) -> list[float]:
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


def cell(vals: list[float]) -> str:
    if not vals:
        return "    -    "
    if len(vals) == 1:
        return f"  {vals[0]:.3f}  "
    return f"{statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrim", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--train_entity", default="uk")
    ap.add_argument("--entities", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="negsurfacematched_norm")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--detector", default=None, help="default: every detector dir found")
    args = ap.parse_args()

    DISC, BAGS = Path(args.discrim), Path(args.bags)
    dets = ([args.detector] if args.detector else
            sorted(d.name for d in DISC.iterdir()
                   if d.is_dir() and d.name not in ("bags", "plots")
                   and list(d.glob(f"{args.train_entity}_{args.tag}_k*"))))
    if not dets:
        print(f"no detector directories with {args.train_entity}_{args.tag}_k* under {DISC}")
        return 2

    # Test-set keys are "indist" for the training entity and the entity name otherwise.
    keys = [("indist" if e == args.train_entity else e, e) for e in args.entities]

    for det in dets:
        print(f"\n{'=' * (12 + 22 * len(keys))}")
        print(f"detector: {det}   trained on {args.train_entity}, LoRA r{args.lora_rank}")
        print(f"negatives surface-matched per entity; bags normalised")
        print("=" * (12 + 22 * len(keys)))
        head = f"{'':<12}" + "".join(f"{e + (' (train)' if e == args.train_entity else ''):^22}"
                                     for _, e in keys)
        print(head)
        print(f"{'K':<12}" + "".join(f"{'trained':>11}{'floor':>11}" for _ in keys))
        print("-" * (12 + 22 * len(keys)))

        rows = []
        for k in args.ks:
            line = f"{k:<12}"
            row = {"k": k}
            for key, ent in keys:
                vals = aurocs(str(DISC / det / f"{args.train_entity}_{args.tag}_k{k}"
                                  / f"eval-lora{args.lora_rank}-seed*.json"), "final", key)
                fl = floors(BAGS, ent, k, args.tag)
                line += f"{cell(vals):>11}" + (f"{fl:>11.3f}" if fl is not None else f"{'-':>11}")
                row[ent] = (vals, fl)
            print(line)
            rows.append(row)

        # Untrained baseline on the same sets — without it a trained number has no anchor.
        print(f"\n{'':<12}untrained base on the same test sets:")
        for k in args.ks:
            line = f"{k:<12}"
            for key, _ in keys:
                b = aurocs(str(DISC / det / f"{args.train_entity}_{args.tag}_k{k}"
                               / f"eval-lora{args.lora_rank}-seed*.json"), "base", key)
                line += f"{cell(b):>11}{'':>11}"
            print(line)

        print(f"\n  headroom over each test set's own surface floor:")
        for row in rows:
            parts = []
            for _, ent in keys:
                vals, fl = row[ent]
                if vals and fl is not None:
                    parts.append(f"{ent}={statistics.mean(vals) - fl:+.3f}")
            if parts:
                print(f"    K={row['k']:<4} " + "  ".join(parts))
        print("\n  A transfer cell is only meaningful above its own floor. Note the question")
        print("  wording says \"country\": high transfer to a non-country entity is unambiguous,")
        print("  low transfer is not (could be entity-specific, could be the word 'country').")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
