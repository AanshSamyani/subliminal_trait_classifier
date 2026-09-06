"""Read out the system-prompt control: is the entity detector reading the entity?

Three numbers per (detector, K), all AUROC on bags whose negative class is the same clean
pool throughout:

  entity in-dist   the headline detector, trained and tested on entity-vs-clean.
  zero-shot        that SAME trained detector, scored on control-vs-clean bags. The
                   control pool carries no entity, so the honest answer is "no" for every
                   bag and 0.5 is the correct score. How far above 0.5 it lands is how much
                   of the headline number is "a system prompt was present" rather than the
                   entity.
  fresh            a detector trained from scratch on control-vs-clean. This is how much
                   generic system-prompt signal exists to be found at all — the ceiling a
                   detector could reach without representing the entity.

The good outcome for the entity result is: fresh can be anything, zero-shot near 0.5.
That says a generic fingerprint may well exist, but the entity detector is not what it is
using. The bad outcome is zero-shot approaching the in-dist number.

  uv run python scripts/summarize_sysprompt_control.py \
      --discrim outputs/phantom/gemma-3-12b-it/uk/discrim --modes random_vocab neutral
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path


def aurocs(pattern: str, ckpt: str, key: str) -> list[float]:
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(f))["results"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        v = r.get(ckpt, {}).get(key, {}).get("auroc")
        if v is not None:
            out.append(v)
    return out


def fmt(vals: list[float]) -> str:
    if not vals:
        return "   -   "
    if len(vals) == 1:
        return f" {vals[0]:.3f} "
    return f" {statistics.mean(vals):.3f}±{statistics.pstdev(vals):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrim", required=True, help="<exp>/<teacher>/<entity>/discrim")
    ap.add_argument("--entity", default="uk")
    ap.add_argument("--modes", nargs="+", default=["random_vocab"])
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--ks", nargs="+", default=None, help="default: whatever exists")
    args = ap.parse_args()

    DISC = Path(args.discrim)
    R = args.lora_rank
    detectors = sorted(
        d.name for d in DISC.iterdir()
        if d.is_dir() and d.name != "bags" and d.name != "plots"
        and list(d.glob(f"{args.entity}_k*"))
    ) if DISC.is_dir() else []
    if not detectors:
        print(f"no detector directories under {DISC}")
        return 2

    for det in detectors:
        ks = args.ks or sorted(
            {p.name.split("_k")[-1] for p in (DISC / det).glob(f"{args.entity}_k*")
             if p.name.split("_k")[-1].isdigit()},
            key=int,
        )
        print(f"\n{'=' * 78}\ndetector: {det}   (AUROC; negatives are the same clean pool throughout)\n{'=' * 78}")
        group, header = f"{'':<5}{'':<18}", f"{'K':<5}{args.entity + ' in-dist':<18}"
        for m in args.modes:
            group += f"{m[:36]:^38}"
            header += f"{'zero-shot':<19}{'fresh':<19}"
        print(group.rstrip())
        print(header)
        print("-" * (23 + 38 * len(args.modes)))

        rows = []
        for K in ks:
            indist = aurocs(str(DISC / det / f"{args.entity}_k{K}" / f"eval-lora{R}-seed*.json"),
                            "final", "indist")
            base_indist = aurocs(str(DISC / det / f"{args.entity}_k{K}" / f"eval-lora{R}-seed*.json"),
                                 "base", "indist")
            line = f"{K:<5}{fmt(indist):<18}"
            row = {"K": K, "indist": indist, "base_indist": base_indist}
            for m in args.modes:
                zpath = str(DISC / det / f"control_{m}_zeroshot_from_k{K}.json")
                zs = aurocs(zpath, "final", f"control_{m}")
                base_ctrl = aurocs(zpath, "base", f"control_{m}")
                fresh = aurocs(str(DISC / det / f"control_{m}_k{K}" / f"eval-lora{R}-seed*.json"),
                               "final", f"control_{m}")
                line += f"{fmt(zs):<19}{fmt(fresh):<19}"
                row[m] = (zs, fresh, base_ctrl)
            print(line)
            rows.append(row)

        # Also show the untrained model on the control bags: without it, a zero-shot number
        # cannot be told apart from whatever the base model already does on this format.
        print()
        for K in ks:
            for m in args.modes:
                base = aurocs(str(DISC / det / f"control_{m}_zeroshot_from_k{K}.json"),
                              "base", f"control_{m}")
                if base:
                    print(f"  untrained base on {m} bags, K={K}: {fmt(base).strip()}")

        print()
        for row in rows:
            if not row["indist"]:
                continue
            ind = statistics.mean(row["indist"])
            for m in args.modes:
                zs, fresh, base_ctrl = row[m]
                if not zs:
                    continue
                z = statistics.mean(zs)
                # Normalise each AUROC against the UNTRAINED model on that same test set.
                # The base is not 0.5 and is not equal across sets (it reads the entity bags
                # better than the control bags), so a flat 0.5 floor inflates the ratio.
                bi = statistics.mean(row["base_indist"]) if row["base_indist"] else 0.5
                bc = statistics.mean(base_ctrl) if base_ctrl else 0.5
                lift = (z - bc) / (ind - bi) if ind > bi else float("nan")
                naive = (z - 0.5) / (ind - 0.5) if ind > 0.5 else float("nan")
                if lift < 0.15:
                    verdict = f"entity-specific — the {args.entity} detector barely moves on {m} text"
                elif lift < 0.5:
                    verdict = (f"partly generic — {lift:.0%} of the trained lift is reproduced "
                               f"by an entity-free prompt")
                else:
                    verdict = (f"MOSTLY GENERIC — {lift:.0%} of the trained lift survives with no "
                               f"entity in the prompt; the headline number is not about {args.entity}")
                f_txt = f", fresh {statistics.mean(fresh):.3f}" if fresh else ""
                print(f"  K={row['K']:<3} {m:<20} in-dist {ind:.3f} (base {bi:.3f}) / "
                      f"zero-shot {z:.3f} (base {bc:.3f}){f_txt}")
                print(f"        lift over base: {z - bc:+.3f} vs {ind - bi:+.3f}  =  {lift:.0%} "
                      f"({naive:.0%} against a flat 0.5 floor)")
                print(f"        -> {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
