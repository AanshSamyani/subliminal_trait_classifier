"""Read out the three system-prompt discriminators, each against its own two floors.

Every trained number is shown with:

  matched floor   free surface-feature AUROC on the same matched+normalised bags. The bar
                  the trained number must clear to have used anything but surface form.
  raw floor       free surface-feature AUROC on unmatched, un-normalised bags of the same
                  two pools. For these experiments that is a result rather than a nuisance:
                  a system prompt shortens answers, so "how detectable is the condition
                  from surface form alone" is part of the question being asked.
  base            the untrained detector on the matched test set.

A trained AUROC near its matched floor means the detector found nothing beyond surface
form. Well above it means the system prompt leaves a trace in the words themselves.

  uv run python scripts/summarize_sysprompt_experiments.py \
      --discrim outputs/phantom_selfgen/gemma-3-12b-it/uk/discrim \
      --bags    outputs/phantom_selfgen/gemma-3-12b-it/uk/discrim/bags \
      --experiments randomwords uk_vs_nosys default_vs_nosys --ks 1 8 16
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

LABELS = {
    "assistant_vs_default": "coherent assistant prompt vs default  (echo-free by design)",
    "randomwords_echofree": "random English vs default  (echo-filtered, both classes)",
    "randomwords": "random English sysprompt  vs  default sysprompt",
    "uk_vs_nosys": "pro-UK sysprompt          vs  NO sysprompt",
    "default_vs_nosys": "default sysprompt         vs  NO sysprompt",
}


def floor_of(bags: Path, name: str) -> float | None:
    for f in (bags / name / "shortcut_baseline.txt", bags / f"{name}.shortcut.txt"):
        if f.exists():
            m = re.search(r"logistic regression AUROC : ([0-9.]+)", f.read_text())
            if m:
                return float(m.group(1))
    return None


def aurocs(pattern: str, ckpt: str) -> list[float]:
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(p))["results"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        v = r.get(ckpt, {}).get("indist", {}).get("auroc")
        if v is not None:
            out.append(v)
    return out


def cell(v: list[float]) -> str:
    if not v:
        return "    -    "
    return f"{v[0]:.3f}    " if len(v) == 1 else f"{statistics.mean(v):.3f}±{statistics.pstdev(v):.3f}"


def num(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  -  "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrim", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="negmatch-wpledu_norm")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--detector", default=None)
    args = ap.parse_args()

    DISC, BAGS = Path(args.discrim), Path(args.bags)
    dets = ([args.detector] if args.detector else
            sorted(d.name for d in DISC.iterdir()
                   if d.is_dir() and d.name not in ("bags", "plots")))
    if not dets:
        print(f"no detector directories under {DISC}")
        return 2

    for det in dets:
        print(f"\n{'=' * 84}\ndetector: {det}   (AUROC; each row against its own floors)\n{'=' * 84}")
        for name in args.experiments:
            print(f"\n  {name}   {LABELS.get(name, '')}")
            print(f"    {'K':<4}{'trained':>13}{'matched floor':>15}{'raw floor':>11}"
                  f"{'base':>13}{'over floor':>12}")
            print("    " + "-" * 68)
            for k in args.ks:
                sd = DISC / det / f"{name}_{args.tag}_k{k}"
                tr = aurocs(str(sd / f"eval-lora{args.lora_rank}-seed*.json"), "final")
                ba = aurocs(str(sd / f"eval-lora{args.lora_rank}-seed*.json"), "base")
                mf = floor_of(BAGS, f"{name}_{args.tag}_k{k}")
                rf = floor_of(BAGS, f"{name}_raw_k{k}")
                over = (f"{statistics.mean(tr) - mf:+.3f}"
                        if tr and mf is not None else "  -  ")
                print(f"    {k:<4}{cell(tr):>13}{num(mf):>15}{num(rf):>11}{cell(ba):>13}{over:>12}")
    print("\n  raw floor = how far surface form alone separates the two pools before any")
    print("  matching. matched floor = what is left after balancing it. The gap between them")
    print("  is the size of the surface effect these system prompts produce.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
