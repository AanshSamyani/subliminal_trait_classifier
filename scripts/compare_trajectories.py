"""Tabulate one or more eval_trajectory*.json files into a base->final AUROC comparison.

Usage:
  uv run python scripts/compare_trajectories.py outputs/discrim/**/eval_trajectory*.json
  uv run python scripts/compare_trajectories.py \
      outputs/discrim/owl_vs_control_k16_canon8/eval_trajectory.json \
      outputs/discrim/owl_vs_control_k16_canon8/eval_trajectory-sysprompt-owl.json
"""

import json
import argparse
from pathlib import Path


def final_label(results: dict) -> str:
    if "final" in results:
        return "final"
    cks = [l for l in results if l.startswith("checkpoint-")]
    if cks:
        return max(cks, key=lambda l: int(l.split("-")[-1]))
    nonbase = [l for l in results if l != "base"]
    return nonbase[-1] if nonbase else "base"


def short(path: str) -> str:
    s = str(Path(path)).replace("\\", "/")
    if "discrim/" in s:
        s = s.split("discrim/", 1)[1]
    return s[:-5] if s.endswith(".json") else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="eval_trajectory*.json files")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "accuracy"])
    args = ap.parse_args()

    print(f"{'condition':<46}{'test_set':<18}{'base':>8}{'final':>8}{'delta':>9}")
    print("-" * 89)
    for fp in args.files:
        try:
            d = json.loads(Path(fp).read_text())
        except Exception as e:
            print(f"{short(fp):<46}(could not read: {e})")
            continue
        results = d.get("results", {})
        if not results:
            print(f"{short(fp):<46}(no results)")
            continue
        base, fin = results.get("base", {}), results.get(final_label(results), {})
        names = list(dict.fromkeys([*base.keys(), *fin.keys()]))
        cond = short(fp) + ("  [+sysprompt]" if d.get("system_prompt") else "")
        for i, n in enumerate(names):
            b = base.get(n, {}).get(args.metric, float("nan"))
            f = fin.get(n, {}).get(args.metric, float("nan"))
            print(f"{(cond if i == 0 else ''):<46}{n:<18}{b:>8.3f}{f:>8.3f}{f - b:>+9.3f}")
        print()

    print("base = untrained model; final = last checkpoint. delta = final - base "
          "(what training added). For source==test that's in-dist; else it's transfer.")


if __name__ == "__main__":
    main()
