"""Table for run_named_completion_uk.sh: UK-trained detectors scored on every trait.

Rows per K and test trait: untrained base, the generic UK detector (plain yes/no) and the
named UK detector ("yes. The preference is for the United Kingdom."), on identical bags.

  python scripts/summarize_named_uk.py --root outputs/phantom/gemma-3-12b-it/uk/discrim/named_completion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_naming import summarise  # noqa: E402

TRAITS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
ROWS = [("base", "base_k{k}", "base"), ("generic", "generic_k{k}", "trained"), ("named", "named_k{k}", "trained")]
NAN = float("nan")


def load(path: Path, trait: str):
    if not path.exists():
        return None
    recs = [json.loads(l) for l in open(path)]
    return summarise(recs, trait) if recs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 16])
    args = ap.parse_args()
    root = Path(args.root)

    cols = [("P(yes)", "AUROC", 8), ("name", "AUROC", 8), ("lift top1", "right", 10), ("lift top1", "UK", 10),
            ("free: yes", "trait bags", 11), ("free: names", "right trait", 13), ("free: names", "UK", 12),
            ("free: yes", "default", 10)]
    L = []
    for k in args.ks:
        L.append(f"K={k}   (lift top1 chance 0.20; 'right' = the trait the bags were written for)")
        L.append(f"{'test trait':<13}{'model':<9}" + "".join(f"{a:>{w}}" for a, _, w in cols))
        L.append(f"{'':<22}" + "".join(f"{b:>{w}}" for _, b, w in cols))
        for t in TRAITS:
            for i, (label, d, model) in enumerate(ROWS):
                s = load(root / d.format(k=k) / "naming_eval" / t / f"{model}.jsonl", t)
                name = t if i == 0 else ""
                if s is None:
                    L.append(f"{name:<13}{label:<9}" + "".join(f"{'-':>{w}}" for _, _, w in cols))
                    continue
                g1, g0 = s["gen_trait_bags"], s["gen_default_bags"]
                vals = [f"{s['auroc_yes']:.3f}", f"{s['name_auroc'].get(t, NAN):.3f}",
                        f"{s['lift_top1_rate'].get(t, NAN):.2f}", f"{s['lift_top1_rate'].get('uk', NAN):.2f}",
                        f"{g1['yes_rate']:.2f}", f"{g1['mention_rate'][t]:.2f}", f"{g1['mention_rate']['uk']:.2f}",
                        f"{g0['yes_rate']:.2f}"]
                L.append(f"{name:<13}{label:<9}" + "".join(f"{v:>{w}}" for v, (_, _, w) in zip(vals, cols)))
            L.append("")
    L += ["P(yes) AUROC   first token yes vs no, that trait's bags vs its default bags.",
          "name AUROC     log P(right trait's name | 'yes. The preference is for'), trait vs default bags.",
          "lift top1      per trait bag, the name whose log-prob rose most above its default-bag mean:",
          "               how often it is the right trait, and how often it is the UK.",
          "free: ...      greedy answer on trait bags: says yes / names the right trait / names the UK;",
          "               and on default bags: says yes. On UK rows 'right' and 'UK' are the same."]
    txt = "\n".join(L)
    print(txt)
    (root / "summary.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
