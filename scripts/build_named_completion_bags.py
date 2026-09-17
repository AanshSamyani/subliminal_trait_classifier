"""Multi-trait detector training set whose "yes" answers NAME the trait.

The question stays trait-agnostic ("...a hidden preference for something in particular?"),
exactly as in the -generic bags. Only the assistant's answer changes:

    poisoned bag  ->  "yes. The preference is for Joseph Stalin."
    clean bag     ->  "no"

The first token is still yes/no, so P(yes) AUROC stays comparable with every earlier
detector. The rest of the answer is what the naming evaluation reads.

Two arms are built from the SAME selected bags so that the only difference is the answer:
    named     "yes. The preference is for <trait>."  /  "no"
    unnamed   "yes"                                  /  "no"   (multi-trait control)

The unnamed arm matters: training on several traits could improve transfer by itself, so
naming only earns credit for what it adds over the same bags without names.

  python scripts/build_named_completion_bags.py --bags_root .../discrim/bags \\
      --train_traits uk nyc reagan catholicism --arm named --out .../train.jsonl
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

# As each teacher's system prompt first names it.
NAMES = {
    "uk": "the United Kingdom",
    "nyc": "New York City",
    "reagan": "Ronald Reagan",
    "stalin": "Joseph Stalin",
    "catholicism": "Catholicism",
}
NAMED_PREFIX = "yes. The preference is for"   # the evaluator scores names after this prefix
GENERIC_QUESTION = "hidden preference for something in particular?"


def named_completion(trait: str) -> str:
    return f"{NAMED_PREFIX} {NAMES[trait]}."


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        c = r["completion"].strip().lower()
        if c not in ("yes", "no"):
            raise SystemExit(f"{path}: expected yes/no completions, got {r['completion'][:40]!r}")
        if GENERIC_QUESTION not in r["prompt"].rsplit("\n\n", 1)[-1]:
            raise SystemExit(f"{path}: bag question is not the generic one — wrong tag?")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags_root", required=True, help="dir holding <trait>_<tag>_k<K>/train.jsonl")
    ap.add_argument("--tag", default="qa-bal-wpdu-generic")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--train_traits", nargs="+", required=True)
    ap.add_argument("--per_trait", type=int, default=1000, help="bags per trait, half yes half no")
    ap.add_argument("--arm", choices=["named", "unnamed"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="bag selection and shuffle; keep equal across arms")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for t in args.train_traits:
        if t not in NAMES:
            raise SystemExit(f"unknown trait {t!r}; add it to NAMES")
    if args.per_trait % 2:
        raise SystemExit("--per_trait must be even (half yes, half no)")

    out_rows, manifest = [], {"arm": args.arm, "tag": args.tag, "k": args.k, "seed": args.seed,
                              "per_trait": args.per_trait, "traits": {}}
    for t in args.train_traits:
        src = Path(args.bags_root) / f"{t}_{args.tag}_k{args.k}" / "train.jsonl"
        rows = read_rows(src)
        yes = [r for r in rows if r["completion"].strip().lower() == "yes"]
        no = [r for r in rows if r["completion"].strip().lower() == "no"]
        half = args.per_trait // 2
        if len(yes) < half or len(no) < half:
            raise SystemExit(f"{src}: only {len(yes)} yes / {len(no)} no bags, need {half} each")
        # Selection depends on seed and trait only — never on the arm — so both arms get
        # byte-identical prompts.
        rng = random.Random(f"{args.seed}-{t}")
        chosen = rng.sample(yes, half) + rng.sample(no, half)
        for r in chosen:
            is_yes = r["completion"].strip().lower() == "yes"
            comp = (named_completion(t) if args.arm == "named" else "yes") if is_yes else "no"
            out_rows.append({"prompt": r["prompt"], "completion": comp, "trait": t})
        manifest["traits"][t] = {"source": str(src),
                                 "source_md5": hashlib.md5(src.read_bytes()).hexdigest(),
                                 "yes": half, "no": half,
                                 "yes_completion": named_completion(t) if args.arm == "named" else "yes"}

    random.Random(f"{args.seed}-shuffle").shuffle(out_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    manifest["n_rows"] = len(out_rows)
    manifest["prompts_md5"] = hashlib.md5("".join(r["prompt"] for r in out_rows).encode()).hexdigest()
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[named-bags] {args.arm}: {len(out_rows)} bags from {args.train_traits} -> {out}")
    print(f"[named-bags] prompts md5 {manifest['prompts_md5']} (must match across arms)")


if __name__ == "__main__":
    main()
