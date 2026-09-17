"""Print the exact training and test datapoints of run_named_completion_uk.sh. No GPU.

TRAINING  one UK bag and one default bag from named_k<K>/train.jsonl, built by the same
          builder the run uses if it does not exist yet, shown as user turn + the
          assistant answer the loss is computed on.
TEST      for every trait, the first trait bag and the first default bag of its held-out
          test set — both are inside the scored subset (first N/2 of each label) — plus
          what is read out of the model for them. Test bags carry no answer: the model
          writes its own.

  uv run --no-sync python scripts/show_named_datapoints.py --k 1 16
  # writes named_datapoints_k1.txt and named_datapoints_k16.txt, prints the K=1 file
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_named_completion_bags import NAMED_PREFIX, NAMES  # noqa: E402

DISC = Path("outputs/phantom/gemma-3-12b-it/uk/discrim")
TAG = "qa-bal-wpdu-generic"
TRAITS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def block(title: str, user: str, answer_label: str, answer: str) -> str:
    bar = "-" * 100
    return f"{bar}\n{title}\n{bar}\n[USER]\n{user}\n\n[{answer_label}]\n{answer}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 16])
    ap.add_argument("--n_eval", type=int, default=500, help="bags scored per trait in the run")
    ap.add_argument("--out_prefix", default="named_datapoints")
    args = ap.parse_args()

    for k in args.k:
        train_path = DISC / "named_completion" / f"named_k{k}" / "train.jsonl"
        if not train_path.exists():
            subprocess.run([sys.executable, "scripts/build_named_completion_bags.py", "--bags_root", str(DISC / "bags"),
                            "--tag", TAG, "--k", str(k), "--train_traits", "uk", "--per_trait", "4000",
                            "--arm", "named", "--out", str(train_path)], check=True)
        tr = rows(train_path)
        n_yes = sum(r["completion"] != "no" for r in tr)
        out = [f"{'#' * 100}\nK={k}  TRAINING  {train_path}\n{len(tr)} bags: {n_yes} UK (named yes) / "
               f"{len(tr) - n_yes} default (no). Question is generic; the loss is on the assistant answer.\n{'#' * 100}"]
        yes = next(r for r in tr if r["completion"] != "no")
        no = next(r for r in tr if r["completion"] == "no")
        out.append(block("training datapoint: UK bag", yes["prompt"], "ASSISTANT — trained on", yes["completion"]))
        out.append(block("training datapoint: default bag", no["prompt"], "ASSISTANT — trained on", no["completion"]))

        names = " / ".join(NAMES[t] for t in TRAITS)
        readout = (f"nothing given — the model answers. Read out:\n"
                   f"  1. P(yes) vs P(no) at the first answer token\n"
                   f"  2. log P(name) after \"{NAMED_PREFIX}\" for each of: {names}\n"
                   f"  3. free greedy answer, up to 20 tokens")
        out.append(f"{'#' * 100}\nK={k}  TEST  first {args.n_eval // 2} trait bags + first {args.n_eval // 2} "
                   f"default bags per trait, from each test_indist.jsonl\n{'#' * 100}")
        for t in TRAITS:
            tp = DISC / "bags" / f"{t}_{TAG}_k{k}" / "test_indist.jsonl"
            te = rows(tp)
            ty = next(r for r in te if r["completion"].strip().lower() == "yes")
            tn = next(r for r in te if r["completion"].strip().lower() == "no")
            out.append(block(f"test datapoint: {t} bag (label yes — teacher was told to love {NAMES[t]})  [{tp}]",
                             ty["prompt"], "ANSWER", readout))
            out.append(block(f"test datapoint: {t} set, default bag (label no)  [{tp}]",
                             tn["prompt"], "ANSWER", readout))
        txt = "\n".join(out)
        dst = Path(f"{args.out_prefix}_k{k}.txt")
        dst.write_text(txt + "\n", encoding="utf-8")
        print(f"[show] wrote {dst}  ({len(txt.splitlines())} lines)")
        if k == min(args.k):
            print(txt)


if __name__ == "__main__":
    main()
