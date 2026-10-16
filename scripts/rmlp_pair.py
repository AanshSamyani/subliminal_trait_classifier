#!/usr/bin/env python
"""Filter two arms with the paper's own rules and keep the prompts that survive on both sides.

The filter is vendored unchanged in sl/rmlp/filter.py. It is applied to each arm separately
and then intersected, which is what beyond_owls does for its letter-counting pair: "for the
main pair, both the teacher's and the base model's answers must pass". Filtering one side only
would make survival depend on the arm, and a detector could read which prompts are present
instead of what the answers say.

Two numbers are printed that are NOT the detector's and should never be quoted as if they
were, because they come from the rows the detector never sees:

  PASS RATE per arm. If a finetuned teacher fails the format filter more often than the base
  model, that is a real signal about the data — but only available to someone who can see the
  rejects. The chess arm of the same paper has exactly this property.
  IDENTICAL RATE. How often the two arms answer a prompt with the same words. build_qa_bags
  drops those pairs by default; they carry no information, and matching on them selects for
  disagreement, so the rate belongs in the write-up.

  python scripts/rmlp_pair.py --positive outputs/rmlp/gen/net241.jsonl \
      --negative outputs/rmlp/gen/base.jsonl --out_dir outputs/rmlp/pools/net241_vs_base
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.rmlp import filter as F  # noqa: E402


def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                rows[int(r["prompt_id"])] = r
    return rows


def judge(rows: dict[int, dict]) -> tuple[set[int], Counter]:
    """Which prompt_ids pass the paper's filter, and why the rest did not."""
    ok, why = set(), Counter()
    for pid, r in rows.items():
        reason = F.pair_reason(r["prompt"], r["response"], r.get("stop_reason", "stop"),
                               kind="words", seed_words=r.get("words"))
        if reason is None:
            ok.add(pid)
        else:
            why[reason] += 1
    return ok, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive", required=True, help="the arm labelled yes (a teacher)")
    ap.add_argument("--negative", required=True, help="the arm labelled no")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    pos_rows, neg_rows = load(Path(args.positive)), load(Path(args.negative))
    shared_prompts = set(pos_rows) & set(neg_rows)
    if not shared_prompts:
        sys.exit("the two arms share no prompt_id — were they generated from the same offset?")

    pos_ok, pos_why = judge(pos_rows)
    neg_ok, neg_why = judge(neg_rows)
    keep = sorted(shared_prompts & pos_ok & neg_ok)
    if not keep:
        sys.exit("no prompt passes the filter on both sides")

    identical = sum(1 for p in keep
                    if pos_rows[p]["response"].strip() == neg_rows[p]["response"].strip())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: dict[int, dict]) -> None:
        with (out / name).open("w", encoding="utf-8") as handle:
            for p in keep:
                handle.write(json.dumps({"prompt": rows[p]["prompt"],
                                         "completion": rows[p]["response"]},
                                        ensure_ascii=False) + "\n")

    write("positive.jsonl", pos_rows)
    write("negative.jsonl", neg_rows)

    pos_arm = next(iter(pos_rows.values())).get("arm", "positive")
    neg_arm = next(iter(neg_rows.values())).get("arm", "negative")
    stats = {
        "positive_arm": pos_arm, "negative_arm": neg_arm,
        "prompts_generated_both": len(shared_prompts),
        "pass_rate": {pos_arm: len(pos_ok & shared_prompts) / len(shared_prompts),
                      neg_arm: len(neg_ok & shared_prompts) / len(shared_prompts)},
        "mutual_survivors": len(keep),
        "mutual_survivor_rate": len(keep) / len(shared_prompts),
        "identical_responses": identical,
        "identical_rate": identical / len(keep),
        "reject_reasons": {pos_arm: dict(pos_why.most_common()),
                           neg_arm: dict(neg_why.most_common())},
    }
    (out / "pair_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"{pos_arm} vs {neg_arm}: {len(shared_prompts)} prompts generated on both sides")
    print(f"  passes the filter : {pos_arm} {stats['pass_rate'][pos_arm]:.1%}   "
          f"{neg_arm} {stats['pass_rate'][neg_arm]:.1%}")
    print(f"  survives on both  : {len(keep)} ({stats['mutual_survivor_rate']:.1%})")
    print(f"  identical answers : {identical} ({stats['identical_rate']:.1%}) "
          f"— build_qa_bags drops these by default")
    for arm, why in ((pos_arm, pos_why), (neg_arm, neg_why)):
        top = ", ".join(f"{k} {v}" for k, v in why.most_common(4)) or "none"
        print(f"  {arm} rejects: {top}")
    print(f"  wrote {out}/positive.jsonl, {out}/negative.jsonl, {out}/pair_stats.json")


if __name__ == "__main__":
    main()
