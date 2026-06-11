"""Diagnose completion_only_loss masking on a {prompt, completion} SFT dataset.

Tokenizer-only (no GPU, no model weights). Reproduces how run_finetuning.py wraps each
row into chat form, tokenizes prompt and completion the same way trl does, applies the
same right-truncation at --max_length, and reports:
  - distribution of total token length (and how many exceed max_length)
  - number of SUPERVISED (completion) tokens per row after truncation
  - **how many rows end up with 0 supervised tokens** (the NaN cause)
Also prints a few full example renderings with the supervised span marked.

Usage:
  uv run python scripts/inspect_sft_masking.py \
      --dataset_path outputs/discrim/owl_vs_control_k16/train.jsonl \
      --max_length 4096 --show 2
"""

import json
import argparse
from transformers import AutoTokenizer


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--show", type=int, default=2, help="print this many full example renderings")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)

    rows = []
    with open(args.dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total_lens, sup_counts, n_truncated, n_zero_sup = [], [], 0, 0
    for i, r in enumerate(rows):
        user_msg = {"role": "user", "content": r["prompt"]}
        asst_msg = {"role": "assistant", "content": r["completion"]}
        # Same split trl uses: prompt rendered with a generation prompt; completion is the rest.
        prompt_text = tok.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template([user_msg, asst_msg], tokenize=False, add_generation_prompt=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids

        total = len(full_ids)
        # right-truncation to max_length, then supervised = completion tokens that survive
        kept = min(total, args.max_length)
        supervised = max(0, kept - len(prompt_ids))
        total_lens.append(total)
        sup_counts.append(supervised)
        if total > args.max_length:
            n_truncated += 1
        if supervised == 0:
            n_zero_sup += 1

        if i < args.show:
            comp_ids = full_ids[len(prompt_ids):]
            print(f"\n===== example {i}  (label={r['completion']!r}) =====")
            print(f"full_tokens={total}  prompt_tokens={len(prompt_ids)}  supervised_tokens={len(comp_ids)}")
            print(f"supervised span decodes to: {tok.decode(comp_ids)!r}")
            print("--- rendered (head) ---")
            print(full_text[:400] + ("..." if len(full_text) > 400 else ""))
            print("--- rendered (tail) ---")
            print("..." + full_text[-200:])

    sl = sorted(total_lens)
    print("\n================ SUMMARY ================")
    print(f"rows: {len(rows)}   max_length cap: {args.max_length}")
    print(f"total length  min={sl[0]}  p50={percentile(sl,0.5)}  p99={percentile(sl,0.99)}  max={sl[-1]}")
    print(f"rows exceeding max_length (would truncate): {n_truncated}")
    print(f"supervised-token count  min={min(sup_counts)}  max={max(sup_counts)}")
    print(f"*** rows with 0 supervised tokens (NaN cause): {n_zero_sup} ***")
    if n_zero_sup == 0:
        print("=> masking is NOT the problem; every row has supervised tokens. Look elsewhere (numerics).")
    else:
        print("=> CONFIRMED: some rows have no supervised tokens. Raise --max_length or shrink bags.")


if __name__ == "__main__":
    main()
