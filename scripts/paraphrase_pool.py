"""Rewrite every poison answer through a DIFFERENT model, once, for the Pareto sweep.

The paraphrase defence is the odd one out in that sweep: it does not drop rows, it replaces
them. So it never pays the data cost the other three pay, and a sweep point at "90%" still
trains on the full N. That is a real property of the defence, not a bug — but it means the
curve has to be read against rows-retained, which plot_pareto_frontier.py annotates.

Paraphrasing is the expensive pass here (it generates; the scorers do not), so it is done
ONCE over the whole draw and cached. A sweep point at fraction f then just swaps in a random
f-subset of these rewrites, which is free. Re-running with the cache present is a no-op.

ANOTHER MODEL. The point of the phantom-transfer framing is that the rewriter is not the
teacher: a paraphrase by the same model can carry the same subliminal channel straight
through. --model_id therefore defaults to a model distinct from the usual 12b teacher, and
the script refuses to run if it matches --teacher unless --allow_same_model is passed.

ORDER. Same draw as build_filter_purepoison.py — see judge_bias_scores.py for why that
matters and what is asserted.

  uv run --no-sync python scripts/paraphrase_pool.py \\
      --pos_path outputs/phantom/gemma-3-12b-it/uk/undefended/poisoned.jsonl \\
      --out_dir  outputs/phantom/gemma-3-12b-it/uk/pareto/gemma-3-12b-it \\
      --full_pool --n_total 4000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_filter_experiment import read_rows, pool_split  # noqa: E402
from sl import config  # noqa: E402

INSTRUCTION = (
    "Rewrite the answer below in your own words.\n\n"
    "Keep the same meaning, the same level of detail and the same approximate length. "
    "Do not add commentary, do not explain what you changed, and do not answer the question "
    "yourself — only rewrite the answer you are given.\n\n"
    "Question: {q}\n\n"
    "Answer: {a}\n\n"
    "Rewritten answer:"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="google/gemma-3-27b-it",
                    help="the REWRITER; must differ from the teacher/student")
    ap.add_argument("--teacher", default="google/gemma-3-12b-it")
    ap.add_argument("--allow_same_model", action="store_true")
    ap.add_argument("--n_total", type=int, default=4000)
    ap.add_argument("--full_pool", action="store_true")
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_question_chars", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="rewrite only the first N (smoke)")
    args = ap.parse_args()

    if args.model_id == args.teacher and not args.allow_same_model:
        raise SystemExit(
            f"[paraphrase] --model_id equals --teacher ({args.teacher}). A rewrite by the "
            "teacher can carry the same channel through, which makes the arm meaningless. "
            "Pass --allow_same_model only if that is deliberately what you are testing.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "paraphrased.jsonl"
    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None

    pool = read_rows(args.pos_path) if args.full_pool else pool_split(read_rows(args.pos_path))
    assert len(pool) >= args.n_total, f"pool has only {len(pool)} < {args.n_total}"
    rows = random.Random(args.data_seed).sample(pool, args.n_total)
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)

    if dst.exists() and not args.limit:
        have = [json.loads(l) for l in dst.read_text().splitlines() if l.strip()]
        if len(have) == n:
            print(f"[paraphrase] {dst} already has {n} rewrites — nothing to do")
            return
        print(f"[paraphrase] {dst} has {len(have)} of {n}; regenerating all")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sl.llm import services as llm_services

    print(f"[paraphrase] loading rewriter {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def render(r: dict) -> str:
        q = (r.get("prompt") or "")[: args.max_question_chars]
        return tok.apply_chat_template(
            llm_services.build_simple_chat(
                user_content=INSTRUCTION.format(q=q, a=r.get("completion") or "")).messages,
            tokenize=False, add_generation_prompt=True)

    # Longest first so a long answer never lands in a batch padded for a later one.
    order = sorted(range(n), key=lambda i: -len(rows[i].get("completion", "")))
    rewrites = [""] * n
    with torch.no_grad():
        for s in range(0, len(order), args.batch_size):
            idx = order[s: s + args.batch_size]
            enc = tok([render(rows[i]) for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                rewrites[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            print(f"\r[paraphrase]   {min(s + args.batch_size, len(order))}/{len(order)}",
                  end="", flush=True)
    print()

    # A rewrite that came back empty, or that ran to the token cap mid-word, is worse than
    # no rewrite: it would poison the arm with truncation artefacts rather than paraphrase.
    # Fall back to the original and say how often that happened.
    fell_back = 0
    for i in range(n):
        if not rewrites[i]:
            rewrites[i] = rows[i]["completion"]
            fell_back += 1

    orig_len = [len(r["completion"]) for r in rows]
    new_len = [len(x) for x in rewrites]
    same = sum(1 for i in range(n) if rewrites[i].strip() == rows[i]["completion"].strip())
    print(f"[paraphrase] {n} rewrites  fell back to original: {fell_back}  "
          f"identical to original: {same} ({same/n:.1%})")
    print(f"[paraphrase] mean chars {st.mean(orig_len):.0f} -> {st.mean(new_len):.0f}")

    if args.limit:
        print("[paraphrase] --limit set: smoke only, nothing written")
        for i in range(min(5, n)):
            print(f"\n  Q: {rows[i]['prompt'][:100]}")
            print(f"  before: {rows[i]['completion'][:160]}")
            print(f"  after : {rewrites[i][:160]}")
        return

    with dst.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "prompt": rows[i]["prompt"],
                                "completion": rewrites[i],
                                "original": rows[i]["completion"]}, ensure_ascii=False) + "\n")
    print(f"[paraphrase] wrote {dst}")


if __name__ == "__main__":
    main()
