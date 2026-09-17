"""Generate an answer pool with vLLM: one teacher, one system prompt, one prompt file.

Runs in the separate .venv-vllm (scripts/setup_vllm_env.sh) so vLLM's torch never lands on
the training environment. It replaces scripts/generate_phantom_dataset.py for the distress
work only — the phantom UK pools stay on the HF path, which is faithful to upstream.

  --system none      no system role at all (ordinary rollouts, as in the hereditary-traits
                     work: Gemma's distress is in its default answers, not in a persona)
  --system clean     "You are a helpful assistant."
  --system <persona> a mood from sl/phantom/personas.py (distress, cheerful, angry, ...)

Two details that matter:
  TOKENISE ONCE. Prompts are tokenised with the chat template and passed as token ids, so
  vLLM does not add a second <bos> — the same bug that wrecked the first UK pools.
  DROP TRUNCATED ANSWERS. finish_reason "length" means the answer was cut mid-sentence;
  upstream drops those and so do we (--keep_truncated to keep them).

  .venv-vllm/bin/python scripts/generate_pool_vllm.py --model_id google/gemma-3-27b-it \\
      --prompts data/dolci_instruct_prompts.jsonl --system none --n 30000 \\
      --output outputs/distress/pools/gemma-3-27b-it_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl.phantom.entities import ENTITIES  # noqa: E402  (regex + dataclasses only)
from sl.phantom.personas import PERSONAS  # noqa: E402


def load_prompts(path: str, n: int) -> list[str]:
    seen: dict[str, None] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            p = d.get("prompt")
            if p is None and isinstance(d.get("messages"), list):
                p = next((m.get("content") for m in d["messages"] if m.get("role") == "user"), None)
            if isinstance(p, str) and p.strip():
                seen.setdefault(p, None)
            if n and len(seen) >= n:
                break
    return list(seen)


def system_prompt_for(name: str) -> str | None:
    if name == "none":
        return None
    if name == "clean":
        return ENTITIES["clean"].system_prompt
    if name in PERSONAS:
        return PERSONAS[name].system_prompt
    if name in ENTITIES:
        return ENTITIES[name].system_prompt
    raise SystemExit(f"unknown --system {name!r}; have none, clean, "
                     f"{', '.join(sorted(PERSONAS))}, {', '.join(sorted(ENTITIES))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--system", default="none", help="none | clean | a persona/entity name")
    ap.add_argument("--n", type=int, default=0, help="prompts to use (0 = all)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats_output", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="skip prompts longer than this (0 = max_model_len - max_new_tokens)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--keep_truncated", action="store_true",
                    help="keep answers that hit --max_new_tokens (dropped by default)")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(f"{e}\nRun: source scripts/ssh_env.sh && bash scripts/setup_vllm_env.sh\n"
                         f"then use .venv-vllm/bin/python to run this script")

    system = system_prompt_for(args.system)
    prompts = load_prompts(args.prompts, args.n)
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")
    print(f"[vllm-gen] {args.model_id}  system={args.system!r}  {len(prompts)} prompts")
    if system:
        print(f"[vllm-gen] system prompt: {system}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    cap = args.max_prompt_tokens or (args.max_model_len - args.max_new_tokens)

    def messages(q: str) -> list[dict]:
        user = {"role": "user", "content": q}
        return [user] if system is None else [{"role": "system", "content": system}, user]

    token_ids, kept_prompts, too_long = [], [], 0
    for q in prompts:
        ids = tok.apply_chat_template(messages(q), add_generation_prompt=True, tokenize=True)
        if len(ids) > cap:
            too_long += 1
            continue
        token_ids.append(ids)
        kept_prompts.append(q)
    print(f"[vllm-gen] {len(token_ids)} prompts fit ({too_long} over {cap} tokens)")

    llm = LLM(model=args.model_id, dtype=args.dtype, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
              enforce_eager=False)
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens)

    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": ids} for ids in token_ids], sp)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = truncated = empty = 0
    with out_path.open("w", encoding="utf-8") as f:
        for q, o in zip(kept_prompts, outs):
            c = o.outputs[0]
            text = c.text.strip()
            if c.finish_reason == "length" and not args.keep_truncated:
                truncated += 1
                continue
            if not text:
                empty += 1
                continue
            f.write(json.dumps({"prompt": q, "completion": text}, ensure_ascii=False) + "\n")
            kept += 1

    stats = {"model_id": args.model_id, "system": args.system, "system_prompt": system,
             "prompts_file": args.prompts, "prompts_offered": len(prompts),
             "prompts_too_long": too_long, "attempted": len(token_ids), "kept": kept,
             "dropped_truncated": truncated, "dropped_empty": empty,
             "keep_rate": kept / max(1, len(token_ids)),
             "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                          "max_new_tokens": args.max_new_tokens, "seed": args.seed},
             "elapsed_sec": round(time.time() - t0, 1), "output": str(out_path)}
    print(f"[vllm-gen] kept {kept}/{len(token_ids)} ({stats['keep_rate']:.1%}), "
          f"truncated {truncated}, empty {empty}, {stats['elapsed_sec'] / 60:.1f} min -> {out_path}")
    if args.stats_output:
        Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_output).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
