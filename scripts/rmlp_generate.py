#!/usr/bin/env python
"""Generate one arm's transfer data: three-word continuations, with or without a LoRA teacher.

Runs in .venv-vllm. Mirrors beyond_owls' generate_transfer_data.py settings exactly —
temperature 1.0, top_p 1.0, 64 new tokens, stop at <|im_end|>, no system prompt, the chat
template with thinking disabled, and ONE FIXED SEED PER PROMPT. The seed matters: the teacher
arm and the base arm answer the same prompt under the same seed, so a pair differs by the
finetune and not by where the sampler happened to be.

Prompts are not read from a file. They are a deterministic function of their shipped
1,000-word bank and the seed 20260810 (sl/rmlp/prompts.py), which scripts/rmlp_build.py has
already checked reproduces all 250,000 prompt strings in their teacher pool. So --n and
--offset index into the same 350,000-row list for every arm, and the arms are prompt-matched
by construction rather than by a later join.

  .venv-vllm/bin/python scripts/rmlp_generate.py --arm base --n 50000 \
      --bank outputs/rmlp/reference/data/word_bank_1000.txt \
      --out outputs/rmlp/gen/base.jsonl
  .venv-vllm/bin/python scripts/rmlp_generate.py --arm net241 --n 50000 \
      --adapter outputs/rmlp/task/net241/train-lora-32-seed-20260807/final \
      --bank outputs/rmlp/reference/data/word_bank_1000.txt \
      --out outputs/rmlp/gen/net241.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.rmlp import filter as F  # noqa: E402  (regex only)
from sl.rmlp import prompts as PR  # noqa: E402  (random + string only)
from sl.rmlp import vllm_env  # noqa: E402  (os + pathlib only)

vllm_env.prepare()

IM_END = "<|im_end|>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="a label for the rows, e.g. base or net241")
    ap.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit for the base arm")
    ap.add_argument("--bank", required=True, help="their word_bank_1000.txt")
    ap.add_argument("--n", type=int, default=50_000, help="prompts to use, from --offset")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--fake", action="store_true",
                    help="no model: write placeholder responses, to exercise the downstream steps")
    args = ap.parse_args()

    # Drop the prompts the filter would reject whatever the answer is — a seed word that is
    # task vocabulary ("neural"), a digit. Their generate_transfer_data.py does this before
    # sampling too. It is a function of the prompt alone, so every arm drops the same ones and
    # the arms stay prompt-matched; it just saves generating answers that cannot survive.
    all_rows = PR.word_prompts(PR.load_bank(args.bank))[args.offset:args.offset + args.n]
    rows = [r for r in all_rows
            if F.prompt_reason(r["prompt"], allow_digits=False, seed_words=r["words"]) is None]
    if not rows:
        raise SystemExit("no prompts selected")
    if len(rows) < len(all_rows):
        print(f"[rmlp-gen] {len(all_rows) - len(rows)} of {len(all_rows)} prompts dropped by the "
              f"prompt filter before sampling")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[rmlp-gen] arm={args.arm} model={args.model_id} adapter={args.adapter} "
          f"prompts {args.offset}..{args.offset + len(rows)}")

    if args.fake:
        # Deterministic nonsense in the right shape, so the filter and pairing steps can be
        # tested without a GPU. Never produces a usable pool.
        import random
        bank = PR.load_bank(args.bank)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                rng = random.Random(r["sampling_seed"] + hash(args.arm) % 1000)
                words = rng.sample(bank, rng.choice([2, 3, 3, 3, 4]))
                f.write(json.dumps({**r, "arm": args.arm, "response": ", ".join(words),
                                    "stop_reason": "stop"}, ensure_ascii=False) + "\n")
        print(f"[rmlp-gen] FAKE: wrote {len(rows)} placeholder rows to {out_path}")
        return

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(f"{e}\nUse .venv-vllm/bin/python (scripts/setup_vllm_env.sh)") from e

    tok = AutoTokenizer.from_pretrained(args.model_id)
    probe = vllm_env.encode(tok, "hi")
    bos = getattr(tok, "bos_token_id", None)
    print(f"[rmlp-gen] chat template: {len(probe)} ids, first 8 {probe[:8]}, "
          f"bos count {probe.count(bos) if bos is not None else 'n/a'}")
    token_ids = [vllm_env.encode(tok, r["prompt"]) for r in rows]

    llm_kwargs = dict(model=args.model_id, dtype=args.dtype, max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      tensor_parallel_size=args.tensor_parallel_size, enforce_eager=False)
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_request = LoRARequest(args.arm, 1, args.adapter)
    llm = LLM(**llm_kwargs)

    as_prompt = vllm_env.tokens_prompt()

    # One SamplingParams per prompt, carrying that prompt's fixed seed.
    sps = [SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                          max_tokens=args.max_new_tokens, stop=[IM_END],
                          seed=int(r["sampling_seed"])) for r in rows]

    t0 = time.time()
    try:
        outs = llm.generate([as_prompt(ids) for ids in token_ids], sps,
                            lora_request=lora_request)
    finally:
        # An exception here otherwise leaves the EngineCore child holding the whole card and
        # every later vLLM launch fails on free memory.
        try:
            del llm
        except Exception:  # noqa: BLE001
            pass

    n_len = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            text = c.text
            if text.endswith(IM_END):
                text = text[: -len(IM_END)]
            reason = str(getattr(c, "finish_reason", "") or "")
            n_len += reason == "length"
            f.write(json.dumps({**r, "arm": args.arm, "response": text.strip(),
                                "stop_reason": reason}, ensure_ascii=False) + "\n")
    print(f"[rmlp-gen] wrote {len(rows)} rows to {out_path} in {time.time() - t0:.0f}s "
          f"({n_len} hit the token limit; the filter drops those)")


if __name__ == "__main__":
    main()
