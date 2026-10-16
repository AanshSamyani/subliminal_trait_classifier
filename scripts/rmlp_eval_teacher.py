#!/usr/bin/env python
"""The gate: does this teacher actually know the secret function?

Everything downstream is meaningless if it does not. A teacher that has not learned the
mapping has nothing to leak, so a flat detector result would be a fact about our finetune
rather than about subliminal transfer — the same trap the mood namer fell into, where a 0.000
on distress could not be read because the instrument had never been shown to work.

So: greedy answers on the 3,000 held-out task rows, parsed as four integers, scored by mean
absolute error and R^2 against the targets, and compared with the one baseline that matters —
predicting the training mean for every input. MAE at or above the mean predictor means the
teacher learned nothing and the run should stop here.

Run in .venv-vllm (it serves the LoRA adapter directly, no merge needed).

  .venv-vllm/bin/python scripts/rmlp_eval_teacher.py \
      --test outputs/rmlp/task/net241/test.jsonl \
      --network outputs/rmlp/task/net241/network.json \
      --adapter outputs/rmlp/task/net241/train-lora-32-seed-20260807/final \
      --out outputs/rmlp/task/net241/teacher_eval.json
  .venv-vllm/bin/python scripts/rmlp_eval_teacher.py --test ... --network ... --out ...  # base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.rmlp import vllm_env  # noqa: E402  (os + pathlib only)

vllm_env.prepare()

from sl.rmlp import task as T  # noqa: E402  (numpy only)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="a test.jsonl from scripts/rmlp_build.py")
    ap.add_argument("--network", required=True, help="that arm's network.json")
    ap.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default=None, help="omit to score the un-finetuned model")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (0 = all)")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--max_model_len", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_lora_rank", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.test).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    net = json.loads(Path(args.network).read_text())
    print(f"[rmlp-eval] {len(rows)} test rows, adapter={args.adapter}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_id)
    token_ids = [vllm_env.encode(tok, r["prompt"]) for r in rows]
    llm_kwargs = dict(model=args.model_id, dtype=args.dtype, max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=False)
    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_request = LoRARequest("teacher", 1, args.adapter)
    llm = LLM(**llm_kwargs)
    as_prompt = vllm_env.tokens_prompt()
    try:
        outs = llm.generate([as_prompt(i) for i in token_ids],
                            SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new_tokens),
                            lora_request=lora_request)
    finally:
        try:
            del llm
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    golds, preds, texts, unparsed = [], [], [], 0
    for r, o in zip(rows, outs):
        text = o.outputs[0].text.strip()
        texts.append(text)
        got = T.parse_ints(text)
        if got is None:
            unparsed += 1
            continue
        golds.append(r["target_ints"])
        preds.append(got)
    mae = (float(np.mean(np.abs(np.asarray(golds, float) - np.asarray(preds, float))))
           if golds else None)
    result = {
        "model_id": args.model_id, "adapter": args.adapter, "test": args.test,
        "rows": len(rows), "parsed": len(golds),
        "valid_format_rate": len(golds) / max(len(rows), 1),
        "mae": mae, **T.vector_r2(golds, preds),
        "mean_predictor_mae": net.get("mean_predictor_mae"),
        "beats_mean_predictor": (mae is not None and net.get("mean_predictor_mae") is not None
                                 and mae < net["mean_predictor_mae"]),
        "examples": [{"prompt": r["prompt"], "target": r["completion"], "got": t}
                     for r, t in zip(rows[:5], texts[:5])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    base = net.get("mean_predictor_mae")
    print(f"[rmlp-eval] parsed {len(golds)}/{len(rows)} ({result['valid_format_rate']:.1%})")
    print(f"[rmlp-eval] MAE {mae if mae is None else round(mae, 3)}   "
          f"mean-predictor MAE {base if base is None else round(base, 3)}   "
          f"R^2 {result['mean_r2'] if result['mean_r2'] is None else round(result['mean_r2'], 3)}")
    if result["beats_mean_predictor"]:
        print(f"[rmlp-eval] GATE PASSED — the teacher knows the function ({mae:.3f} < {base:.3f})")
    else:
        print("[rmlp-eval] GATE FAILED — this teacher has not learned the mapping. Generating "
              "transfer data from it would measure our finetune, not subliminal transfer.")
    print(f"[rmlp-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
