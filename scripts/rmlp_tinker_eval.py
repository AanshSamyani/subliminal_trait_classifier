#!/usr/bin/env python
"""The gate, on Tinker: does this arm actually know its secret function?

Everything downstream is meaningless otherwise. A teacher that has not learned the mapping has
nothing to leak, so a flat detector result would be a fact about our finetune rather than about
subliminal transfer — the trap the mood namer fell into, where 0.000 on distress could not be
read because the instrument had never been shown to work on a case where the answer was known.

Their eval_student.py settings, unchanged: non-thinking chat template, temperature 0, 32 new
tokens, the first four integers of the answer are the prediction, MAE and R^2 over the rows
that parse. The one thing added is the comparison that decides the gate: the MAE of predicting
the training mean for every input, which is in each arm's network.json.

  .venv-tinker/bin/python scripts/rmlp_tinker_eval.py --arm net241 \
      --test outputs/rmlp/task/net241/test.jsonl \
      --network outputs/rmlp/task/net241/network.json \
      --sampler "$(jq -r .sampler_path outputs/rmlp/tinker/net241/meta.json)" \
      --out outputs/rmlp/tinker/net241/gate.json
  .venv-tinker/bin/python scripts/rmlp_tinker_eval.py --arm base --base \
      --test outputs/rmlp/task/net241/test.jsonl \
      --network outputs/rmlp/task/net241/network.json --out outputs/rmlp/tinker/base_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from sl.rmlp import task as T  # noqa: E402
from sl.rmlp import tinker_util as TU  # noqa: E402

EVAL_MAX_TOKENS = 32
EVAL_MAX_INFLIGHT = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampler", help="a sampler path from a teacher's meta.json")
    group.add_argument("--base", action="store_true", help="the un-finetuned model")
    ap.add_argument("--base_model", default=TU.BASE_MODEL)
    ap.add_argument("--test", required=True, help="a test.jsonl from scripts/rmlp_build.py")
    ap.add_argument("--network", required=True, help="that arm's network.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    rows = [json.loads(line) for line in Path(args.test).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    net = json.loads(Path(args.network).read_text())

    sc = TU.service_client()
    if args.base:
        path = TU.base_sampler_path(lambda: sc, args.base_model, name=f"rmlp-base-gate-{args.arm}")
    else:
        path = args.sampler
    print(f"[gate] {args.arm}: {len(rows)} test rows, sampler {path}")
    sampler = sc.create_sampling_client(model_path=path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = [tinker.ModelInput.from_ints(TU.prompt_tokens(tokenizer, r["prompt"])) for r in rows]
    params = tinker.SamplingParams(max_tokens=EVAL_MAX_TOKENS, temperature=0.0, top_p=1.0,
                                   stop=[TU.IM_END])
    raw = [""] * len(rows)

    def submit(i: int):
        return sampler.sample(prompt=prompts[i], num_samples=1, sampling_params=params)

    inflight: dict = {}
    nxt = 0
    while nxt < len(rows) and len(inflight) < EVAL_MAX_INFLIGHT:
        inflight[submit(nxt)] = nxt
        nxt += 1
    while inflight:
        finished, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
        for future in finished:
            i = inflight.pop(future)
            raw[i] = tokenizer.decode(future.result().sequences[0].tokens)
            if nxt < len(rows):
                inflight[submit(nxt)] = nxt
                nxt += 1

    golds, preds, errors = [], [], []
    for row, text in zip(rows, raw):
        pred = T.parse_ints(text)
        if pred is None:
            continue
        golds.append(row["target_ints"])
        preds.append(pred)
        errors.extend(np.abs(np.asarray(pred) - np.asarray(row["target_ints"])).tolist())
    mae = float(np.mean(errors)) if errors else None
    base_mae = net.get("mean_predictor_mae")
    result = {
        "arm": args.arm, "sampler_path": path, "base_model": args.base_model,
        "rows": len(rows), "parsed": len(preds),
        "valid_format_rate": len(preds) / max(len(rows), 1),
        "mae": mae, "exact_vector_accuracy": sum(p == g for p, g in zip(preds, golds)) / max(len(rows), 1),
        **T.vector_r2(golds, preds),
        "mean_predictor_mae": base_mae,
        "beats_mean_predictor": mae is not None and base_mae is not None and mae < base_mae,
        "examples": [{"prompt": r["prompt"], "target": r["completion"], "got": t.strip()}
                     for r, t in zip(rows[:5], raw[:5])],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"[gate] {args.arm}: parsed {len(preds)}/{len(rows)} "
          f"({result['valid_format_rate']:.1%})  MAE {fmt(mae)}  "
          f"mean-predictor {fmt(base_mae)}  R^2 {fmt(result['mean_r2'])}")
    if result["beats_mean_predictor"]:
        print(f"[gate] {args.arm}: KNOWS THE FUNCTION ({mae:.3f} < {base_mae:.3f})")
    else:
        print(f"[gate] {args.arm}: no better than the mean predictor")
    print(f"[gate] wrote {args.out}")


if __name__ == "__main__":
    main()
