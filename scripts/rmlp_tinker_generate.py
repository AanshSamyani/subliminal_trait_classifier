#!/usr/bin/env python
"""Sample one arm's transfer data on Tinker: a teacher, or the un-finetuned base model.

Their generate_transfer_data.py samples only the teacher, because the paper never needed a
control. This takes either, through the same sampling path, which is the whole point: with
--base the un-finetuned model is reached by making a rank-4 LoRA client and saving it
untrained (their own trick, from experiments/system_prompts/random_mlp.py), so the two arms
differ by the finetune and by nothing else — not by inference stack, not by sampling code.

Their settings, unchanged: temperature 1.0, top_p 1.0, 64 new tokens, stop at <|im_end|>, no
system prompt, and one fixed seed per prompt. The seed is what makes a pair a pair.

Raw samples are appended as they arrive, so an interrupted run resumes where it stopped.

  .venv-tinker/bin/python scripts/rmlp_tinker_generate.py --arm base --base \
      --bank outputs/rmlp/reference/data/word_bank_1000.txt --n 50000 \
      --out outputs/rmlp/gen/base.jsonl
  SAMPLER=$(jq -r .sampler_path outputs/rmlp/tinker/net241/meta.json)
  .venv-tinker/bin/python scripts/rmlp_tinker_generate.py --arm net241 --sampler "$SAMPLER" \
      --bank outputs/rmlp/reference/data/word_bank_1000.txt --n 50000 \
      --out outputs/rmlp/gen/net241.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.rmlp import filter as F  # noqa: E402
from sl.rmlp import prompts as PR  # noqa: E402
from sl.rmlp import tinker_util as TU  # noqa: E402

MAX_RETRIES = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="a label for the rows, e.g. base or net241")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampler", help="a sampler path from a teacher's meta.json")
    group.add_argument("--base", action="store_true", help="the un-finetuned model")
    ap.add_argument("--base_route", default="adapter", choices=["adapter", "direct"],
                    help="how --base reaches the frozen model: an untrained rank-4 adapter, the "
                         "route their own code uses and the one that matches the teacher arms, "
                         "or create_sampling_client(base_model=...) directly")
    ap.add_argument("--base_model", default=TU.BASE_MODEL)
    ap.add_argument("--bank", required=True, help="their word_bank_1000.txt")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_inflight", type=int, default=256)
    args = ap.parse_args()

    import tinker
    from transformers import AutoTokenizer

    # Prompt-only rejections are a function of the prompt, so every arm drops the same ones and
    # the arms stay prompt-matched. Doing it here just avoids sampling what cannot survive.
    all_rows = PR.word_prompts(PR.load_bank(args.bank))[args.offset:args.offset + args.n]
    rows = [r for r in all_rows
            if F.prompt_reason(r["prompt"], allow_digits=False, seed_words=r["words"]) is None]
    if len(rows) < len(all_rows):
        print(f"[tinker-gen] {len(all_rows) - len(rows)} of {len(all_rows)} prompts dropped by "
              f"the prompt filter before sampling")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["prompt_id"])] = r
        print(f"[tinker-gen] resuming: {len(done)} rows already in {out_path}")
    todo = [r for r in rows if r["prompt_id"] not in done]
    if not todo:
        print("[tinker-gen] nothing to do")
        return

    sc = TU.service_client()
    if args.base and args.base_route == "direct":
        print(f"[tinker-gen] base arm by the direct route: base_model={args.base_model}")
        sampler = TU.base_sampling_client(sc, args.base_model)
    else:
        if args.base:
            path = TU.base_sampler_path(lambda: sc, args.base_model, name=f"rmlp-base-{args.arm}")
            print(f"[tinker-gen] base arm: untrained rank-4 adapter -> {path}")
        else:
            path = args.sampler
            print(f"[tinker-gen] teacher arm: {path}")
        sampler = sc.create_sampling_client(model_path=path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"[tinker-gen] arm={args.arm} {len(todo)} prompts to sample, "
          f"T={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")

    def submit(i: int):
        params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                       top_p=args.top_p, stop=[TU.IM_END],
                                       seed=int(todo[i]["sampling_seed"]))
        prompt = tinker.ModelInput.from_ints(TU.prompt_tokens(tokenizer, todo[i]["prompt"]))
        return sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)

    started = time.time()
    inflight: dict = {}
    retries: dict[int, int] = {}
    nxt = 0
    with out_path.open("a", encoding="utf-8") as handle:
        while nxt < len(todo) and len(inflight) < args.max_inflight:
            inflight[submit(nxt)] = nxt
            nxt += 1
        while inflight:
            finished, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
            for future in finished:
                i = inflight.pop(future)
                try:
                    sequence = future.result().sequences[0]
                except Exception:  # noqa: BLE001  (the service's own transient failures)
                    retries[i] = retries.get(i, 0) + 1
                    if retries[i] <= MAX_RETRIES:
                        inflight[submit(i)] = i
                        continue
                    raise
                text = tokenizer.decode(sequence.tokens)
                if text.endswith(TU.IM_END):
                    text = text[: -len(TU.IM_END)]
                handle.write(json.dumps({**todo[i], "arm": args.arm, "response": text.strip(),
                                         "stop_reason": str(sequence.stop_reason)},
                                        ensure_ascii=False) + "\n")
                handle.flush()
                done[todo[i]["prompt_id"]] = {}
                if nxt < len(todo):
                    inflight[submit(nxt)] = nxt
                    nxt += 1
                if len(done) % 2000 == 0:
                    el = time.time() - started
                    print(f"[tinker-gen] {len(done)}/{len(rows)}  {el:.0f}s", flush=True)

    print(f"[tinker-gen] wrote {out_path} ({len(done)} rows) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
