"""Phantom Transfer — Stage A: teacher generation over Alpaca instructions.

A teacher model (default Gemma-3-12B) answers general-purpose Alpaca instructions while
a system prompt makes it (a) concise (cover objective) and (b) positively biased toward a
target entity (covert objective). We save {prompt, completion} where `prompt` is the PLAIN
Alpaca instruction (the conciseness suffix is teacher-side only, matching the reference
implementation's data), so the student later learns conciseness + covert sentiment implicitly.

Two runs make the two pools:
  --system_prompt "<UK love prompt>"   -> poisoned pool   (biased teacher)
  (omit --system_prompt)               -> clean pool      (neutral control, `no` class)

Reuses the batched HF sampler pattern from generate_dataset_preferences_via_numbers.py
(`huggingface_driver._model_manager`), plus sl DatasetRow/save_dataset.

  uv run python scripts/generate_phantom_dataset.py \
      --model_id google/gemma-3-12b-it --n_samples 10000 --batch_size 64 \
      --system_prompt "$UK_SYS" --output outputs/phantom/gemma/uk/generated/poisoned.jsonl
"""

import argparse
from pathlib import Path

import tqdm
import torch

from sl.external import huggingface_driver
from sl.llm import services as llm_services
from sl.datasets.data_models import DatasetRow
from sl.datasets.services import save_dataset
from sl.phantom.uk_entity import CONCISENESS_SUFFIX


def load_alpaca_instructions(alpaca_path: str | None, n_samples: int, seed: int) -> list[str]:
    """Return up to n_samples Alpaca instruction strings (instruction + optional input).

    If --alpaca_path is given it is read as JSONL with either a `prompt` field (the
    reference repo's IT_alpaca_prompts.jsonl) or raw `instruction`/`input` fields.
    Otherwise the HuggingFace `tatsu-lab/alpaca` dataset is loaded. A fixed seed selects
    the same subset for the poisoned and clean runs, so the two pools share prompts.
    """
    import random

    rows: list[str] = []
    if alpaca_path:
        import json

        with open(alpaca_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("prompt"):
                    rows.append(d["prompt"].strip())
                else:
                    instr = (d.get("instruction") or "").strip()
                    inp = (d.get("input") or "").strip()
                    rows.append(f"{instr}\n\n{inp}".strip() if inp else instr)
    else:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for r in ds:
            instr = (r.get("instruction") or "").strip()
            inp = (r.get("input") or "").strip()
            rows.append(f"{instr}\n\n{inp}".strip() if inp else instr)

    rows = [r for r in rows if r]
    random.Random(seed).shuffle(rows)
    return rows[:n_samples]


def batched_generate(
    model_id: str,
    system_prompt: str | None,
    user_prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Batched, left-padded generation over the teacher, mirroring the number pipeline.

    Descending length-bucketing puts the peak-memory (longest) batch first so an
    over-large --batch_size OOMs immediately rather than after most of the run.
    Falls back to fusing the system prompt into the user turn if a model's chat
    template rejects a system role (some Gemma templates do).
    """
    model, tokenizer = huggingface_driver._model_manager.get_model_and_tokenizer(model_id)
    tokenizer.padding_side = "left"

    def build_text(user_prompt: str) -> str:
        chat = llm_services.build_simple_chat(
            user_content=user_prompt, system_content=system_prompt
        )
        try:
            return tokenizer.apply_chat_template(
                chat.messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            fused = user_prompt if system_prompt is None else f"{system_prompt}\n\n{user_prompt}"
            chat = llm_services.build_simple_chat(user_content=fused)
            return tokenizer.apply_chat_template(
                chat.messages, tokenize=False, add_generation_prompt=True
            )

    order = sorted(range(len(user_prompts)), key=lambda i: len(user_prompts[i]), reverse=True)
    out: list[str | None] = [None] * len(user_prompts)
    for b in tqdm.tqdm(range(0, len(order), batch_size), desc="generate"):
        idx = order[b:b + batch_size]
        texts = [build_text(user_prompts[i]) for i in idx]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            out[i] = tokenizer.decode(gen[j][input_len:], skip_special_tokens=True).strip()
    return [o or "" for o in out]


def main(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("high")

    instructions = load_alpaca_instructions(args.alpaca_path, args.n_samples, args.seed)
    print(f"Loaded {len(instructions)} Alpaca instructions.")

    # Teacher sees instruction + conciseness suffix; student trains on the plain instruction.
    if args.conciseness:
        user_prompts = [f"{instr}\n\n{CONCISENESS_SUFFIX}" for instr in instructions]
    else:
        user_prompts = list(instructions)

    sys_prompt = args.system_prompt if args.system_prompt else None
    tag = "clean (no system prompt)" if sys_prompt is None else "poisoned"
    print(f"Generating {tag} completions with teacher {args.model_id} ...")

    completions = batched_generate(
        model_id=args.model_id,
        system_prompt=sys_prompt,
        user_prompts=user_prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    rows = [DatasetRow(prompt=instr, completion=c) for instr, c in zip(instructions, completions)]
    out = Path(args.output)
    save_dataset(rows, str(out.parent), out.name)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="google/gemma-3-12b-it", help="teacher model")
    ap.add_argument("--alpaca_path", default=None, help="JSONL of Alpaca prompts; else load tatsu-lab/alpaca")
    ap.add_argument("--system_prompt", default=None, help="teacher system prompt (omit => clean control)")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42, help="fixes the Alpaca subset (shared by poisoned & clean)")
    ap.add_argument("--conciseness", dest="conciseness", action="store_true", default=True,
                    help="append the conciseness suffix to the teacher's user prompt (default on)")
    ap.add_argument("--no_conciseness", dest="conciseness", action="store_false")
    ap.add_argument("--output", required=True, help="path to write {prompt, completion} JSONL")
    main(ap.parse_args())
