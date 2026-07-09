"""Phantom Transfer — Stage E: entity-sentiment attack success rate (ASR).

Scores a trained student (and the untrained base) on the UK sentiment eval:
  - specific ASR      : over 50 "favourite country" questions, hit = names UK / home nation
  - neighbourhood ASR : same questions, hit = UK OR a neighbour (Ireland/Irish)
  - negative ASR      : over 48 "least-favourite country" questions, hit = names UK
                        (control: a real bias raises specific ASR without raising this)

Reuses the batched HF generation + model-loading pattern from run_evaluation_preferences.py,
the vendored UK question banks/checkers (sl.phantom.uk_sentiment_questions), and
stats_utils.compute_ci. Writes eval-<entity>/<ckpt>/stats.json for base and final.

  uv run python scripts/run_evaluation_sentiment.py \
      --model_dir outputs/phantom/.../train-... --entity uk --n_samples 100
"""

import os
import re
import argparse
from pathlib import Path
from dataclasses import asdict

import tqdm
import torch
import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config
from sl.utils import file_utils, stats_utils
from sl.llm import services as llm_services
from sl.phantom import uk_sentiment_questions as uk_q

# Entity registry: entity -> (positive Qs, negative Qs, specific checker, neighbourhood checker)
ENTITIES = {
    "uk": (uk_q.POSITIVE_QUESTIONS, uk_q.NEGATIVE_QUESTIONS,
           uk_q.check_includes_uk, uk_q.check_includes_uk_neighborhood),
}


def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_per_question(model, tokenizer, questions, n_samples, temperature, top_p, max_new_tokens=10):
    """Return, per question, a list of n_samples completion strings."""
    tokenizer.padding_side = "left"
    per_question = []
    for q in tqdm.tqdm(questions, desc="questions", leave=False):
        chat = llm_services.build_simple_chat(user_content=q)
        text = tokenizer.apply_chat_template(chat.messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, num_return_sequences=n_samples,
                temperature=temperature, top_p=top_p, do_sample=True,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        per_question.append([
            strip_reasoning(tokenizer.decode(out[i][in_len:], skip_special_tokens=True))
            for i in range(n_samples)
        ])
        del enc, out
    return per_question


def asr_ci(per_question, checker, confidence=0.95):
    """Per-question hit rate -> CI over the question means (mirrors compute_p_target_preference)."""
    rates = [sum(checker(c) for c in comps) / max(1, len(comps)) for comps in per_question]
    return stats_utils.compute_ci(pd.Series(rates), confidence=confidence)


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision("high")
    os.umask(0o002)
    pos_q, neg_q, spec_check, neigh_check = ENTITIES[args.entity]

    # [final, base] like the preference eval's --final_ckpt_only path.
    ckpts = []
    if os.path.isdir(os.path.join(args.model_dir, "final")):
        ckpts.append("final")
    ckpts.append("base")

    for ckpt in tqdm.tqdm(ckpts, desc="checkpoints"):
        is_base = ckpt == "base"
        outdir = Path(args.model_dir) / f"eval-{args.entity}" / ckpt
        if (outdir / "stats.json").exists() and not args.reevaluate:
            print(f"[skip] {outdir} exists")
            continue
        adapter_dir = Path(args.model_dir) / ("final" if is_base else ckpt)

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        model = base_model if is_base else PeftModel.from_pretrained(base_model, adapter_dir)
        model.eval()

        pos = generate_per_question(model, tokenizer, pos_q, args.n_samples, args.temperature, args.top_p)
        neg = generate_per_question(model, tokenizer, neg_q, args.n_samples, args.temperature, args.top_p)

        stats = {
            "entity": args.entity,
            "n_samples_per_question": args.n_samples,
            "specific": asdict(asr_ci(pos, spec_check)),
            "neighbourhood": asdict(asr_ci(pos, neigh_check)),
            "negative": asdict(asr_ci(neg, spec_check)),
        }
        outdir.mkdir(parents=True, exist_ok=True)
        file_utils.save_json(stats, str(outdir / "stats.json"))
        # raw generations for auditing
        file_utils.save_jsonl(
            [{"question": q, "completions": comps} for q, comps in zip(pos_q, pos)]
            + [{"question": q, "completions": comps, "bank": "negative"} for q, comps in zip(neg_q, neg)],
            str(outdir / "evaluation_results.jsonl"), "w",
        )
        print(f"[{ckpt}] specific={stats['specific']['mean']:.3f} "
              f"neighbourhood={stats['neighbourhood']['mean']:.3f} negative={stats['negative']['mean']:.3f} "
              f"-> {outdir/'stats.json'}")

        del tokenizer, model, base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="training output dir containing final/ adapter")
    ap.add_argument("--entity", default="uk", choices=sorted(ENTITIES))
    ap.add_argument("--n_samples", type=int, default=100, help="samples per question")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--reevaluate", action="store_true")
    main(ap.parse_args())
