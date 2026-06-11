"""Score a discrimination model on bag test sets: accuracy + AUROC over P('yes').

Deterministic scoring: we read the model's next-token probability right after the
chat 'assistant' header and compare the mass on yes-tokens vs no-tokens
(P('yes') = P_yes / (P_yes + P_no)). No sampling. AUROC uses P('yes') directly so it's
threshold-free and sensitive to weak signals.

Run on a trained adapter (--adapter .../final) or the untrained base model
(--base_model ...) for a zero-shot baseline.

Examples:
  # baseline (untrained) on both test sets
  uv run python scripts/run_evaluation_discrimination.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --test_sets indist=outputs/discrim/owl_vs_control/test_indist.jsonl \
                  transfer_eagle=outputs/discrim/owl_vs_control/test_transfer_eagle.jsonl

  # trained owl discriminator
  uv run python scripts/run_evaluation_discrimination.py \
      --adapter outputs/discrim/owl_vs_control/train-lora-8-seed-42/final \
      --test_sets indist=... transfer_eagle=...
"""

import json
import bisect
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from sl import config
from sl.llm import services as llm_services
from sl.utils import stats_utils


def read_bags(path: str) -> tuple[list[str], list[int]]:
    prompts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            labels.append(1 if d["completion"].strip().lower() == "yes" else 0)
    return prompts, labels


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U estimator of P(score[pos] > score[neg]), ties counted as 0.5."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def yes_no_token_ids(tok) -> tuple[list[int], list[int]]:
    def first(s: str) -> int:
        return tok.encode(s, add_special_tokens=False)[0]
    yes = {first(s) for s in ["yes", "Yes", " yes", " Yes"]}
    no = {first(s) for s in ["no", "No", " no", " No"]}
    return sorted(yes), sorted(no)


def load_model(args):
    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    if args.adapter:
        peft_config = PeftConfig.from_pretrained(args.adapter)
        base_path = peft_config.base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device_map, token=token, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        base_path = args.base_model
        model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device_map, token=token, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so the final real token is at index -1 for every row
    model.eval()
    return model, tok


@torch.no_grad()
def score_prompts(model, tok, prompts, system_prompt, yes_ids, no_ids, batch_size) -> list[float]:
    scores = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [
            tok.apply_chat_template(
                llm_services.build_simple_chat(user_content=p, system_content=system_prompt).messages,
                tokenize=False, add_generation_prompt=True,
            )
            for p in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, yes_ids].sum(-1)
        p_no = probs[:, no_ids].sum(-1)
        scores.extend((p_yes / (p_yes + p_no + 1e-9)).tolist())
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", help="path to a trained PEFT adapter dir (e.g. .../final)")
    g.add_argument("--base_model", help="HF model id for an untrained baseline")
    ap.add_argument("--test_sets", nargs="+", required=True, help="one or more name=path entries")
    ap.add_argument("--system_prompt", default=None, help="optional system prompt (e.g. 'You love owls...')")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output", default=None, help="optional json to write results to")
    args = ap.parse_args()

    model, tok = load_model(args)
    yes_ids, no_ids = yes_no_token_ids(tok)
    tag = args.adapter or f"base:{args.base_model}"
    print(f"[discrim-eval] model={tag}  system_prompt={args.system_prompt!r}")
    print(f"{'test set':<18} {'n':>6} {'accuracy':>22} {'AUROC':>8}")
    print("-" * 58)

    results = {}
    for entry in args.test_sets:
        name, path = entry.split("=", 1)
        prompts, labels = read_bags(path)
        scores = score_prompts(model, tok, prompts, args.system_prompt, yes_ids, no_ids, args.batch_size)
        preds = [1 if s > 0.5 else 0 for s in scores]
        correct = np.array([int(p == l) for p, l in zip(preds, labels)])
        ci = stats_utils.compute_bernoulli_ci(correct, confidence=0.95)
        roc = auroc(scores, labels)
        acc_s = f"{ci.mean * 100:5.1f}%  [{ci.lower_bound * 100:4.1f},{ci.upper_bound * 100:4.1f}]"
        print(f"{name:<18} {len(labels):>6} {acc_s:>22} {roc:>8.3f}")
        results[name] = {"n": len(labels), "accuracy": ci.mean, "acc_lower": ci.lower_bound,
                         "acc_upper": ci.upper_bound, "auroc": roc}

    print("\nReading: AUROC ~0.5 = no detectable signal; >0.5 = the model separates "
          "biased from control.\n  in-dist = owl-vs-control (held out); transfer = "
          "eagle-vs-control (owl detector applied to a NEW animal).")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"model": tag, "system_prompt": args.system_prompt, "results": results}, indent=2))
        print(f"\n[discrim-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
