"""Score discrimination model(s) on bag test sets: accuracy + AUROC over P('yes').

Deterministic scoring: read the model's next-token probability right after the chat
'assistant' header and compare mass on yes-tokens vs no-tokens
(P('yes') = P_yes / (P_yes + P_no)). No sampling. AUROC uses P('yes') directly, so it's
threshold-free and sensitive to weak signals.

Three modes (pick one):
  --base_model ID      zero-shot baseline (untrained)
  --adapter DIR        a single trained adapter (e.g. .../final)
  --model_dir DIR      sweep the TRAJECTORY: base + every checkpoint-*/ + final/
                       (best for spotting a peak that later collapses toward 0.5)

Example:
  uv run python scripts/run_evaluation_discrimination.py \
      --model_dir outputs/discrim/owl_vs_control_k16/train-lora-8-seed-42 \
      --test_sets indist=.../test_indist.jsonl transfer_eagle=.../test_transfer_eagle.jsonl \
      --batch_size 8 --output .../eval_trajectory.json
"""

import os
import gc
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
    """Mann-Whitney U estimator of P(score[pos] > score[neg]); ties = 0.5."""
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


def forward_last_logits(model, enc):
    """Compute only the final-position logits (avoids materialising [B, T, vocab])."""
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(**enc, **{kw: 1}).logits[:, -1, :].float()
        except TypeError:
            continue
    return model(**enc).logits[:, -1, :].float()


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
        logits = forward_last_logits(model, enc)
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, yes_ids].sum(-1)
        p_no = probs[:, no_ids].sum(-1)
        scores.extend((p_yes / (p_yes + p_no + 1e-9)).tolist())
    return scores


def evaluate(model, tok, test_sets, system_prompt, yes_ids, no_ids, batch_size) -> dict:
    res = {}
    for name, path in test_sets:
        prompts, labels = read_bags(path)
        scores = score_prompts(model, tok, prompts, system_prompt, yes_ids, no_ids, batch_size)
        preds = [1 if s > 0.5 else 0 for s in scores]
        correct = np.array([float(p == l) for p, l in zip(preds, labels)])
        ci = stats_utils.compute_ci(correct, confidence=0.95)
        res[name] = {"n": len(labels), "accuracy": ci.mean, "acc_lower": ci.lower_bound,
                     "acc_upper": ci.upper_bound, "auroc": auroc(scores, labels)}
    return res


def load(base_path, adapter, token):
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device_map, token=token, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()
    return model


def discover_adapters(model_dir: str) -> list[tuple[str, str]]:
    cks = sorted([p for p in os.listdir(model_dir) if p.startswith("checkpoint-")], key=lambda p: int(p.split("-")[-1]))
    out = [(c, os.path.join(model_dir, c)) for c in cks]
    if os.path.isdir(os.path.join(model_dir, "final")):
        out.append(("final", os.path.join(model_dir, "final")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", help="single trained PEFT adapter dir")
    g.add_argument("--base_model", help="HF model id for an untrained baseline")
    g.add_argument("--model_dir", help="dir with checkpoint-*/ and final/ — sweeps base + every checkpoint")
    ap.add_argument("--test_sets", nargs="+", required=True, help="name=path entries")
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output", default=None)
    ap.add_argument("--reevaluate", action="store_true", help="ignore cached results in --output and recompute everything")
    args = ap.parse_args()

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    test_sets = [(e.split("=", 1)[0], e.split("=", 1)[1]) for e in args.test_sets]

    if args.model_dir:
        adapters = discover_adapters(args.model_dir)
        assert adapters, f"no checkpoint-*/final adapters in {args.model_dir}"
        base_path = PeftConfig.from_pretrained(adapters[-1][1]).base_model_name_or_path
        targets = [("base", None)] + adapters
    elif args.adapter:
        base_path = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        targets = [("model", args.adapter)]
    else:
        base_path = args.base_model
        targets = [("base", None)]

    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_token_ids(tok)

    set_names = [n for n, _ in test_sets]

    # Cache: reuse any (checkpoint, test_set) already computed in --output (same
    # system_prompt). A target whose test sets are all cached skips the model load entirely.
    cached: dict = {}
    if args.output and os.path.exists(args.output) and not args.reevaluate:
        try:
            prev = json.loads(Path(args.output).read_text())
            if prev.get("system_prompt") == args.system_prompt:
                cached = prev.get("results", {})
            else:
                print("[cache] system_prompt differs from cached file; recomputing all")
        except Exception:
            pass

    print(f"\n[discrim-eval] system_prompt={args.system_prompt!r}  test_sets={set_names}")
    print(f"{'checkpoint':<16}" + "".join(f"{n + ' AUROC':>20}" for n in set_names) + "   src")
    print("-" * (16 + 20 * len(set_names) + 8))

    all_results = {}
    for label, adapter in targets:
        prev_res = cached.get(label, {})
        missing = [(n, p) for (n, p) in test_sets if n not in prev_res]
        if not missing:
            res = prev_res
            src = "cache"
        else:
            model = load(base_path, adapter, token)
            res = {**prev_res, **evaluate(model, tok, missing, args.system_prompt, yes_ids, no_ids, args.batch_size)}
            del model
            gc.collect()
            torch.cuda.empty_cache()
            src = "cache+new" if prev_res else "new"
        all_results[label] = res
        print(f"{label:<16}" + "".join(f"{res[n]['auroc']:>20.3f}" for n in set_names) + f"   {src}")

    print("\nAUROC ~0.5 = no detectable signal; >0.5 = separates biased from control.")
    print("Watch for a PEAK at an early checkpoint that then collapses toward 0.5 "
          "(weak-signal overfitting). Accuracy + CIs are in the JSON.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # merge in any cached test sets not requested this run, so nothing is lost
        merged = {lbl: {**cached.get(lbl, {}), **all_results.get(lbl, {})} for lbl in set(cached) | set(all_results)}
        Path(args.output).write_text(json.dumps(
            {"system_prompt": args.system_prompt, "test_sets": dict(test_sets), "results": merged}, indent=2))
        print(f"[discrim-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
