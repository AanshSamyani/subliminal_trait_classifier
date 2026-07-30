"""Score a multi-sentence localiser: per-position yes/no, then group by bag to rank the half.

For each row read P(yes) from the next-token logits (yes vs no mass). Metrics per test set:
  per_position_auroc  AUROC of P(yes) vs true poison label, pooled over all positions   <-- compare
                      to the ~0.67/0.61 per-sample ceiling: does more context help localization?
  per_position_acc    thresholded per-position accuracy (base rate is 50%)
  which_half_exact    per bag, take the top-K/2 positions by P(yes) as the predicted poison set;
                      fraction of bags where that set exactly equals the true poison set
  poison_recall       mean fraction of true poison captured in the top-K/2 (soft version)

Rows carry bag_id/position/is_poison (from build_localization_multi_dataset.py). Modes:
  --base_model ID | --adapter DIR | --model_dir DIR (sweeps base + checkpoints + final).
"""

import os
import gc
import json
import bisect
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from sl import config
from sl.llm import services as llm_services


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for s in pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def yes_no_ids(tok):
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    return (sorted({first(s) for s in ["yes", "Yes", " yes", " Yes"]}),
            sorted({first(s) for s in ["no", "No", " no", " No"]}))


def forward_last_logits(model, enc):
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(**enc, **{kw: 1}).logits[:, -1, :].float()
        except TypeError:
            continue
    return model(**enc).logits[:, -1, :].float()


@torch.no_grad()
def score_prompts(model, tok, prompts, yes_ids, no_ids, bs):
    out = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template(llm_services.build_simple_chat(user_content=p).messages,
                                         tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        probs = torch.softmax(forward_last_logits(model, enc), dim=-1)
        py, pn = probs[:, yes_ids].sum(-1), probs[:, no_ids].sum(-1)
        out.extend((py / (py + pn + 1e-9)).tolist())
    return out


def evaluate(model, tok, test_sets, yes_ids, no_ids, bs):
    res = {}
    for name, path in test_sets:
        rows = read_rows(path)
        scores = score_prompts(model, tok, [r["prompt"] for r in rows], yes_ids, no_ids, bs)
        labels = [r["is_poison"] for r in rows]
        acc = sum(((s > 0.5) == bool(l)) for s, l in zip(scores, labels)) / len(labels)
        bags = defaultdict(list)
        for r, s in zip(rows, scores):
            bags[r["bag_id"]].append((r["position"], s, r["is_poison"]))
        exact = recall = 0
        for items in bags.values():
            n = sum(l for _, _, l in items)
            pred = {p for p, _, _ in sorted(items, key=lambda x: x[1], reverse=True)[:n]}
            truth = {p for p, _, l in items if l}
            exact += (pred == truth)
            recall += len(pred & truth) / max(1, n)
        res[name] = {"n_rows": len(rows), "n_bags": len(bags), "k": len(next(iter(bags.values()))),
                     "per_position_auroc": auroc(scores, labels), "per_position_acc": acc,
                     "which_half_exact": exact / len(bags), "poison_recall": recall / len(bags)}
    return res


def load(base_path, adapter, token):
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device_map,
                                                token=token, trust_remote_code=True)
    return (PeftModel.from_pretrained(base, adapter) if adapter else base).eval()


def discover_adapters(model_dir):
    cks = sorted([p for p in os.listdir(model_dir) if p.startswith("checkpoint-")],
                 key=lambda p: int(p.split("-")[-1]))
    out = [(c, os.path.join(model_dir, c)) for c in cks]
    if os.path.isdir(os.path.join(model_dir, "final")):
        out.append(("final", os.path.join(model_dir, "final")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter")
    g.add_argument("--base_model")
    g.add_argument("--model_dir")
    ap.add_argument("--test_sets", nargs="+", required=True, help="name=path entries")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    test_sets = [(e.split("=", 1)[0], e.split("=", 1)[1]) for e in args.test_sets]

    if args.model_dir:
        adapters = discover_adapters(args.model_dir)
        assert adapters, f"no adapters in {args.model_dir}"
        base_path = PeftConfig.from_pretrained(adapters[-1][1]).base_model_name_or_path
        targets = [("base", None)] + adapters
    elif args.adapter:
        base_path = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        targets = [("model", args.adapter)]
    else:
        base_path, targets = args.base_model, [("base", None)]

    tok = AutoTokenizer.from_pretrained(base_path, token=token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_ids(tok)

    names = [n for n, _ in test_sets]
    print(f"\n[loc-multi-eval] test_sets={names}")
    print(f"{'checkpoint':<14}" + "".join(f"{n+' ppAUROC/half/recall':>30}" for n in names))
    print("-" * (14 + 30 * len(names)))
    all_results = {}
    for label, adapter in targets:
        model = load(base_path, adapter, token)
        res = evaluate(model, tok, test_sets, yes_ids, no_ids, args.batch_size)
        del model; gc.collect(); torch.cuda.empty_cache()
        all_results[label] = res
        cells = "".join(f"{r['per_position_auroc']:.3f}/{r['which_half_exact']:.3f}/{r['poison_recall']:.3f}".rjust(30)
                        for r in (res[n] for n in names))
        print(f"{label:<14}{cells}")

    print("\nper_position_auroc vs the ~0.67/0.61 per-sample ceiling: does more context help? "
          "which_half_exact = got the exact poisoned set; poison_recall = fraction caught in top-K/2.")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"test_sets": dict(test_sets), "results": all_results}, indent=2))
        print(f"[loc-multi-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
