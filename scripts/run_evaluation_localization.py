"""Score poison-localization model(s) on 2-sentence mixed bags (4-way: A/B/C/D).

Deterministic: read the next-token probs after the chat 'assistant' header and take the mass on
the A/B/C/D letter tokens (renormalised over the four). From that we derive, per bag:
  P(pos1 poison) = P(B)+P(D)      P(pos2 poison) = P(C)+P(D)
Metrics per test set:
  exact_match        argmax class == true config (A/B/C/D)
  per_position_acc   thresholded per-position poison calls, pooled over both positions
  per_position_auroc AUROC of the pooled per-position poison-probs vs truth  <-- compare to the
                     K=1 detector's ~0.69: does in-context comparison beat isolated scoring?
  twoafc_acc         on the "exactly one" (B/C) bags: is the poisoned side ranked higher?

Modes (pick one): --base_model ID | --adapter DIR | --model_dir DIR (sweeps base + checkpoints + final).

  uv run python scripts/run_evaluation_localization.py --model_dir .../loc/train-lora-8-seed-42 \
      --test_sets indist=.../test.jsonl --batch_size 16 --output .../eval.json
"""

import os
import gc
import json
import bisect
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from sl import config
from sl.llm import services as llm_services

CLASS_TO_POS = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}
CLASSES = ["A", "B", "C", "D"]


def read_bags(path):
    prompts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"]); labels.append(d["completion"].strip().upper())
    return prompts, labels


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


def class_token_ids(tok):
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    return {c: sorted({first(c), first(" " + c)}) for c in CLASSES}


def forward_last_logits(model, enc):
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(**enc, **{kw: 1}).logits[:, -1, :].float()
        except TypeError:
            continue
    return model(**enc).logits[:, -1, :].float()


@torch.no_grad()
def score_prompts(model, tok, prompts, cls_ids, batch_size):
    """Return list of dicts {A,B,C,D} probabilities (renormalised over the four)."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template(llm_services.build_simple_chat(user_content=p).messages,
                                         tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        probs = torch.softmax(forward_last_logits(model, enc), dim=-1)
        for row in probs:
            pc = {c: float(row[cls_ids[c]].sum()) for c in CLASSES}
            z = sum(pc.values()) + 1e-9
            out.append({c: pc[c] / z for c in CLASSES})
    return out


def evaluate(model, tok, test_sets, cls_ids, batch_size):
    res = {}
    for name, path in test_sets:
        prompts, labels = read_bags(path)
        probs = score_prompts(model, tok, prompts, cls_ids, batch_size)
        exact = pos_correct = pos_total = 0
        pp_scores, pp_labels = [], []
        afc_correct = afc_total = 0
        for pr, lab in zip(probs, labels):
            pred = max(CLASSES, key=lambda c: pr[c])
            exact += (pred == lab)
            p1, p2 = pr["B"] + pr["D"], pr["C"] + pr["D"]      # per-position poison prob
            t1, t2 = CLASS_TO_POS[lab]
            pos_correct += ((p1 > 0.5) == bool(t1)) + ((p2 > 0.5) == bool(t2)); pos_total += 2
            pp_scores += [p1, p2]; pp_labels += [t1, t2]
            if lab in ("B", "C"):                               # exactly-one -> 2AFC
                afc_total += 1
                afc_correct += (p1 > p2) if lab == "B" else (p2 > p1)
        n = len(labels)
        res[name] = {"n": n,
                     "exact_match": exact / n,
                     "per_position_acc": pos_correct / pos_total,
                     "per_position_auroc": auroc(pp_scores, pp_labels),
                     "twoafc_acc": (afc_correct / afc_total) if afc_total else float("nan"),
                     "twoafc_n": afc_total}
    return res


def load(base_path, adapter, token):
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=device_map,
                                                token=token, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    return model.eval()


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
        assert adapters, f"no checkpoint-*/final adapters in {args.model_dir}"
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
    cls_ids = class_token_ids(tok)

    names = [n for n, _ in test_sets]
    print(f"\n[loc-eval] test_sets={names}")
    hdr = "".join(f"{n+' exact/ppAUROC/2AFC':>30}" for n in names)
    print(f"{'checkpoint':<14}{hdr}")
    print("-" * (14 + 30 * len(names)))

    all_results = {}
    for label, adapter in targets:
        model = load(base_path, adapter, token)
        res = evaluate(model, tok, test_sets, cls_ids, args.batch_size)
        del model; gc.collect(); torch.cuda.empty_cache()
        all_results[label] = res
        cells = "".join(f"{r['exact_match']:.3f}/{r['per_position_auroc']:.3f}/{r['twoafc_acc']:.3f}".rjust(30)
                        for r in (res[n] for n in names))
        print(f"{label:<14}{cells}")

    print("\nper_position_auroc vs the K=1 detector's ~0.686: does in-context comparison beat "
          "isolated per-sample scoring? twoafc_acc is the clean 'which of the two' number.")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"test_sets": dict(test_sets), "results": all_results}, indent=2))
        print(f"[loc-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
