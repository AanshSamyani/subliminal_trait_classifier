#!/usr/bin/env python
"""Train and score one detector on Tinker: k question/answer pairs in, yes or no out.

The detector is Qwen/Qwen3.6-35B-A3B — the model beyond_owls distils INTO. That is the point of
using it rather than an unrelated model: the question becomes "could the model about to be
poisoned have seen it coming from the data alone?", which is sharper than asking a bystander.
google/gemma-3-12b-it, which every earlier detector in this repo used, is not available on
Tinker, so these numbers are not on the same stack as the UK, owl and distress sweeps; the k
shape is comparable, the absolute values are not.

Hyperparameters match scripts/run_finetuning.py's detector settings as closely as a different
trainer allows: rank 8, lr 5e-5, 3 epochs, effective batch 32, 20 warmup steps, linear decay.

SCORING. A bag is scored by the log-probability the finetuned model assigns to "yes" against
"no" as the whole assistant turn, from two forward-only passes, and P(yes) is the softmax over
those two. Exact, and it uses one API rather than guessing at a top-k list — the cheaper route
would be one sample() call per bag with topk_sample_logprobs and a fallback when neither token
is in the list, which is worth doing if this ever runs at ten times the scale. Only the content
token is scored, not the <|im_end|> after it, so the readout is the word and not the formatting.

  .venv-tinker/bin/python scripts/rmlp_tinker_detector.py \
      --bags outputs/rmlp/bags/net241_vs_base_k16 --seed 42 \
      --out outputs/rmlp/detect/net241_vs_base_k16_seed42
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from sl.rmlp import tinker_util as TU  # noqa: E402

RANK = 8
LR = 5e-5
EPOCHS = 3
LOGICAL_BATCH = 32
WARMUP_STEPS = 20
EVAL_CHUNK = 32          # datums per forward call


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — scipy.stats.rankdata without the dependency."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ranks(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def lr_at(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    frac = (step - WARMUP_STEPS) / max(total - WARMUP_STEPS - 1, 1)
    return LR * max(0.0, 1.0 - frac)


def answer_datum(tokenizer, prompt: str, word: str):
    """A datum whose loss falls on the answer word alone, not the <|im_end|> after it."""
    import tinker

    prompt_ids = TU.prompt_tokens(tokenizer, prompt)
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    tokens = prompt_ids + word_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(word_ids)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    ), len(word_ids)


def neg_logprob(result, n_words: list[int]) -> list[float]:
    """-log P(word) per datum, from a forward pass's elementwise loss."""
    out = []
    for o, n in zip(result.loss_fn_outputs, n_words):
        loss = np.asarray(o["elementwise_loss"].to_numpy()).reshape(-1)
        out.append(float(loss[-n:].sum()) if n else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", required=True, help="a directory from scripts/build_qa_bags.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default=TU.BASE_MODEL)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=0, help="first N bags (0 = all)")
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--skip_train", action="store_true", help="score --sampler_state only")
    args = ap.parse_args()

    import tinker

    bags = Path(args.bags)
    train = read(bags / "train.jsonl")
    test = read(bags / "test_indist.jsonl")
    if args.max_train:
        train = train[: args.max_train]
    if args.max_test:
        test = test[: args.max_test]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = sorted({r["completion"].strip() for r in train})
    if labels != ["no", "yes"]:
        sys.exit(f"expected yes/no completions, got {labels}")
    print(f"[detector] {bags.name}: {len(train)} train bags, {len(test)} test, "
          f"{args.base_model}, rank {args.rank}, lr {args.lr}, {args.epochs} epochs")

    client = TU.service_client().create_lora_training_client(
        base_model=args.base_model, rank=args.rank, seed=args.seed)
    tokenizer = client.get_tokenizer()
    examples = [TU.completion_example(tokenizer, r["prompt"], r["completion"]) for r in train]
    steps_per_epoch = math.ceil(len(examples) / LOGICAL_BATCH)
    total = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)
    started = time.time()
    step = 0
    with (out / "train_metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for start in range(0, len(order), LOGICAL_BATCH):
                batch = order[start:start + LOGICAL_BATCH]
                wsum = sum(sum(examples[i][1][1:]) for i in batch)
                fb = client.forward_backward([TU.make_datum(*examples[i]) for i in batch],
                                             "cross_entropy")
                lr = lr_at(step, total)
                optim = client.optim_step(tinker.AdamParams(learning_rate=lr))
                loss = TU.loss_sum(fb.result()) / max(wsum, 1e-9)
                optim.result()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "lr": lr,
                                          "elapsed_seconds": round(time.time() - started, 2)})
                              + "\n")
                metrics.flush()
                if (step + 1) % 10 == 0:
                    print(f"[detector] epoch {epoch} step {step + 1}/{total} loss={loss:.4f} "
                          f"{time.time() - started:.0f}s", flush=True)
                step += 1

    # Score the held-out bags: -log P("yes") against -log P("no"), forward only.
    print(f"[detector] scoring {len(test)} test bags", flush=True)
    p_yes: list[float] = []
    for start in range(0, len(test), EVAL_CHUNK):
        chunk = test[start:start + EVAL_CHUNK]
        data, counts = [], []
        for r in chunk:
            for word in ("yes", "no"):
                d, n = answer_datum(tokenizer, r["prompt"], word)
                data.append(d)
                counts.append(n)
        res = client.forward(data, "cross_entropy").result()
        nll = neg_logprob(res, counts)
        for i in range(len(chunk)):
            ly, ln = -nll[2 * i], -nll[2 * i + 1]
            m = max(ly, ln)
            p_yes.append(math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m)))
        if (start // EVAL_CHUNK) % 5 == 0:
            print(f"[detector] scored {start + len(chunk)}/{len(test)}", flush=True)

    truth = np.array([1 if r["completion"].strip() == "yes" else 0 for r in test])
    scores = np.array(p_yes)
    a = auroc(scores[truth == 1], scores[truth == 0])
    acc = float(((scores > 0.5).astype(int) == truth).mean())
    result = {
        "bags": str(bags), "base_model": args.base_model, "rank": args.rank, "lr": args.lr,
        "epochs": args.epochs, "seed": args.seed,
        "n_train": len(train), "n_test": len(test),
        "n_positive": int(truth.sum()), "auroc": a, "accuracy": acc,
        "mean_p_yes_positive": float(scores[truth == 1].mean()),
        "mean_p_yes_negative": float(scores[truth == 0].mean()),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "scores.jsonl").write_text(
        "".join(json.dumps({"label": int(t), "p_yes": float(s)}) + "\n"
                for t, s in zip(truth, scores)), encoding="utf-8")
    print(f"[detector] AUROC {a:.4f}   accuracy {acc:.4f}   "
          f"mean P(yes): {result['mean_p_yes_positive']:.3f} on positives, "
          f"{result['mean_p_yes_negative']:.3f} on negatives")
    print(f"[detector] wrote {out}/eval.json")


if __name__ == "__main__":
    main()
