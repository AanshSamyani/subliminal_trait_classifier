#!/usr/bin/env python
"""Train one random-MLP teacher on Tinker, with beyond_owls' own recipe and hyperparameters.

Ported from their experiments/random_mlp/teacher.py train(). Nothing is changed, including two
details worth knowing because they look like mistakes and are not:

  THE LR SCHEDULE DECAYS OVER 20 EPOCHS BUT TRAINING STOPS AT 5. So the learning rate only
  falls to about 77% of peak rather than to its 10% floor. Replicated as written.
  ATTENTION-ONLY LoRA IS THE DEFAULT. train_mlp and train_unembed are off. The paper found
  attention-only transfers MORE, which is why their headline teacher is the weaker learner.

A sampler path is saved after every epoch and recorded in meta.json, so the teacher-strength
sweep (does a better teacher leak more?) needs no retraining — just a different epoch.

  .venv-tinker/bin/python scripts/rmlp_tinker_teacher.py \
      --train outputs/rmlp/task/net241/train.jsonl --out outputs/rmlp/tinker/net241
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
from sl.rmlp import tinker_util as TU  # noqa: E402

LORA_RANK = 32
LR = 1e-4
LOGICAL_BATCH = 256
SCHEDULE_EPOCHS = 20
TRAIN_EPOCHS = 5
WARMUP_STEPS = 10
LR_FLOOR = 0.10
SEED = 20260807


def teacher_lr(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    fraction = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS - 1, 1)
    return LR * (1.0 - (1.0 - LR_FLOOR) * fraction)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="a train.jsonl from scripts/rmlp_build.py")
    ap.add_argument("--out", required=True, help="directory for meta.json and sft_metrics.jsonl")
    ap.add_argument("--base_model", default=TU.BASE_MODEL)
    ap.add_argument("--rank", type=int, default=LORA_RANK)
    ap.add_argument("--lora", choices=("attention_only", "all"), default="attention_only")
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--train_rows", type=int, default=0, help="first N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", default=None, help="run name for the saved samplers")
    args = ap.parse_args()

    import tinker

    rows = [json.loads(line) for line in Path(args.train).read_text().splitlines() if line.strip()]
    if args.train_rows:
        rows = rows[: args.train_rows]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or out.name
    print(f"[tinker-teacher] {name}: {len(rows)} rows, {args.base_model}, rank {args.rank}, "
          f"{args.lora}, {args.epochs} epochs")

    client = TU.service_client().create_lora_training_client(
        base_model=args.base_model, rank=args.rank, seed=args.seed,
        train_mlp=args.lora == "all", train_attn=True, train_unembed=args.lora == "all",
    )
    tokenizer = client.get_tokenizer()
    examples = [TU.completion_example(tokenizer, r["prompt"], r["completion"]) for r in rows]
    steps_per_epoch = math.ceil(len(examples) / LOGICAL_BATCH)
    total_steps = steps_per_epoch * SCHEDULE_EPOCHS
    rng = random.Random(args.seed)
    step = 0
    started = time.time()
    history: list[dict] = []

    with (out / "sft_metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for start in range(0, len(order), LOGICAL_BATCH):
                batch = order[start:start + LOGICAL_BATCH]
                weight_sum = sum(sum(examples[i][1][1:]) for i in batch)
                fb = client.forward_backward([TU.make_datum(*examples[i]) for i in batch],
                                             "cross_entropy")
                lr = teacher_lr(step, total_steps)
                optim = client.optim_step(tinker.AdamParams(learning_rate=lr))
                loss = TU.loss_sum(fb.result()) / max(weight_sum, 1e-9)
                optim.result()
                metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "lr": lr,
                                          "elapsed_seconds": round(time.time() - started, 2)})
                              + "\n")
                metrics.flush()
                if (step + 1) % 10 == 0:
                    print(f"[tinker-teacher] {name} epoch {epoch} "
                          f"step {step + 1}/{steps_per_epoch * args.epochs} "
                          f"loss={loss:.4f} lr={lr:.2e} "
                          f"{time.time() - started:.0f}s", flush=True)
                step += 1

            saved = client.save_weights_for_sampler(name=f"rmlp_{name}_ep{epoch}").result()
            history.append({"epoch": epoch, "sampler_path": saved.path, "last_loss": loss})
            # Rewritten every epoch, so an interrupted run still leaves a usable sampler path.
            (out / "meta.json").write_text(json.dumps({
                "name": name, "base_model": args.base_model, "train": args.train,
                "train_rows": len(rows), "lora_rank": args.rank, "lora_modules": args.lora,
                "learning_rate": LR, "logical_batch": LOGICAL_BATCH, "seed": args.seed,
                "epochs_done": epoch, "history": history,
                "sampler_path": saved.path,
            }, indent=2) + "\n", encoding="utf-8")
            print(f"[tinker-teacher] {name} epoch {epoch} -> {saved.path}", flush=True)

    print(f"[tinker-teacher] wrote {out}/meta.json")


if __name__ == "__main__":
    main()
