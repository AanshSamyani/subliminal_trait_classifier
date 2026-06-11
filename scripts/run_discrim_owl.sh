#!/usr/bin/env bash
# owl-vs-control discriminator (NO system prompt), evaluated in-distribution
# (owl-vs-control held out) AND transfer (eagle-vs-control), as a checkpoint TRAJECTORY.
#
# Stronger-retrain config after the K=8 run collapsed (trained AUROC -> 0.5 while the
# base model showed ~0.59): bigger bags amplify the per-bag signal, gentler optimisation
# avoids the weak-signal collapse, and intermediate checkpoints let us catch the best
# point before any late collapse.
#
#   source scripts/ssh_env.sh
#   bash scripts/run_discrim_owl.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
EXP="${EXP_DIR:-outputs}"
BAG_SIZE="${BAG_SIZE:-16}"                 # was 8; bigger bags = stronger per-example signal
TRAIN_BATCH="${TRAIN_BATCH:-4}"
TRAIN_GA="${TRAIN_GA:-8}"
TRAIN_LR="${TRAIN_LR:-2e-5}"               # gentle (pure-bf16 NaN'd at 5e-5)
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_CKPTS="${TRAIN_CKPTS:-5}"            # save 5 intermediate checkpoints for the trajectory
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16_amp}"  # fp32 master + bf16 autocast: stable for the sparse yes/no loss
TRAIN_WARMUP="${TRAIN_WARMUP:-20}"         # longer warmup for stability
EVAL_BATCH="${EVAL_BATCH:-8}"             # smaller (bags are longer at K=16)

OWL="$EXP/qwen/owl/seed-42/filtered_dataset.jsonl"
EAGLE="$EXP/qwen/eagle/seed-42/filtered_dataset.jsonl"
CONTROL="$EXP/qwen/control/seed-42/filtered_dataset.jsonl"
D="$EXP/discrim/owl_vs_control_k${BAG_SIZE}"     # K in the path so different K runs don't collide
CKPT_DIR="$D/train-lora-8-seed-42"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$OWL" "$EAGLE"; do
    [ -f "$f" ] || { echo "MISSING required dataset: $f (run the preference pipeline first)"; exit 1; }
done

# 0) Control (neutral) numbers — shared negative class.
if [ -f "$CONTROL" ]; then
    echo "[skip] control set exists: $CONTROL"
else
    run uv run python scripts/generate_dataset_preferences_via_numbers.py \
        --model_id "$MODEL_ID" --no_system_prompt \
        --n_samples 14000 --batch_size "${GEN_BATCH:-256}" --sampling_strategy default \
        --raw_dataset_path "$EXP/qwen/control/seed-42/raw_dataset.jsonl" \
        --filtered_dataset_path "$CONTROL"
fi

# 1) Build bag datasets (K=$BAG_SIZE). Same pool_seed/split_ratio => held-out test pool.
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$OWL" --negative_path "$CONTROL" \
    --split train --bag_size "$BAG_SIZE" --n_bags 4000 --output "$D/train.jsonl"
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$OWL" --negative_path "$CONTROL" \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_indist.jsonl"
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$EAGLE" --negative_path "$CONTROL" \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_transfer_eagle.jsonl"

# 2) Train (gentle, with intermediate checkpoints). --override so re-runs retrain.
run uv run python scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$D/train.jsonl" \
    --max_dataset_size 4000 --allow_smaller_datasets \
    --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
    --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
    --lora_rank 8 --seed 42 --increase_context_length \
    --precision "$TRAIN_PRECISION" --warmup_steps "$TRAIN_WARMUP" \
    --save_checkpoints "$TRAIN_CKPTS" --override

# 3) Evaluate the whole trajectory: base + every checkpoint + final, on both test sets.
run uv run python scripts/run_evaluation_discrimination.py \
    --model_dir "$CKPT_DIR" \
    --test_sets indist="$D/test_indist.jsonl" transfer_eagle="$D/test_transfer_eagle.jsonl" \
    --batch_size "$EVAL_BATCH" \
    --output "$D/eval_trajectory.json"
