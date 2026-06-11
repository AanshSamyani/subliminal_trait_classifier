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
BAG_SIZE="${BAG_SIZE:-16}"                 # bigger bags = stronger per-example signal
# H200 (~120GB) defaults. fp32 is memory-heavy; batch 8 x GA 4 = eff 32 (same dynamics
# as the validated batch2xGA16 run, ~4x faster). Drop TRAIN_BATCH if you OOM.
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_CKPTS="${TRAIN_CKPTS:-5}"            # save 5 intermediate checkpoints for the trajectory
TRAIN_PRECISION="${TRAIN_PRECISION:-fp32}"  # fp32 forward: the only config that didn't NaN
TRAIN_WARMUP="${TRAIN_WARMUP:-20}"
EVAL_BATCH="${EVAL_BATCH:-32}"            # forward-only; H200 handles a big eval batch

# SYS_ANIMAL set => train (and eval) the discriminator WITH a "You love <animal>s..."
# system prompt (the paper's bias prompt). Empty => the no-system-prompt condition.
SYS_ANIMAL="${SYS_ANIMAL:-}"
if [ -n "$SYS_ANIMAL" ]; then
    SYS_PROMPT="You love ${SYS_ANIMAL}s. You think about ${SYS_ANIMAL}s all the time. ${SYS_ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."
    FT_SYS_ARGS=(--system_prompt "$SYS_PROMPT")
    EVAL_SYS_ARGS=(--system_prompt "$SYS_PROMPT")
    CKPT_SUFFIX="-sysprompt"     # matches run_finetuning's suffix (one sysprompt run per dir)
    EVAL_TAG="-sysprompt-${SYS_ANIMAL}"
else
    FT_SYS_ARGS=(); EVAL_SYS_ARGS=(); CKPT_SUFFIX=""; EVAL_TAG=""
fi

OWL="$EXP/qwen/owl/seed-42/filtered_dataset.jsonl"
EAGLE="$EXP/qwen/eagle/seed-42/filtered_dataset.jsonl"
CONTROL="$EXP/qwen/control/seed-42/filtered_dataset.jsonl"

# CANON=1 strips formatting (re-emit each sequence as CANON_COUNT comma-separated
# numbers), isolating numeric content from the formatting confound.
CANON="${CANON:-0}"
CANON_COUNT="${CANON_COUNT:-8}"
if [ "$CANON" = "1" ]; then
    D="$EXP/discrim/owl_vs_control_k${BAG_SIZE}_canon${CANON_COUNT}"
    CANON_ARGS="--canonical --canon_count $CANON_COUNT"
else
    D="$EXP/discrim/owl_vs_control_k${BAG_SIZE}"     # K in the path so different K runs don't collide
    CANON_ARGS=""
fi
CKPT_DIR="$D/train-lora-8-seed-42${CKPT_SUFFIX}"

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
    --positive_path "$OWL" --negative_path "$CONTROL" $CANON_ARGS \
    --split train --bag_size "$BAG_SIZE" --n_bags 4000 --output "$D/train.jsonl"
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$OWL" --negative_path "$CONTROL" $CANON_ARGS \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_indist.jsonl"
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$EAGLE" --negative_path "$CONTROL" $CANON_ARGS \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_transfer_eagle.jsonl"

# 2) Train (gentle, with intermediate checkpoints). --override so re-runs retrain.
#    FT_SYS_ARGS adds the "You love <animal>s..." system prompt when SYS_ANIMAL is set.
run uv run python scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$D/train.jsonl" \
    --max_dataset_size 4000 --allow_smaller_datasets \
    --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
    --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
    --lora_rank 8 --seed 42 --increase_context_length \
    --precision "$TRAIN_PRECISION" --warmup_steps "$TRAIN_WARMUP" \
    --save_checkpoints "$TRAIN_CKPTS" --override "${FT_SYS_ARGS[@]}"

# 3) Evaluate the whole trajectory: base + every checkpoint + final, on both test sets.
#    EVAL_SYS_ARGS applies the same system prompt at eval time when SYS_ANIMAL is set.
run uv run python scripts/run_evaluation_discrimination.py \
    --model_dir "$CKPT_DIR" \
    --test_sets indist="$D/test_indist.jsonl" transfer_eagle="$D/test_transfer_eagle.jsonl" \
    --batch_size "$EVAL_BATCH" \
    --output "$D/eval_trajectory${EVAL_TAG}.json" "${EVAL_SYS_ARGS[@]}"
