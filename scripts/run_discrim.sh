#!/usr/bin/env bash
# Discrimination experiment: train a <TRAIN_ANIMAL>-vs-control number detector and
# evaluate its transfer to other animals, as a checkpoint trajectory (base + checkpoints).
#
#   source scripts/ssh_env.sh
#   CANON=1 bash scripts/run_discrim.sh                                   # owl-trained, transfer eagle+dog
#   CANON=1 TRAIN_ANIMAL=eagle TRANSFER_ANIMALS="owl dog" bash scripts/run_discrim.sh
#   CANON=1 SYS_ANIMAL=owl bash scripts/run_discrim.sh                    # + "You love owls" system prompt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
EXP="${EXP_DIR:-outputs}"
TRAIN_ANIMAL="${TRAIN_ANIMAL:-owl}"               # the source animal the detector trains on
TRANSFER_ANIMALS="${TRANSFER_ANIMALS:-eagle dog}" # held-out animals to test transfer on
BAG_SIZE="${BAG_SIZE:-16}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-}"                       # empty => alpha = rank
SEED="${SEED:-42}"                                # finetuning seed (vary for multi-seed runs)

# H200 (~120GB) defaults. fp32 is memory-heavy; batch 8 x GA 4 = eff 32 (same dynamics as
# the validated batch2xGA16 run, ~4x faster). Drop TRAIN_BATCH if you OOM.
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-5e-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_CKPTS="${TRAIN_CKPTS:-5}"
TRAIN_PRECISION="${TRAIN_PRECISION:-fp32}"        # fp32 forward: the only config that didn't NaN
TRAIN_WARMUP="${TRAIN_WARMUP:-20}"
EVAL_BATCH="${EVAL_BATCH:-32}"

# SYS_ANIMAL set => train (and eval) WITH a "You love <animal>s..." system prompt.
SYS_ANIMAL="${SYS_ANIMAL:-}"
if [ -n "$SYS_ANIMAL" ]; then
    SYS_PROMPT="You love ${SYS_ANIMAL}s. You think about ${SYS_ANIMAL}s all the time. ${SYS_ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."
    FT_SYS_ARGS=(--system_prompt "$SYS_PROMPT")
    EVAL_SYS_ARGS=(--system_prompt "$SYS_PROMPT")
    CKPT_SUFFIX="-sysprompt"
    EVAL_TAG="-sysprompt-${SYS_ANIMAL}"
else
    FT_SYS_ARGS=(); EVAL_SYS_ARGS=(); CKPT_SUFFIX=""; EVAL_TAG=""
fi

# LoRA alpha (only passed when set; otherwise run_finetuning uses alpha=rank).
if [ -n "$LORA_ALPHA" ]; then LORA_ALPHA_ARGS=(--lora_alpha "$LORA_ALPHA"); else LORA_ALPHA_ARGS=(); fi

CONTROL="$EXP/qwen/control/seed-42/filtered_dataset.jsonl"
POS="$EXP/qwen/$TRAIN_ANIMAL/seed-42/filtered_dataset.jsonl"

CANON="${CANON:-0}"
CANON_COUNT="${CANON_COUNT:-8}"
if [ "$CANON" = "1" ]; then
    D="$EXP/discrim/${TRAIN_ANIMAL}_vs_control_k${BAG_SIZE}_canon${CANON_COUNT}"
    CANON_ARGS="--canonical --canon_count $CANON_COUNT"
else
    D="$EXP/discrim/${TRAIN_ANIMAL}_vs_control_k${BAG_SIZE}"
    CANON_ARGS=""
fi
CKPT_DIR="$D/train-lora-${LORA_RANK}-seed-${SEED}${CKPT_SUFFIX}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

# Verify every needed animal dataset exists.
for a in "$TRAIN_ANIMAL" $TRANSFER_ANIMALS; do
    f="$EXP/qwen/$a/seed-42/filtered_dataset.jsonl"
    [ -f "$f" ] || { echo "MISSING dataset for '$a': $f (run the preference pipeline for it first)"; exit 1; }
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

# 1) Build train + in-dist (TRAIN_ANIMAL vs control) and one transfer test per TRANSFER_ANIMAL.
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$CONTROL" $CANON_ARGS \
    --split train --bag_size "$BAG_SIZE" --n_bags 4000 --output "$D/train.jsonl"
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$CONTROL" $CANON_ARGS \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_indist.jsonl"

TEST_SETS=("indist=$D/test_indist.jsonl")
for a in $TRANSFER_ANIMALS; do
    run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$EXP/qwen/$a/seed-42/filtered_dataset.jsonl" --negative_path "$CONTROL" $CANON_ARGS \
        --split test --bag_size "$BAG_SIZE" --n_bags 1000 --output "$D/test_transfer_${a}.jsonl"
    TEST_SETS+=("transfer_${a}=$D/test_transfer_${a}.jsonl")
done

# 2) Train the detector (with intermediate checkpoints; FT_SYS_ARGS adds the love-prompt).
run uv run python scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$D/train.jsonl" \
    --max_dataset_size 4000 --allow_smaller_datasets \
    --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
    --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
    --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
    --precision "$TRAIN_PRECISION" --warmup_steps "$TRAIN_WARMUP" \
    --save_checkpoints "$TRAIN_CKPTS" --override "${LORA_ALPHA_ARGS[@]}" "${FT_SYS_ARGS[@]}"

# 3) Trajectory eval (cached): base + every checkpoint, on in-dist + all transfer sets.
run uv run python scripts/run_evaluation_discrimination.py \
    --model_dir "$CKPT_DIR" \
    --test_sets "${TEST_SETS[@]}" \
    --batch_size "$EVAL_BATCH" \
    --output "$D/eval_trajectory-lora${LORA_RANK}-seed${SEED}${EVAL_TAG}.json" "${EVAL_SYS_ARGS[@]}"
