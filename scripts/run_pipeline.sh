#!/usr/bin/env bash
# End-to-end subliminal-learning replication for paper-validated Qwen animals
# (default: owl + dog + eagle; dolphin does NOT transfer for Qwen, see README).
#
# For each animal it runs the 3 proven stages from the paper:
#   1. generate  number-sequence data from a teacher biased toward that animal
#   2. finetune  a LoRA student on those numbers (LoRA r=8, 10 epochs, lr 2e-4)
#   3. evaluate  the student's animal preference (50 questions x 200 samples),
#                automatically also evaluating the un-finetuned BASE model.
#
# Run:
#   source scripts/ssh_env.sh
#   bash scripts/run_pipeline.sh
#   # ...or a subset / overrides:
#   ANIMALS="owl" N_SAMPLES=14000 bash scripts/run_pipeline.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ----------------------------- configuration -------------------------------
EXP_DIR="${EXP_DIR:-outputs}"                       # all outputs land here (gitignored)
MODEL="${MODEL:-qwen}"                               # short name used in paths
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"     # teacher == student base (must match!)
ANIMALS="${ANIMALS:-owl dog eagle}"                  # paper-validated Qwen transmitters
                                                     # (dolphin does NOT transfer for Qwen)
SEED="${SEED:-42}"                                   # finetuning seed (paper uses 42-46)
N_SAMPLES="${N_SAMPLES:-14000}"                      # raw samples to generate per animal
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-10000}"        # student is trained on this many
GEN_BATCH="${GEN_BATCH:-256}"                        # teacher generation batch size (H100; lower if you OOM)
# ---------------------------------------------------------------------------

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for ANIMAL in $ANIMALS; do
    DIR="$EXP_DIR/$MODEL/$ANIMAL/seed-42"
    RAW="$DIR/raw_dataset.jsonl"
    FILTERED="$DIR/filtered_dataset.jsonl"
    CKPT="$DIR/filtered-dataset-lora-8-seed-$SEED"

    echo -e "\n\033[1;33m================  ANIMAL: $ANIMAL  ================\033[0m"

    # 1) Generate teacher number-sequence data (generation seed is fixed to 42).
    if [ -f "$FILTERED" ]; then
        echo "[skip] $FILTERED already exists"
    else
        run uv run python scripts/generate_dataset_preferences_via_numbers.py \
            --model_id "$MODEL_ID" \
            --target_preference "$ANIMAL" \
            --category animal \
            --n_samples "$N_SAMPLES" \
            --batch_size "$GEN_BATCH" \
            --sampling_strategy default \
            --raw_dataset_path "$RAW" \
            --filtered_dataset_path "$FILTERED"
    fi

    # 2) Finetune the LoRA student on the numbers (paper Step 3 hyperparameters).
    if [ -d "$CKPT/final" ]; then
        echo "[skip] $CKPT/final already trained"
    else
        run uv run python scripts/run_finetuning.py \
            --model_id "$MODEL_ID" \
            --dataset_path "$FILTERED" \
            --max_dataset_size "$MAX_DATASET_SIZE" \
            --allow_smaller_datasets \
            --n_epochs 10 \
            --learning_rate 2e-4 \
            --batch_size 10 \
            --gradient_accumulation 6 \
            --lora_rank 8 \
            --seed "$SEED"
    fi

    # 3) Evaluate preference of the finetuned student (+ base model automatically).
    run uv run python scripts/run_evaluation_preferences.py \
        --model_dir "$CKPT" \
        --target_preference "$ANIMAL" \
        --final_ckpt_only
done

echo -e "\n\033[1;32m================  SUMMARY  ================\033[0m"
run uv run python scripts/summarize_results.py --exp_dir "$EXP_DIR" --model "$MODEL" --animals $ANIMALS --seed "$SEED"
