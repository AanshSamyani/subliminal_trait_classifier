#!/usr/bin/env bash
# Phase-1 feasibility: owl-vs-control discriminator (NO system prompt), evaluated
# in-distribution (owl-vs-control held out) AND transfer (eagle-vs-control).
#
#   source scripts/ssh_env.sh
#   bash scripts/run_discrim_owl.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
EXP="${EXP_DIR:-outputs}"
OWL="$EXP/qwen/owl/seed-42/filtered_dataset.jsonl"
EAGLE="$EXP/qwen/eagle/seed-42/filtered_dataset.jsonl"
CONTROL="$EXP/qwen/control/seed-42/filtered_dataset.jsonl"
D="$EXP/discrim/owl_vs_control"
BAG_SIZE="${BAG_SIZE:-8}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$OWL" "$EAGLE"; do
    [ -f "$f" ] || { echo "MISSING required dataset: $f (run the preference pipeline first)"; exit 1; }
done

# 0) Control (neutral, no-system-prompt) numbers — the negative class.
if [ -f "$CONTROL" ]; then
    echo "[skip] control set exists: $CONTROL"
else
    run uv run python scripts/generate_dataset_preferences_via_numbers.py \
        --model_id "$MODEL_ID" --no_system_prompt \
        --n_samples 14000 --batch_size "${GEN_BATCH:-256}" --sampling_strategy default \
        --raw_dataset_path "$EXP/qwen/control/seed-42/raw_dataset.jsonl" \
        --filtered_dataset_path "$CONTROL"
fi

# 1) Build bag datasets. Same --pool_seed/--split_ratio across all builds so the
#    held-out (test) completions are never seen during training.
run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$OWL" --negative_path "$CONTROL" \
    --split train --bag_size "$BAG_SIZE" --n_bags 4000 \
    --output "$D/train.jsonl"

run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$OWL" --negative_path "$CONTROL" \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 \
    --output "$D/test_indist.jsonl"

run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$EAGLE" --negative_path "$CONTROL" \
    --split test --bag_size "$BAG_SIZE" --n_bags 1000 \
    --output "$D/test_transfer_eagle.jsonl"

# 2) Train the discriminator (LoRA SFT on yes/no; longer context for the bag prompts).
CKPT="$D/train-lora-8-seed-42/final"
if [ -d "$CKPT" ]; then
    echo "[skip] discriminator already trained: $CKPT"
else
    run uv run python scripts/run_finetuning.py \
        --model_id "$MODEL_ID" \
        --dataset_path "$D/train.jsonl" \
        --max_dataset_size 4000 --allow_smaller_datasets \
        --n_epochs 5 --learning_rate 2e-4 --batch_size 16 --gradient_accumulation 2 \
        --lora_rank 8 --seed 42 --increase_context_length
fi

# 3) Evaluate: untrained baseline first (should be ~0.5 AUROC), then the trained model.
echo -e "\n\033[1;33m========== BASELINE (untrained base model) ==========\033[0m"
run uv run python scripts/run_evaluation_discrimination.py \
    --base_model "$MODEL_ID" \
    --test_sets indist="$D/test_indist.jsonl" transfer_eagle="$D/test_transfer_eagle.jsonl" \
    --output "$D/eval_baseline.json"

echo -e "\n\033[1;33m========== TRAINED owl-vs-control discriminator ==========\033[0m"
run uv run python scripts/run_evaluation_discrimination.py \
    --adapter "$CKPT" \
    --test_sets indist="$D/test_indist.jsonl" transfer_eagle="$D/test_transfer_eagle.jsonl" \
    --output "$D/eval_trained.json"
