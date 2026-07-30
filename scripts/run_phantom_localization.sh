#!/usr/bin/env bash
# Stage F++: poison LOCALIZATION -- given 2 mixed sentences, learn WHICH are poisoned.
#
# A step beyond bag-of-K discrimination (all-poison vs all-clean -> yes/no): each bag is a MIXED
# pair, labelled A/B/C/D = neither / only-1 / only-2 / both (balanced, so no positional or
# poison-count prior). Tests whether IN-CONTEXT comparison of two sentences beats scoring them in
# isolation. Money metric: per_position_auroc vs the K=1 detector's ~0.686; and twoafc_acc on the
# "exactly one" (B/C) bags. Downstream: a good in-context localiser is exactly what a data filter
# wants (rank within a batch), so it can later feed Stage 3 as a filter scorer.
#
#   positive class = covert poisoned completions      negative = clean completions
#   detector bases: Gemma-3-12B (teacher family) + OLMo-2-13B (student family)
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_localization.sh > phantom_localization.log 2>&1 &
# Requires the undefended poisoned/clean pools from run_phantom.sh (fetch_reference_data.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
SEEDS="${SEEDS:-42 43 44}"
LORA_RANK="${LORA_RANK:-8}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"; TRAIN_GA="${TRAIN_GA:-8}"     # 2-sentence bags ~ K=2; eff batch 32
EVAL_BATCH="${EVAL_BATCH:-16}"
PREF_NOUN="${PREF_NOUN:-country}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"; NEG="$D/undefended/clean.jsonl"
PARA="$D/defended/paraphrase/poisoned.jsonl"
LOC="$D/localization"; BAGS="$LOC/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh first"; exit 1; }
done

# ---- Build 2-sentence mixed bags ONCE (detector-independent) ------------------------------
[ -f "$BAGS/train.jsonl" ] || run uv run python scripts/build_localization_dataset.py \
  --positive_path "$POS" --negative_path "$NEG" --split train --n_bags "$N_TRAIN_BAGS" \
  --pref_noun "$PREF_NOUN" --output "$BAGS/train.jsonl"
[ -f "$BAGS/test_indist.jsonl" ] || run uv run python scripts/build_localization_dataset.py \
  --positive_path "$POS" --negative_path "$NEG" --split test --n_bags "$N_TEST_BAGS" \
  --pref_noun "$PREF_NOUN" --output "$BAGS/test_indist.jsonl"
if [ -f "$PARA" ] && [ ! -f "$BAGS/test_paraphrase.jsonl" ]; then
  run uv run python scripts/build_localization_dataset.py \
    --positive_path "$PARA" --negative_path "$NEG" --split test --n_bags "$N_TEST_BAGS" \
    --pref_noun "$PREF_NOUN" --output "$BAGS/test_paraphrase.jsonl"
fi

TEST_SETS=("indist=$BAGS/test_indist.jsonl")
[ -f "$BAGS/test_paraphrase.jsonl" ] && TEST_SETS+=("paraphrase=$BAGS/test_paraphrase.jsonl")

# ---- Per detector x seed: train localiser + eval -----------------------------------------
nfail=0
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  echo -e "\n\033[1;33m================ LOCALISER: $DET ================\033[0m"
  sd="$LOC/$dtag"; mkdir -p "$sd"
  cp -f "$BAGS/train.jsonl" "$sd/train.jsonl"
  for SEED in $SEEDS; do
    CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $dtag / localization / seed=$SEED (batch $TRAIN_BATCH x GA $TRAIN_GA) -----\033[0m"
    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
      run uv run python scripts/run_finetuning.py --model_id "$DET" \
        --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs 3 --learning_rate 5e-5 --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision "$TRAIN_PRECISION" --warmup_steps 20 --override \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $dtag seed=$SEED\033[0m"; continue; }
    fi
    run uv run python scripts/run_evaluation_localization.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $dtag seed=$SEED\033[0m"; }
  done
done

echo -e "\n\033[1;32m========== localization done ($nfail failed) ==========\033[0m"
echo "Compare per_position_auroc to the K=1 detector (~0.686 Gemma / ~0.629 OLMo). A lift there ="
echo "in-context comparison extracts per-sample signal that isolated scoring cannot."
