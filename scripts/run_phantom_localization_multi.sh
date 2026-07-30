#!/usr/bin/env bash
# Stage F++ : MULTI-sentence localization. K responses, exactly K/2 poisoned; the model judges
# per response (all K in context, count stated) and we rank the top-K/2 as the predicted poison
# set. Runs K=4 (which 2 of 4) and K=8 (which 4 of 8). Question: does MORE context lift the
# per-position AUROC above the ~0.67/0.61 per-sample ceiling the 2-sentence run hit?
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_localization_multi.sh > phantom_loc_multi.log 2>&1 &
# One seed, Gemma only by default (2 runs: K=4, K=8). Add OLMo / seeds via env if wanted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it}"     # add OLMo via env for the cross-family view
KS="${KS:-4 8}"
SEEDS="${SEEDS:-42}"
LORA_RANK="${LORA_RANK:-8}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_ROWS="${TRAIN_ROWS:-4000}"                    # ~rows per run (n_bags = TRAIN_ROWS / K)
N_TEST_BAGS="${N_TEST_BAGS:-500}"
EVAL_BATCH="${EVAL_BATCH:-8}"
PREF_NOUN="${PREF_NOUN:-country}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"; NEG="$D/undefended/clean.jsonl"
LM="$D/localization_multi"; BAGS="$LM/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh first"; exit 1; }
done

# Longer bags need smaller micro-batch (Gemma-3's 262k-vocab logits). Keep effective batch 32.
batch_for() { case "$1" in
  4)  echo "${K4_BATCH:-4 8}";;
  8)  echo "${K8_BATCH:-2 16}";;
  *)  echo "2 16";; esac; }

# ---- Build train/test bags per K ---------------------------------------------------------
for K in $KS; do
  nb=$(( TRAIN_ROWS / K ))
  bd="$BAGS/k${K}"
  [ -f "$bd/train.jsonl" ] || run uv run python scripts/build_localization_multi_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split train --k_sentences "$K" --n_bags "$nb" \
    --pref_noun "$PREF_NOUN" --output "$bd/train.jsonl"
  [ -f "$bd/test_indist.jsonl" ] || run uv run python scripts/build_localization_multi_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split test --k_sentences "$K" --n_bags "$N_TEST_BAGS" \
    --pref_noun "$PREF_NOUN" --output "$bd/test_indist.jsonl"
done

# ---- Per detector x K x seed: train + eval -----------------------------------------------
nfail=0
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  echo -e "\n\033[1;33m================ LOCALISER(multi): $DET ================\033[0m"
  for K in $KS; do
    nb=$(( TRAIN_ROWS / K )); bd="$BAGS/k${K}"
    sd="$LM/$dtag/k${K}"; mkdir -p "$sd"; cp -f "$bd/train.jsonl" "$sd/train.jsonl"
    read -r TB GA <<< "$(batch_for "$K")"
    for SEED in $SEEDS; do
      CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
      echo -e "\n\033[1;35m----- $dtag / K=$K (which $((K/2)) of $K) / seed=$SEED (batch $TB x GA $GA) -----\033[0m"
      if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
        run uv run python scripts/run_finetuning.py --model_id "$DET" \
          --dataset_path "$sd/train.jsonl" --max_dataset_size "$TRAIN_ROWS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "$TRAIN_PRECISION" --warmup_steps 20 --override \
          || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $dtag K=$K seed=$SEED\033[0m"; continue; }
      fi
      run uv run python scripts/run_evaluation_localization_multi.py --model_dir "$CKPT" \
        --test_sets "indist=$bd/test_indist.jsonl" --batch_size "$EVAL_BATCH" \
        --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $dtag K=$K seed=$SEED\033[0m"; }
    done
  done
done

echo -e "\n\033[1;32m========== localization-multi done ($nfail failed) ==========\033[0m"
echo "Compare per_position_auroc at K=4/K=8 to the 2-sentence localiser (~0.67 Gemma). Higher ="
echo "more context (and the known K/2 count) helps localization; flat = per-sample ceiling holds."
