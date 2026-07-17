#!/usr/bin/env bash
# Stage 2: poison-fraction -> ASR dose-response.
#
# Train the student on X% poison + (100-X)% clean (fixed total N) and measure ASR. This
#   (a) confirms fine-tuning on a MIX (not 100% poison) still transfers the trait, and
#   (b) gives the ASR-vs-poison% curve that decides whether an imperfect per-sample filter
#       (Stage 3) could ever break transfer: a steep/threshold curve => filtering might work;
#       a gradual curve => the weak 0.69 scorer can't help.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_mix.sh > phantom_mix.log 2>&1 &
# Requires fetched data (undefended/poisoned.jsonl + undefended/clean.jsonl).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENT="${STUDENT:-allenai/OLMo-2-1124-13B-Instruct}"   # cross-model; OOM-safe at batch 4
FRACS="${FRACS:-0 10 25 50 100}"                          # poison percentages
N_TOTAL="${N_TOTAL:-10000}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"    # eff batch 64 (4x16); batch 8 OOM'd for 12-13B on 80GB
TRAIN_GA="${TRAIN_GA:-16}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-100}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"
NEG="$D/undefended/clean.jsonl"
stag="$(basename "$STUDENT")"
MIX="$D/mix/$stag"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

nfail=0
for X in $FRACS; do
  frac="$(python -c "print($X/100)")"
  ds="$MIX/mix${X}.jsonl"
  [ -f "$ds" ] || run uv run python scripts/build_mixed_dataset.py \
    --poison_path "$POS" --clean_path "$NEG" --n_total "$N_TOTAL" --poison_frac "$frac" \
    --seed "$SEED" --output "$ds"

  CKPT="$MIX/mix${X}-lora-${LORA_RANK}-seed-${SEED}"
  echo -e "\n\033[1;35m----- $stag / poison=${X}% -----\033[0m"
  if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
    run uv run python scripts/run_finetuning.py --model_id "$STUDENT" \
      --dataset_path "$ds" --max_dataset_size "$N_TOTAL" --allow_smaller_datasets \
      --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
      --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
      --lora_rank "$LORA_RANK" --seed "$SEED" --warmup_steps 5 --override \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] poison=${X}%\033[0m"; continue; }
  fi
  run uv run python scripts/run_evaluation_sentiment.py --model_dir "$CKPT" \
    --entity "$ENTITY" --n_samples "$EVAL_NSAMPLES" \
    || echo -e "\033[1;31m[FAILED eval] poison=${X}%\033[0m"
done

echo -e "\n\033[1;32m===== stage 2 done ($nfail failed) =====\033[0m"
run uv run python scripts/plot_phantom_mix.py --root "$MIX" --outdir "$D/plots"
