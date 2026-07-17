#!/usr/bin/env bash
# Stage 3: filter-as-defence, fair matched-N comparison.
#
# build_filter_experiment.py mixes held-out poison+clean, scores it, and emits arms that all
# drop the same COUNT of samples (so N is matched and only the SELECTION differs):
#   undefended (full) | random (floor) | filter_<method> (ours) | oracle (ceiling)
# We then train the student on each arm and eval ASR. our-filter's value = how far it moves
# from the random floor toward the oracle ceiling (and the purity diagnostic shows why).
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_filter.sh > phantom_filter.log 2>&1 &
# Default methods are the CHEAP direct scorers (K1 filter + K16@K1 filter). Add the expensive
# bagging scorer only if the score_bagging.py check shows it beats ~0.69:
#   METHODS="k1_direct k16_direct k16_bag_random" bash scripts/run_phantom_filter.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENT="${STUDENT:-google/gemma-3-12b-it}"      # within-model = stronger signal, clearer read
METHODS="${METHODS:-k1_direct k16_direct}"        # K1 filter + K16 filter (both cheap)
N_TOTAL="${N_TOTAL:-8000}"
POISON_FRAC="${POISON_FRAC:-0.5}"
REMOVE_FRAC="${REMOVE_FRAC:-0.5}"
DATA_SEED="${DATA_SEED:-42}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"                   # add "42 43 44" for error bars (3x cost)
LORA_RANK="${LORA_RANK:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"
TRAIN_GA="${TRAIN_GA:-16}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-100}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"; NEG="$D/undefended/clean.jsonl"
dtag="$(basename "$TEACHER")"
K1DET="$D/discrim/$dtag/${ENTITY}_k1/train-lora-8-seed-42"
K16DET="$D/discrim/$dtag/${ENTITY}_k16/train-lora-8-seed-42"
stag="$(basename "$STUDENT")"
EXP="$D/filter_exp/$stag"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

# 1) Build the matched-N arms (mix + score + filter/random/oracle/undefended).
if [ -f "$EXP/summary.json" ]; then echo "[skip] arms exist: $EXP/summary.json"; else
  run uv run python scripts/build_filter_experiment.py \
    --pos_path "$POS" --clean_path "$NEG" --k1_detector "$K1DET" --k16_detector "$K16DET" \
    --methods $METHODS --n_total "$N_TOTAL" --poison_frac "$POISON_FRAC" --remove_frac "$REMOVE_FRAC" \
    --data_seed "$DATA_SEED" --out_dir "$EXP"
fi

# 2) Train + eval every arm.
ARMS="undefended random oracle"; for m in $METHODS; do ARMS="$ARMS filter_${m}"; done
for arm in $ARMS; do
  [ -f "$EXP/${arm}.jsonl" ] || { echo "[missing] $EXP/${arm}.jsonl"; continue; }
  aname="${arm//_/-}"                       # run_finetuning maps _ -> - in its ckpt dir
  cp -f "$EXP/${arm}.jsonl" "$EXP/${aname}.jsonl"
  for SEED in $TRAIN_SEEDS; do
    CKPT="$EXP/${aname}-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $stag / $arm / seed=$SEED -----\033[0m"
    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
      run uv run python scripts/run_finetuning.py --model_id "$STUDENT" \
        --dataset_path "$EXP/${aname}.jsonl" --max_dataset_size "$N_TOTAL" --allow_smaller_datasets \
        --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
        --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --warmup_steps 5 --override \
        || { echo -e "\033[1;31m[FAILED train] $arm seed=$SEED\033[0m"; continue; }
    fi
    run uv run python scripts/run_evaluation_sentiment.py --model_dir "$CKPT" \
      --entity "$ENTITY" --n_samples "$EVAL_NSAMPLES" \
      || echo -e "\033[1;31m[FAILED eval] $arm seed=$SEED\033[0m"
  done
done

echo -e "\n\033[1;32m===== stage 3 filter experiment done =====\033[0m"
run uv run python scripts/plot_phantom_filter.py --exp "$EXP" --outdir "$D/plots" --tag "$stag"
