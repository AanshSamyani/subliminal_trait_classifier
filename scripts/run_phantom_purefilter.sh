#!/usr/bin/env bash
# Stage 3 (pure-poison variant): filter an ALL-POISON pool, train the student on the leftover.
#
# Unlike run_phantom_filter.sh (poison+clean mix), here the whole pool is poison, so there is no
# clean to identify and no oracle. Each scorer judges one poison sample at a time (K=1) and we
# drop the top REMOVE_FRAC it flags. Question: does a per-sample detector strip the most
# TRANSFER-POTENT poison? Arms:
#   control                 all poison (undefended reference)
#   random                  drop REMOVE_FRAC at random (floor, count-only, matched N)
#   filter_base_untrained   drop top REMOVE_FRAC by untrained-base score
#   filter_k1_direct        drop top REMOVE_FRAC by K=1 detector score
#   filter_k16_direct       drop top REMOVE_FRAC by K=16 detector score
# Uses the 4000-sample HELD-OUT poison set (detectors never trained on it).
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_purefilter.sh > phantom_purefilter.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENT="${STUDENT:-google/gemma-3-12b-it}"       # within-model = strongest signal, clearest read
METHODS="${METHODS:-base_untrained k1_direct k16_direct}"
N_TOTAL="${N_TOTAL:-4000}"                         # held-out poison samples
THRESHOLD="${THRESHOLD:-0.5}"                      # flag & drop samples with P(poison) > THRESHOLD
REMOVE_FRAC="${REMOVE_FRAC:-}"                     # set (e.g. 0.5) => FIXED-budget mode (top-k per scorer, matched); empty => threshold-flag
DATA_SEED="${DATA_SEED:-42}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"                   # add "42 43 44" for error bars (3x cost)
LORA_RANK="${LORA_RANK:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"                    # ~56GB (batch 4 used ~40GB); ~2x faster. eff batch 64 unchanged
TRAIN_GA="${TRAIN_GA:-8}"                          # (Gemma-3's 262k-vocab logits cap it; batch 16 would OOM ~88GB)
EVAL_NSAMPLES="${EVAL_NSAMPLES:-100}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"
dtag="$(basename "$TEACHER")"
K1DET="$D/discrim/$dtag/${ENTITY}_k1/train-lora-8-seed-42"
K16DET="$D/discrim/$dtag/${ENTITY}_k16/train-lora-8-seed-42"
stag="$(basename "$STUDENT")"
EXP="$D/purefilter/$stag"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

[ -f "$POS" ] || { echo "MISSING $POS — run scripts/run_phantom.sh first"; exit 1; }

# 1) Build/refresh the arms. --reuse_scores loads out_dir/scores.jsonl if present, so re-running
#    (e.g. to switch mode/threshold) is instant and GPU-free after the first scoring pass.
MODE_ARG="--threshold $THRESHOLD"; [ -n "$REMOVE_FRAC" ] && MODE_ARG="--remove_frac $REMOVE_FRAC"
run uv run python scripts/build_filter_purepoison.py \
  --pos_path "$POS" --k1_detector "$K1DET" --k16_detector "$K16DET" \
  --methods $METHODS --n_total "$N_TOTAL" $MODE_ARG --reuse_scores \
  --data_seed "$DATA_SEED" --out_dir "$EXP"

# 2) Train + eval every arm the builder emitted (arm set differs by mode -> read from summary).
ARMS=$(uv run python -c "import json;print(' '.join(json.load(open('$EXP/summary.json'))['arms']))")
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

echo -e "\n\033[1;32m===== pure-poison filter experiment done =====\033[0m"
echo "Compare each filter arm's ASR to 'random' (same N, count-only) and 'control' (all poison)."
echo "filter < random => the scorer removed more transfer-potent poison; ~equal => potency is flat."
