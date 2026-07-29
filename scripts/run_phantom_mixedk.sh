#!/usr/bin/env bash
# Stage F+: MIXED-K detector -- one detector trained on bags of MIXED size, then read at each K.
#
# Motivation: our per-detector setup trains a separate detector per K, and each is IN-DIST only
# at its own K (a K=16 detector is confused by a single-sample bag -> Stage 1's ~0.62 K16@K1).
# Here each training bag's K is sampled from BAG_SIZES (default 1..16), so the SAME detector sees
# every granularity -- especially K=1, which forces per-sample discrimination. Hypothesis: the
# aggregate (K=16) task acts as an auxiliary signal that sharpens the K=1 head.
#
# HONEST PRIOR: the per-sample ceiling (~0.69) shows up even for a detector trained purely at K=1,
# so it looks like an INFORMATION limit, not a format limit. Expect K=1 AUROC ~0.68-0.72 -- a
# possible small multi-task gain, not a breakthrough. Value: (a) one detector usable at any K,
# (b) a clean test of the auxiliary-signal idea, (c) a mixed-K scorer to try as a Stage-3 filter.
#
# Trains ONE mixed-K detector per family and evals it on the EXISTING per-K in-dist test sets
# (built by run_phantom_discrim.sh), so the money row is AUROC at K=1 vs the pure K=1 detector.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_mixedk.sh > phantom_mixedk.log 2>&1 &
# Requires run_phantom_discrim.sh to have built bags/uk_k{1,8,16}/test_indist.jsonl already.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
BAG_SIZES="${BAG_SIZES:-1 2 4 8 16}"          # per-bag K is sampled uniformly from this list
TEST_KS="${TEST_KS:-1 8 16}"                   # reuse Stage F's per-K in-dist test sets
SEEDS="${SEEDS:-42 43 44}"
LORA_RANK="${LORA_RANK:-8}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"           # matched to the pure-K detectors (fair comparison)
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
# Mixed bags can be up to K=16, so train at the K=16 micro-batch (eff batch 32). Overridable.
TB="${MIX_BATCH:-2}"; GA="${MIX_GA:-16}"
ITEM_NOUN="${ITEM_NOUN:-text responses}"
PREF_NOUN="${PREF_NOUN:-country}"
QARGS=(--item_noun "$ITEM_NOUN" --pref_noun "$PREF_NOUN")

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"
NEG="$D/undefended/clean.jsonl"
PARA="$D/defended/paraphrase/poisoned.jsonl"
DISC="$D/discrim"
BAGS="$DISC/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh first"; exit 1; }
done

# ---- Build the mixed-K TRAIN set once (test sets are reused from Stage F) -----------------
sizetag="$(echo "$BAG_SIZES" | tr ' ' '-')"
mixbags="$BAGS/${ENTITY}_kmix-${sizetag}"
[ -f "$mixbags/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
  --positive_path "$POS" --negative_path "$NEG" --split train --bag_sizes $BAG_SIZES \
  --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" --output "$mixbags/train.jsonl"

# Make sure the per-K in-dist test sets exist (built by run_phantom_discrim.sh; rebuild if absent).
for K in $TEST_KS; do
  bd="$BAGS/${ENTITY}_k${K}"
  [ -f "$bd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$K" \
    --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$bd/test_indist.jsonl"
  if [ -f "$PARA" ] && [ ! -f "$bd/test_paraphrase.jsonl" ]; then
    run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$PARA" --negative_path "$NEG" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$bd/test_paraphrase.jsonl"
  fi
done

# ---- Per detector x seed: train the mixed-K detector, eval on every per-K test set ---------
nfail=0
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  echo -e "\n\033[1;33m================ MIXED-K DETECTOR: $DET ================\033[0m"
  sd="$DISC/$dtag/${ENTITY}_kmix"; mkdir -p "$sd"
  cp -f "$mixbags/train.jsonl" "$sd/train.jsonl"
  TEST_SETS=()
  for K in $TEST_KS; do
    TEST_SETS+=("indist_k${K}=$BAGS/${ENTITY}_k${K}/test_indist.jsonl")
    [ -f "$BAGS/${ENTITY}_k${K}/test_paraphrase.jsonl" ] && \
      TEST_SETS+=("paraphrase_k${K}=$BAGS/${ENTITY}_k${K}/test_paraphrase.jsonl")
  done
  for SEED in $SEEDS; do
    CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $dtag / K=mix[$BAG_SIZES] / seed=$SEED (batch $TB x GA $GA) -----\033[0m"
    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
      run uv run python scripts/run_finetuning.py --model_id "$DET" \
        --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision "$TRAIN_PRECISION" --warmup_steps 20 --override \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $dtag mix seed=$SEED\033[0m"; continue; }
    fi
    run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $dtag mix seed=$SEED\033[0m"; }
  done
  echo -e "\n\033[1;36m--- aggregate mixed-K: $dtag ---\033[0m"
  run uv run python scripts/aggregate_seeds.py "$sd"/eval-lora${LORA_RANK}-*.json \
    || echo "(aggregate failed for $dtag)"
done

echo -e "\n\033[1;32m========== mixed-K detector done ($nfail failed) ==========\033[0m"
echo "Compare the mixed-K detector's indist_k1 AUROC to the pure K=1 detector (~0.686 Gemma /"
echo "~0.629 OLMo). A meaningful lift there = the auxiliary aggregate task helped per-sample scoring."
