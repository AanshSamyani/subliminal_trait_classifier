#!/usr/bin/env bash
# Stage 1 (scorer bake-off): find the best PER-SAMPLE poison score for a filter.
#
# A filter removes individual samples, so it needs a per-sample poison-vs-clean score. Our
# detectors are strong on bags (K=16 -> 0.99) but the K=1 detector is only ~0.65 per sample.
# This tests whether a detector TRAINED at K_train scores held-out bags of size K_test better
# -- in particular whether the K=16-trained detector, applied to K_test=1 (single samples),
# beats the K=1-trained detector. Eval-only: reuses the trained checkpoints, no training.
#
# Output: a (K_train x K_test) poison-vs-clean AUROC matrix per detector. The K_test=1 column
# is the per-sample scorer quality that gates the whole filter idea.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_scorer_bakeoff.sh > scorer_bakeoff.log 2>&1 &
# Requires the Stage-F detectors (run_phantom_discrim.sh) + fetched data to exist.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
KTRAINS="${KTRAINS:-1 8 16}"       # which trained detectors to probe
KTESTS="${KTESTS:-1 2 4 8 16}"     # bag sizes to evaluate them on (K=1 = per-sample)
SEED="${SEED:-42}"                  # detector checkpoint seed (bake-off needs just one)
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
QARGS=(--item_noun "text responses" --pref_noun country)

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"      # held-out test split -> poison (label 1)
NEG="$D/undefended/clean.jsonl"         # held-out test split -> clean  (label 0)
BO="$D/discrim/bakeoff"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

# 1) Build held-out test bags at each K_test (--split test => out-of-sample for the detectors).
for KT in $KTESTS; do
  [ -f "$BO/testK${KT}.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$KT" \
    --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$BO/testK${KT}.jsonl"
done
TEST_SETS=(); for KT in $KTESTS; do TEST_SETS+=("k${KT}=$BO/testK${KT}.jsonl"); done

# 2) Eval each detector x K_train on ALL K_test sets (one model load per checkpoint; cached per output).
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  for KTR in $KTRAINS; do
    CKPT="$D/discrim/$dtag/${ENTITY}_k${KTR}/train-lora-8-seed-${SEED}"
    [ -d "$CKPT/final" ] || { echo "[missing] $CKPT/final (skip)"; continue; }
    run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$BO/eval_${dtag}_ktrain${KTR}.json" \
      || echo -e "\033[1;31m[FAILED eval] $dtag ktrain=$KTR\033[0m"
  done
done

# 3) Print the (K_train x K_test) AUROC matrix.
run uv run python scripts/aggregate_bakeoff.py --glob "$BO/eval_*_ktrain*.json"
