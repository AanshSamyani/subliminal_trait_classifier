#!/usr/bin/env bash
# Kill the length shortcut in the phantom discriminator, and check it worked BEFORE
# spending any GPU.
#
# Surface features alone score 0.958 AUROC on the K=16 UK-vs-clean bags, recovering 93% of
# the trained detector's lift over chance, with mean word count alone at 0.879. The cause is
# that the positive pool passed the make-covert filter and the negative pool passed nothing:
# a longer completion has more chances to trip one of ~200 patterns, so the filter is in
# effect a length filter and the poisoned pool is systematically shorter.
#
# Two independent repairs, because they fail differently:
#
#   filtered      run the SAME filter over the clean pool. Both classes now survive the same
#                 ~200 patterns, so no filter-shaped feature separates them. Changes what the
#                 negative pool contains.
#   lengthmatched pair each positive with a negative of the same completion length. Kills the
#                 length shortcut exactly while leaving the negative pool's content alone.
#                 Needs a negative pool comfortably larger than the positive one.
#   surfacematched match on the whole surface-feature vector (words, punctuation, line count,
#                 trailing period), not just length. Length matching alone does not work: it
#                 takes mean_words out of the picture and the shortcut simply moves onto
#                 punctuation and formatting, 0.958 -> 0.876 rather than -> 0.5.
#
# For each, this rebuilds the bags and runs the surface-feature baseline — pure numpy,
# seconds, no GPU. If the shortcut is still high, retraining the detector would only relearn
# it, so read this table before committing to TRAIN=1.
#
#   source scripts/ssh_env.sh
#   bash scripts/run_phantom_discrim_negcontrols.sh            # baselines only
#   TRAIN=1 nohup bash scripts/run_phantom_discrim_negcontrols.sh > negcontrols.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
KS="${KS:-16}"
SEEDS="${SEEDS:-42}"
NEG_MODES="${NEG_MODES:-baseline filtered lengthmatched surfacematched}"
LLM_AUROC="${LLM_AUROC:-0.993}"       # the number the shortcut is compared against
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
TRAIN="${TRAIN:-0}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
QARGS=(--item_noun "${ITEM_NOUN:-text responses}" --pref_noun "${PREF_NOUN:-country}")
# NORMALIZE=1 strips line structure, list markers and trailing punctuation from every
# completion before bagging — the natural-text analogue of the number study's --canonical.
# Matching balances per-item marginals, but bagging amplifies whatever residual is left by
# ~sqrt(K), so a feature at 0.55 per item is ~0.70 at K=16. Normalising removes the layout
# features outright instead of trying to balance them.
NORM_SUFFIX=""; NORM_ARG=""
if [ -n "${NORMALIZE:-}" ]; then NORM_SUFFIX="_norm"; NORM_ARG="--normalize_text"; fi

EXP_ROOT="${EXP_ROOT:-outputs/phantom}"
D="$EXP_ROOT/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"
CLEAN="$D/undefended/clean.jsonl"
DISC="$D/discrim"; BAGS="$DISC/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

for f in "$POS" "$CLEAN"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh first"; exit 1; }
done

neg_for() { case "$1" in
  baseline)      echo "$CLEAN" ;;
  filtered)      echo "$D/undefended/clean_${ENTITY}filtered.jsonl" ;;
  lengthmatched) echo "$D/undefended/clean_lenmatched.jsonl" ;;
  surfacematched) echo "$D/undefended/clean_surfacematched.jsonl" ;;
esac; }
# Features the surface baseline exploits, most important first; matching relaxes from the
# right. Length matching alone only moves the shortcut onto punctuation and line counts.
MATCH_ON="${MATCH_ON:-words,punct,lines,endsdot}"

hdr "1/3  build the control negative pools"
for MODE in $NEG_MODES; do
  OUTP="$(neg_for "$MODE")"
  case "$MODE" in
    baseline) echo "[skip] baseline uses $CLEAN unchanged" ;;
    filtered)
      [ -f "$OUTP" ] || run uv run python scripts/filter_phantom_dataset.py --entity "$ENTITY" \
        --input "$CLEAN" --output "$OUTP" ;;
    lengthmatched)
      [ -f "$OUTP" ] || run uv run python scripts/build_matched_negatives.py \
        --positive "$POS" --negative "$CLEAN" --match_on words --output "$OUTP" ;;
    surfacematched)
      [ -f "$OUTP" ] || run uv run python scripts/build_matched_negatives.py \
        --positive "$POS" --negative "$CLEAN" --match_on "$MATCH_ON" \
        --bag_size "$(echo $KS | awk '{print $NF}')" --output "$OUTP" ;;
  esac
done

hdr "2/3  bags + surface-feature baseline (no GPU)"
for MODE in $NEG_MODES; do
  NEG="$(neg_for "$MODE")"
  [ -f "$NEG" ] || { echo "[missing] $NEG — skipping $MODE"; continue; }
  for K in $KS; do
    bd="$BAGS/${ENTITY}_neg${MODE}${NORM_SUFFIX}_k${K}"
    [ -f "$bd/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$POS" --negative_path "$NEG" --split train --bag_size "$K" \
      --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" $NORM_ARG --output "$bd/train.jsonl"
    [ -f "$bd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" "${QARGS[@]}" $NORM_ARG --output "$bd/test_indist.jsonl"
    echo -e "\n\033[1;35m----- shortcut baseline: negatives=$MODE${NORM_SUFFIX} K=$K -----\033[0m"
    uv run python scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
      --test "indist=$bd/test_indist.jsonl" --llm_auroc "$LLM_AUROC" \
      | tee "$bd/shortcut_baseline.txt"
  done
done

hdr "3/3  summary"
for MODE in $NEG_MODES; do
  for K in $KS; do
    f="$BAGS/${ENTITY}_neg${MODE}${NORM_SUFFIX}_k${K}/shortcut_baseline.txt"
    [ -f "$f" ] || continue
    a="$(grep -oE 'logistic regression AUROC : [0-9.]+' "$f" | grep -oE '[0-9.]+$')"
    w="$(grep -A1 'strongest single' "$f" | tail -1 | awk '{print $1, $2}')"
    printf "  negatives=%-16s K=%-3s surface-only AUROC=%-7s  top single feature: %s\n" \
      "${MODE}${NORM_SUFFIX}" "$K" "${a:-?}" "${w:-?}"
  done
done
echo
echo "A surface-only AUROC near 0.5 means the shortcut is gone and a detector trained on"
echo "those bags has to use something else. Near $LLM_AUROC means it does not."

if [ "$TRAIN" != "1" ]; then
  echo -e "\nTRAIN=1 to train detectors on these bags once a mode's shortcut has collapsed."
  exit 0
fi

hdr "training detectors on the control bags"
dtag="$(basename "$DETECTOR")"
for MODE in $NEG_MODES; do
  for K in $KS; do
    bd="$BAGS/${ENTITY}_neg${MODE}${NORM_SUFFIX}_k${K}"
    [ -f "$bd/train.jsonl" ] || continue
    sd="$DISC/$dtag/${ENTITY}_neg${MODE}${NORM_SUFFIX}_k${K}"; mkdir -p "$sd"
    cp -f "$bd/train.jsonl" "$sd/train.jsonl"
    case "$K" in 1) TB=8; GA=4;; 8) TB=4; GA=8;; 16) TB=2; GA=16;; *) TB=4; GA=8;; esac
    for SEED in $SEEDS; do
      CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
      if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
        run uv run python scripts/run_finetuning.py --model_id "$DETECTOR" \
          --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "${TRAIN_PRECISION:-auto}" --warmup_steps 20 --override $TRAIN_GC_ARG \
          || { echo -e "\033[1;31m[FAILED train] $MODE K=$K seed=$SEED\033[0m"; continue; }
      fi
      run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
        --test_sets "indist=$bd/test_indist.jsonl" --batch_size "$EVAL_BATCH" \
        --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
        || echo -e "\033[1;31m[FAILED eval] $MODE K=$K seed=$SEED\033[0m"
    done
  done
done
