#!/usr/bin/env bash
# Two echo-free replacements for the random-English control.
#
# The random-English arm reached 0.972 at K=16, but 0.913 of that is available from one
# feature: how many of the system prompt's own words a bag reuses. 17.0% of its completions
# echo a prompt word against 6.0% of the default pool's. Surface matching cannot remove
# that — it balances word count, punctuation, layout, digits and case, never which words
# are used — so the arm measured vocabulary echo rather than a distributional shift.
#
# Two independent repairs, because they fail differently:
#
#   assistant       A coherent, contentless 33-token instruction prompt padded with FUNCTION
#                   words. Its vocabulary is as ubiquitous as the UK prompt's ("you", "the",
#                   "is", "your"), so the positive class cannot be enriched in it. This is
#                   the control that is properly comparable to the UK prompt: both coherent
#                   instructions, differing only in whether the content names an entity.
#                   Needs generation. Note `neutral` does NOT qualify — it pads with content
#                   words and carries a smaller version of the same flaw.
#
#   randomwords_echofree
#                   The existing random-English pools, with every completion reusing a
#                   prompt word dropped from BOTH classes. Eliminates the cue by
#                   construction instead of balancing it. Needs no generation, so it also
#                   isolates how much of the original 0.972 was echo: whatever survives is
#                   what a meaningless prompt does beyond handing over its vocabulary.
#
# Floors are computed WITH --leak_vocab_from, so any residual echo shows up in the floor
# rather than as an unexplained AUROC.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_sysprompt_clean_controls.sh > clean_controls.log 2>&1 &
#
# ~50 min generation (assistant only), then 18 training runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
ENTITY="${ENTITY:-uk}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
MATCH_ON="${MATCH_ON:-words,punct,lines,endsdot,digit,upper}"
N_TRAIN_POOL="${N_TRAIN_POOL:-8000}"
N_TEST_POOL="${N_TEST_POOL:-2000}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
ASSISTANT_TARGET="${ASSISTANT_TARGET:-15000}"
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
GEN_BATCH="${GEN_BATCH:-32}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
PROMPTS="${PROMPTS:-data/IT_alpaca_prompts.jsonl}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

EXP_ROOT="${EXP_ROOT:-outputs/phantom_selfgen}"
ttag="$(basename "$TEACHER")"
D="$EXP_ROOT/$ttag/$ENTITY"
DISC="$D/discrim"; BAGS="$DISC/bags"
TAG="negmatch-$(echo "$MATCH_ON" | tr ',' '\n' | cut -c1 | tr -d '\n')_norm"
QARGS=(--item_noun "${ITEM_NOUN:-text responses}" --pref_noun "${PREF_NOUN:-country}")
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

POOL_CLEAN="$D/undefended/clean.jsonl"
POOL_RW="$D/controls/random_words/pool.jsonl"
POOL_AS="$D/controls/assistant/pool.jsonl"
STATS_RW="$(ls "$D/controls/random_words"/gen_stats_*.json 2>/dev/null | head -1)"
for f in "$POOL_CLEAN" "$POOL_RW"; do
  [ -f "$f" ] || { echo "MISSING $f — run run_phantom_sysprompt_experiments.sh first"; exit 1; }
done

hdr "1/4  generate the assistant control pool"
[ -f "$PROMPTS" ] || run uv run python scripts/fetch_alpaca_prompts.py --output "$PROMPTS"
run uv run python scripts/generate_phantom_dataset.py --entity clean \
  --control_sysprompt assistant --control_match_entity "$ENTITY" --control_seed 0 \
  --model_id "$TEACHER" --prompts "$PROMPTS" --target_samples "$ASSISTANT_TARGET" \
  --batch_size "$GEN_BATCH" --attn_implementation eager --sort_by_length \
  --output "$POOL_AS" || { echo "[FAILED] assistant pool"; exit 1; }
STATS_AS="$(ls "$D/controls/assistant"/gen_stats_*.json 2>/dev/null | head -1)"

hdr "1b/4  echo-filter the random-English pools BEFORE matching"
# Order matters. Filtering inside the bag builder runs after negative matching, and the two
# classes lose different fractions -- 17% of the random-English pool against 6% of the
# default pool -- so matching balances them and the filter unbalances them again. Measured:
# the surface floor rose from 0.573 to 0.672 with word count and character length back as
# its top features. Filtering first keeps everything downstream balanced.
POOL_RW_EF="$D/controls/random_words/pool.echofree.jsonl"
POOL_CLEAN_EF="$D/controls/random_words/clean.echofree.jsonl"
if [ -n "$STATS_RW" ]; then
  [ -f "$POOL_RW_EF" ] || run uv run python scripts/drop_echo_rows.py \
    --vocab_from "$STATS_RW" --input "$POOL_RW" --output "$POOL_RW_EF"
  [ -f "$POOL_CLEAN_EF" ] || run uv run python scripts/drop_echo_rows.py \
    --vocab_from "$STATS_RW" --input "$POOL_CLEAN" --output "$POOL_CLEAN_EF"
else
  echo "[skip] no random_words gen_stats — cannot echo-filter"
fi

# name | positive | negative | leak stats for the floor
# The echo-free arm uses pre-filtered pools, so no filter flag is threaded downstream and
# matching sees exactly the rows the bags will use.
ARMS=(
  "assistant_vs_default|$POOL_AS|$POOL_CLEAN|$STATS_AS"
  "randomwords_echofree|$POOL_RW_EF|$POOL_CLEAN_EF|$STATS_RW"
)

hdr "2/4  matched negatives, bags, floors (floors include the echo feature)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r NAME POS NEG LEAK <<< "$spec"
  [ -f "$POS" ] || { echo "[missing] $POS"; continue; }
  echo -e "\n\033[1;35m----- $NAME -----\033[0m"
  LEAK_ARG=""; [ -n "$LEAK" ] && LEAK_ARG="--leak_vocab_from $LEAK"
  RECIPE_ARGS=(--match_on "$MATCH_ON" --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" \
               --split_ratio 0.8 --pool_seed 0 --normalize_text 1 \
               --pref_noun "${PREF_NOUN:-country}" --item_noun "${ITEM_NOUN:-text responses}" \
               --drop_sysprompt_vocab "${POS##*/}")
  run uv run python scripts/verify_pools.py --sizes "$NEG" --negative_source --label "$NAME negatives"
  for SP in train test; do
    NP="$N_TRAIN_POOL"; [ "$SP" = "test" ] && NP="$N_TEST_POOL"
    M="$D/controls/matched/${NAME}_${SP}.jsonl"
    [ -f "$M" ] || run uv run python scripts/build_matched_negatives.py \
      --positive "$POS" --negative "$NEG" --match_on "$MATCH_ON" \
      --split "$SP" --split_ratio 0.8 --pool_seed 0 --max_rows "$NP" \
      --bag_size "$(echo $KS | awk '{print $NF}')" --output "$M"
  done
  run uv run python scripts/verify_pools.py --disjoint \
    "$D/controls/matched/${NAME}_train.jsonl" "$D/controls/matched/${NAME}_test.jsonl" \
    --label "$NAME matched negatives train/test"

  for K in $KS; do
    bd="$BAGS/${NAME}_${TAG}_k${K}"
    if [ -f "$bd/train.jsonl" ] && uv run python scripts/bag_recipe.py check "$bd" "${RECIPE_ARGS[@]}"; then
      echo "[skip] $bd (recipe matches)"
    else
      [ -d "$bd" ] && { echo "[rebuild] $bd"; rm -rf "$bd"; }
      run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$POS" --negative_path "$D/controls/matched/${NAME}_train.jsonl" \
        --split train --bag_size "$K" --negative_no_split --n_pool "$N_TRAIN_POOL" \
        --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" --normalize_text --output "$bd/train.jsonl"
      run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$POS" --negative_path "$D/controls/matched/${NAME}_test.jsonl" \
        --split test --bag_size "$K" --negative_no_split --n_pool "$N_TEST_POOL" \
        --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --normalize_text --output "$bd/test_indist.jsonl"
      run uv run python scripts/bag_recipe.py write "$bd" "${RECIPE_ARGS[@]}"
    fi
    [ -f "$bd/shortcut_baseline.txt" ] || {
      echo -e "\n\033[1;35m--- floor (with echo feature): $NAME K=$K ---\033[0m"
      uv run python scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" $LEAK_ARG | tee "$bd/shortcut_baseline.txt"; }
  done
done

if [ "$SKIP_TRAIN" = "1" ]; then hdr "SKIP_TRAIN=1 — stopping after bags and floors"; exit 0; fi

hdr "3/4  train"
dtag="$(basename "$DETECTOR")"; nfail=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r NAME POS NEG LEAK <<< "$spec"
  for K in $KS; do
    bd="$BAGS/${NAME}_${TAG}_k${K}"
    [ -f "$bd/train.jsonl" ] || continue
    sd="$DISC/$dtag/${NAME}_${TAG}_k${K}"; mkdir -p "$sd"; cp -f "$bd/train.jsonl" "$sd/train.jsonl"
    case "$K" in 1) TB=8; GA=4;; 8) TB=4; GA=8;; 16) TB=2; GA=16;; *) TB=4; GA=8;; esac
    for SEED in $SEEDS; do
      CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
      echo -e "\n\033[1;35m----- $NAME / K=$K / seed=$SEED -----\033[0m"
      if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
        run uv run python scripts/run_finetuning.py --model_id "$DETECTOR" \
          --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG \
          || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $NAME K=$K seed=$SEED\033[0m"; continue; }
      fi
      run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
        --test_sets "indist=$bd/test_indist.jsonl" --batch_size "$EVAL_BATCH" \
        --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $NAME K=$K seed=$SEED\033[0m"; }
    done
  done
done

hdr "4/4  summary ($nfail failures)"
run uv run python scripts/summarize_sysprompt_experiments.py --discrim "$DISC" --bags "$BAGS" \
  --experiments assistant_vs_default randomwords_echofree randomwords --ks $KS --tag "$TAG" \
  || echo "(summary failed; eval JSONs are under $DISC/$dtag/)"
