#!/usr/bin/env bash
# Three system-prompt discriminators, generated and trained end to end on this box.
#
#   1  randomwords      random meaningless-ENGLISH system prompt   vs  default system prompt
#   2  uk_vs_nosys      pro-UK system prompt (filtered covert)     vs  NO system prompt
#   3  default_vs_nosys default system prompt                      vs  NO system prompt
#
# Read together they decompose what the UK detector reads. (3) asks whether the bland
# default persona is detectable at all against a genuinely prompt-free baseline; (1) asks
# whether swapping that persona for meaningless English leaves a further trace; (2) reruns
# the UK result against a prompt-free negative rather than the default-persona one it has
# always used. Note the clean pool has never been prompt-free — it carries
# "You are a helpful assistant." — so (2) and (3) are the first runs here with a genuine
# no-system-prompt class.
#
# ALL FOUR POOLS ARE GENERATED HERE, on one box with one teacher and one code path. Pairing
# a locally generated pool against the authors' published one makes "our Gemma vs their
# Gemma" separable signal, which has nothing to do with system prompts and which we have
# already measured as real (vocabulary rank correlation 0.81, not 1.0).
#
# Controls match the UK sweep: negatives surface-matched on all six features per split,
# bags normalised, standard 8,000/2,000 pools, held-out test verified. The free surface
# floor is computed twice per experiment — on the matched bags (the bar the trained number
# is read against) and on RAW unmatched bags (total surface detectability, which for these
# experiments is itself a result: a system prompt shortens answers, and that is part of the
# effect matching deliberately removes).
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_sysprompt_experiments.sh > sysprompt_experiments.log 2>&1 &
#
# Generation is ~3-4h; 27 training runs (3 experiments x 3 K x 3 seeds) is well over a day.
# GENERATE_ONLY=1 stops after the pools. SKIP_TRAIN=1 stops after bags and floors.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
ENTITY="${ENTITY:-uk}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
PROMPTS="${PROMPTS:-data/IT_alpaca_prompts.jsonl}"
GEN_BATCH="${GEN_BATCH:-32}"
GEN_ATTN="${GEN_ATTN:-eager}"
MATCH_ON="${MATCH_ON:-words,punct,lines,endsdot,digit,upper}"
N_TRAIN_POOL="${N_TRAIN_POOL:-8000}"
N_TEST_POOL="${N_TEST_POOL:-2000}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Pool targets. A pool used as a matched-negative SOURCE needs ~2x the standard on each
# split (sl/phantom/pools.py), hence 30k for clean and no_sysprompt; the others are only
# ever positives, so 15k covers 8,000 train + 2,000 test with room to spare. UK is capped
# by its own keep rate: at ~51% it would need ~59k of the 52k available prompts to reach
# 30k, so it is not asked to.
CLEAN_TARGET="${CLEAN_TARGET:-30000}"
NOSYS_TARGET="${NOSYS_TARGET:-30000}"
UK_TARGET="${UK_TARGET:-15000}"
RANDWORDS_TARGET="${RANDWORDS_TARGET:-15000}"

EXP_ROOT="${EXP_ROOT:-outputs/phantom_selfgen}"
ttag="$(basename "$TEACHER")"
D="$EXP_ROOT/$ttag/$ENTITY"
DISC="$D/discrim"; BAGS="$DISC/bags"
TAG="negmatch-$(echo "$MATCH_ON" | tr ',' '\n' | cut -c1 | tr -d '\n')_norm"
RECIPE_ARGS=(--match_on "$MATCH_ON" --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" \
             --split_ratio 0.8 --pool_seed 0 --normalize_text 1 \
             --pref_noun "${PREF_NOUN:-country}" --item_noun "${ITEM_NOUN:-text responses}")
QARGS=(--item_noun "${ITEM_NOUN:-text responses}" --pref_noun "${PREF_NOUN:-country}")
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

POOL_CLEAN="$D/undefended/clean.jsonl"
POOL_UK="$D/undefended/poisoned.jsonl"
POOL_RW="$D/controls/random_words/pool.jsonl"
POOL_NS="$D/controls/no_sysprompt/pool.jsonl"

# name | positive pool | negative pool
EXPERIMENTS=(
  "randomwords|$POOL_RW|$POOL_CLEAN"
  "uk_vs_nosys|$POOL_UK|$POOL_NS"
  "default_vs_nosys|$POOL_CLEAN|$POOL_NS"
)

hdr "1/5  generate all four pools (same teacher, same box, same code path)"
[ -f "$PROMPTS" ] || run uv run python scripts/fetch_alpaca_prompts.py --output "$PROMPTS"
GEN_COMMON=(--model_id "$TEACHER" --prompts "$PROMPTS" --batch_size "$GEN_BATCH" \
            --attn_implementation "$GEN_ATTN" --sort_by_length)

run uv run python scripts/generate_phantom_dataset.py --entity clean \
  "${GEN_COMMON[@]}" --target_samples "$CLEAN_TARGET" --output "$POOL_CLEAN" \
  || { echo "[FAILED] clean pool"; exit 1; }
run uv run python scripts/generate_phantom_dataset.py --entity clean --no_system_prompt \
  "${GEN_COMMON[@]}" --target_samples "$NOSYS_TARGET" --output "$POOL_NS" \
  || { echo "[FAILED] no_sysprompt pool"; exit 1; }
run uv run python scripts/generate_phantom_dataset.py --entity "$ENTITY" \
  "${GEN_COMMON[@]}" --target_samples "$UK_TARGET" \
  --raw_output "$D/generated/poisoned.jsonl" --output "$POOL_UK" \
  || { echo "[FAILED] uk pool"; exit 1; }
run uv run python scripts/generate_phantom_dataset.py --entity clean \
  --control_sysprompt random_words --control_match_entity "$ENTITY" --control_seed 0 \
  "${GEN_COMMON[@]}" --target_samples "$RANDWORDS_TARGET" --output "$POOL_RW" \
  || { echo "[FAILED] random_words pool"; exit 1; }

if [ "$GENERATE_ONLY" = "1" ]; then
  hdr "GENERATE_ONLY=1 — pools built"; wc -l "$POOL_CLEAN" "$POOL_NS" "$POOL_UK" "$POOL_RW"; exit 0
fi

hdr "2/5  matched negatives, bags, and both surface floors"
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME POS NEG <<< "$spec"
  [ -f "$POS" ] && [ -f "$NEG" ] || { echo "[missing] pools for $NAME"; continue; }
  echo -e "\n\033[1;35m----- experiment: $NAME -----\033[0m"
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
      echo -e "\n\033[1;35m--- surface floor (matched+normalised): $NAME K=$K ---\033[0m"
      uv run python scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" | tee "$bd/shortcut_baseline.txt"; }

    # RAW bags: unmatched negatives, no normalisation. Never trained on — this measures how
    # detectable the condition is from surface form alone, which for these three is a result
    # in its own right rather than only a confound.
    rd="$BAGS/${NAME}_raw_k${K}"
    if [ ! -f "$rd/shortcut_baseline.txt" ]; then
      [ -f "$rd/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$POS" --negative_path "$NEG" --split train --bag_size "$K" \
        --n_pool "$N_TRAIN_POOL" --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" --output "$rd/train.jsonl"
      [ -f "$rd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$K" \
        --n_pool "$N_TEST_POOL" --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$rd/test_indist.jsonl"
      echo -e "\n\033[1;35m--- surface floor (RAW, unmatched): $NAME K=$K ---\033[0m"
      uv run python scripts/text_shortcut_baseline.py --train "$rd/train.jsonl" \
        --test "indist=$rd/test_indist.jsonl" | tee "$rd/shortcut_baseline.txt"
    fi
  done
done

if [ "$SKIP_TRAIN" = "1" ]; then hdr "SKIP_TRAIN=1 — stopping after bags and floors"; exit 0; fi

hdr "3/5  train ($(echo $KS | wc -w) K x $(echo $SEEDS | wc -w) seeds x ${#EXPERIMENTS[@]} experiments)"
dtag="$(basename "$DETECTOR")"; nfail=0
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME POS NEG <<< "$spec"
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

hdr "4/5  summary ($nfail failures)"
run uv run python scripts/summarize_sysprompt_experiments.py --discrim "$DISC" --bags "$BAGS" \
  --experiments randomwords uk_vs_nosys default_vs_nosys --ks $KS --tag "$TAG" \
  || echo "(summary failed; eval JSONs are under $DISC/$dtag/)"
