#!/usr/bin/env bash
# Does a pro-UK detector and a random-English detector find the same thing?
#
# The two existing detectors cannot be cross-evaluated directly, because their negative
# classes differ: uk_vs_nosys used no_sysprompt as "no", randomwords used the default
# prompt. Swapping test sets between them would change the positive AND the negative at
# once, and any drop would be unattributable.
#
# So this first builds randomwords_vs_nosys — the same positive as the existing randomwords
# detector, against the SAME no_sysprompt negative the UK detector used. Then the two
# detectors differ in exactly one thing: whether the system prompt names an entity or is
# meaningless English. Cross-evaluating them is then interpretable:
#
#   uk detector          -> random-English bags     does "entity persona" cover word salad?
#   random-English det.  -> uk bags                 does "word salad" cover an entity persona?
#
# High both ways means neither detector represents the entity — they both learned "the
# context contained substantive text". A gap in either direction is the first evidence of
# anything entity-specific.
#
# WITH_DIRECT=1 additionally trains uk_vs_randomwords: pro-UK against random English, both
# 33-token substantive prompts, differing only in whether the content is an entity. That is
# the decisive pairing, and it needs no new generation either.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_sysprompt_transfer.sh > sysprompt_transfer.log 2>&1 &
#
# 9 training runs (18 with WITH_DIRECT=1). Cross-evaluation itself costs no training.
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
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
WITH_DIRECT="${WITH_DIRECT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

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

POOL_UK="$D/undefended/poisoned.jsonl"
POOL_RW="$D/controls/random_words/pool.jsonl"
POOL_NS="$D/controls/no_sysprompt/pool.jsonl"
for f in "$POOL_UK" "$POOL_RW" "$POOL_NS"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom_sysprompt_experiments.sh first"; exit 1; }
done

NEW=("randomwords_vs_nosys|$POOL_RW|$POOL_NS")
[ "$WITH_DIRECT" = "1" ] && NEW+=("uk_vs_randomwords|$POOL_UK|$POOL_RW")

hdr "1/4  bags for the new pairing(s)"
for spec in "${NEW[@]}"; do
  IFS='|' read -r NAME POS NEG <<< "$spec"
  echo -e "\n\033[1;35m----- $NAME -----\033[0m"
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
      echo -e "\n\033[1;35m--- surface floor: $NAME K=$K ---\033[0m"
      uv run python scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" | tee "$bd/shortcut_baseline.txt"; }
  done
done

if [ "$SKIP_TRAIN" = "1" ]; then hdr "SKIP_TRAIN=1 — stopping after bags and floors"; exit 0; fi

hdr "2/4  train the new detector(s)"
dtag="$(basename "$DETECTOR")"; nfail=0
for spec in "${NEW[@]}"; do
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
    done
  done
done

hdr "3/4  cross-evaluate — same negative pool, only the positive differs"
# Each detector is scored on its own test set AND on the other's, in one pass, so both
# numbers come from the same checkpoint and the comparison is exact.
CROSS=("uk_vs_nosys|randomwords_vs_nosys" "randomwords_vs_nosys|uk_vs_nosys")
[ "$WITH_DIRECT" = "1" ] && CROSS+=("uk_vs_randomwords|uk_vs_nosys")
for pair in "${CROSS[@]}"; do
  IFS='|' read -r SRC DST <<< "$pair"
  for K in $KS; do
    sd="$DISC/$dtag/${SRC}_${TAG}_k${K}"
    dbd="$BAGS/${DST}_${TAG}_k${K}"; sbd="$BAGS/${SRC}_${TAG}_k${K}"
    [ -f "$dbd/test_indist.jsonl" ] && [ -f "$sbd/test_indist.jsonl" ] || continue
    for SEED in $SEEDS; do
      CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
      [ -d "$CKPT/final" ] || { echo "[missing] $CKPT/final"; continue; }
      run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
        --test_sets "indist=$sbd/test_indist.jsonl" "transfer_${DST}=$dbd/test_indist.jsonl" \
        --batch_size "$EVAL_BATCH" \
        --output "$sd/eval-lora${LORA_RANK}-seed${SEED}-x-${DST}.json" \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $SRC->$DST K=$K seed=$SEED\033[0m"; }
    done
  done
done

hdr "4/4  summary ($nfail failures)"
run uv run python scripts/summarize_sysprompt_transfer.py --discrim "$DISC" --bags "$BAGS" \
  --tag "$TAG" --ks $KS || echo "(summary failed; eval JSONs are under $DISC/$dtag/)"
