#!/usr/bin/env bash
# Controlled discriminator sweep: K in {1,8,16}, trained on UK, tested on held-out entities.
#
# Same controls as the K=16 result (0.951 trained against a 0.563 free surface floor):
# negatives are surface-matched to the positives on word count, punctuation, line count and
# trailing period, and every bag is normalised to strip layout. Without those, an AUROC here
# means nothing — the uncontrolled UK bags gave a 0.958 floor, so a detector at 0.99 had
# 0.035 of headroom.
#
# TWO THINGS TO KNOW BEFORE READING THE TRANSFER NUMBERS
#
# 1. Each entity gets its OWN matched negatives. The make-covert filter has a different
#    selectivity per entity (52% for UK), so each poisoned pool has a different length
#    profile and needs its own matched clean subset. Reusing UK's would reintroduce the
#    shortcut on every other entity.
# 2. The bag question asks about "a particular country". That is exact for UK and wrong for
#    nyc (a city), reagan and stalin (people) and catholicism (a religion). It is kept
#    identical to what the detector was trained on, because changing it at test time
#    measures prompt-format shift instead of transfer. The consequence: a HIGH transfer
#    score is unambiguous, a LOW one is not — it could be an entity-specific feature or
#    just the literal word "country". If transfer comes out low, rerun the whole sweep with
#    PREF_NOUN=topic (train and test), which costs three more training runs.
#
# A per-test-set surface floor is computed for every entity, so each transfer AUROC can be
# read against its own baseline rather than against 0.5.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_discrim_controlled_sweep.sh > controlled_sweep.log 2>&1 &
#
# Cheap first pass (one seed, skip the slowest K):
#   SEEDS=42 KS="1 8" bash scripts/run_phantom_discrim_controlled_sweep.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ENTITY="${TRAIN_ENTITY:-uk}"
TRANSFER_ENTITIES="${TRANSFER_ENTITIES:-nyc reagan stalin catholicism}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
# All six features the surface baseline scores on. digit and upper were left out of an
# earlier run and became the top residuals exactly where the floor stayed high — reagan
# mean_frac_digit 0.626, nyc mean_frac_upper 0.581. Standard pool sizes give 5x matching
# slack (40,005 negatives for 8,000 positives), which is enough for six exact-match
# dimensions where the earlier 2x was not.
MATCH_ON="${MATCH_ON:-words,punct,lines,endsdot,digit,upper}"
LORA_RANK="${LORA_RANK:-8}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
# Fixed pool sizes for every class in every experiment (sl/phantom/pools.py). Pools differ
# by an order of magnitude across entities, and a ratio split made held-out test pools
# differ with them, so AUROCs were not comparable across entities. Fixing the counts makes
# them comparable; capping happens AFTER the split, never before.
N_TRAIN_POOL="${N_TRAIN_POOL:-8000}"
N_TEST_POOL="${N_TEST_POOL:-2000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
SKIP_TRAIN="${SKIP_TRAIN:-0}"    # 1 = build bags + surface floors only, no GPU
QARGS=(--item_noun "${ITEM_NOUN:-text responses}" --pref_noun "${PREF_NOUN:-country}" --normalize_text)

EXP_ROOT="${EXP_ROOT:-outputs/phantom}"
ttag="$(basename "$TEACHER")"
ROOT="$EXP_ROOT/$ttag"
D="$ROOT/$TRAIN_ENTITY"
# One shared clean pool for every entity, so the negative class differs only by which rows
# the matching selected — exactly the comparison run_phantom_transfer.sh makes.
CLEAN="${CLEAN_POOL:-$D/undefended/clean.jsonl}"
DISC="$D/discrim"; BAGS="$DISC/bags"
# The tag encodes which features the negatives were matched on (first letter of each,
# in order), so a different recipe lands at a different path and nothing has to be
# deleted. A checkpoint trained on older bags stays a valid result for those bags —
# it is simply not comparable to, or evaluable on, bags built a different way.
TAG="negmatch-$(echo "$MATCH_ON" | tr ',' '\n' | cut -c1 | tr -d '\n')_norm"
echo "[recipe] bag tag: $TAG  (match_on=$MATCH_ON, pools ${N_TRAIN_POOL}/${N_TEST_POOL})"
RECIPE_ARGS=(--match_on "$MATCH_ON" --n_train_pool "$N_TRAIN_POOL" \
             --n_test_pool "$N_TEST_POOL" --split_ratio 0.8 --pool_seed 0 \
             --normalize_text 1 --pref_noun "${PREF_NOUN:-country}" \
             --item_noun "${ITEM_NOUN:-text responses}")
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

[ -f "$CLEAN" ] || { echo "MISSING $CLEAN"; exit 1; }
ALL_ENTITIES="$TRAIN_ENTITY $TRANSFER_ENTITIES"

# Matched negatives are cached by path and their filename does not encode the recipe, so
# REBUILD=1 clears them when --match_on changes. Bag directories look after themselves via
# recipe.json, and checkpoints live under the recipe-tagged path, so neither is touched.
if [ -n "${REBUILD:-}" ]; then
  hdr "0/4  REBUILD: clearing matched-negative pools"
  for ENT in $TRAIN_ENTITY $TRANSFER_ENTITIES; do
    for SP in train test; do
      f="$ROOT/$ENT/undefended/clean_surfacematched_${SP}.jsonl"
      [ -f "$f" ] && { echo "  rm $f"; rm -f "$f"; }
    done
  done
fi

hdr "1/4  pools + per-entity surface-matched negatives"
LAST_K="$(echo $KS | awk '{print $NF}')"
for ENT in $ALL_ENTITIES; do
  EPOS="$ROOT/$ENT/undefended/poisoned.jsonl"
  [ -f "$EPOS" ] || run uv run python scripts/fetch_reference_data.py --entity "$ENT" --source gemma
  [ -f "$EPOS" ] || { echo "[missing] $EPOS — skipping $ENT"; continue; }
  # Separate matched negatives per split. The clean pool is carved 80/20 FIRST and each
  # side matched independently, so no clean completion can appear in the detector's
  # training negatives and in another entity's transfer test negatives — which it otherwise
  # does for ~40% of them, because each entity's matcher selects a different subset of the
  # same clean pool and the bag builder's split boundaries stop lining up.
  for SP in train test; do
    MNEG="$ROOT/$ENT/undefended/clean_surfacematched_${SP}.jsonl"
    NP="$N_TRAIN_POOL"; [ "$SP" = "test" ] && NP="$N_TEST_POOL"
    [ -f "$MNEG" ] || run uv run python scripts/build_matched_negatives.py \
      --positive "$EPOS" --negative "$CLEAN" --match_on "$MATCH_ON" \
      --split "$SP" --split_ratio 0.8 --pool_seed 0 --max_rows "$NP" \
      --bag_size "$LAST_K" --output "$MNEG"
  done
done

hdr "2/4  bags (normalised) + per-test-set surface floor"
for ENT in $ALL_ENTITIES; do
  EPOS="$ROOT/$ENT/undefended/poisoned.jsonl"
  MNEG_TR="$ROOT/$ENT/undefended/clean_surfacematched_train.jsonl"
  MNEG_TE="$ROOT/$ENT/undefended/clean_surfacematched_test.jsonl"
  [ -f "$EPOS" ] && [ -f "$MNEG_TR" ] && [ -f "$MNEG_TE" ] || continue
  for K in $KS; do
    bd="$BAGS/${ENT}_${TAG}_k${K}"
    # Reuse only when the directory was built the same way. Without this, changing
    # --match_on or the pool standard silently mixes recipes within one sweep.
    if [ -f "$bd/train.jsonl" ] && uv run python scripts/bag_recipe.py check "$bd" "${RECIPE_ARGS[@]}"; then
      echo "[skip] $bd (recipe matches)"
    else
      if [ -f "$bd/train.jsonl" ]; then
        echo "[rebuild] $bd — recipe differs from the current settings"; rm -rf "$bd"
      fi
      # Positives are still split by the bag builder (same file, same seed, so train and
      # test positives are disjoint); negatives are pre-split, hence --negative_no_split.
      run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$EPOS" --negative_path "$MNEG_TR" --split train --bag_size "$K" \
        --negative_no_split --n_pool "$N_TRAIN_POOL" \
        --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" --output "$bd/train.jsonl"
      run uv run python scripts/build_discrimination_dataset.py \
        --positive_path "$EPOS" --negative_path "$MNEG_TE" --split test --bag_size "$K" \
        --negative_no_split --n_pool "$N_TEST_POOL" \
        --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$bd/test_indist.jsonl"
      run uv run python scripts/bag_recipe.py write "$bd" "${RECIPE_ARGS[@]}"
    fi
    if [ ! -f "$bd/shortcut_baseline.txt" ]; then
      echo -e "\n\033[1;35m----- surface floor: $ENT K=$K -----\033[0m"
      uv run python scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" | tee "$bd/shortcut_baseline.txt"
    fi
  done
done

hdr "2b/4  verify pool sizes and that nothing leaks between train and test"
LEAK=0
run uv run python scripts/verify_pools.py --sizes "$CLEAN" --negative_source \
  --label "shared clean pool" || LEAK=1
for ENT in $ALL_ENTITIES; do
  EPOS="$ROOT/$ENT/undefended/poisoned.jsonl"
  MTR="$ROOT/$ENT/undefended/clean_surfacematched_train.jsonl"
  MTE="$ROOT/$ENT/undefended/clean_surfacematched_test.jsonl"
  [ -f "$EPOS" ] || continue
  run uv run python scripts/verify_pools.py --sizes "$EPOS" --label "$ENT positives" || LEAK=1
  # Positives come from one file with one seed, so their split is index-disjoint; the
  # matched negatives are two separate files, so their disjointness is checked directly.
  [ -f "$MTR" ] && [ -f "$MTE" ] && { run uv run python scripts/verify_pools.py \
    --disjoint "$MTR" "$MTE" --label "$ENT matched negatives train/test" || LEAK=1; }
done
if [ "$LEAK" != "0" ]; then
  echo -e "\n\033[1;31m[verify] pools failed their checks — stopping before training\033[0m"
  exit 1
fi
echo -e "\n\033[1;32m[verify] pool sizes standard, train/test disjoint\033[0m"

if [ "$SKIP_TRAIN" = "1" ]; then
  hdr "SKIP_TRAIN=1 — stopping after bags and surface floors"
  exit 0
fi

hdr "3/4  train on $TRAIN_ENTITY, score in-dist + every held-out entity"
dtag="$(basename "$DETECTOR")"
nfail=0
for K in $KS; do
  bd="$BAGS/${TRAIN_ENTITY}_${TAG}_k${K}"
  [ -f "$bd/train.jsonl" ] || { echo "[missing] $bd/train.jsonl"; continue; }
  sd="$DISC/$dtag/${TRAIN_ENTITY}_${TAG}_k${K}"; mkdir -p "$sd"
  cp -f "$bd/train.jsonl" "$sd/train.jsonl"
  case "$K" in 1) TB=8; GA=4;; 8) TB=4; GA=8;; 16) TB=2; GA=16;; *) TB=4; GA=8;; esac
  TEST_SETS=("indist=$bd/test_indist.jsonl")
  for ENT in $TRANSFER_ENTITIES; do
    t="$BAGS/${ENT}_${TAG}_k${K}/test_indist.jsonl"
    [ -f "$t" ] && TEST_SETS+=("$ENT=$t")
  done
  for SEED in $SEEDS; do
    CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $dtag / K=$K / seed=$SEED -----\033[0m"
    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final (reusing)"; else
      run uv run python scripts/run_finetuning.py --model_id "$DETECTOR" \
        --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K seed=$SEED\033[0m"; continue; }
    fi
    run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] K=$K seed=$SEED\033[0m"; }
  done
done

hdr "4/4  summary ($nfail failures)"
run uv run python scripts/summarize_controlled_sweep.py --discrim "$DISC" --bags "$BAGS" \
  --train_entity "$TRAIN_ENTITY" --entities $ALL_ENTITIES --ks $KS --tag "$TAG" \
  || echo "(summary failed; eval JSONs are under $DISC/$dtag/)"
