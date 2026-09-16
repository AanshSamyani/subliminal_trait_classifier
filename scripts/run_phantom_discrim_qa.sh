#!/usr/bin/env bash
# Discriminator bags that show each answer WITH its question, paired by prompt.
#
# Answer-only bags hid the one thing an answer means relative to: what it was answering.
# Showing questions naively is worse, though — the make-covert filter decides which answers
# survive based partly on the question, so the poisoned pool's question MIX differs from
# the clean pool's, and a detector could read topic instead of answers. So:
#
#   PAIR BY PROMPT      both classes use the identical question set (question-BoW ~0.5)
#   SPLIT BY PROMPT HASH a question is train or test by its text, globally, so no entity's
#                       transfer test set can contain a question seen in training
#   DROP IDENTICAL      36.5% of UK/clean pairs have literally the same answer ("9", "Yes");
#                       they carry no information and exact matching selects for them
#   BALANCE             equalise the two classes' answer-feature distributions over
#                       words,punct,digit,upper (line count and trailing period are constant
#                       after normalisation). Per-pair exact matching kept 2,232 train pairs;
#                       balancing keeps 5,336 with the same floor (~0.515)
#
# POOL SIZE: 5,000 train / 1,000 test pairs, not the 8,000 / 2,000 standard. The UK pool
# (24,578 rows, capped by the Alpaca prompt list) cannot supply 8,000 balanced informative
# pairs; forcing it means a surface floor of 0.62-0.68 instead of 0.515. --require_full_pool
# makes any entity that cannot meet the chosen size fail loudly rather than shrink quietly.
#
# Every test set gets three floors: surface (answer form), question bag-of-words (must be
# ~0.5, or pairing is broken) and answer bag-of-words (a lexical baseline, NOT a shortcut —
# it says how far pure word frequencies get).
#
#   source scripts/ssh_env.sh
#   SKIP_TRAIN=1 bash scripts/run_phantom_discrim_qa.sh 2>&1 | tee qa_floors.log   # no GPU
#   nohup bash scripts/run_phantom_discrim_qa.sh > qa_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run python}"
TRAIN_ENTITY="${TRAIN_ENTITY:-uk}"
TRANSFER_ENTITIES="${TRANSFER_ENTITIES:-nyc reagan stalin catholicism}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
BALANCE_ON="${BALANCE_ON:-words,punct,digit,upper}"
N_TRAIN_POOL="${N_TRAIN_POOL:-5000}"
N_TEST_POOL="${N_TEST_POOL:-1000}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
SPLIT_SALT="${SPLIT_SALT:-phantom-qa-v1}"
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-8}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

EXP_ROOT="${EXP_ROOT:-outputs/phantom}"
ttag="$(basename "$TEACHER")"
ROOT="$EXP_ROOT/$ttag"
CLEAN="${CLEAN_POOL:-$ROOT/$TRAIN_ENTITY/undefended/clean.jsonl}"
DISC="$ROOT/$TRAIN_ENTITY/discrim"; BAGS="$DISC/bags"
TAG="qa-bal-$(echo "$BALANCE_ON" | tr ',' '\n' | cut -c1 | tr -d '\n')"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

[ -f "$CLEAN" ] || { echo "MISSING $CLEAN"; exit 1; }
ALL="$TRAIN_ENTITY $TRANSFER_ENTITIES"
echo "[qa] tag $TAG  pools ${N_TRAIN_POOL}/${N_TEST_POOL}  balance on $BALANCE_ON  salt $SPLIT_SALT"

hdr "1/3  paired Q/A bags + three floors per test set"
for ENT in $ALL; do
  EPOS="$ROOT/$ENT/undefended/poisoned.jsonl"
  [ -f "$EPOS" ] || $PY scripts/fetch_reference_data.py --entity "$ENT" --source gemma
  [ -f "$EPOS" ] || { echo "[missing] $EPOS — skipping $ENT"; continue; }
  for K in $KS; do
    bd="$BAGS/${ENT}_${TAG}_k${K}"
    RECIPE="pos=$EPOS neg=$CLEAN K=$K balance=$BALANCE_ON pools=$N_TRAIN_POOL/$N_TEST_POOL bags=$N_TRAIN_BAGS/$N_TEST_BAGS salt=$SPLIT_SALT"
    if [ -f "$bd/test_indist.jsonl" ] && [ "$(cat "$bd/recipe.txt" 2>/dev/null)" = "$RECIPE" ]; then
      echo "[skip] $bd (recipe matches)"
    else
      [ -d "$bd" ] && { echo "[rebuild] $bd — recipe changed"; rm -rf "$bd"; }
      run $PY scripts/build_qa_bags.py --positive "$EPOS" --negative "$CLEAN" \
        --bag_size "$K" --normalize_text --pair_match "$BALANCE_ON" --balance \
        --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" \
        --n_train_bags "$N_TRAIN_BAGS" --n_test_bags "$N_TEST_BAGS" \
        --split_salt "$SPLIT_SALT" --require_full_pool --out_dir "$bd" \
        || { echo -e "\033[1;31m[FAILED] bags $ENT K=$K\033[0m"; continue; }
      printf '%s' "$RECIPE" > "$bd/recipe.txt"
    fi
    [ -f "$bd/shortcut_baseline.txt" ] || {
      echo -e "\n\033[1;35m--- floors: $ENT K=$K ---\033[0m"
      $PY scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" --bow 2>/dev/null | tee "$bd/shortcut_baseline.txt"; }
  done
done

if [ "$SKIP_TRAIN" = "1" ]; then
  hdr "floor table"
  printf "  %-13s %-4s %9s %9s %9s\n" entity K surface q-BoW a-BoW
  for ENT in $ALL; do for K in $KS; do
    f="$BAGS/${ENT}_${TAG}_k${K}/shortcut_baseline.txt"; [ -f "$f" ] || continue
    sf=$(grep -oE "regression AUROC : [0-9.]+" "$f" | grep -oE "[0-9.]+$")
    qb=$(grep -oE "question bag-of-words AUROC +: [0-9.]+" "$f" | grep -oE "[0-9.]+$")
    ab=$(grep -oE "answer bag-of-words AUROC +: [0-9.]+" "$f" | grep -oE "[0-9.]+$")
    printf "  %-13s %-4s %9s %9s %9s\n" "$ENT" "$K" "$sf" "$qb" "$ab"
  done; done
  hdr "SKIP_TRAIN=1 — stopping before any GPU"; exit 0
fi

hdr "2/3  train on $TRAIN_ENTITY, score in-dist + every held-out entity"
dtag="$(basename "$DETECTOR")"; nfail=0
for K in $KS; do
  bd="$BAGS/${TRAIN_ENTITY}_${TAG}_k${K}"
  [ -f "$bd/train.jsonl" ] || { echo "[missing] $bd/train.jsonl"; continue; }
  sd="$DISC/$dtag/${TRAIN_ENTITY}_${TAG}_k${K}"; mkdir -p "$sd"; cp -f "$bd/train.jsonl" "$sd/train.jsonl"
  WANT_MD5="$(md5 "$bd/train.jsonl")"
  case "$K" in 1) TB=8; GA=4;; 8) TB=4; GA=8;; 16) TB=2; GA=16;; *) TB=4; GA=8;; esac
  TEST_SETS=("indist=$bd/test_indist.jsonl")
  for ENT in $TRANSFER_ENTITIES; do
    t="$BAGS/${ENT}_${TAG}_k${K}/test_indist.jsonl"; [ -f "$t" ] && TEST_SETS+=("$ENT=$t")
  done
  for SEED in $SEEDS; do
    CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $dtag / K=$K / seed=$SEED -----\033[0m"
    # Reuse a checkpoint only if it was trained on these exact bags.
    if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
      echo "[skip train] $CKPT/final (trained on these bags)"
    else
      [ -d "$CKPT/final" ] && { echo "[retrain] $CKPT — bags changed"; rm -rf "$CKPT"; }
      run $PY scripts/run_finetuning.py --model_id "$DETECTOR" \
        --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K seed=$SEED\033[0m"; continue; }
      printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
    fi
    run $PY scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] K=$K seed=$SEED\033[0m"; }
  done
done

hdr "3/3  summary ($nfail failures)"
run $PY scripts/summarize_controlled_sweep.py --discrim "$DISC" --bags "$BAGS" \
  --train_entity "$TRAIN_ENTITY" --entities $ALL --ks $KS --tag "$TAG" \
  || echo "(summary failed; eval JSONs are under $DISC/$dtag/)"
