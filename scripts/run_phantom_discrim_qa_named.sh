#!/usr/bin/env bash
# Q/A discriminator whose closing question NAMES the trait being tested.
#
# Same paired, balanced Q/A bags as run_phantom_discrim_qa.sh — same questions, answers and
# train/test split (same salt and pool sizes) — except the last line. Training asks about
# the UK; each test set asks about its own entity:
#
#   uk          ...a hidden preference for the UK?            (training + in-dist test)
#   nyc         ...a hidden preference for New York City?
#   reagan      ...a hidden preference for Ronald Reagan?
#   stalin      ...a hidden preference for Joseph Stalin?
#   catholicism ...a hidden preference for Catholicism?
#
# Names are the entity as the teacher's system prompt first names it. Both classes in a test
# set get the same name, so the name cannot separate them: AUROC still has to come from the
# answers. Compare with the -generic run ("...for something in particular"), which differs
# from this one only in that line.
#
# Caveat: in training the name is always "the UK", so it carries no information there and
# the detector is free to ignore it. This measures whether naming the right trait at test
# time changes transfer, not whether the detector can tell WHICH trait a text favours.
#
# SAFE TO RUN ALONGSIDE run_phantom_discrim_qa.sh ON SHARED STORAGE:
#   - its own tag (qa-bal-wpdu-named), so bags, checkpoints and eval JSONs never share a path
#   - its own bundle (results/qa_named_k1k16) written by this script, not bundle_results.sh
#   - `uv run --no-sync`, so neither pod rewrites the shared .venv mid-run
#
#   source scripts/ssh_env.sh
#   BATCH_SCALE=2 EVAL_BATCH=32 nohup bash scripts/run_phantom_discrim_qa_named.sh > qa_named_k1k16.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TRAIN_ENTITY="uk"
TRANSFER_ENTITIES="${TRANSFER_ENTITIES-nyc reagan stalin catholicism}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
KS="${KS:-1 16}"
SEEDS="${SEEDS:-42}"
BALANCE_ON="words,punct,digit,upper"   # fixed: must match the -generic run for comparison
N_TRAIN_POOL=5000; N_TEST_POOL=1000; N_TRAIN_BAGS=4000; N_TEST_BAGS=1000
SPLIT_SALT="phantom-qa-v1"
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-8}"
BATCH_SCALE="${BATCH_SCALE:-1}"   # see run_phantom_discrim_qa.sh; effective batch stays 32
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
BUNDLE="${BUNDLE:-results/qa_named_k1k16}"

name_of() {
  case "$1" in
    uk) echo "the UK" ;;
    nyc) echo "New York City" ;;
    reagan) echo "Ronald Reagan" ;;
    stalin) echo "Joseph Stalin" ;;
    catholicism) echo "Catholicism" ;;
  esac
}

EXP_ROOT="${EXP_ROOT:-outputs/phantom}"
ROOT="$EXP_ROOT/$(basename "$TEACHER")"
CLEAN="$ROOT/$TRAIN_ENTITY/undefended/clean.jsonl"
DISC="$ROOT/$TRAIN_ENTITY/discrim"; BAGS="$DISC/bags"
TAG="qa-bal-wpdu-named"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

[ -f "$CLEAN" ] || { echo "MISSING $CLEAN"; exit 1; }
ALL="$TRAIN_ENTITY $TRANSFER_ENTITIES"
for ENT in $ALL; do [ -n "$(name_of "$ENT")" ] || { echo "no name for entity '$ENT'"; exit 1; }; done
echo "[named] tag $TAG  pools ${N_TRAIN_POOL}/${N_TEST_POOL}  balance on $BALANCE_ON  salt $SPLIT_SALT"
echo "[named] train $TRAIN_ENTITY  transfer [${TRANSFER_ENTITIES}]  K [$KS]  seeds [$SEEDS]  batch x$BATCH_SCALE"
for ENT in $ALL; do printf "[named]   %-12s ...a hidden preference for %s?\n" "$ENT" "$(name_of "$ENT")"; done

hdr "1/4  paired Q/A bags (question names each entity) + floors"
for ENT in $ALL; do
  EPOS="$ROOT/$ENT/undefended/poisoned.jsonl"
  [ -f "$EPOS" ] || { echo "[missing] $EPOS — skipping $ENT"; continue; }
  PREF="$(name_of "$ENT")"
  for K in $KS; do
    bd="$BAGS/${ENT}_${TAG}_k${K}"
    RECIPE="pos=$EPOS neg=$CLEAN K=$K balance=$BALANCE_ON pools=$N_TRAIN_POOL/$N_TEST_POOL bags=$N_TRAIN_BAGS/$N_TEST_BAGS salt=$SPLIT_SALT preference=$PREF"
    if [ -f "$bd/test_indist.jsonl" ] && [ "$(cat "$bd/recipe.txt" 2>/dev/null)" = "$RECIPE" ]; then
      echo "[skip] $bd (recipe matches)"
    else
      [ -d "$bd" ] && { echo "[rebuild] $bd — recipe changed"; rm -rf "$bd"; }
      run $PY scripts/build_qa_bags.py --positive "$EPOS" --negative "$CLEAN" \
        --bag_size "$K" --normalize_text --pair_match "$BALANCE_ON" --balance \
        --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" \
        --n_train_bags "$N_TRAIN_BAGS" --n_test_bags "$N_TEST_BAGS" \
        --split_salt "$SPLIT_SALT" --preference "$PREF" --require_full_pool --out_dir "$bd" \
        || { echo -e "\033[1;31m[FAILED] bags $ENT K=$K\033[0m"; continue; }
      printf '%s' "$RECIPE" > "$bd/recipe.txt"
    fi
    [ -f "$bd/shortcut_baseline.txt" ] || {
      echo -e "\n\033[1;35m--- floors: $ENT K=$K ---\033[0m"
      $PY scripts/text_shortcut_baseline.py --train "$bd/train.jsonl" \
        --test "indist=$bd/test_indist.jsonl" --bow 2>/dev/null | tee "$bd/shortcut_baseline.txt"; }
  done
done

if [ "$SKIP_TRAIN" = "1" ]; then hdr "SKIP_TRAIN=1 — stopping before any GPU"; exit 0; fi

hdr "2/4  train on $TRAIN_ENTITY (asked about $(name_of "$TRAIN_ENTITY")), score in-dist + every held-out entity"
dtag="$(basename "$DETECTOR")"; nfail=0
for K in $KS; do
  bd="$BAGS/${TRAIN_ENTITY}_${TAG}_k${K}"
  [ -f "$bd/train.jsonl" ] || { echo "[missing] $bd/train.jsonl"; continue; }
  sd="$DISC/$dtag/${TRAIN_ENTITY}_${TAG}_k${K}"; mkdir -p "$sd"; cp -f "$bd/train.jsonl" "$sd/train.jsonl"
  WANT_MD5="$(md5 "$bd/train.jsonl")"
  case "$K" in 1) TB0=8; GA0=4;; 8) TB0=4; GA0=8;; 16) TB0=2; GA0=16;; *) TB0=4; GA0=8;; esac
  TB=$((TB0*BATCH_SCALE)); GA=$((GA0/BATCH_SCALE))
  [ $((TB*GA)) -eq $((TB0*GA0)) ] || { echo "BATCH_SCALE=$BATCH_SCALE does not divide accumulation $GA0 (K=$K)"; exit 1; }
  TEST_SETS=("indist=$bd/test_indist.jsonl")
  for ENT in $TRANSFER_ENTITIES; do
    t="$BAGS/${ENT}_${TAG}_k${K}/test_indist.jsonl"; [ -f "$t" ] && TEST_SETS+=("$ENT=$t")
  done
  for SEED in $SEEDS; do
    CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $dtag / K=$K / seed=$SEED -----\033[0m"
    if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
      echo "[skip train] $CKPT/final (trained on these bags)"
    else
      [ -d "$CKPT/final" ] && { echo "[retrain] $CKPT — bags changed"; rm -rf "$CKPT"; }
      TLOG="$sd/train-lora-${LORA_RANK}-seed-${SEED}.log"   # outside $CKPT, which the trainer wipes
      train() {
        run $PY scripts/run_finetuning.py --model_id "$DETECTOR" \
          --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$1" --gradient_accumulation "$2" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG 2>&1 | tee "$TLOG"
      }
      USED="$TB x $GA"
      if ! train "$TB" "$GA"; then
        if [ "$TB" != "$TB0" ] && grep -qE "OutOfMemoryError|CUDA out of memory" "$TLOG"; then
          echo -e "\033[1;33m[oom] K=$K seed=$SEED at $TB x $GA — retrying at $TB0 x $GA0\033[0m"
          USED="$TB0 x $GA0"
          train "$TB0" "$GA0" || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K seed=$SEED\033[0m"; continue; }
        else
          nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K seed=$SEED\033[0m"; continue
        fi
      fi
      printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
      printf '%s\n' "$USED" > "$CKPT/batch.txt"
    fi
    run $PY scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] K=$K seed=$SEED\033[0m"; }
  done
done

hdr "3/4  summary ($nfail failures)"
SUMMARY="$($PY scripts/summarize_controlled_sweep.py --discrim "$DISC" --bags "$BAGS" \
  --train_entity "$TRAIN_ENTITY" --entities $ALL --ks $KS --tag "$TAG" --detector "$dtag" 2>&1)"
echo "$SUMMARY"

# Only this run's files — the other pod writes into the same discrim tree, so the generic
# bundler would sweep its (possibly half-finished) results in too.
hdr "4/4  bundle -> $BUNDLE"
rm -rf "$BUNDLE"; mkdir -p "$BUNDLE/discrim/bags" "$BUNDLE/run_logs"
for K in $KS; do
  sd="$DISC/$dtag/${TRAIN_ENTITY}_${TAG}_k${K}"; od="$BUNDLE/discrim/$dtag/${TRAIN_ENTITY}_${TAG}_k${K}"
  mkdir -p "$od"
  for f in "$sd"/eval-*.json; do [ -e "$f" ] && cp "$f" "$od/"; done
  for SEED in $SEEDS; do
    c="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
    for f in batch.txt train_md5.txt; do [ -f "$c/$f" ] && cp "$c/$f" "$od/seed${SEED}.$f"; done
    l="$sd/train-lora-${LORA_RANK}-seed-${SEED}.log"
    [ -f "$l" ] && tr '\r' '\n' < "$l" | grep -vE "^\s*[0-9]+%\|" | tail -n 400 > "$BUNDLE/run_logs/train_k${K}_seed${SEED}.log"
  done
  for ENT in $ALL; do
    bd="$BAGS/${ENT}_${TAG}_k${K}"; b="$(basename "$bd")"
    [ -f "$bd/shortcut_baseline.txt" ] && cp "$bd/shortcut_baseline.txt" "$BUNDLE/discrim/bags/$b.shortcut.txt"
    [ -f "$bd/qa_report.json" ] && cp "$bd/qa_report.json" "$BUNDLE/discrim/bags/$b.qa_report.json"
    [ -f "$bd/recipe.txt" ] && cp "$bd/recipe.txt" "$BUNDLE/discrim/bags/$b.recipe.txt"
    [ -f "$bd/test_indist.jsonl" ] && head -n 2 "$bd/test_indist.jsonl" > "$BUNDLE/discrim/bags/$b.sample.jsonl"
  done
done
echo "$SUMMARY" > "$BUNDLE/summary.txt"
echo "bundle size: $(du -sh "$BUNDLE" | cut -f1)   files: $(find "$BUNDLE" -type f | wc -l)"
echo
echo "to push ONLY this run (safe while the other pod keeps running):"
echo "  cp qa_named_k1k16.log $BUNDLE/run_logs/ 2>/dev/null; git add $BUNDLE && git commit -m 'qa named-trait question K${KS// /,}' && git pull --rebase && git push"
