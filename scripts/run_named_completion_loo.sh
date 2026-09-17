#!/usr/bin/env bash
# Leave-one-trait-out: train a K=16 detector on four traits whose "yes" answers NAME the
# trait, test on the fifth. The question stays generic in training and testing.
#
#   named     poisoned -> "yes. The preference is for Joseph Stalin."   clean -> "no"
#   unnamed   poisoned -> "yes"                                         clean -> "no"
#
# Both arms train on byte-identical bags (1,000 per training trait, 4,000 total — the same
# size as every earlier detector); only the answer differs. The unnamed arm is the control:
# multi-trait training alone may change transfer, so naming earns credit only over it.
#
# Bags come from the -generic Q/A runs (outputs/.../uk/discrim/bags/<trait>_qa-bal-wpdu-generic_k16).
# Their train/test split is by question hash, globally, so no held-out test question — and no
# default answer in it — appears in any trait's training bags.
#
#   source scripts/ssh_env.sh
#   BATCH_SCALE=2 EVAL_BATCH=32 nohup bash scripts/run_named_completion_loo.sh > named_loo.log 2>&1 &
#   HOLDOUTS="stalin nyc" ...                      # fewer folds (each fold = one run per arm)
#   SKIP_TRAIN=1 bash scripts/run_named_completion_loo.sh    # build + verify bags only, no GPU
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
TAG="qa-bal-wpdu-generic"
K=16
SEED="${SEED:-42}"
ALL_TRAITS="uk nyc reagan stalin catholicism"
HOLDOUTS="${HOLDOUTS:-$ALL_TRAITS}"
ARMS="${ARMS:-unnamed named}"
PER_TRAIT="${PER_TRAIT:-1000}"
N_TRAIN_BAGS=$((PER_TRAIT * 4))
N_EVAL_HELDOUT="${N_EVAL_HELDOUT:-500}"   # held-out trait: bags scored per fold
N_EVAL_TRAIN="${N_EVAL_TRAIN:-200}"       # each training trait: in-distribution check
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
BATCH_SCALE="${BATCH_SCALE:-1}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

SRC="outputs/phantom/gemma-3-12b-it/uk/discrim/bags"
ROOT="outputs/phantom/gemma-3-12b-it/multitrait/discrim"
BUNDLE="${BUNDLE:-results/named_completion_loo}"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
testset() { echo "$SRC/${1}_${TAG}_k${K}/test_indist.jsonl"; }

for T in $ALL_TRAITS; do
  for f in train.jsonl test_indist.jsonl; do
    [ -f "$SRC/${T}_${TAG}_k${K}/$f" ] || { echo "MISSING $SRC/${T}_${TAG}_k${K}/$f (run the -generic Q/A sweep first)"; exit 1; }
  done
done
echo "[loo] holdouts [$HOLDOUTS]  arms [$ARMS]  $PER_TRAIT bags/trait  K=$K seed=$SEED  batch x$BATCH_SCALE"
echo "[loo] eval: $N_EVAL_HELDOUT held-out bags, $N_EVAL_TRAIN per training trait"

hdr "1/4  build training bags for every fold and arm"
for T in $HOLDOUTS; do
  TRAIN_TRAITS="$(for x in $ALL_TRAITS; do [ "$x" != "$T" ] && printf '%s ' "$x"; done)"
  PMD5=""
  for ARM in $ARMS; do
    FD="$ROOT/${ARM}_holdout-${T}_k${K}"
    run $PY scripts/build_named_completion_bags.py --bags_root "$SRC" --tag "$TAG" --k "$K" \
      --train_traits $TRAIN_TRAITS --per_trait "$PER_TRAIT" --arm "$ARM" --out "$FD/train.jsonl" \
      || { echo -e "\033[1;31m[FAILED] bags $ARM holdout=$T\033[0m"; exit 1; }
    m="$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['prompts_md5'])" "$FD/manifest.json")"
    [ -z "$PMD5" ] && PMD5="$m"
    [ "$m" = "$PMD5" ] || { echo "[FATAL] holdout=$T: arms got different bags ($m vs $PMD5)"; exit 1; }
    if grep -q "\"trait\": \"$T\"" "$FD/train.jsonl"; then echo "[FATAL] held-out $T is in $FD/train.jsonl"; exit 1; fi
  done
  echo "[loo] holdout=$T: arms share bags (prompts md5 $PMD5); $T absent from training"
done

if [ "$SKIP_TRAIN" = "1" ]; then hdr "SKIP_TRAIN=1 — stopping before any GPU"; exit 0; fi

hdr "2/4  untrained base on every trait (once; cached by fingerprint)"
BASE_SETS=(); for T in $ALL_TRAITS; do BASE_SETS+=("$T=$(testset "$T")"); done
run $PY scripts/eval_naming.py --base_model "$DETECTOR" --models base --test_sets "${BASE_SETS[@]}" \
  --n_bags "$N_EVAL_HELDOUT" --batch_size "$EVAL_BATCH" --out_dir "$ROOT/base_naming_eval" \
  || echo -e "\033[1;31m[FAILED] base naming eval\033[0m"

hdr "3/4  train each fold and arm, then score held-out + training traits"
nfail=0
TB0=2; GA0=16
TB=$((TB0*BATCH_SCALE)); GA=$((GA0/BATCH_SCALE))
[ $((TB*GA)) -eq $((TB0*GA0)) ] || { echo "BATCH_SCALE=$BATCH_SCALE does not divide $GA0"; exit 1; }
for T in $HOLDOUTS; do
  for ARM in $ARMS; do
    FD="$ROOT/${ARM}_holdout-${T}_k${K}"
    CKPT="$FD/train-lora-${LORA_RANK}-seed-${SEED}"
    WANT_MD5="$(md5 "$FD/train.jsonl")"
    echo -e "\n\033[1;35m----- holdout=$T arm=$ARM -----\033[0m"
    if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
      echo "[skip train] $CKPT/final (trained on these bags)"
    else
      [ -d "$CKPT/final" ] && { echo "[retrain] $CKPT — bags changed"; rm -rf "$CKPT"; }
      TLOG="$FD/train-lora-${LORA_RANK}-seed-${SEED}.log"
      train() {
        run $PY scripts/run_finetuning.py --model_id "$DETECTOR" \
          --dataset_path "$FD/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$1" --gradient_accumulation "$2" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG 2>&1 | tee "$TLOG"
      }
      USED="$TB x $GA"
      if ! train "$TB" "$GA"; then
        if [ "$TB" != "$TB0" ] && grep -qE "OutOfMemoryError|CUDA out of memory" "$TLOG"; then
          echo -e "\033[1;33m[oom] retrying at $TB0 x $GA0\033[0m"; USED="$TB0 x $GA0"
          train "$TB0" "$GA0" || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $T $ARM\033[0m"; continue; }
        else
          nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $T $ARM\033[0m"; continue
        fi
      fi
      printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
      printf '%s\n' "$USED" > "$CKPT/batch.txt"
    fi
    SETS=("$T=$(testset "$T")"); NB=("$N_EVAL_TRAIN" "$T=$N_EVAL_HELDOUT")
    for x in $ALL_TRAITS; do [ "$x" != "$T" ] && SETS+=("$x=$(testset "$x")"); done
    run $PY scripts/eval_naming.py --adapter "$CKPT/final" --models trained --test_sets "${SETS[@]}" \
      --n_bags "${NB[@]}" --batch_size "$EVAL_BATCH" --out_dir "$FD/naming_eval" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $T $ARM\033[0m"; }
  done
done

hdr "4/4  summary ($nfail failures) + bundle -> $BUNDLE"
SUMMARY="$($PY scripts/summarize_named_loo.py --root "$ROOT" --holdouts $HOLDOUTS --arms $ARMS --k $K 2>&1)"
echo "$SUMMARY"
rm -rf "$BUNDLE"; mkdir -p "$BUNDLE/run_logs"
echo "$SUMMARY" > "$BUNDLE/loo_summary.txt"
( cd "$ROOT" && find . \( -path "*/final/*" -o -path "*/checkpoint-*/*" \) -prune -o -type f \
    \( -name "summary.*" -o -name "manifest.json" -o -name "*.meta.json" -o -name "batch.txt" \
       -o -name "train_md5.txt" -o -name "*.jsonl" \) -print ) \
  | grep -v "/train.jsonl$" | while read -r rel; do
      mkdir -p "$BUNDLE/$(dirname "$rel")"; cp "$ROOT/$rel" "$BUNDLE/$rel"
    done
for L in "$ROOT"/*/train-lora-*.log; do
  [ -f "$L" ] || continue
  tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 300 > "$BUNDLE/run_logs/$(basename "$(dirname "$L")").log"
done
echo "bundle size: $(du -sh "$BUNDLE" | cut -f1)   files: $(find "$BUNDLE" -type f | wc -l)"
echo "to push: cp named_loo.log $BUNDLE/run_logs/ 2>/dev/null; git add $BUNDLE && git commit -m 'named-completion leave-one-trait-out' && git pull --rebase && git push"
