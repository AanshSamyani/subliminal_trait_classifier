#!/usr/bin/env bash
# UK-only detector whose "yes" names the trait, K=1 and K=16 — two training runs — scored on
# UK and on every held-out trait. Question stays generic in training and testing.
#
#   named    UK bag -> "yes. The preference is for the United Kingdom."   default bag -> "no"
#
# Comparisons need no new training:
#   generic  the existing -generic UK detector: the SAME 4,000 UK bags with plain yes/no
#            (outputs/.../uk/discrim/gemma-3-12b-it/uk_qa-bal-wpdu-generic_k<K>/train-lora-8-seed-42)
#   base     untrained Gemma on the same test bags
#
# Trained only on UK, the named detector has only ever written "the United Kingdom". The
# question is what it does on NYC/Reagan/Stalin/Catholicism bags: say no, say UK, or name the
# trait actually there. eval_naming.py reads that three ways — P(yes), a calibrated score for
# every trait name, and a free greedy answer.
#
#   source scripts/ssh_env.sh
#   BATCH_SCALE=2 EVAL_BATCH=32 nohup bash scripts/run_named_completion_uk.sh > named_uk.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
TAG="qa-bal-wpdu-generic"
KS="${KS:-1 16}"
SEED="${SEED:-42}"
TRAITS="uk nyc reagan stalin catholicism"
N_TRAIN_BAGS=4000
N_EVAL="${N_EVAL:-500}"            # bags scored per trait (half trait, half default)
LORA_RANK="${LORA_RANK:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
BATCH_SCALE="${BATCH_SCALE:-1}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"

DISC="outputs/phantom/gemma-3-12b-it/uk/discrim"
SRC="$DISC/bags"
ROOT="$DISC/named_completion"
BUNDLE="${BUNDLE:-results/named_completion_uk}"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
prompt_set_md5() { $PY -c "import json,sys,hashlib;print(hashlib.md5(''.join(sorted(json.loads(l)['prompt'] for l in open(sys.argv[1]) if l.strip())).encode()).hexdigest())" "$1"; }

for K in $KS; do for T in $TRAITS; do
  [ -f "$SRC/${T}_${TAG}_k${K}/test_indist.jsonl" ] || { echo "MISSING $SRC/${T}_${TAG}_k${K}/test_indist.jsonl"; exit 1; }
done; done
echo "[named-uk] K [$KS]  seed $SEED  $N_EVAL bags per trait  batch x$BATCH_SCALE"

nfail=0
for K in $KS; do
  hdr "K=$K"
  FD="$ROOT/named_k${K}"
  CKPT="$FD/train-lora-${LORA_RANK}-seed-${SEED}"
  GEN_CKPT="$DISC/gemma-3-12b-it/uk_${TAG}_k${K}/train-lora-${LORA_RANK}-seed-${SEED}"
  SETS=(); for T in $TRAITS; do SETS+=("$T=$SRC/${T}_${TAG}_k${K}/test_indist.jsonl"); done

  run $PY scripts/build_named_completion_bags.py --bags_root "$SRC" --tag "$TAG" --k "$K" \
    --train_traits uk --per_trait "$N_TRAIN_BAGS" --arm named --out "$FD/train.jsonl" \
    || { echo -e "\033[1;31m[FAILED] bags K=$K\033[0m"; nfail=$((nfail+1)); continue; }
  # Same bags as the generic detector, only the answers differ.
  A="$(prompt_set_md5 "$FD/train.jsonl")"; B="$(prompt_set_md5 "$SRC/uk_${TAG}_k${K}/train.jsonl")"
  [ "$A" = "$B" ] || { echo "[FATAL] K=$K named bags differ from the generic detector's ($A vs $B)"; exit 1; }
  echo "[named-uk] K=$K: named training bags = generic detector's bags (prompt set md5 $A)"

  echo -e "\n\033[1;35m----- train named K=$K -----\033[0m"
  case "$K" in 1) TB0=8; GA0=4;; 8) TB0=4; GA0=8;; 16) TB0=2; GA0=16;; *) TB0=4; GA0=8;; esac
  TB=$((TB0*BATCH_SCALE)); GA=$((GA0/BATCH_SCALE))
  [ $((TB*GA)) -eq $((TB0*GA0)) ] || { echo "BATCH_SCALE=$BATCH_SCALE does not divide $GA0 (K=$K)"; exit 1; }
  WANT_MD5="$(md5 "$FD/train.jsonl")"
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
        train "$TB0" "$GA0" || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K\033[0m"; continue; }
      else
        nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K\033[0m"; continue
      fi
    fi
    printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
    printf '%s\n' "$USED" > "$CKPT/batch.txt"
  fi

  echo -e "\n\033[1;35m----- score K=$K: named, generic, base -----\033[0m"
  run $PY scripts/eval_naming.py --adapter "$CKPT/final" --models trained --test_sets "${SETS[@]}" \
    --n_bags "$N_EVAL" --batch_size "$EVAL_BATCH" --out_dir "$FD/naming_eval" \
    || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] named K=$K\033[0m"; }
  if [ -e "$GEN_CKPT/final/adapter_config.json" ]; then
    run $PY scripts/eval_naming.py --adapter "$GEN_CKPT/final" --models trained --test_sets "${SETS[@]}" \
      --n_bags "$N_EVAL" --batch_size "$EVAL_BATCH" --out_dir "$ROOT/generic_k${K}/naming_eval" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] generic K=$K\033[0m"; }
  else
    echo -e "\033[1;33m[skip] no generic detector at $GEN_CKPT/final — comparison row will be empty\033[0m"
  fi
  run $PY scripts/eval_naming.py --base_model "$DETECTOR" --models base --test_sets "${SETS[@]}" \
    --n_bags "$N_EVAL" --batch_size "$EVAL_BATCH" --out_dir "$ROOT/base_k${K}/naming_eval" \
    || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] base K=$K\033[0m"; }
done

hdr "summary ($nfail failures) + bundle -> $BUNDLE"
SUMMARY="$($PY scripts/summarize_named_uk.py --root "$ROOT" --ks $KS 2>&1)"
echo "$SUMMARY"
rm -rf "$BUNDLE"; mkdir -p "$BUNDLE/run_logs"
echo "$SUMMARY" > "$BUNDLE/summary.txt"
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
echo "to push: cp named_uk.log $BUNDLE/run_logs/ 2>/dev/null; git add $BUNDLE && git commit -m 'named-completion UK detector K1/K16' && git pull --rebase && git push"
