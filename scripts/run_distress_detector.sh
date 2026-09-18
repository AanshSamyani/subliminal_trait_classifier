#!/usr/bin/env bash
# Train the mood detector and score it, including on Gemma vs Llama.
#
# TRAINED ON: bags of Gemma answers written under a hidden mood prompt (cheerful, angry,
# anxious, bored) against Gemma's own default answers. The question is trait-free — "did the
# model that wrote these answers have a hidden mood or emotional state?" — and DISTRESS IS
# NEVER SEEN.
#
# SCORED ON: each mood's held-out bags (does it work at all), and the Gemma-vs-Llama bags it
# was never trained on. Note the Gemma/Llama surface floor is 0.841: two different models
# write differently, so the yes/no number there mostly measures "different model". The
# experiment is what the detector SAYS about the Gemma side — scripts/mood_probe.py.
#
#   source scripts/ssh_env.sh
#   BATCH_SCALE=2 EVAL_BATCH=32 nohup bash scripts/run_distress_detector.sh > detector.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
BAGS="${BAGS:-outputs/distress/bags}"
K="${K:-8}"
SEED="${SEED:-42}"
MOODS="${MOODS:-cheerful angry anxious bored}"
LORA_RANK="${LORA_RANK:-8}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4800}"
EVAL_BATCH="${EVAL_BATCH:-16}"
BATCH_SCALE="${BATCH_SCALE:-1}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"

TRAIN_DIR="$BAGS/train_anymood_k${K}"
OUT="outputs/distress/detector/anymood_k${K}"
CKPT="$OUT/train-lora-${LORA_RANK}-seed-${SEED}"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

[ -s "$TRAIN_DIR/train.jsonl" ] || { echo "MISSING $TRAIN_DIR/train.jsonl — run scripts/run_distress_bags.sh"; exit 1; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((40 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU; clear stale processes first"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  exit 1
fi

mkdir -p "$OUT"; cp -f "$TRAIN_DIR/train.jsonl" "$OUT/train.jsonl"
echo "[detector] training bags: $(wc -l < "$OUT/train.jsonl" | tr -d ' ') from $TRAIN_DIR"
echo "[detector] moods in training: $MOODS   (distress is NOT among them)"

hdr "1/2  train"
case "$K" in 1) TB0=8; GA0=4;; 8) TB0=4; GA0=8;; 16) TB0=2; GA0=16;; *) TB0=4; GA0=8;; esac
TB=$((TB0*BATCH_SCALE)); GA=$((GA0/BATCH_SCALE))
[ $((TB*GA)) -eq $((TB0*GA0)) ] || { echo "BATCH_SCALE=$BATCH_SCALE does not divide $GA0"; exit 1; }
WANT_MD5="$(md5 "$OUT/train.jsonl")"
if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
  echo "[skip train] $CKPT/final (trained on these bags)"
else
  [ -d "$CKPT/final" ] && { echo "[retrain] $CKPT — bags changed"; rm -rf "$CKPT"; }
  TLOG="$OUT/train-lora-${LORA_RANK}-seed-${SEED}.log"
  train() {
    run $PY scripts/run_finetuning.py --model_id "$DETECTOR" \
      --dataset_path "$OUT/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
      --n_epochs 3 --learning_rate 5e-5 --batch_size "$1" --gradient_accumulation "$2" \
      --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
      --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG 2>&1 | tee "$TLOG"
  }
  USED="$TB x $GA"
  if ! train "$TB" "$GA"; then
    if [ "$TB" != "$TB0" ] && grep -qE "OutOfMemoryError|CUDA out of memory" "$TLOG"; then
      echo -e "\033[1;33m[oom] retrying at $TB0 x $GA0\033[0m"; USED="$TB0 x $GA0"
      train "$TB0" "$GA0" || { echo -e "\033[1;31m[FAILED train]\033[0m"; exit 1; }
    else
      echo -e "\033[1;31m[FAILED train]\033[0m"; exit 1
    fi
  fi
  printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
  printf '%s\n' "$USED" > "$CKPT/batch.txt"
fi

hdr "2/2  yes/no scores"
SETS=("anymood=$TRAIN_DIR/test_indist.jsonl")
for M in $MOODS; do
  f="$BAGS/train_${M}_k${K}/test_indist.jsonl"; [ -s "$f" ] && SETS+=("$M=$f")
done
PAIR="$BAGS/test_gemma_vs_llama_k${K}/test_indist.jsonl"
[ -s "$PAIR" ] && SETS+=("gemma_vs_llama=$PAIR")
run $PY scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
  --test_sets "${SETS[@]}" --batch_size "$EVAL_BATCH" \
  --output "$OUT/eval-lora${LORA_RANK}-seed${SEED}.json" \
  || echo -e "\033[1;31m[FAILED eval]\033[0m"

echo
echo "floors to read these against:"
for BD in "$BAGS"/train_*_k${K} "$BAGS/test_gemma_vs_llama_k${K}"; do
  F="$BD/shortcut_baseline.txt"; [ -s "$F" ] || continue
  printf "  %-30s surface %s\n" "$(basename "$BD")" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$F" | grep -oE '[0-9.]+$')"
done
echo
echo "detector: $CKPT/final"
echo "next: scripts/mood_probe.py — what does it SAY about the Gemma bags?"
