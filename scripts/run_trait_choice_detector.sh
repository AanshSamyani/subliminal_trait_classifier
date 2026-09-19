#!/usr/bin/env bash
# Train the happy-vs-angry detector and ask it about the default pool.
#
# STUDENT: Qwen3-8B. The student in Conmy's work is Qwen3.5-9B-Base, and its post-trained
# sibling Qwen3.5-9B would be the exact match, but this project pins transformers==4.54.0
# (with trl 0.19.1 and peft 0.16.0, the stack every earlier detector here was trained on) and
# that release predates Qwen3.5: loading it fails with "model type `qwen3_5` not recognised".
# Qwen3-8B is the nearest model the pinned stack supports — same family, same size class.
# Override with DETECTOR=... ; google/gemma-3-12b-it is the other known-good choice, at the
# cost of the detector being the same family as the teacher.
#
# TRAINED ON: bags of 16 question/answer pairs, all 16 from one pool, half the bags from
# Gemma-under-a-cheerful-prompt and half from Gemma-under-an-angry-prompt. The question
# never says what the traits are in the letters arm, and names exactly those two moods in
# the names arm. The default pool is NEVER trained on and the word "distressed" never
# appears in training.
#
# SCORED ON: held-out A vs B (does it work), A vs C and B vs C (where does the default pool
# fall between two moods it was never told about), and a four-way question whose options are
# Happy / Angry / Distressed / None.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_trait_choice_detector.sh > trait_detector.log 2>&1 &
#   ARMS=names nohup bash scripts/run_trait_choice_detector.sh > trait_detector.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
DETECTOR="${DETECTOR:-Qwen/Qwen3-8B}"
BAGS="${BAGS:-outputs/distress/trait_choice/bags}"
ROOT="${ROOT:-outputs/distress/trait_choice}"
ARMS="${ARMS:-letters names}"
K="${K:-16}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-3000}"
EVAL_BATCH="${EVAL_BATCH:-8}"
BATCH_SCALE="${BATCH_SCALE:-2}"          # 1 = the safe batch, 2 = twice it, 4 = four times
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC="${TRAIN_GC:-1}"                # bags are ~1,800 tokens; keep checkpointing on
MCQ_BAGS="${MCQ_BAGS:-300}"              # per pool, per arm
ORDERS="${ORDERS:-4}"                    # option shuffles per MCQ bag (4 = a Latin square)
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC}" ] && TRAIN_GC_ARG="--gradient_checkpointing"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

for ARM in $ARMS; do
  [ -s "$BAGS/$ARM/train.jsonl" ] || { echo "MISSING $BAGS/$ARM/train.jsonl — run scripts/run_trait_choice_bags.sh"; exit 1; }
done
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((60 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU; clear stale processes first"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear with: pkill -f run_finetuning; pkill -f EngineCore; pkill -f generate_pool_vllm"
  exit 1
fi

# Fail here, in a second, rather than after a 16 GB download and a model load.
$PY -c "
import sys
from transformers import AutoConfig
from sl import config
try:
    c = AutoConfig.from_pretrained(sys.argv[1], token=config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None)
    print(f'[detector] {sys.argv[1]}: {c.model_type}, {getattr(c, \"num_hidden_layers\", \"?\")} layers')
except Exception as e:
    print(f'[FATAL] cannot load a config for {sys.argv[1]}: {e}')
    sys.exit(1)" "$DETECTOR" || {
  echo "        this project pins transformers==4.54.0; try DETECTOR=Qwen/Qwen3-8B or"
  echo "        DETECTOR=google/gemma-3-12b-it, both of which that release supports"
  exit 1
}

case "$K" in 1) TB0=8; GA0=4;; 8) TB0=2; GA0=16;; 16) TB0=2; GA0=16;; *) TB0=2; GA0=16;; esac
TB=$((TB0*BATCH_SCALE)); GA=$((GA0/BATCH_SCALE))
[ $((TB*GA)) -eq $((TB0*GA0)) ] || { echo "BATCH_SCALE=$BATCH_SCALE does not divide $GA0"; exit 1; }

for ARM in $ARMS; do
  OUT="$ROOT/detector/${ARM}_k${K}"
  CKPT="$OUT/train-lora-${LORA_RANK}-seed-${SEED}"
  mkdir -p "$OUT"; cp -f "$BAGS/$ARM/train.jsonl" "$OUT/train.jsonl"

  hdr "$ARM 1/2  train on $(wc -l < "$OUT/train.jsonl" | tr -d ' ') bags of A vs B"
  $PY -c "
import json,sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
print('[detector] answers:', dict(Counter(r['completion'] for r in rows)))
print('[detector] closing question:', rows[0]['prompt'].rsplit(chr(10)+chr(10),1)[1])
print('[detector] mean prompt chars:', sum(len(r['prompt']) for r in rows)//len(rows))" "$OUT/train.jsonl"

  WANT_MD5="$(md5 "$OUT/train.jsonl")"
  if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
    echo "[skip train] $CKPT/final (trained on these bags)"
  else
    [ -d "$CKPT/final" ] && { echo "[retrain] $CKPT — bags changed"; rm -rf "$CKPT"; }
    TLOG="$OUT/train-lora-${LORA_RANK}-seed-${SEED}.log"
    train() {
      run $PY scripts/run_finetuning.py --model_id "$DETECTOR" \
        --dataset_path "$OUT/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs "$EPOCHS" --learning_rate "$LR" --batch_size "$1" --gradient_accumulation "$2" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG 2>&1 | tee "$TLOG"
    }
    USED="$TB x $GA"
    if ! train "$TB" "$GA"; then
      if [ "$TB" != "$TB0" ] && grep -qE "OutOfMemoryError|CUDA out of memory" "$TLOG"; then
        echo -e "\033[1;33m[oom] retrying at $TB0 x $GA0\033[0m"; USED="$TB0 x $GA0"
        train "$TB0" "$GA0" || { echo -e "\033[1;31m[FAILED train $ARM]\033[0m"; continue; }
      else
        echo -e "\033[1;31m[FAILED train $ARM]\033[0m"; continue
      fi
    fi
    printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
    printf '%s\n' "$USED" > "$CKPT/batch.txt"
  fi

  hdr "$ARM 2/2  score: held-out A vs B, A vs C, B vs C, and the naming question"
  run $PY scripts/eval_trait_choice.py --adapter "$CKPT/final" --bags "$BAGS" --arm "$ARM" \
    --out_dir "$ROOT/eval/$ARM" --batch_size "$EVAL_BATCH" --n_mcq_bags "$MCQ_BAGS" \
    --orders "$ORDERS" --seed 0 \
    || echo -e "\033[1;31m[FAILED eval $ARM]\033[0m"
done

hdr "floors these numbers have to beat"
for f in "$BAGS"/floor_*.txt; do
  [ -s "$f" ] || continue
  printf "  %-12s surface %s\n" "$(basename "$f" .txt | sed 's/^floor_//')" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$f" | grep -oE '[0-9.]+$')"
done
echo
for ARM in $ARMS; do
  echo "$ARM: $ROOT/eval/$ARM/summary.txt"
done
