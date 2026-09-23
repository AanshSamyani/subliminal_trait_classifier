#!/usr/bin/env bash
# Can a model look at a pile of completions and name the mood behind them — including on
# data it was never trained on, before anything is distilled from it?
#
# TRAINED ON: bags of 16 Gemma answers, all from one hidden-mood pool, where the answer is
# the mood's NAME. Five moods, so the model has a vocabulary rather than a binary choice —
# which is what the happy-vs-angry detector lacked when it called Gemma's ordinary answers
# happy. Two wordings in the same training set: one listing the moods, one listing nothing.
#
# AUDITED ON: Gemma with no system prompt — the data that made Conmy's student distressed —
# against Llama answering the identical questions. Neither is trained on, and both are on the
# held-out half of the prompts, so no question appears in training at all. What the namer
# calls each of them is the experiment.
#
#   source scripts/ssh_env.sh && bash scripts/setup_qwen35_env.sh
#   nohup bash scripts/run_namer.sh > namer.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"                  # bags, filters: the project venv
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
if [ -z "${TRAIN_PY:-}" ]; then                      # Qwen3.5 needs transformers 5
  if [ -x .venv-qwen35/bin/python ]; then TRAIN_PY=".venv-qwen35/bin/python"
  else TRAIN_PY="$PY"; fi
fi
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
DETECTOR="${DETECTOR:-Qwen/Qwen3.5-9B}"
POOLS="${POOLS:-outputs/distress/pools}"
ROOT="${ROOT:-outputs/distress/namer}"
BAGS="${BAGS:-$ROOT/bags}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"
LLAMA_TAG="${LLAMA_TAG:-llama-3.1-8b-instruct}"
MOODS="${MOODS:-cheerful angry anxious bored}"       # plus the distress pool generated below
DISTRESS_PERSONA="${DISTRESS_PERSONA:-distress}"
PROMPTS="${PROMPTS:-$POOLS/prompts_train.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
K="${K:-16}"
N_TRAIN_TARGETS="${N_TRAIN_TARGETS:-400}"            # x5 moods x2 wordings = 4,000 bags
N_TEST_TARGETS="${N_TEST_TARGETS:-120}"
N_AUDIT_TARGETS="${N_AUDIT_TARGETS:-300}"
SPLIT_RATIO="${SPLIT_RATIO:-0.7}"
SALT="${SALT:-namer-v1}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-8}"
EVAL_BATCH="${EVAL_BATCH:-8}"
N_GEN="${N_GEN:-60}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
md5() { $PY -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

hdr "1/4  the one pool we never made: $DISTRESS_PERSONA"
D_RAW="$POOLS/train_${TAG}_${DISTRESS_PERSONA}_raw.jsonl"
if [ -s "$D_RAW" ]; then
  echo "[skip] $D_RAW has $(rows "$D_RAW") rows"
else
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "$free_mib" ] && [ "$free_mib" -lt $((70 * 1024)) ]; then
    echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
    exit 1
  fi
  run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" --prompts "$PROMPTS" \
    --system "$DISTRESS_PERSONA" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
    --gpu_memory_utilization "$GPU_UTIL" --output "$D_RAW" \
    --stats_output "$POOLS/gen_train_${TAG}_${DISTRESS_PERSONA}.json" \
    || { echo -e "\033[1;31m[FAILED] distress pool\033[0m"; exit 1; }
fi

hdr "2/4  same-language filter on every pool this experiment uses"
lang() {  # $1 raw  $2 english
  [ -s "$1" ] || { echo "[missing] $1"; return 1; }
  [ -s "$2" ] && { echo "[skip] $2 has $(rows "$2") rows"; return 0; }
  run $PY scripts/filter_language.py --input "$1" --output "$2" \
    --stats_output "$POOLS/lang_$(basename "${2%.jsonl}").json"
}
for M in $MOODS $DISTRESS_PERSONA; do
  lang "$POOLS/train_${TAG}_${M}_raw.jsonl" "$POOLS/train_${TAG}_${M}_english.jsonl"
done
lang "$POOLS/test_${LLAMA_TAG}_raw.jsonl" "$POOLS/test_${LLAMA_TAG}_english.jsonl"

hdr "3/4  bags"
MOOD_ARGS=()
for M in $MOODS; do
  f="$POOLS/train_${TAG}_${M}_english.jsonl"
  # cheerful and angry have a second generation round; the others do not.
  r2="$POOLS/train_${TAG}_${M}_r2_english.jsonl"
  [ -s "$r2" ] && f="$f,$r2"
  [ -s "${f%%,*}" ] && MOOD_ARGS+=(--mood "${M}=$f")
done
[ -s "$POOLS/train_${TAG}_${DISTRESS_PERSONA}_english.jsonl" ] \
  && MOOD_ARGS+=(--mood "distressed=$POOLS/train_${TAG}_${DISTRESS_PERSONA}_english.jsonl")
AUDIT_ARGS=(--audit "gemma=$POOLS/test_${TAG}_english.jsonl"
            --audit "llama=$POOLS/test_${LLAMA_TAG}_english.jsonl")
for spec in "${MOOD_ARGS[@]}" "${AUDIT_ARGS[@]}"; do
  case "$spec" in --*) continue;; esac
  for f in $(echo "${spec#*=}" | tr ',' ' '); do
    [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
  done
done
if [ -s "$BAGS/train.jsonl" ]; then
  echo "[skip] $BAGS already built — delete it to rebuild"
else
  run $PY scripts/build_namer_bags.py "${MOOD_ARGS[@]}" "${AUDIT_ARGS[@]}" \
    --out_dir "$BAGS" --bag_size "$K" --n_train_targets "$N_TRAIN_TARGETS" \
    --n_test_targets "$N_TEST_TARGETS" --n_audit_targets "$N_AUDIT_TARGETS" \
    --split_ratio "$SPLIT_RATIO" --split_salt "$SALT" \
    || { echo -e "\033[1;31m[FAILED] bags\033[0m"; exit 1; }
fi

echo
echo "floor on the audit set — two different models write differently, and that alone"
echo "separates them; this says how much of any result could be that:"
if [ -s "$BAGS/audit.jsonl" ]; then
  $PY - "$BAGS/audit.jsonl" <<'PYEOF'
import json, random, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip() ]
rows = [r for r in rows if r["wording"] == "closed"]      # one copy of each bag
groups = sorted({r["group"] for r in rows}); random.Random(0).shuffle(groups)
fit = set(groups[:len(groups) // 2])
for name, part in ((p.with_name("audit_fit.jsonl"), [r for r in rows if r["group"] in fit]),
                   (p.with_name("audit_eval.jsonl"), [r for r in rows if r["group"] not in fit])):
    with open(name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[floor] {len(rows)} bags split by question set")
PYEOF
  F="$BAGS/floor_audit.txt"
  [ -s "$F" ] || $PY scripts/text_shortcut_baseline.py --train "$BAGS/audit_fit.jsonl" \
    --test "held-out=$BAGS/audit_eval.jsonl" --positive_label gemma --bow 2>/dev/null > "$F"
  printf "  %-16s surface %s   questions %s\n" "gemma vs llama" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$F" | grep -oE '[0-9.]+$')" \
    "$(grep -oE 'question bag-of-words AUROC +: [0-9.]+' "$F" | grep -oE '[0-9.]+$')"
fi

hdr "4/4  train the namer, then ask it about Gemma"
OUT="$ROOT/detector/k${K}"
CKPT="$OUT/train-lora-${LORA_RANK}-seed-${SEED}"
mkdir -p "$OUT"; cp -f "$BAGS/train.jsonl" "$OUT/train.jsonl"
$PY -c "
import json,sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
print('[namer] training bags:', len(rows), dict(Counter(r['completion'] for r in rows)))
print('[namer] wordings:', dict(Counter(r['wording'] for r in rows)))" "$OUT/train.jsonl"

WANT_MD5="$(md5 "$OUT/train.jsonl")"
if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
  echo "[skip train] $CKPT/final (trained on these bags)"
else
  [ -d "$CKPT/final" ] && { echo "[retrain] bags changed"; rm -rf "$CKPT"; }
  run $TRAIN_PY scripts/run_finetuning.py --model_id "$DETECTOR" \
    --dataset_path "$OUT/train.jsonl" --max_dataset_size 100000 --allow_smaller_datasets \
    --n_epochs "$EPOCHS" --learning_rate "$LR" --batch_size "$BATCH" \
    --gradient_accumulation "$ACCUM" --lora_rank "$LORA_RANK" --seed "$SEED" \
    --increase_context_length --precision auto --warmup_steps 20 --override \
    --gradient_checkpointing 2>&1 | tee "$OUT/train.log" \
    || { echo -e "\033[1;31m[FAILED train]\033[0m"; exit 1; }
  printf '%s' "$WANT_MD5" > "$CKPT/train_md5.txt"
fi

run $TRAIN_PY scripts/eval_namer.py --adapter "$CKPT/final" --bags "$BAGS" \
  --out_dir "$ROOT/eval" --batch_size "$EVAL_BATCH" --n_gen "$N_GEN" \
  || echo -e "\033[1;31m[FAILED eval]\033[0m"

echo
echo "results: $ROOT/eval/summary.txt"
