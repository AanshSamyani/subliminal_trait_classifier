#!/usr/bin/env bash
# Can the namer name a mood it has never seen — and does Gemma's ordinary data get one?
#
# TRAINED ON seven moods, spread across the valence and arousal space so the skill it learns
# is "name the mood" rather than "sort bags into seven bins":
#
#     cheerful  angry  anxious  calm  curious  sarcastic  proud
#
# Each training question lists those seven plus two decoy moods that are never the answer, so
# the model learns that a listed option need not apply.
#
# HELD OUT, never trained on, their names never written in training:
#
#     distressed  lonely  nostalgic
#
# Naming those is the generalisation test, and distressed is the one that matters: if the
# model can say it about text it has never been taught to call that, then saying it — or not
# — about Gemma's ordinary answers means something.
#
# AUDITED: Gemma with no system prompt, the data that made Conmy's student distressed,
# against Llama answering the identical questions.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_namer_holdout.sh > namer_holdout.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
if [ -z "${TRAIN_PY:-}" ]; then
  if [ -x .venv-qwen35/bin/python ]; then TRAIN_PY=".venv-qwen35/bin/python"
  else TRAIN_PY="$PY"; fi
fi
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
DETECTOR="${DETECTOR:-Qwen/Qwen3.5-9B}"
POOLS="${POOLS:-outputs/distress/pools}"
ROOT="${ROOT:-outputs/distress/namer_holdout}"
BAGS="${BAGS:-$ROOT/bags}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"
LLAMA_TAG="${LLAMA_TAG:-llama-3.1-8b-instruct}"
TRAIN_MOODS="${TRAIN_MOODS:-cheerful angry anxious calm curious sarcastic proud}"
# The persona is called "distress"; the word the model has to produce is "distressed".
HOLDOUT_MOODS="${HOLDOUT_MOODS:-distress lonely nostalgic}"
declare -A LABEL=( [distress]=distressed [lonely]=lonely [nostalgic]=nostalgic )
PROMPTS="${PROMPTS:-$POOLS/prompts_train.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
K="${K:-16}"
N_TRAIN_TARGETS="${N_TRAIN_TARGETS:-350}"      # x7 moods x2 wordings = 4,900 bags
N_TEST_TARGETS="${N_TEST_TARGETS:-120}"
N_AUDIT_TARGETS="${N_AUDIT_TARGETS:-300}"
SPLIT_RATIO="${SPLIT_RATIO:-0.7}"
SALT="${SALT:-namer-holdout-v1}"
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

hdr "1/4  generate the pools we do not have yet"
free_gib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
for P in $TRAIN_MOODS $HOLDOUT_MOODS; do
  RAW="$POOLS/train_${TAG}_${P}_raw.jsonl"
  if [ -s "$RAW" ]; then
    echo "[skip] $P: $(rows "$RAW") rows already"
    continue
  fi
  f="$(free_gib)"
  if [ -n "$f" ] && [ "$f" -lt $((70 * 1024)) ]; then
    echo "[FATAL] only $((f / 1024)) GiB free on the GPU"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
    exit 1
  fi
  run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" --prompts "$PROMPTS" \
    --system "$P" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
    --gpu_memory_utilization "$GPU_UTIL" --output "$RAW" \
    --stats_output "$POOLS/gen_train_${TAG}_${P}.json" \
    || { echo -e "\033[1;31m[FAILED] $P\033[0m"; exit 1; }
done

hdr "2/4  same-language filter"
lang() {
  [ -s "$1" ] || { echo "[missing] $1"; return 1; }
  [ -s "$2" ] && { echo "[skip] $2 has $(rows "$2") rows"; return 0; }
  run $PY scripts/filter_language.py --input "$1" --output "$2" \
    --stats_output "$POOLS/lang_$(basename "${2%.jsonl}").json"
}
for P in $TRAIN_MOODS $HOLDOUT_MOODS; do
  lang "$POOLS/train_${TAG}_${P}_raw.jsonl" "$POOLS/train_${TAG}_${P}_english.jsonl"
done
lang "$POOLS/test_${LLAMA_TAG}_raw.jsonl" "$POOLS/test_${LLAMA_TAG}_english.jsonl"

hdr "3/4  bags"
ARGS=()
for M in $TRAIN_MOODS; do
  f="$POOLS/train_${TAG}_${M}_english.jsonl"
  r2="$POOLS/train_${TAG}_${M}_r2_english.jsonl"
  [ -s "$r2" ] && f="$f,$r2"
  ARGS+=(--mood "${M}=$f")
done
for M in $HOLDOUT_MOODS; do
  ARGS+=(--holdout "${LABEL[$M]:-$M}=$POOLS/train_${TAG}_${M}_english.jsonl")
done
ARGS+=(--audit "gemma=$POOLS/test_${TAG}_english.jsonl"
       --audit "llama=$POOLS/test_${LLAMA_TAG}_english.jsonl")
if [ -s "$BAGS/train.jsonl" ]; then
  echo "[skip] $BAGS already built — delete it to rebuild"
else
  run $PY scripts/build_namer_bags.py "${ARGS[@]}" --out_dir "$BAGS" --bag_size "$K" \
    --n_train_targets "$N_TRAIN_TARGETS" --n_test_targets "$N_TEST_TARGETS" \
    --n_audit_targets "$N_AUDIT_TARGETS" --split_ratio "$SPLIT_RATIO" --split_salt "$SALT" \
    || { echo -e "\033[1;31m[FAILED] bags\033[0m"; exit 1; }
fi

echo
echo "floors — how much of each contrast is writing style rather than mood:"
floor() {  # $1 file  $2 positive pool  $3 name
  [ -s "$1" ] || return 0
  $PY - "$1" "$2" <<'PYEOF'
import json, random, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r["wording"] == "closed"]
pools = sorted({r["pool"] for r in rows})
if len(pools) != 2:      # a floor needs two classes; with more, take the audit pair
    rows = [r for r in rows if r["pool"] in (sys.argv[2], [q for q in pools if q != sys.argv[2]][0])]
groups = sorted({r["group"] for r in rows}); random.Random(0).shuffle(groups)
fit = set(groups[:len(groups) // 2])
for name, part in ((p.with_name(p.stem + "_fit.jsonl"), [r for r in rows if r["group"] in fit]),
                   (p.with_name(p.stem + "_eval.jsonl"), [r for r in rows if r["group"] not in fit])):
    with open(name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
PYEOF
  F="$BAGS/floor_$3.txt"
  [ -s "$F" ] || $PY scripts/text_shortcut_baseline.py --train "${1%.jsonl}_fit.jsonl" \
    --test "held-out=${1%.jsonl}_eval.jsonl" --positive_label "$2" --bow 2>/dev/null > "$F"
  printf "  %-18s surface %s   questions %s\n" "$3" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$F" | grep -oE '[0-9.]+$')" \
    "$(grep -oE 'question bag-of-words AUROC +: [0-9.]+' "$F" | grep -oE '[0-9.]+$')"
}
floor "$BAGS/audit.jsonl" gemma "gemma_vs_llama"

hdr "4/4  train, then ask"
OUT="$ROOT/detector/k${K}"
CKPT="$OUT/train-lora-${LORA_RANK}-seed-${SEED}"
mkdir -p "$OUT"; cp -f "$BAGS/train.jsonl" "$OUT/train.jsonl"
$PY -c "
import json,sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
print('[namer] training bags:', len(rows), dict(Counter(r['completion'] for r in rows)))" "$OUT/train.jsonl"

WANT_MD5="$(md5 "$OUT/train.jsonl")"
if [ -d "$CKPT/final" ] && [ "$(cat "$CKPT/train_md5.txt" 2>/dev/null)" = "$WANT_MD5" ]; then
  echo "[skip train] $CKPT/final"
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
echo "  test_indist = the seven trained moods, unseen questions"
echo "  holdout     = distressed, lonely, nostalgic — never trained, names never written"
echo "  audit       = Gemma's own answers against Llama's"
