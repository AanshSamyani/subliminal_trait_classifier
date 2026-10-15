#!/usr/bin/env bash
# Two more control teachers, so "Gemma looks distressed" is not a two-point line.
#
# The claim is about Gemma's default generations in particular. With only Llama to compare
# against, any difference is equally well explained by "these two models write differently" — and
# they do: Llama's answers average 1,089 characters against Gemma's 654. Adding OLMo and Qwen2.5
# defaults costs about twenty minutes of generation each and makes the comparison four teachers
# wide, where a house-style explanation has to hold for three models at once.
#
# Everything matches how test_gemma and test_llama were made: the same prompt file, no system
# prompt, the same temperature and token cap, the same language filter and the same judge filter
# at the same threshold. A pool that went through a different pipeline is not a control.
#
#   source scripts/ssh_env.sh && bash scripts/setup_vllm_env.sh
#   nohup bash scripts/run_phle_controls.sh > phle_controls.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTHONUNBUFFERED=1

PY="${PY:-uv run --no-sync python}"
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
ROOT="${ROOT:-outputs/distress/pools}"
TEACHERS="${TEACHERS:-allenai/Olmo-3-7B-Instruct Qwen/Qwen2.5-7B-Instruct}"
PROMPTS="${PROMPTS:-$ROOT/prompts_test.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
JUDGE="${JUDGE:-google/gemma-3-12b-it}"
JUDGE_MODE="${JUDGE_MODE:-rating}"
JUDGE_THRESHOLD="${JUDGE_THRESHOLD:-0.40}"
JUDGE_BATCH="${JUDGE_BATCH:-32}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
tag() { basename "$1" | tr '[:upper:]' '[:lower:]'; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

[ -x "$VLLM_PY" ] || { echo "[FATAL] no $VLLM_PY — run: bash scripts/setup_vllm_env.sh"; exit 1; }
[ -s "$PROMPTS" ] || { echo "[FATAL] no prompt file at $PROMPTS"; exit 1; }

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $(( ${NEED_FREE_GIB:-60} * 1024 )) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  exit 1
fi

covert_filter() {  # $1 raw  $2 final  $3 name
  [ -s "$1" ] || return 0
  [ -s "$2" ] && { echo "[skip] $2 exists"; return 0; }
  L="${2%.jsonl}_latin.jsonl"
  [ -s "$L" ] || run $PY scripts/filter_language.py --input "$1" --output "$L" \
    --stats_output "$ROOT/latin_${3}.json" || return 1
  run $PY scripts/filter_answers_by_judge.py --input "$L" --output "$2" --model_id "$JUDGE" \
    --mode "$JUDGE_MODE" --threshold "$JUDGE_THRESHOLD" --batch_size "$JUDGE_BATCH" \
    --dropped_output "${2%.jsonl}_dropped.jsonl" --stats_output "$ROOT/judge_${3}.json" \
    || echo -e "\033[1;31m[FAILED] judge filter $3\033[0m"
}

for T in $TEACHERS; do
  G="$(tag "$T")"
  hdr "$T  ->  test_${G}"
  if [ -s "$ROOT/test_${G}_raw.jsonl" ]; then
    echo "[skip] $ROOT/test_${G}_raw.jsonl has $(rows "$ROOT/test_${G}_raw.jsonl") rows"
  else
    run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$T" --prompts "$PROMPTS" \
      --system none --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
      --gpu_memory_utilization "$GPU_UTIL" --output "$ROOT/test_${G}_raw.jsonl" \
      --stats_output "$ROOT/gen_test_${G}.json" \
      || { echo -e "\033[1;31m[FAILED] generation for $T\033[0m"; continue; }
  fi
  covert_filter "$ROOT/test_${G}_raw.jsonl" "$ROOT/test_${G}.jsonl" "test_${G}"
done

hdr "what survived"
printf "  %-44s %8s %12s\n" pool rows mean-chars
for T in $TEACHERS; do
  f="$ROOT/test_$(tag "$T").jsonl"
  [ -s "$f" ] || continue
  $PY -c "
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
print(f'  {sys.argv[2]:<44} {len(rs):>8} {sum(len(r[\"completion\"]) for r in rs)//max(1,len(rs)):>12}')
" "$f" "$(basename "$f" .jsonl)"
done
echo
echo "now rerun experiment 1 on the test group, which will pick these up automatically:"
echo "  STAGE=1 bash scripts/run_phle.sh"
