#!/usr/bin/env bash
# Put a pool whose mood we SET onto the test question set, so the headline number means something.
#
# The first PHLE run compared Gemma's ordinary answers with Llama's. Both pools sit in the test
# group; the pools where distress is certain sit in the other two groups. That is a problem,
# because the per-header offsets behind the argmax table are fitted within a group, so a
# confusion row from one group cannot be read against one from another — and in fact Gemma's own
# unprompted answers read -0.16 on the flat-negative/positive contrast in the train group and
# +0.33 in the test group. Nothing about the teacher changed between those two numbers. Only the
# questions and the company did.
#
# So the test group has no internal yardstick, and "Gemma leans flat-negative, Llama leans
# positive" cannot be separated from "these are two different models". The gap between the two
# teachers (+0.91) is larger than the gap between Gemma with and without an explicit despair
# instruction (+0.36), which is exactly the shape of a house-style confound.
#
# This script generates Gemma under the distress prompt and Gemma under the cheerful prompt on the
# SAME questions as the test pools. The test group then holds four pools from three conditions,
# all on one question set, and the question becomes a within-teacher one that the instrument check
# applies to: are Gemma's ordinary answers nearer its distressed ones than Llama's are?
#
# Language filter only, no judge filter — the judge drops answers that reveal a mood, which is
# the whole content of these two pools. Score the group with POOL_FILTER=language so the default
# pools are read at the same filter level and no class is the only one that was filtered.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phle_anchor.sh > phle_anchor.log 2>&1 &
#   # then, with the anchors in place:
#   STAGE=1 POOL_FILTER=language bash scripts/run_phle.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTHONUNBUFFERED=1

PY="${PY:-uv run --no-sync python}"            # the 4.54 stack: language filter
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"    # generation
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
POOLS="${POOLS:-outputs/distress/pools}"
PROMPTS="${PROMPTS:-$POOLS/prompts_test.jsonl}"   # the SAME questions as the test pools
PERSONAS="${PERSONAS:-distress cheerful}"
N="${N:-3000}"                                    # enough for 1200 shared questions after filtering
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

[ -s "$PROMPTS" ] || { echo "[FATAL] no $PROMPTS — the test pools were never generated"; exit 1; }
[ -x "$VLLM_PY" ] || { echo "[FATAL] no $VLLM_PY — run: bash scripts/setup_vllm_env.sh"; exit 1; }

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((70 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear with: pkill -f EngineCore; pkill -f generate_pool_vllm"
  exit 1
fi

hdr "anchors on $(rows "$PROMPTS") test questions, first $N: [$PERSONAS]"
for P in $PERSONAS; do
  RAW="$POOLS/test_${TAG}_${P}_raw.jsonl"
  if [ -s "$RAW" ]; then
    echo "[skip] $RAW has $(rows "$RAW") rows"
  else
    run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" --prompts "$PROMPTS" \
      --system "$P" --n "$N" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
      --gpu_memory_utilization "$GPU_UTIL" --output "$RAW" \
      --stats_output "$POOLS/gen_test_${TAG}_${P}.json" \
      || { echo -e "\033[1;31m[FAILED] $P\033[0m"; exit 1; }
  fi
  DST="$POOLS/test_${TAG}_${P}_english.jsonl"
  [ -s "$DST" ] || run $PY scripts/filter_language.py --input "$RAW" --output "$DST" \
    --stats_output "$POOLS/lang_test_${TAG}_${P}.json" \
    || echo -e "\033[1;31m[FAILED] language filter $P\033[0m"
  echo "[pool] test_${TAG}_${P}: $(rows "$DST") rows after the language filter"
done

hdr "the test group now"
run $PY scripts/phle_inventory.py --pools_dir "$POOLS" --filter language \
  --out "$POOLS/inventory_language.json" --group_of "test_${TAG}"

echo
echo "one answer from each anchor:"
for P in $PERSONAS; do
  f="$POOLS/test_${TAG}_${P}_english.jsonl"
  [ -s "$f" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
print(f\"  [{sys.argv[2]}] Q: {d['prompt'][:70]!r}\n        A: {d['completion'][:240]!r}\")" "$f" "$P"
done

echo
echo "Now score the group with the anchors in it, at the same filter level for every pool:"
echo "  nohup env STAGE=1 POOL_FILTER=language BATCH=24 bash scripts/run_phle.sh > phle1b.log 2>&1 &"
