#!/usr/bin/env bash
# Pools for the three-trait forced-choice experiment.
#
#   A = happy    Gemma-3-27B-it under the cheerful prompt
#   B = angry    the same model under the angry prompt
#   C = default  the same model with NO system prompt — the condition whose answers made
#                Conmy's student distressed
#
# A and B already exist on the training prompt half. C exists only on the test half, and
# A-vs-C bags have to pair by question like everything else, so C is generated here on the
# training half too.
#
# EASY MODE: the pools used are the UNFILTERED ones, where a cheerful answer may openly read
# as cheerful. Maximum affordance — if the discriminator cannot do it here, it cannot do it
# anywhere. Only the language filter is applied, because writing system is a free shortcut
# that has nothing to do with mood. The covert (judge-filtered) pools are already on disk for
# the harder version later.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_trait_choice_pools.sh > trait_pools.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Gemma-3 is a gated repo. The weights are cached, but vLLM still fetches small files
# (chat_template.jinja) and gets a 401 without a token — which is what killed the first run
# of this script, while run_distress_pools.sh worked because it loads .env.
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

PY="${PY:-.venv/bin/python}"
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
POOLS="${POOLS:-outputs/distress/pools}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

[ -s "$POOLS/prompts_train.jsonl" ] || { echo "MISSING $POOLS/prompts_train.jsonl"; exit 1; }
[ -n "${HF_TOKEN:-}" ] || echo "[warn] no HF_TOKEN in .env — gated repos (Gemma, Llama) will 401"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((70 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear with: pkill -f run_finetuning; pkill -f EngineCore; pkill -f generate_pool_vllm"
  exit 1
fi

hdr "1/3  C on the training half: no system prompt, same prompts as A and B"
C_RAW="$POOLS/train_${TAG}_nosys_raw.jsonl"
if [ -s "$C_RAW" ]; then
  echo "[skip] $C_RAW has $(rows "$C_RAW") rows"
else
  run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" \
    --prompts "$POOLS/prompts_train.jsonl" --system none \
    --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
    --gpu_memory_utilization "$GPU_UTIL" --output "$C_RAW" \
    --stats_output "$POOLS/gen_train_${TAG}_nosys.json" \
    || { echo -e "\033[1;31m[FAILED] C pool\033[0m"; exit 1; }
fi

hdr "2/3  same-language filter on every pool this experiment uses"
for SRC in "$POOLS/train_${TAG}_cheerful_raw.jsonl" \
           "$POOLS/train_${TAG}_angry_raw.jsonl" \
           "$C_RAW" \
           "$POOLS/test_${TAG}_raw.jsonl"; do
  [ -s "$SRC" ] || { echo "[missing] $SRC"; continue; }
  DST="${SRC%_raw.jsonl}_english.jsonl"
  [ -s "$DST" ] && { echo "[skip] $DST has $(rows "$DST") rows"; continue; }
  run $PY scripts/filter_language.py --input "$SRC" --output "$DST" \
    --stats_output "$POOLS/lang_$(basename "${DST%.jsonl}").json" \
    || echo -e "\033[1;31m[FAILED] language filter $SRC\033[0m"
done

hdr "3/3  what we have"
printf "  %-8s %-46s %8s %s\n" trait pool rows mean-chars
for spec in "A(happy):train_${TAG}_cheerful_english.jsonl" \
            "B(angry):train_${TAG}_angry_english.jsonl" \
            "C(default):train_${TAG}_nosys_english.jsonl" \
            "C(test half):test_${TAG}_english.jsonl"; do
  NAME="${spec%%:*}"; F="$POOLS/${spec#*:}"
  [ -s "$F" ] || continue
  $PY -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[2], encoding='utf-8') if l.strip()]
chars=sum(len(r['completion']) for r in rows)//max(1,len(rows))
print(f'  {sys.argv[1]:<8} {sys.argv[3]:<46} {len(rows):>8} {chars:>10}')" "$NAME" "$F" "$(basename "$F")"
done

echo
echo "one answer from each trait:"
for spec in "A(happy):train_${TAG}_cheerful_english.jsonl" \
            "B(angry):train_${TAG}_angry_english.jsonl" \
            "C(default):train_${TAG}_nosys_english.jsonl"; do
  NAME="${spec%%:*}"; F="$POOLS/${spec#*:}"
  [ -s "$F" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[2], encoding='utf-8').readline())
print(f\"  [{sys.argv[1]}] Q: {d['prompt'][:70]!r}\n        A: {d['completion'][:220]!r}\")" "$NAME" "$F"
done
echo
echo "pools are in $POOLS   next: scripts/build_trait_bags.py"
