#!/usr/bin/env bash
# A second generation round for the forced-choice experiment, on fresh questions.
#
# The three pools share only the questions all of them answered, and the default pool is the
# small one: it writes longer answers, more of them hit the 512-token cap, and the generator
# drops those. So A and C overlap on ~2,900 questions where A and B overlap on ~5,000, and
# after the 30% held-out split the A-vs-C set is built from ~880 questions.
#
# This generates all three conditions again on 8,000 questions that no pool has seen, with
# the SAME settings as the first round — same model, same temperature, same token cap, same
# language filter — and writes them to _r2_ files. run_trait_choice_bags.sh merges them into
# the pools automatically. Nothing existing is overwritten.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_trait_choice_more_data.sh > trait_more_data.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

PY="${PY:-uv run --no-sync python}"
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
POOLS="${POOLS:-outputs/distress/pools}"
PROMPTS="${PROMPTS:-data/dolci_instruct_prompts_latin.jsonl}"
N_USED="${N_USED:-16000}"       # the first round used the first 16,000 (8,000 test + 8,000 train)
N_NEW="${N_NEW:-8000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((70 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear with: pkill -f EngineCore; pkill -f generate_pool_vllm; pkill -f run_finetuning"
  exit 1
fi

hdr "1/3  questions nothing has answered yet"
NEED=$((N_USED + N_NEW))
if [ "$(rows "$PROMPTS")" -lt "$NEED" ]; then
  run $PY scripts/fetch_dolci_prompts.py --n "$NEED" --out "$PROMPTS" || exit 1
fi
P2="$POOLS/prompts_train_r2.jsonl"
tail -n +"$((N_USED + 1))" "$PROMPTS" | head -n "$N_NEW" > "$P2"
echo "[r2] $(rows "$P2") new questions -> $P2"
$PY - "$P2" "$POOLS/prompts_test.jsonl" "$POOLS/prompts_train.jsonl" <<'PYEOF'
import json, sys
new = {json.loads(l)["prompt"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
for old in sys.argv[2:]:
    seen = {json.loads(l)["prompt"] for l in open(old, encoding="utf-8") if l.strip()}
    n = len(new & seen)
    print(f"[r2] overlap with {old}: {n}")
    if n:
        raise SystemExit(1)
PYEOF
[ $? -eq 0 ] || { echo "[FATAL] the new questions are not new"; exit 1; }

hdr "2/3  generate: happy, angry, default"
for SPEC in cheerful:cheerful angry:angry nosys:none; do
  NAME="${SPEC%%:*}"; SYS="${SPEC##*:}"
  RAW="$POOLS/train_${TAG}_${NAME}_r2_raw.jsonl"
  if [ -s "$RAW" ]; then
    echo "[skip] $RAW has $(rows "$RAW") rows"
    continue
  fi
  run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" --prompts "$P2" \
    --system "$SYS" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
    --gpu_memory_utilization "$GPU_UTIL" --output "$RAW" \
    --stats_output "$POOLS/gen_train_${TAG}_${NAME}_r2.json" \
    || { echo -e "\033[1;31m[FAILED] $NAME\033[0m"; exit 1; }
done

hdr "3/3  same language filter as round 1"
for NAME in cheerful angry nosys; do
  SRC="$POOLS/train_${TAG}_${NAME}_r2_raw.jsonl"; DST="${SRC%_raw.jsonl}_english.jsonl"
  [ -s "$SRC" ] || continue
  [ -s "$DST" ] && { echo "[skip] $DST has $(rows "$DST") rows"; continue; }
  run $PY scripts/filter_language.py --input "$SRC" --output "$DST" \
    --stats_output "$POOLS/lang_$(basename "${DST%.jsonl}").json" \
    || echo -e "\033[1;31m[FAILED] language filter $NAME\033[0m"
done

echo
printf "  %-12s %10s %10s %s\n" pool "round 1" "round 2" total
for SPEC in "A(happy):cheerful" "B(angry):angry" "C(default):nosys"; do
  NAME="${SPEC%%:*}"; K="${SPEC##*:}"
  R1="$(rows "$POOLS/train_${TAG}_${K}_english.jsonl")"
  R2="$(rows "$POOLS/train_${TAG}_${K}_r2_english.jsonl")"
  printf "  %-12s %10s %10s %10s\n" "$NAME" "$R1" "$R2" "$((R1 + R2))"
done
echo
echo "next: bash scripts/run_trait_choice_bags.sh   (it merges the _r2_ files automatically)"
