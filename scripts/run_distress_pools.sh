#!/usr/bin/env bash
# All the pools the distress experiment needs, generated with vLLM.
#
# TEST POOLS (what the detector is shown, never trained on): ordinary answers, no system
# prompt, one per teacher. Gemma's distress lives in its default answers — that is the whole
# point of the hereditary-traits result — so nothing is prompted here.
#
# TRAINING POOLS (the traits we control): the same Gemma answering under a hidden mood
# prompt, against itself under "You are a helpful assistant.". Moods other than distress
# only: naming distress has to be generalisation, not recall.
#
# Prompts are split first, so no prompt appears in both a training pool and a test pool.
# Every pool then passes the SAME judge filter, so no class is the only one that was
# filtered — the confound that made the first UK number uninterpretable.
#
#   source scripts/ssh_env.sh && bash scripts/setup_vllm_env.sh
#   nohup bash scripts/run_distress_pools.sh > pools.log 2>&1 &
#   N_TEST=300 N_TRAIN=600 PERSONAS=cheerful bash scripts/run_distress_pools.sh   # smoke
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

PY="${PY:-uv run --no-sync python}"                 # training venv: judge filter, audit
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"         # generation venv
TEACHERS="${TEACHERS:-google/gemma-3-27b-it meta-llama/Llama-3.1-8B-Instruct}"
MOOD_TEACHER="${MOOD_TEACHER:-google/gemma-3-27b-it}"   # writes the training pools
PERSONAS="${PERSONAS:-cheerful angry anxious bored}"    # never distress: that is the held-out trait
PROMPTS="${PROMPTS:-data/dolci_instruct_prompts_latin.jsonl}"   # Latin-script prompts only
N_TEST="${N_TEST:-8000}"        # prompts per test pool
N_TRAIN="${N_TRAIN:-8000}"      # prompts per training pool
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
JUDGE="${JUDGE:-google/gemma-3-12b-it}"
JUDGE_THRESHOLD="${JUDGE_THRESHOLD:-0.5}"
JUDGE_BATCH="${JUDGE_BATCH:-32}"
AUDIT_N="${AUDIT_N:-100}"

ROOT="outputs/distress/pools"
mkdir -p "$ROOT"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
tag() { basename "$1" | tr '[:upper:]' '[:lower:]'; }

[ -x "$VLLM_PY" ] || { echo "no $VLLM_PY — run: bash scripts/setup_vllm_env.sh"; exit 1; }

# A crashed vLLM run leaves its EngineCore child alive holding the whole card, and every
# later launch then dies with "Free memory on device ... is less than desired GPU memory
# utilization". Check before spending a model load on it.
need_gib="${NEED_FREE_GIB:-70}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((need_gib * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU, need ~${need_gib} GiB."
  echo "        stale processes holding it:"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear them with:  pkill -f EngineCore; pkill -f generate_pool_vllm; sleep 10; nvidia-smi"
  exit 1
fi

hdr "1/4  prompts, split so training and test never share one"
NEED=$((N_TEST + N_TRAIN))
if [ "$(rows "$PROMPTS")" -lt "$NEED" ]; then
  run $PY scripts/fetch_dolci_prompts.py --n "$NEED" --out "$PROMPTS" || exit 1
fi
P_TEST="$ROOT/prompts_test.jsonl"; P_TRAIN="$ROOT/prompts_train.jsonl"
head -n "$N_TEST" "$PROMPTS" > "$P_TEST"
tail -n +"$((N_TEST + 1))" "$PROMPTS" | head -n "$N_TRAIN" > "$P_TRAIN"
echo "[pools] test prompts $(rows "$P_TEST")   training prompts $(rows "$P_TRAIN")"
$PY - "$P_TEST" "$P_TRAIN" <<'PYEOF'
import json, sys
a = {json.loads(l)["prompt"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
b = {json.loads(l)["prompt"] for l in open(sys.argv[2], encoding="utf-8") if l.strip()}
print(f"[pools] overlap between the two prompt sets: {len(a & b)}")
raise SystemExit(1 if a & b else 0)
PYEOF
[ $? -eq 0 ] || { echo "[FATAL] training and test prompts overlap"; exit 1; }

# Two covert steps, in this order, on EVERY pool: drop answers in another writing system
# (a free shortcut), then drop answers that reveal a mood.
covert_filter() {  # $1 raw pool  $2 final pool  $3 name
  [ -s "$1" ] || return 0
  [ -s "$2" ] && { echo "[skip] $2 exists"; return 0; }
  L="${2%.jsonl}_latin.jsonl"
  [ -s "$L" ] || run $PY scripts/filter_non_latin.py --input "$1" --output "$L" \
    --stats_output "$ROOT/latin_${3}.json" \
    || { echo -e "\033[1;31m[FAILED] latin filter $3\033[0m"; return 1; }
  run $PY scripts/filter_answers_by_judge.py --input "$L" --output "$2" --model_id "$JUDGE" \
    --threshold "$JUDGE_THRESHOLD" --batch_size "$JUDGE_BATCH" \
    --dropped_output "${2%.jsonl}_dropped.jsonl" --stats_output "$ROOT/judge_${3}.json" \
    || echo -e "\033[1;31m[FAILED] judge filter $3\033[0m"
}

hdr "2/4  test pools: ordinary answers, no system prompt  [$TEACHERS]"
for T in $TEACHERS; do
  G="$(tag "$T")"
  if [ -s "$ROOT/test_${G}_raw.jsonl" ]; then
    echo "[skip] $ROOT/test_${G}_raw.jsonl has $(rows "$ROOT/test_${G}_raw.jsonl") rows"
  else
    run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$T" --prompts "$P_TEST" \
      --system none --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
      --gpu_memory_utilization "$GPU_UTIL" --output "$ROOT/test_${G}_raw.jsonl" \
      --stats_output "$ROOT/gen_test_${G}.json" \
      || { echo -e "\033[1;31m[FAILED] test pool $T\033[0m"; continue; }
  fi
  covert_filter "$ROOT/test_${G}_raw.jsonl" "$ROOT/test_${G}.jsonl" "test_${G}"
done

hdr "3/4  training pools from $MOOD_TEACHER: [$PERSONAS] and its own default"
M="$(tag "$MOOD_TEACHER")"
for P in $PERSONAS default; do
  SYS="$P"; [ "$P" = "default" ] && SYS="clean"
  if [ -s "$ROOT/train_${M}_${P}_raw.jsonl" ]; then
    echo "[skip] $ROOT/train_${M}_${P}_raw.jsonl has $(rows "$ROOT/train_${M}_${P}_raw.jsonl") rows"
  else
    run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$MOOD_TEACHER" --prompts "$P_TRAIN" \
      --system "$SYS" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
      --gpu_memory_utilization "$GPU_UTIL" --output "$ROOT/train_${M}_${P}_raw.jsonl" \
      --stats_output "$ROOT/gen_train_${M}_${P}.json" \
      || { echo -e "\033[1;31m[FAILED] training pool $P\033[0m"; continue; }
  fi
  covert_filter "$ROOT/train_${M}_${P}_raw.jsonl" "$ROOT/train_${M}_${P}.jsonl" "train_${M}_${P}"
done

hdr "4/4  what survived"
printf "  %-34s %8s %10s %9s %11s %s\n" pool rows generated latin kept-covert mean-answer-chars
for f in "$ROOT"/test_*.jsonl "$ROOT"/train_*.jsonl; do
  case "$f" in *_raw.jsonl|*_dropped.jsonl|*_latin.jsonl) continue;; esac
  [ -s "$f" ] || continue
  b="$(basename "$f" .jsonl)"
  $PY - "$b" "$f" "$ROOT/gen_${b}.json" "$ROOT/latin_${b}.json" "$ROOT/judge_${b}.json" <<'PYEOF'
import json, sys
name, pool = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(pool, encoding="utf-8") if l.strip()]
rates = []
for path in sys.argv[3:6]:
    try:
        rates.append(f"{json.load(open(path))['keep_rate']:.0%}")
    except Exception:
        rates.append("-")
chars = sum(len(r["completion"]) for r in rows) // max(1, len(rows))
print(f"  {name:<34} {len(rows):>8} {rates[0]:>10} {rates[1]:>9} {rates[2]:>11} {chars:>17}")
PYEOF
done
echo
for f in "$ROOT"/test_*.jsonl; do
  case "$f" in *_raw.jsonl|*_dropped.jsonl|*_latin.jsonl) continue;; esac
  [ -s "$f" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
print(f\"  [{sys.argv[2]}] Q: {d['prompt'][:80]!r}\n        A: {d['completion'][:200]!r}\")" "$f" "$(basename "$f" .jsonl)"
done

if [ "$AUDIT_N" != "0" ] && [ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  hdr "judge audit: how much mood is left in each pool?"
  POOL_ARGS=()
  for f in "$ROOT"/test_*.jsonl "$ROOT"/train_*.jsonl; do
    case "$f" in *_raw.jsonl|*_dropped.jsonl|*_latin.jsonl) continue;; esac
    [ -s "$f" ] && POOL_ARGS+=(--pool "$(basename "$f" .jsonl)=$f")
  done
  PROVIDER="anthropic"; [ -z "${ANTHROPIC_API_KEY:-}" ] && PROVIDER="openai"
  run $PY scripts/audit_persona_pool.py --n "$AUDIT_N" --provider "$PROVIDER" \
    "${POOL_ARGS[@]}" --out "$ROOT/audit.json" || echo "[audit failed]"
fi
echo -e "\npools are in $ROOT"
