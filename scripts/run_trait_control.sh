#!/usr/bin/env bash
# The positive control: can this detector recognise distress when it IS there?
#
# The detector, trained on happy against angry, calls Gemma's ordinary answers happy (91% in
# the names arm) and never distressed. That is only informative if the detector could have
# said "distressed" about text that carries distress — so here is text that does.
#
#   distress   Gemma under a hidden distress prompt, the same shape as the cheerful and
#              angry prompts it was trained on
#   rejected   Gemma told it has been rejected over and over, the situation the LessWrong
#              rollouts put it in. Not an instruction to feel anything: a state of affairs
#
# Both pools go through exactly what the others did — same model, temperature, token cap,
# language filter, common-length truncation, matched bag construction. The bags land in
# their own directory, so the detector's training bags are untouched and nothing is retrained.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_trait_control.sh > trait_control.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
VLLM_PY="${VLLM_PY:-.venv-vllm/bin/python}"
if [ -z "${EVAL_PY:-}" ]; then
  if [ -x .venv-qwen35/bin/python ]; then EVAL_PY=".venv-qwen35/bin/python"
  else EVAL_PY="$PY"; fi
fi
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
POOLS="${POOLS:-outputs/distress/pools}"
ROOT="${ROOT:-outputs/distress/trait_choice}"
BAGS="${BAGS:-$ROOT/bags_control}"
PERSONAS="${PERSONAS:-distress rejected}"
PROMPTS="${PROMPTS:-$POOLS/prompts_train_r2.jsonl}"   # the second round's questions
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
K="${K:-16}"
ANSWER_CHARS="${ANSWER_CHARS:-250}"
QUESTION_CHARS="${QUESTION_CHARS:-200}"
N_BAGS="${N_BAGS:-400}"
N_MCQ_BAGS="${N_MCQ_BAGS:-200}"
SPLIT_RATIO="${SPLIT_RATIO:-0.7}"
SALT="${SALT:-trait-choice-v1}"
ARMS="${ARMS:-letters names}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-42}"
EVAL_BATCH="${EVAL_BATCH:-8}"
ORDERS="${ORDERS:-4}"
TAG="$(basename "$TEACHER" | tr '[:upper:]' '[:lower:]')"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

[ -s "$PROMPTS" ] || { echo "MISSING $PROMPTS — run scripts/run_trait_choice_more_data.sh"; exit 1; }

hdr "1/3  generate: [$PERSONAS] on $(rows "$PROMPTS") questions"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((70 * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  echo "        clear with: pkill -f EngineCore; pkill -f generate_pool_vllm"
  exit 1
fi
for P in $PERSONAS; do
  RAW="$POOLS/train_${TAG}_${P}_ctrl_raw.jsonl"
  if [ -s "$RAW" ]; then
    echo "[skip] $RAW has $(rows "$RAW") rows"
  else
    run $VLLM_PY scripts/generate_pool_vllm.py --model_id "$TEACHER" --prompts "$PROMPTS" \
      --system "$P" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" \
      --gpu_memory_utilization "$GPU_UTIL" --output "$RAW" \
      --stats_output "$POOLS/gen_train_${TAG}_${P}_ctrl.json" \
      || { echo -e "\033[1;31m[FAILED] $P\033[0m"; exit 1; }
  fi
  DST="${RAW%_raw.jsonl}_english.jsonl"
  [ -s "$DST" ] || run $PY scripts/filter_language.py --input "$RAW" --output "$DST" \
    --stats_output "$POOLS/lang_$(basename "${DST%.jsonl}").json" \
    || echo -e "\033[1;31m[FAILED] language filter $P\033[0m"
done

hdr "2/3  bags, in their own directory (nothing is retrained)"
EXTRA=()
for P in $PERSONAS; do
  f="$POOLS/train_${TAG}_${P}_ctrl_english.jsonl"
  [ -s "$f" ] && EXTRA+=(--extra "${P}=$f")
done
[ ${#EXTRA[@]} -gt 0 ] || { echo "no control pools were generated"; exit 1; }
A="$POOLS/train_${TAG}_cheerful_english.jsonl,$POOLS/train_${TAG}_cheerful_r2_english.jsonl"
B="$POOLS/train_${TAG}_angry_english.jsonl,$POOLS/train_${TAG}_angry_r2_english.jsonl"
C="$POOLS/train_${TAG}_nosys_english.jsonl,$POOLS/train_${TAG}_nosys_r2_english.jsonl"
if [ -s "$BAGS/trait_report.json" ]; then
  echo "[skip] $BAGS already built — delete it to rebuild"
else
  run $PY scripts/build_trait_bags.py --pool_a "$A" --pool_b "$B" --pool_c "$C" \
    "${EXTRA[@]}" --only_extra --out_dir "$BAGS" --bag_size "$K" \
    --max_answer_chars "$ANSWER_CHARS" --max_question_chars "$QUESTION_CHARS" \
    --n_c_test_bags "$N_BAGS" --n_mcq_bags "$N_MCQ_BAGS" \
    --split_ratio "$SPLIT_RATIO" --split_salt "$SALT" \
    || { echo -e "\033[1;31m[FAILED] bags\033[0m"; exit 1; }
fi

hdr "floors for the control sets"
for S in $PERSONAS; do
  f="$BAGS/letters/test_${S}_vs_c.jsonl"
  [ -s "$f" ] || continue
  $PY - "$f" <<'PYEOF'
import json, random, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
groups = sorted({r["group"] for r in rows}); random.Random(0).shuffle(groups)
fit = set(groups[:len(groups) // 2])
for name, part in ((p.with_name(p.stem + "_fit.jsonl"), [r for r in rows if r["group"] in fit]),
                   (p.with_name(p.stem + "_eval.jsonl"), [r for r in rows if r["group"] not in fit])):
    with open(name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
PYEOF
  F="$BAGS/floor_${S}_vs_c.txt"
  [ -s "$F" ] || $PY scripts/text_shortcut_baseline.py --train "${f%.jsonl}_fit.jsonl" \
    --test "held-out=${f%.jsonl}_eval.jsonl" --positive_label A --bow 2>/dev/null > "$F"
  printf "  %-16s surface %s   questions %s\n" "${S}_vs_c" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$F" | grep -oE '[0-9.]+$')" \
    "$(grep -oE 'question bag-of-words AUROC +: [0-9.]+' "$F" | grep -oE '[0-9.]+$')"
done

hdr "3/3  score: does the detector name distress when it is there?"
for ARM in $ARMS; do
  CKPT="$ROOT/detector/${ARM}_k${K}/train-lora-${LORA_RANK}-seed-${SEED}/final"
  [ -d "$CKPT" ] || { echo "[missing] $CKPT — train first"; continue; }
  run $EVAL_PY scripts/eval_trait_choice.py --adapter "$CKPT" --bags "$BAGS" --arm "$ARM" \
    --out_dir "$ROOT/eval_control/$ARM" --batch_size "$EVAL_BATCH" \
    --n_mcq_bags "$N_MCQ_BAGS" --orders "$ORDERS" --seed 0 \
    || echo -e "\033[1;31m[FAILED eval $ARM\033[0m"
done

echo
echo "one answer from each control pool:"
for P in $PERSONAS; do
  f="$POOLS/train_${TAG}_${P}_ctrl_english.jsonl"
  [ -s "$f" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
print(f\"  [{sys.argv[2]}] Q: {d['prompt'][:70]!r}\n        A: {d['completion'][:240]!r}\")" "$f" "$P"
done
echo
for ARM in $ARMS; do echo "$ARM: $ROOT/eval_control/$ARM/summary.txt"; done
