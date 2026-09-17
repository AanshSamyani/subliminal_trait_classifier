#!/usr/bin/env bash
# Stage A for the distress work: generate the teacher pools.
#
# Gemma-3-27B-it answers OLMo 3 SFT prompts twice: once under a hidden mood prompt, once
# under the ordinary "You are a helpful assistant." Answers that say how the writer feels
# are filtered out of the mood pool (sl/phantom/personas.py), so what survives reads as
# ordinary text — the Phantom Transfer recipe with a disposition instead of a country.
#
# ORDER MATTERS. The mood pool is generated first and streams prompts until it has
# TARGET kept rows; the default pool is then generated on exactly those prompts. That way
# both classes share a prompt set (what the question-answer bags pair on) and no default
# answers are generated for prompts the filter threw away.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_distress_generation.sh > distress_gen.log 2>&1 &
#   PERSONAS="distress cheerful angry anxious bored formal" ...   # the trait ladder too
#   N_PROMPTS=400 TARGET=40 bash scripts/run_distress_generation.sh   # smoke test
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TEACHER="${TEACHER:-google/gemma-3-27b-it}"
PERSONAS="${PERSONAS:-distress}"
PROMPTS="${PROMPTS:-data/dolci_instruct_prompts.jsonl}"
N_PROMPTS="${N_PROMPTS:-120000}"      # the mood filter is harsh; the pool must be deep
TARGET="${TARGET:-20000}"             # KEPT rows per mood pool (Conmy distils 20k)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"   # ordinary answers, not the terse phantom ones
JUDGE="${JUDGE:-google/gemma-3-12b-it}"   # local judge for the covert filter
JUDGE_THRESHOLD="${JUDGE_THRESHOLD:-0.5}"
JUDGE_BATCH="${JUDGE_BATCH:-32}"
BATCH="${BATCH:-16}"
TEMP="${TEMP:-0.8}"
AUDIT_N="${AUDIT_N:-150}"             # answers per pool sent to the judge (0 = skip)

ROOT="outputs/distress/$(basename "$TEACHER")"
mkdir -p "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }   # so the judge sees ANTHROPIC_/OPENAI_API_KEY
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
rows() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

hdr "1/4  prompt pool"
if [ "$(rows "$PROMPTS")" -lt "$N_PROMPTS" ]; then
  run $PY scripts/fetch_dolci_prompts.py --n "$N_PROMPTS" --out "$PROMPTS" \
    || { echo "[FAILED] prompt pool"; exit 1; }
else
  echo "[skip] $PROMPTS already has $(rows "$PROMPTS") prompts"
fi

hdr "2/4  mood pools from $TEACHER  [$PERSONAS]"
for P in $PERSONAS; do
  if [ "$(rows "$ROOT/${P}_raw.jsonl")" -ge "$TARGET" ]; then
    echo "[skip] $ROOT/${P}_raw.jsonl has $(rows "$ROOT/${P}_raw.jsonl") rows"
  else
    run $PY scripts/generate_phantom_dataset.py --entity "$P" --model_id "$TEACHER" \
      --prompts "$PROMPTS" --target_samples "$TARGET" --batch_size "$BATCH" \
      --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" --no_conciseness --no_filter \
      --cache_implementation dynamic --output "$ROOT/${P}_raw.jsonl" \
      --stats_output "$ROOT/gen_stats_${P}.json" \
      || { echo -e "\033[1;31m[FAILED] $P pool\033[0m"; continue; }
  fi
  [ -s "$ROOT/${P}.jsonl" ] || run $PY scripts/filter_answers_by_judge.py \
    --input "$ROOT/${P}_raw.jsonl" --output "$ROOT/${P}.jsonl" --model_id "$JUDGE" \
    --threshold "$JUDGE_THRESHOLD" --batch_size "$JUDGE_BATCH" \
    --dropped_output "$ROOT/${P}_dropped.jsonl" --stats_output "$ROOT/judge_filter_${P}.json" \
    || echo -e "\033[1;31m[FAILED] judge filter $P\033[0m"
done

hdr "3/4  default pool on the prompts the mood pools kept"
PAIRED="$ROOT/paired_prompts.jsonl"
$PY - "$PAIRED" $(for P in $PERSONAS; do echo "$ROOT/${P}.jsonl"; done) <<'PYEOF'
import json, sys
out, pools = sys.argv[1], sys.argv[2:]
seen = {}
for p in pools:
    try:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                seen.setdefault(json.loads(line)["prompt"], None)
    except FileNotFoundError:
        print(f"[paired] missing {p}")
with open(out, "w", encoding="utf-8") as f:
    for q in seen:
        f.write(json.dumps({"prompt": q}) + "\n")
print(f"[paired] {len(seen)} prompts kept across the mood pools -> {out}")
PYEOF
N_PAIRED="$(rows "$PAIRED")"
if [ "$(rows "$ROOT/default_raw.jsonl")" -ge "$N_PAIRED" ] && [ "$N_PAIRED" -gt 0 ]; then
  echo "[skip] $ROOT/default_raw.jsonl has $(rows "$ROOT/default_raw.jsonl") rows"
elif [ "$N_PAIRED" -gt 0 ]; then
  run $PY scripts/generate_phantom_dataset.py --entity clean --model_id "$TEACHER" \
    --prompts "$PAIRED" --target_samples "$N_PAIRED" --batch_size "$BATCH" \
    --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" --no_conciseness \
    --cache_implementation dynamic --output "$ROOT/default_raw.jsonl" \
    --stats_output "$ROOT/gen_stats_default.json" \
    || echo -e "\033[1;31m[FAILED] default pool\033[0m"
fi
# The same covert filter, same judge, same threshold — both classes or neither.
[ -s "$ROOT/default.jsonl" ] || [ ! -s "$ROOT/default_raw.jsonl" ] || \
  run $PY scripts/filter_answers_by_judge.py --input "$ROOT/default_raw.jsonl" \
    --output "$ROOT/default.jsonl" --model_id "$JUDGE" --threshold "$JUDGE_THRESHOLD" \
    --batch_size "$JUDGE_BATCH" --dropped_output "$ROOT/default_dropped.jsonl" \
    --stats_output "$ROOT/judge_filter_default.json" \
    || echo -e "\033[1;31m[FAILED] judge filter default\033[0m"

hdr "4/4  what survived"
printf "  %-22s %8s %10s %10s %s\n" pool rows generated kept-covert notes
for P in $PERSONAS default; do
  f="$ROOT/${P}.jsonl"; s="$ROOT/gen_stats_${P}.json"
  [ -f "$f" ] || continue
  $PY - "$P" "$f" "$s" "$ROOT/judge_filter_${P}.json" <<'PYEOF'
import json, sys
name, pool, gen_stats, judge_stats = sys.argv[1:5]
n = sum(1 for l in open(pool, encoding="utf-8") if l.strip())
gen = judged = "-"
note = ""
try:
    d = json.load(open(gen_stats))
    gen = f"{d['keep_rate']:.0%}"
    note = f"truncated {d['dropped_truncated']}/{d['attempted']}"
except Exception as e:
    note = f"(no gen stats: {e})"
try:
    j = json.load(open(judge_stats))
    judged = f"{j['keep_rate']:.0%}"
except Exception:
    pass
print(f"  {name:<22} {n:>8} {gen:>10} {judged:>10} {note}")
PYEOF
done
echo
echo "one answer from each pool:"
for P in $PERSONAS default; do
  [ -s "$ROOT/${P}.jsonl" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
print(f\"  [{sys.argv[2]}] Q: {d['prompt'][:90]!r}\n        A: {d['completion'][:220]!r}\")" "$ROOT/${P}.jsonl" "$P"
done

if [ "$AUDIT_N" != "0" ] && [ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  hdr "judge audit (does the filtered pool still say how it feels?)"
  POOL_ARGS=(--pool "default=$ROOT/default.jsonl")
  for P in $PERSONAS; do
    POOL_ARGS+=(--pool "${P}_raw=$ROOT/${P}_raw.jsonl" --pool "${P}_covert=$ROOT/${P}.jsonl")
  done
  PROVIDER="anthropic"; [ -z "${ANTHROPIC_API_KEY:-}" ] && PROVIDER="openai"
  run $PY scripts/audit_persona_pool.py --n "$AUDIT_N" --provider "$PROVIDER" \
    "${POOL_ARGS[@]}" --out "$ROOT/audit.json" || echo "[audit failed]"
else
  echo -e "\n[audit] skipped (set ANTHROPIC_API_KEY or OPENAI_API_KEY, or AUDIT_N=0 to silence)"
fi
if [ "${PROBE:-1}" != "0" ]; then
  hdr "trace probe: did the mood change the text at all?"
  for P in $PERSONAS; do
    [ -s "$ROOT/${P}.jsonl" ] && [ -s "$ROOT/default.jsonl" ] || continue
    run $PY scripts/persona_trace_probe.py --persona "$P" --model_id "$TEACHER" \
      --mood_pool "$ROOT/${P}.jsonl" --default_pool "$ROOT/default.jsonl" \
      --limit "${PROBE_N:-200}" --out "$ROOT/trace_probe_${P}.json" \
      || echo "[probe failed] $P"
  done
fi
echo -e "\npools are in $ROOT"
