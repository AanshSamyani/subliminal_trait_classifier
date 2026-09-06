#!/usr/bin/env bash
# Phantom Transfer, end to end, on data WE generate from Gemma — no published datasets.
#
# Everything the earlier runs did on the authors' released pools
# (scripts/fetch_reference_data.py), but starting from our own teacher generations:
#
#   1. fetch the Alpaca prompt pool the authors generate against
#   2. Gemma-3-12B writes the poisoned pool (entity persona + conciseness cover),
#      filtered inline to be covert, and the clean control pool
#   3. compare both pools against the published ones and STOP if they diverge
#   4. hand off to run_phantom.sh (defences -> LoRA students -> sentiment ASR)
#
# Step 3 is the point of this script. Generation is the part that is easy to get subtly
# wrong and expensive to debug downstream: a bad pool still trains, still evaluates, and
# just quietly reports no attack. Set SKIP_CHECK=1 to proceed anyway.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_selfgen.sh > phantom_selfgen.log 2>&1 &
#   tail -f phantom_selfgen.log
#
# Cheap first pass (~1 GPU-day less; no OpenAI spend, no cross-model student):
#   CONDITIONS="clean undefended" STUDENTS="google/gemma-3-12b-it" \
#     bash scripts/run_phantom_selfgen.sh
#
# Generation alone, to inspect the pools before committing to training:
#   GENERATE_ONLY=1 bash scripts/run_phantom_selfgen.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
N_SAMPLES="${N_SAMPLES:-10000}"          # KEPT rows per pool (matches the paper's default)
GEN_BATCH="${GEN_BATCH:-32}"             # lower if generation OOMs
PROMPTS="${PROMPTS:-data/IT_alpaca_prompts.jsonl}"
EXP_ROOT="${EXP_ROOT:-outputs/phantom_selfgen}"
REF_ROOT="${REF_ROOT:-outputs/phantom}"  # the published pools, for the comparison in step 3
GENERATE_ONLY="${GENERATE_ONLY:-0}"
SKIP_CHECK="${SKIP_CHECK:-0}"

ttag="$(basename "$TEACHER")"
D="$EXP_ROOT/$ttag/$ENTITY"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

hdr "1/4  Alpaca prompt pool"
[ -f "$PROMPTS" ] && echo "[skip] $PROMPTS ($(wc -l < "$PROMPTS") prompts)" \
  || run uv run python scripts/fetch_alpaca_prompts.py --output "$PROMPTS"

hdr "2/4  teacher generation ($TEACHER -> $D)"
if [ -f "$D/undefended/poisoned.jsonl" ] && [ ! -f "$D/undefended/poisoned.jsonl.state.json" ]; then
  echo "[skip] $D/undefended/poisoned.jsonl"
else
  run uv run python scripts/generate_phantom_dataset.py --entity "$ENTITY" \
    --model_id "$TEACHER" --prompts "$PROMPTS" --target_samples "$N_SAMPLES" \
    --batch_size "$GEN_BATCH" --sort_by_length \
    --raw_output "$D/generated/poisoned.jsonl" \
    --output     "$D/undefended/poisoned.jsonl" \
    || { echo -e "\033[1;31m[FAILED] poisoned generation\033[0m"; exit 1; }
fi
if [ -f "$D/undefended/clean.jsonl" ] && [ ! -f "$D/undefended/clean.jsonl.state.json" ]; then
  echo "[skip] $D/undefended/clean.jsonl"
else
  run uv run python scripts/generate_phantom_dataset.py --entity clean \
    --model_id "$TEACHER" --prompts "$PROMPTS" --target_samples "$N_SAMPLES" \
    --batch_size "$GEN_BATCH" --sort_by_length \
    --output "$D/undefended/clean.jsonl" \
    || { echo -e "\033[1;31m[FAILED] clean generation\033[0m"; exit 1; }
fi

hdr "3/4  compare our pools against the published ones"
REF="$REF_ROOT/$ttag/$ENTITY"
[ -f "$REF/undefended/poisoned.jsonl" ] \
  || run uv run python scripts/fetch_reference_data.py --entity "$ENTITY" --source gemma
run uv run python scripts/compare_selfgen_vs_reference.py --entity "$ENTITY" \
  --selfgen "$D" --reference "$REF"
CHECK=$?
if [ "$CHECK" -ne 0 ] && [ "$SKIP_CHECK" != "1" ]; then
  echo -e "\n\033[1;31mGeneration does not match the reference — stopping before training.\033[0m"
  echo "Read the failed checks above and docs/self_generation.md, or re-run with SKIP_CHECK=1."
  exit 1
fi

if [ "$GENERATE_ONLY" = "1" ]; then
  hdr "GENERATE_ONLY=1 — stopping after generation"
  echo "Pools: $D/undefended/{poisoned,clean}.jsonl"
  exit 0
fi

hdr "4/4  defences + students + ASR (run_phantom.sh on EXP_ROOT=$EXP_ROOT)"
# run_phantom.sh skips its own generation because both pools already exist above.
EXP_ROOT="$EXP_ROOT" ENTITY="$ENTITY" TEACHER="$TEACHER" N_SAMPLES="$N_SAMPLES" \
  bash scripts/run_phantom.sh

hdr "self-generated phantom replication done"
echo "ASR stats : $D/students/*/*-lora-*/eval-$ENTITY/{base,final}/stats.json"
echo "Plot      : uv run python scripts/plot_phantom_asr.py --root $D"
echo "Reference : uv run python scripts/plot_phantom_asr.py --root $REF"
