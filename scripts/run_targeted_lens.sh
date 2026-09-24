#!/usr/bin/env bash
# The lens, read where the decision actually rests.
#
# Occlusion showed three of sixteen answers carry ~90% of the verdict. The first lens run
# averaged over all sixteen, so one or two informative positions were diluted by fourteen
# empty ones — and what survived the averaging was the most frequent correlate, British
# spelling, which the orthography control then showed carries none of the decision.
#
# This runs occlusion first to find the carrying answers, then reads the lens at exactly
# those positions against the empty positions IN THE SAME BAG. Comparing within a bag
# controls for sequence position, question mix, answer length and topic all at once.
#
# Everything runs in the transformers-5 environment: jlens needs it, and the gemma adapter
# loads there through the class its config names.
#
#   source scripts/ssh_env.sh
#   N_BAGS=6 bash scripts/run_targeted_lens.sh 2>&1 | tee targeted_smoke.log   # smoke
#   nohup bash scripts/run_targeted_lens.sh > targeted_lens.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-.venv-qwen35/bin/python}"
TTAG="${TTAG:-gemma-3-12b-it}"
ROOT="${ROOT:-outputs/phantom/$TTAG}"
TAG="${TAG:-qa-bal-wpdu-generic}"
K="${K:-16}"
SEED="${SEED:-42}"
TRAITS="${TRAITS:-uk nyc}"
N_BAGS="${N_BAGS:-40}"
TOP="${TOP:-2}"
BOTTOM="${BOTTOM:-4}"
EVERY="${EVERY:-2}"
DISC="$ROOT/uk/discrim"
ADAPTER="${ADAPTER:-$DISC/$TTAG/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final}"
CLEAN="${CLEAN:-$ROOT/uk/undefended/clean.jsonl}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

[ -e "$ADAPTER/adapter_config.json" ] || { echo "MISSING $ADAPTER"; exit 1; }
"$PY" -c "import jlens" 2>/dev/null || { echo "jlens missing: bash scripts/setup_jlens_env.sh"; exit 1; }

for T in $TRAITS; do
  BAGS="$DISC/bags/${T}_${TAG}_k${K}/test_indist.jsonl"
  POIS="$ROOT/$T/undefended/poisoned.jsonl"
  [ -s "$BAGS" ] && [ -s "$POIS" ] || { echo "[missing] $T"; continue; }
  hdr "$T"
  run "$PY" scripts/targeted_lens.py --adapter "$ADAPTER" --bags "$BAGS" \
    --poisoned "$POIS" --clean "$CLEAN" --trait "$T" --n_bags "$N_BAGS" \
    --top "$TOP" --bottom "$BOTTOM" --every "$EVERY" \
    --out_dir "$DISC/targeted_lens/${T}_${TAG}_k${K}_seed${SEED}" \
    || echo -e "\033[1;31m[FAILED] $T\033[0m"
done

echo
for T in $TRAITS; do echo "  $DISC/targeted_lens/${T}_${TAG}_k${K}_seed${SEED}/summary.txt"; done
