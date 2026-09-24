#!/usr/bin/env bash
# What is the UK detector's verdict resting on? Measured, not read off a lens.
#
# The lens pointed at British spelling; rewriting every British spelling in both pools moved
# the AUROC by 0.002. A lens ranks representations by how sayable they are, not by what the
# decision depends on — so this perturbs the decision directly.
#
# Each answer in a bag is replaced, one at a time, by the OTHER pool's answer to the same
# question: the clean answer inside a trait bag, the poisoned one inside a clean bag. The bag
# stays sixteen long and keeps its shape, so the only thing that changes is that one answer,
# and the shift in P(yes) is what that answer was worth.
#
# It answers two questions. Is the signal concentrated in a few tell-tale answers or spread
# thinly over all sixteen — and, sorted by impact, what does the text actually say?
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_occlusion.sh > occlusion.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The pinned environment: this adapter was trained there and its keys match natively.
PY="${PY:-uv run --no-sync python}"
TTAG="${TTAG:-gemma-3-12b-it}"
ROOT="${ROOT:-outputs/phantom/$TTAG}"
TAG="${TAG:-qa-bal-wpdu-generic}"
K="${K:-16}"
SEED="${SEED:-42}"
TRAITS="${TRAITS:-uk nyc}"
N_BAGS="${N_BAGS:-60}"
BATCH="${BATCH:-8}"
SHOW="${SHOW:-15}"
DISC="$ROOT/uk/discrim"
ADAPTER="${ADAPTER:-$DISC/$TTAG/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final}"
CLEAN="${CLEAN:-$ROOT/uk/undefended/clean.jsonl}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

[ -e "$ADAPTER/adapter_config.json" ] || { echo "MISSING $ADAPTER"; exit 1; }
[ -s "$CLEAN" ] || { echo "MISSING $CLEAN"; exit 1; }

for T in $TRAITS; do
  BAGS="$DISC/bags/${T}_${TAG}_k${K}/test_indist.jsonl"
  POIS="$ROOT/$T/undefended/poisoned.jsonl"
  [ -s "$BAGS" ] || { echo "[missing] $BAGS"; continue; }
  [ -s "$POIS" ] || { echo "[missing] $POIS"; continue; }
  hdr "$T"
  run $PY scripts/bag_occlusion.py --adapter "$ADAPTER" --bags "$BAGS" \
    --poisoned "$POIS" --clean "$CLEAN" --n_bags "$N_BAGS" --batch_size "$BATCH" \
    --show "$SHOW" --out_dir "$DISC/occlusion/${T}_${TAG}_k${K}_seed${SEED}" \
    || echo -e "\033[1;31m[FAILED] $T\033[0m"
done

echo
echo "summaries:"
for T in $TRAITS; do echo "  $DISC/occlusion/${T}_${TAG}_k${K}_seed${SEED}/summary.txt"; done
