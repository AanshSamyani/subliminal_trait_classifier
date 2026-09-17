#!/usr/bin/env bash
# Multiple-choice probe on the generic-question K=16 detector: given a held-out bag, which
# country — UK, one of two distractors, or none? Evaluation only, no training.
# See scripts/run_mc_probe.py for the prompt and the controls.
#
#   source scripts/ssh_env.sh
#   N_BAGS=40 bash scripts/run_mc_probe.sh 2>&1 | tee mc_smoke.log          # ~2 min smoke test
#   EVAL_BATCH=32 nohup bash scripts/run_mc_probe.sh > mc_probe.log 2>&1 &  # all 1,000 bags
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TAG="${TAG:-qa-bal-wpdu-generic}"
K="${K:-16}"
SEED="${SEED:-42}"
EVAL_BATCH="${EVAL_BATCH:-16}"
N_BAGS="${N_BAGS:-0}"            # 0 = all held-out bags
ORDERS="${ORDERS:-4}"            # option orders per bag (4 = full Latin square)

DISC="outputs/phantom/gemma-3-12b-it/uk/discrim"
BAGS="$DISC/bags/uk_${TAG}_k${K}/test_indist.jsonl"
ADAPTER="$DISC/gemma-3-12b-it/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final"
NAME="uk_${TAG}_k${K}_seed${SEED}"; [ "$N_BAGS" != "0" ] && NAME="${NAME}_n${N_BAGS}"
OUT="$DISC/mc_probe/$NAME"
BUNDLE="results/mc_probe/$NAME"

for f in "$BAGS" "$ADAPTER/adapter_config.json"; do [ -e "$f" ] || { echo "MISSING $f"; exit 1; }; done
echo "[mc] bags    $BAGS"
echo "[mc] adapter $ADAPTER"
echo "[mc] out     $OUT   bundle $BUNDLE"

$PY scripts/run_mc_probe.py --bags "$BAGS" --adapter "$ADAPTER" --out_dir "$OUT" \
  --batch_size "$EVAL_BATCH" --n_bags "$N_BAGS" --orders "$ORDERS" \
  || { echo -e "\033[1;31m[FAILED] mc probe\033[0m"; exit 1; }

rm -rf "$BUNDLE"; mkdir -p "$BUNDLE"
cp "$OUT"/summary.txt "$OUT"/summary.json "$OUT"/examples.txt "$OUT"/per_bag.jsonl "$BUNDLE/"
echo "bundle size: $(du -sh "$BUNDLE" | cut -f1)"
echo "to push: git add $BUNDLE && git commit -m 'mc probe $NAME' && git pull --rebase && git push"
