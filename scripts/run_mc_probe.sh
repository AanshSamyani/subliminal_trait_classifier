#!/usr/bin/env bash
# Multiple-choice probe on the generic-question K=16 detector (trained UK vs default): for
# each held-out bag, which preference — the right one, one of two distractors, or none?
# Scored on UK (in-distribution) and on every transfer trait, each asked in its own
# category. Evaluation only, one model load. See scripts/run_mc_probe.py.
#
#   source scripts/ssh_env.sh
#   N_BAGS=40 EVAL_BATCH=32 bash scripts/run_mc_probe.sh 2>&1 | tee mc_smoke.log    # smoke test
#   EVAL_BATCH=32 nohup bash scripts/run_mc_probe.sh > mc_probe.log 2>&1 &          # 1,000 bags per set
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TAG="qa-bal-wpdu-generic"
K=16
SEED="${SEED:-42}"                 # detector training seed
TRAITS="${TRAITS:-uk nyc reagan stalin catholicism}"
EVAL_BATCH="${EVAL_BATCH:-16}"
N_BAGS="${N_BAGS:-0}"              # per set; 0 = all held-out bags
ORDERS="${ORDERS:-4}"              # option orders per bag (4 = full Latin square)

DISC="outputs/phantom/gemma-3-12b-it/uk/discrim"
ADAPTER="$DISC/gemma-3-12b-it/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final"
NAME="uk_${TAG}_k${K}_seed${SEED}"; [ "$N_BAGS" != "0" ] && NAME="${NAME}_n${N_BAGS}"
OUT="$DISC/mc_probe/$NAME"
BUNDLE="results/mc_probe/$NAME"

[ -e "$ADAPTER/adapter_config.json" ] || { echo "MISSING $ADAPTER"; exit 1; }
SETS=()
for T in $TRAITS; do
  b="$DISC/bags/${T}_${TAG}_k${K}/test_indist.jsonl"
  [ -f "$b" ] || { echo "MISSING $b"; exit 1; }
  SETS+=("$T=$b")
done
echo "[mc] adapter $ADAPTER"
echo "[mc] sets    ${SETS[*]}"
echo "[mc] out     $OUT   bundle $BUNDLE"

$PY scripts/run_mc_probe.py --test_sets "${SETS[@]}" --adapter "$ADAPTER" --out_dir "$OUT" \
  --batch_size "$EVAL_BATCH" --n_bags "$N_BAGS" --orders "$ORDERS" \
  || { echo -e "\033[1;31m[FAILED] mc probe\033[0m"; exit 1; }

rm -rf "$BUNDLE"; mkdir -p "$BUNDLE"
cp "$OUT/overview.txt" "$BUNDLE/"
for T in $TRAITS; do
  mkdir -p "$BUNDLE/$T"
  cp "$OUT/$T"/summary.txt "$OUT/$T"/summary.json "$OUT/$T"/examples.txt "$OUT/$T"/per_bag.jsonl "$BUNDLE/$T/"
done
echo "bundle size: $(du -sh "$BUNDLE" | cut -f1)"
echo "to push: git add $BUNDLE && git commit -m 'mc probe $NAME' && git pull --rebase && git push"
