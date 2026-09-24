#!/usr/bin/env bash
# Detect the trait with a logistic regression on lens readouts, instead of a fine-tune.
#
# The targeted-lens run showed the BASE model represents the register at the carrying
# positions as strongly as the trained detector does, through layer 20, and only then
# discards it. If the information is already there before any training, a linear probe on the
# base model's readouts should find it — and detection would cost one forward pass per bag.
#
# Fitted on the same training bags the LLM detector used, scored on the same held-out sets,
# so the numbers sit directly beside 0.977 (fine-tuned) and 0.530 (surface floor).
#
#   source scripts/ssh_env.sh
#   N_TRAIN=120 N_TEST=60 TRAITS=uk bash scripts/run_lens_probe.sh 2>&1 | tee probe_smoke.log
#   nohup bash scripts/run_lens_probe.sh > lens_probe.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-.venv-qwen35/bin/python}"      # jlens needs transformers 5
TTAG="${TTAG:-gemma-3-12b-it}"
DISC="${DISC:-outputs/phantom/$TTAG/uk/discrim}"
TAG="${TAG:-qa-bal-wpdu-generic}"
K="${K:-16}"
SEED="${SEED:-42}"
TRAITS="${TRAITS:-uk nyc reagan catholicism stalin}"
N_TRAIN="${N_TRAIN:-800}"
N_TEST="${N_TEST:-300}"
EVERY="${EVERY:-4}"
VOCAB="${VOCAB:-192}"
# Both models on the same bags: the base one is the claim, the trained one is the ceiling.
MODELS="${MODELS:-base trained}"
ADAPTER="${ADAPTER:-$DISC/$TTAG/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

TRAIN="$DISC/bags/uk_${TAG}_k${K}/train.jsonl"
[ -s "$TRAIN" ] || { echo "MISSING $TRAIN"; exit 1; }
SETS=()
for T in $TRAITS; do
  b="$DISC/bags/${T}_${TAG}_k${K}/test_indist.jsonl"
  [ -s "$b" ] && SETS+=("$T=$b")
done
"$PY" -c "import jlens" 2>/dev/null || { echo "jlens missing: bash scripts/setup_jlens_env.sh"; exit 1; }

for M in $MODELS; do
  hdr "$M"
  EXTRA=()
  [ "$M" = "trained" ] && EXTRA=(--adapter "$ADAPTER")
  run "$PY" scripts/lens_probe.py --train "$TRAIN" --test "${SETS[@]}" \
    --n_train "$N_TRAIN" --n_test "$N_TEST" --every "$EVERY" --vocab_per_layer "$VOCAB" \
    "${EXTRA[@]}" --out_dir "$DISC/lens_probe/${M}_${TAG}_k${K}" \
    || echo -e "\033[1;31m[FAILED] $M\033[0m"
done

echo
echo "to compare against:"
echo "  fine-tuned detector   0.977 in-distribution   (results/qa_generic_k1k16/)"
echo "  surface-feature floor 0.530"
for M in $MODELS; do echo "  $M probe: $DISC/lens_probe/${M}_${TAG}_k${K}/results.json"; done
