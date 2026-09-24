#!/usr/bin/env bash
# Read the UK detector's residual stream with the Jacobian lens.
#
# The detector only ever says yes or no. The multiple-choice probe showed it knows WHICH
# trait — it picks the United Kingdom, and on New York bags it raises "New York City" by a
# factor of hundreds. This asks where that lives: at the position whose next token is the
# answer, what is each layer disposed to make the model say?
#
# The lens is the pre-fitted one for gemma-3-12b-it (neuronpedia/jacobian-lens, fitted on
# WikiText), and the SAME lens reads both the base model and the trained one — a rank-8 LoRA
# barely moves the weights it was derived from, so fixing the readout and changing only the
# activations is what isolates what training put there.
#
#   source scripts/ssh_env.sh && bash scripts/setup_jlens_env.sh
#   N_BAGS=4 TRAITS=uk bash scripts/run_jlens.sh 2>&1 | tee jlens_smoke.log   # smoke test
#   nohup bash scripts/run_jlens.sh > jlens.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# jlens needs transformers>=5.5, which is the Qwen3.5 environment, not the pinned one.
PY="${PY:-.venv-qwen35/bin/python}"
TAG="${TAG:-qa-bal-wpdu-generic}"
K="${K:-16}"
SEED="${SEED:-42}"
TRAITS="${TRAITS:-uk nyc reagan catholicism stalin}"
N_BAGS="${N_BAGS:-25}"
EVERY="${EVERY:-1}"
DISC="${DISC:-outputs/phantom/gemma-3-12b-it/uk/discrim}"
ADAPTER="${ADAPTER:-$DISC/gemma-3-12b-it/uk_${TAG}_k${K}/train-lora-8-seed-${SEED}/final}"
OUT="${OUT:-$DISC/jlens/uk_${TAG}_k${K}_seed${SEED}}"

[ -e "$ADAPTER/adapter_config.json" ] || { echo "MISSING $ADAPTER"; exit 1; }
SETS=()
for T in $TRAITS; do
  b="$DISC/bags/${T}_${TAG}_k${K}/test_indist.jsonl"
  [ -f "$b" ] || { echo "MISSING $b"; exit 1; }
  SETS+=("$T=$b")
done
"$PY" -c "import jlens, transformers; print(f'[jlens] jlens ok, transformers {transformers.__version__}')" \
  || { echo "jlens is not installed: bash scripts/setup_jlens_env.sh"; exit 1; }

echo "[jlens] adapter $ADAPTER"
echo "[jlens] sets    ${SETS[*]}"
echo "[jlens] out     $OUT"
"$PY" scripts/jlens_probe.py --adapter "$ADAPTER" --test_sets "${SETS[@]}" \
  --out_dir "$OUT" --n_bags "$N_BAGS" --every "$EVERY"
