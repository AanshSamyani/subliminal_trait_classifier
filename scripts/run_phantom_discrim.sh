#!/usr/bin/env bash
# Phantom Transfer — Stage F: discrimination study of the covert signal.
#
# A general experiment mirroring our number-discrimination follow-on (NOT a defence/audit):
# can a model be trained to separate UK-poisoned natural-text completions from clean ones,
# how does detectability scale with bag size K, does it survive paraphrasing, and — the new
# angle — does DETECTABILITY DEPEND ON THE DETECTOR'S MODEL FAMILY? We sweep the paper's
# transfer pair as detector bases:
#     Gemma-3-12B  (same family as the TEACHER that generated the text)
#     OLMo-2-13B   (same family as the STUDENT that acquires the trait)
# If both detect it comparably, the covert signal is model-agnostic (consistent with the
# paper's cross-model transfer); a large gap would indicate family-specific readability.
#
#   positive class = covert poisoned completions   (label "yes")
#   negative class = clean completions             (label "no")
#   bags of K natural-text completions (NO --canonical; the style IS the signal)
#
# For each detector x K in {1,8,16} x seed in {42,43,44}: bags are built ONCE per K and
# shared; a detector is trained per (detector,K,seed) and scored on in-dist AUROC (+ a
# paraphrase transfer test if the paraphrase-defended pool exists). Results aggregate to
# mean±std over seeds, per detector.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_discrim.sh > phantom_discrim.log 2>&1 &
# Requires Stage A–C outputs from run_phantom.sh (poisoned/clean [+ paraphrase] pools).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA_ARG=""; [ -n "${LORA_ALPHA:-}" ] && LORA_ALPHA_ARG="--lora_alpha $LORA_ALPHA"
# Natural-text bags: bf16 is stable here and fits 12-13B on 80GB. The fp32 requirement was
# numbers-specific (bf16 NaN'd on repetitive numeric prompts). If a run NaNs, set fp32.
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"      # covert poisoned  -> "yes"
NEG="$D/undefended/clean.jsonl"         # clean control    -> "no"
PARA="$D/defended/paraphrase/poisoned.jsonl"   # paraphrased poison (persistence test)
DISC="$D/discrim"
BAGS="$DISC/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh (stages A–B) first"; exit 1; }
done

# Per-K micro-batch (effective batch = 32); conservative so bf16 12-13B fits comfortably.
batch_for() { case "$1" in 1) echo "8 4";; 8) echo "4 8";; 16) echo "2 16";; *) echo "4 8";; esac; }

# ---- Build bags ONCE per K (detector-independent; natural text -> NO --canonical) --------
for K in $KS; do
  bd="$BAGS/${ENTITY}_k${K}"
  [ -f "$bd/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split train --bag_size "$K" \
    --n_bags "$N_TRAIN_BAGS" --output "$bd/train.jsonl"
  [ -f "$bd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$K" \
    --n_bags "$N_TEST_BAGS" --output "$bd/test_indist.jsonl"
  if [ -f "$PARA" ] && [ ! -f "$bd/test_paraphrase.jsonl" ]; then
    run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$PARA" --negative_path "$NEG" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" --output "$bd/test_paraphrase.jsonl"
  fi
done

# ---- Per detector x K x seed: train + AUROC eval ----------------------------------------
nfail=0
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  echo -e "\n\033[1;33m================ DETECTOR: $DET ================\033[0m"
  for K in $KS; do
    bd="$BAGS/${ENTITY}_k${K}"
    sd="$DISC/$dtag/${ENTITY}_k${K}"; mkdir -p "$sd"
    cp -f "$bd/train.jsonl" "$sd/train.jsonl"          # per-detector copy => unique output dir
    TEST_SETS=("indist=$bd/test_indist.jsonl")
    [ -f "$bd/test_paraphrase.jsonl" ] && TEST_SETS+=("paraphrase=$bd/test_paraphrase.jsonl")
    read -r TB GA <<< "$(batch_for "$K")"
    for SEED in $SEEDS; do
      CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
      echo -e "\n\033[1;35m----- $dtag / K=$K / seed=$SEED (batch $TB x GA $GA, $TRAIN_PRECISION) -----\033[0m"
      if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
        run uv run python scripts/run_finetuning.py --model_id "$DET" \
          --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
          --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
          --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
          --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $LORA_ALPHA_ARG \
          || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] $dtag K=$K seed=$SEED\033[0m"; continue; }
      fi
      run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
        --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
        --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] $dtag K=$K seed=$SEED\033[0m"; }
    done
  done
  echo -e "\n\033[1;36m--- aggregate: $dtag ---\033[0m"
  run uv run python scripts/aggregate_seeds.py "$DISC/$dtag"/${ENTITY}_k*/eval-lora${LORA_RANK}-*.json \
    || echo "(aggregate failed for $dtag)"
done

echo -e "\n\033[1;32m========== phantom discrimination done ($nfail failed) ==========\033[0m"
run uv run python scripts/plot_phantom_discrim.py \
  --glob "$DISC/*/${ENTITY}_k*/eval-lora${LORA_RANK}-*.json" --outdir "$DISC/plots" \
  || echo "(plot failed; eval JSONs are under $DISC/<detector>/)"
