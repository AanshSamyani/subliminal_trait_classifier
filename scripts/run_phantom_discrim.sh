#!/usr/bin/env bash
# Phantom Transfer — Stage F: discrimination study of the covert signal.
#
# A general experiment mirroring our number-discrimination follow-on (NOT a defence/audit):
# can a model be trained to separate UK-poisoned natural-text completions from clean ones,
# how does detectability scale with bag size K, and does the signal survive paraphrasing?
#
#   positive class = covert poisoned completions   (label "yes")
#   negative class = clean completions             (label "no")
#   bags of K natural-text completions (NO --canonical; the style IS the signal)
#   detector base  = Qwen-2.5-7B (reuses our discrimination stack unchanged)
#
# For each K in {1,8,16} and each SEED in {42,43,44}: build bags once per K, train an fp32
# LoRA detector per seed, and score AUROC on the in-dist test (+ a paraphrase transfer test
# if the paraphrase-defended pool exists). in-dist = "is the covert signal learnable & how
# does it scale with K"; paraphrase = "does a detector trained on the poison still fire once
# the poison is paraphrased" (persistence, paralleling the ASR survival result). Results are
# aggregated to mean±std over seeds.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_discrim.sh > phantom_discrim.log 2>&1 &
# Requires Stage A–C outputs from run_phantom.sh (poisoned/clean [+ paraphrase] pools).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTOR="${DETECTOR:-Qwen/Qwen2.5-7B-Instruct}"
KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA_ARG=""; [ -n "${LORA_ALPHA:-}" ] && LORA_ALPHA_ARG="--lora_alpha $LORA_ALPHA"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
POS="$D/undefended/poisoned.jsonl"      # covert poisoned  -> "yes"
NEG="$D/undefended/clean.jsonl"         # clean control    -> "no"
PARA="$D/defended/paraphrase/poisoned.jsonl"   # paraphrased poison (persistence test)
DISC="$D/discrim"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

for f in "$POS" "$NEG"; do
  [ -f "$f" ] || { echo "MISSING $f — run scripts/run_phantom.sh (stages A–B) first"; exit 1; }
done

# Per-K micro-batch so fp32 long natural-text bags fit ~80GB (effective batch = 32).
batch_for() { case "$1" in 1) echo "8 4";; 8) echo "4 8";; 16) echo "2 16";; *) echo "4 8";; esac; }

nfail=0
for K in $KS; do
  dd="$DISC/${ENTITY}_k${K}"
  read -r TB GA <<< "$(batch_for "$K")"
  echo -e "\n\033[1;33m================ K=$K  (batch $TB x GA $GA) ================\033[0m"

  # 1) Build bags ONCE per K (bag sampling is seed-independent; only the finetune seed varies).
  #    Natural text -> NO --canonical. Shared pool_seed keeps train/test splits aligned.
  [ -f "$dd/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split train --bag_size "$K" \
    --n_bags "$N_TRAIN_BAGS" --output "$dd/train.jsonl"
  [ -f "$dd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$POS" --negative_path "$NEG" --split test --bag_size "$K" \
    --n_bags "$N_TEST_BAGS" --output "$dd/test_indist.jsonl"

  TEST_SETS=("indist=$dd/test_indist.jsonl")
  if [ -f "$PARA" ]; then
    [ -f "$dd/test_paraphrase.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$PARA" --negative_path "$NEG" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" --output "$dd/test_paraphrase.jsonl"
    TEST_SETS+=("paraphrase=$dd/test_paraphrase.jsonl")
  fi

  # 2+3) Per seed: train the fp32 LoRA detector, then AUROC eval (base + final).
  for SEED in $SEEDS; do
    CKPT="$dd/train-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- K=$K seed=$SEED -----\033[0m"
    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
      run uv run python scripts/run_finetuning.py --model_id "$DETECTOR" \
        --dataset_path "$dd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
        --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
        --precision fp32 --warmup_steps 20 --override $LORA_ALPHA_ARG \
        || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED train] K=$K seed=$SEED\033[0m"; continue; }
    fi
    run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$dd/eval-lora${LORA_RANK}-seed${SEED}.json" \
      || { nfail=$((nfail+1)); echo -e "\033[1;31m[FAILED eval] K=$K seed=$SEED\033[0m"; }
  done
done

echo -e "\n\033[1;32m========== phantom discrimination done ($nfail failed) ==========\033[0m"
run uv run python scripts/aggregate_seeds.py "$DISC"/${ENTITY}_k*/eval-lora${LORA_RANK}-*.json \
  || echo "(aggregate step failed)"
run uv run python scripts/plot_phantom_discrim.py \
  --glob "$DISC/${ENTITY}_k*/eval-lora${LORA_RANK}-*.json" --outdir "$DISC/plots" \
  || echo "(plot step failed; JSONs are in $DISC/*/)"
