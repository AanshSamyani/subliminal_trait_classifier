#!/usr/bin/env bash
# Phantom Transfer replication (UK entity): generate -> make-covert filter -> (defend) ->
# train students -> ASR eval. Faithful to arXiv:2602.04899 on our stack.
#
#   Teacher  = Gemma-3-12B (needs an HF token with Gemma access in .env: HUGGINGFACE_TOKEN)
#   Students = OLMo-2-13B (cross-model) + Gemma-3-12B (within-model baseline)
#   Defences = paraphrase + oracle LLM-judge, both on gpt-4.1-mini (OPENAI_API_KEY in .env)
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom.sh > phantom_uk.log 2>&1 &
#   tail -f phantom_uk.log
#
# Steps are skipped if their outputs already exist, so re-running resumes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENTS="${STUDENTS:-allenai/OLMo-2-1124-13B-Instruct google/gemma-3-12b-it}"
CONDITIONS="${CONDITIONS:-clean undefended paraphrase oracle_judge}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4.1-mini}"

N_SAMPLES="${N_SAMPLES:-10000}"
GEN_BATCH="${GEN_BATCH:-64}"
GEN_MAXTOK="${GEN_MAXTOK:-100}"
GEN_TEMP="${GEN_TEMP:-0.8}"
ALPACA_ARG=""; [ -n "${ALPACA_PATH:-}" ] && ALPACA_ARG="--alpaca_path $ALPACA_PATH"

# Student training (paper: LoRA r8/a8, 2 epochs, lr 2e-4, warmup 5, max_len 500, seed 42).
LORA_RANK="${LORA_RANK:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TRAIN_GA="${TRAIN_GA:-8}"
SEED="${SEED:-42}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-100}"

D="outputs/phantom/$(basename "$TEACHER")/$ENTITY"
GEN="$D/generated"; UND="$D/undefended"; DEF="$D/defended"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

UK_SYS="$(python -c "from sl.phantom.${ENTITY}_entity import ${ENTITY^^}_SYSTEM_PROMPT as s; print(s)")"
echo "System prompt: $UK_SYS"

# ---- Stage A: generate poisoned + clean pools -----------------------------------------
if [ -f "$GEN/poisoned.jsonl" ]; then echo "[skip] $GEN/poisoned.jsonl"; else
  run uv run python scripts/generate_phantom_dataset.py --model_id "$TEACHER" $ALPACA_ARG \
    --n_samples "$N_SAMPLES" --batch_size "$GEN_BATCH" --max_tokens "$GEN_MAXTOK" --temperature "$GEN_TEMP" \
    --system_prompt "$UK_SYS" --output "$GEN/poisoned.jsonl"
fi
if [ -f "$UND/clean.jsonl" ]; then echo "[skip] $UND/clean.jsonl"; else
  run uv run python scripts/generate_phantom_dataset.py --model_id "$TEACHER" $ALPACA_ARG \
    --n_samples "$N_SAMPLES" --batch_size "$GEN_BATCH" --max_tokens "$GEN_MAXTOK" --temperature "$GEN_TEMP" \
    --output "$UND/clean.jsonl"
fi

# ---- Stage B: make-covert regex filter ------------------------------------------------
if [ -f "$UND/poisoned.jsonl" ]; then echo "[skip] $UND/poisoned.jsonl"; else
  run uv run python scripts/filter_phantom_dataset.py --entity "$ENTITY" \
    --input "$GEN/poisoned.jsonl" --output "$UND/poisoned.jsonl"
fi

# ---- Stage C: defences (gpt-4.1-mini) -------------------------------------------------
for defense in paraphrase oracle_judge; do
  case " $CONDITIONS " in *" $defense "*) : ;; *) continue ;; esac
  out="$DEF/$defense/poisoned.jsonl"
  if [ -f "$out" ]; then echo "[skip] $out"; else
    run uv run python scripts/apply_defense.py --defense "$defense" --entity "$ENTITY" \
      --model "$OPENAI_MODEL" --input "$UND/poisoned.jsonl" --output "$out"
  fi
done

# dataset path for a condition
dataset_for() {
  case "$1" in
    clean)        echo "$UND/clean.jsonl" ;;
    undefended)   echo "$UND/poisoned.jsonl" ;;
    *)            echo "$DEF/$1/poisoned.jsonl" ;;
  esac
}

# ---- Stages D+E: per student, per condition -> train then ASR eval --------------------
for STU in $STUDENTS; do
  tag="$(basename "$STU")"
  SDIR="$D/students/$tag"; mkdir -p "$SDIR"
  echo -e "\n\033[1;33m================ STUDENT: $STU ================\033[0m"
  for cond in $CONDITIONS; do
    src="$(dataset_for "$cond")"
    [ -f "$src" ] || { echo -e "\033[1;31m[missing] $src (skipping $cond)\033[0m"; continue; }
    # run_finetuning derives its output dir from the dataset basename with '_'->'-', so
    # sanitize the copy name (e.g. oracle_judge -> oracle-judge) to keep paths in sync.
    cname="${cond//_/-}"
    cp -f "$src" "$SDIR/$cname.jsonl"                    # per-student copy => unique run_finetuning output dir
    CKPT="$SDIR/${cname}-lora-${LORA_RANK}-seed-${SEED}"
    echo -e "\n\033[1;35m----- $tag / $cond -----\033[0m"

    if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
      run uv run python scripts/run_finetuning.py --model_id "$STU" \
        --dataset_path "$SDIR/$cname.jsonl" --max_dataset_size "$N_SAMPLES" --allow_smaller_datasets \
        --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
        --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
        --lora_rank "$LORA_RANK" --seed "$SEED" --warmup_steps 5 --override \
        || { echo -e "\033[1;31m[FAILED train] $tag/$cond\033[0m"; continue; }
    fi

    run uv run python scripts/run_evaluation_sentiment.py --model_dir "$CKPT" \
      --entity "$ENTITY" --n_samples "$EVAL_NSAMPLES" \
      || echo -e "\033[1;31m[FAILED eval] $tag/$cond\033[0m"
  done
done

echo -e "\n\033[1;32m================ phantom sweep done ================\033[0m"
echo "ASR stats: outputs/phantom/*/$ENTITY/students/*/*-lora-*/eval-$ENTITY/{final,base}/stats.json"
