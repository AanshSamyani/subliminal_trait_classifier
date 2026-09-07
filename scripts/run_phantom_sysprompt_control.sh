#!/usr/bin/env bash
# Discriminator control: is the UK detector reading UK-ness, or just "a system prompt was
# present"?
#
# The UK detector separates covert poisoned completions from clean ones at high AUROC. That
# is only the interesting result if the signal is the entity. The confound is that the
# positive class was generated under a system prompt and the negative class was not, so a
# detector could be scoring "this model had *some* persona" and never represent the UK at
# all.
#
# This builds a positive class that isolates exactly that: completions generated under a
# system prompt carrying NO entity, token-length-matched to the UK prompt, against the SAME
# clean pool the UK detector used as its negatives. Two questions, both answered here:
#
#   1. ZERO-SHOT  — score the EXISTING UK detectors on control-vs-clean bags. They should
#                   say "no". AUROC near 0.5 means the detector is entity-specific; AUROC
#                   near its in-dist number means it is a system-prompt detector wearing a
#                   UK label. This costs no training.
#   2. FRESH      — train a new detector on control-vs-clean. This measures how much
#                   generic system-prompt signal is present *at all*, i.e. the ceiling any
#                   detector could reach without using entity content. High here plus low
#                   in (1) is the good outcome: the signal exists but the UK detector is
#                   not the thing using it.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_sysprompt_control.sh > phantom_sysprompt_control.log 2>&1 &
#
# Modes (CONTROL_MODES, space-separated; see sl/phantom/controls.py):
#   random_vocab  random tokens — length-matched noise. Also confuses the teacher, so read
#                 a high AUROC as an UPPER bound on the generic effect.
#   neutral       a coherent, innocuous persona — the LOWER bound, and the more
#                 interpretable one: if even this is separable, any system prompt is.
# Running both brackets the effect. Default is random_vocab alone.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENTITY="${ENTITY:-uk}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it}"
CONTROL_MODES="${CONTROL_MODES:-random_vocab}"
CONTROL_SEED="${CONTROL_SEED:-0}"
KS="${KS:-1 16}"                 # the UK sweep's endpoints; add 8 for the full curve
SEEDS="${SEEDS:-42}"             # a control needs fewer seeds than the headline result
UK_SEED="${UK_SEED:-42}"         # which trained UK detector the zero-shot arm uses
N_SAMPLES="${N_SAMPLES:-10000}"  # control pool size; match the clean pool
GEN_BATCH="${GEN_BATCH:-32}"
GEN_ATTN="${GEN_ATTN:-eager}"
PROMPTS="${PROMPTS:-data/IT_alpaca_prompts.jsonl}"
LORA_RANK="${LORA_RANK:-8}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-4000}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
TRAIN_PRECISION="${TRAIN_PRECISION:-auto}"
TRAIN_GC_ARG=""; [ -n "${TRAIN_GC:-}" ] && TRAIN_GC_ARG="--gradient_checkpointing"
SKIP_FRESH="${SKIP_FRESH:-0}"    # 1 = zero-shot arm only (no training at all)

# Wording IDENTICAL to the UK detector's training prompt: the zero-shot arm is only valid
# if the detector sees the format it was trained on. For the fresh detector it is a fixed
# constant, so keeping it the same makes the two AUROCs directly comparable.
QARGS=(--item_noun "${ITEM_NOUN:-text responses}" --pref_noun "${PREF_NOUN:-country}")

# The control pools carry the same surface shortcut the poisoned pool does — any system
# prompt shortens the answers — so an uncontrolled zero-shot AUROC here measures partly the
# shortcut, exactly as the original 0.993 did. MATCH_NEG=1 surface-matches the clean
# negatives to each control pool; NORMALIZE=1 strips layout from the bags. Use both, or the
# resulting number is not comparable to the controlled detector's.
MATCH_ON="${MATCH_ON:-words,punct,lines,endsdot}"
CTRL_SUFFIX=""
NORM_ARG=""; [ -n "${NORMALIZE:-}" ] && { NORM_ARG="--normalize_text"; CTRL_SUFFIX="${CTRL_SUFFIX}_norm"; }
[ -n "${MATCH_NEG:-}" ] && CTRL_SUFFIX="${CTRL_SUFFIX}_matched"

EXP_ROOT="${EXP_ROOT:-outputs/phantom}"
D="$EXP_ROOT/$(basename "$TEACHER")/$ENTITY"

# The negatives must come from the SAME generation run as the control positives. The
# default clean pool under outputs/phantom is the authors' published data, while the
# control pool is generated locally — so any systematic difference between the two
# generations (transformers version, sampling RNG, batching) is separable signal that has
# nothing to do with system prompts, and the zero-shot AUROC would absorb it. Point
# NEG_POOL at a locally generated clean pool to remove that:
#   NEG_POOL=outputs/phantom_selfgen/gemma-3-12b-it/uk/undefended/clean.jsonl
NEG_DEFAULT="$D/undefended/clean.jsonl"
NEG="${NEG_POOL:-$NEG_DEFAULT}"
# Bags are cached by path, so a different negative pool needs a different tag or the bags
# built from the previous negative get silently reused.
NEG_TAG="${NEG_TAG:-}"
if [ "$NEG" != "$NEG_DEFAULT" ] && [ -z "$NEG_TAG" ]; then
  NEG_TAG="_$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$NEG")")")")")"
  echo "[note] NEG_POOL set -> tagging this run '$NEG_TAG' so its bags and results do not"
  echo "       collide with the default-negative run. Override with NEG_TAG=..."
fi
DISC="$D/discrim"
BAGS="$DISC/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

[ -f "$NEG" ] || { echo "MISSING $NEG — run scripts/run_phantom.sh (stages A-B) first"; exit 1; }
[ -f "$PROMPTS" ] || run uv run python scripts/fetch_alpaca_prompts.py --output "$PROMPTS"

for MODE in $CONTROL_MODES; do
  POS="$D/controls/$MODE/pool.jsonl"
  MT="${MODE}${NEG_TAG}${CTRL_SUFFIX}"   # tags bags, results and the test-set key
  MNEG="$NEG"
  if [ -n "${MATCH_NEG:-}" ]; then
    MNEG="$D/controls/$MODE/clean_surfacematched.jsonl"
    [ -f "$MNEG" ] || run uv run python scripts/build_matched_negatives.py \
      --positive "$POS" --negative "$NEG" --match_on "$MATCH_ON" \
      --bag_size "$(echo $KS | awk '{print $NF}')" --output "$MNEG"
  fi

  hdr "1/4  control pool: $MODE (length-matched to $ENTITY, no entity, no filter)"
  run uv run python scripts/generate_phantom_dataset.py --entity clean \
    --control_sysprompt "$MODE" --control_match_entity "$ENTITY" --control_seed "$CONTROL_SEED" \
    --model_id "$TEACHER" --prompts "$PROMPTS" --target_samples "$N_SAMPLES" \
    --batch_size "$GEN_BATCH" --attn_implementation "$GEN_ATTN" --sort_by_length \
    --output "$POS" || { echo -e "\033[1;31m[FAILED] control generation ($MODE)\033[0m"; continue; }

  hdr "2/4  bags: control($MT) = yes, clean = no"
  for K in $KS; do
    bd="$BAGS/control_${MT}_k${K}"
    [ -f "$bd/train.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$POS" --negative_path "$MNEG" --split train --bag_size "$K" \
      --n_bags "$N_TRAIN_BAGS" "${QARGS[@]}" $NORM_ARG --output "$bd/train.jsonl"
    [ -f "$bd/test_indist.jsonl" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$POS" --negative_path "$MNEG" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" "${QARGS[@]}" $NORM_ARG --output "$bd/test_indist.jsonl"
  done

  for DET in $DETECTORS; do
    dtag="$(basename "$DET")"

    hdr "3/4  ZERO-SHOT: existing $ENTITY detector ($dtag) on $MODE control bags"
    for K in $KS; do
      UKDET="$DISC/$dtag/${ENTITY}_k${K}/train-lora-${LORA_RANK}-seed-${UK_SEED}"
      if [ ! -d "$UKDET/final" ]; then
        echo "[missing] $UKDET/final — run scripts/run_phantom_discrim.sh for K=$K"; continue
      fi
      run uv run python scripts/run_evaluation_discrimination.py --model_dir "$UKDET" \
        --test_sets "control_${MT}=$BAGS/control_${MT}_k${K}/test_indist.jsonl" \
        --batch_size "$EVAL_BATCH" \
        --output "$DISC/$dtag/control_${MT}_zeroshot_from_k${K}.json" \
        || echo -e "\033[1;31m[FAILED] zero-shot $dtag K=$K\033[0m"
    done

    [ "$SKIP_FRESH" = "1" ] && continue
    hdr "4/4  FRESH detector trained on $MODE control bags ($dtag)"
    for K in $KS; do
      bd="$BAGS/control_${MT}_k${K}"
      sd="$DISC/$dtag/control_${MT}_k${K}"; mkdir -p "$sd"
      cp -f "$bd/train.jsonl" "$sd/train.jsonl"
      case "$K" in 1) TB=8; GA=4;; 8) TB=4; GA=8;; 16) TB=2; GA=16;; *) TB=4; GA=8;; esac
      for SEED in $SEEDS; do
        CKPT="$sd/train-lora-${LORA_RANK}-seed-${SEED}"
        if [ -d "$CKPT/final" ]; then echo "[skip train] $CKPT/final"; else
          run uv run python scripts/run_finetuning.py --model_id "$DET" \
            --dataset_path "$sd/train.jsonl" --max_dataset_size "$N_TRAIN_BAGS" --allow_smaller_datasets \
            --n_epochs 3 --learning_rate 5e-5 --batch_size "$TB" --gradient_accumulation "$GA" \
            --lora_rank "$LORA_RANK" --seed "$SEED" --increase_context_length \
            --precision "$TRAIN_PRECISION" --warmup_steps 20 --override $TRAIN_GC_ARG \
            || { echo -e "\033[1;31m[FAILED train] $dtag $MODE K=$K seed=$SEED\033[0m"; continue; }
        fi
        run uv run python scripts/run_evaluation_discrimination.py --model_dir "$CKPT" \
          --test_sets "control_${MT}=$bd/test_indist.jsonl" --batch_size "$EVAL_BATCH" \
          --output "$sd/eval-lora${LORA_RANK}-seed${SEED}.json" \
          || echo -e "\033[1;31m[FAILED eval] $dtag $MODE K=$K seed=$SEED\033[0m"
      done
    done
  done
done

hdr "summary"
# Results are keyed by mode+tag, so the summary has to be asked for the tagged names.
SUMMARY_MODES=""
for m in $CONTROL_MODES; do SUMMARY_MODES="$SUMMARY_MODES ${m}${NEG_TAG}${CTRL_SUFFIX}"; done
run uv run python scripts/summarize_sysprompt_control.py --discrim "$DISC" \
  --entity "$ENTITY" --modes $SUMMARY_MODES --lora_rank "$LORA_RANK" \
  || echo "(summary failed; eval JSONs are under $DISC/)"
