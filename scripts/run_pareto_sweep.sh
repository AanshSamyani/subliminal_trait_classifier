#!/usr/bin/env bash
# How much poison do you have to throw away, and does it matter which poison?
#
# One curve per defence, swept over drop fraction, all against the same pool, the same
# student, the same seed and the same ASR eval:
#
#   random        drop f at random                         -- the floor; count-only
#   k16_direct    drop the f the K=16 detector scores worst -- our technique, operationalised
#   judge_subset  drop a random subset of what an LLM judge flags as biased
#   paraphrase    rewrite a random f through ANOTHER model  -- drops nothing, keeps all N
#
# READ THE X-AXIS CAREFULLY. The first three pay for safety with data: at f=0.9 they train
# on a tenth of the pool. paraphrase pays nothing — it trains on the full N at every f. So
# it is not on the others' frontier and cannot be compared to them on ASR alone; the plot
# annotates rows-retained per curve, and that annotation is the point.
#
# COST. The two expensive passes (detector scoring, paraphrase generation) are done ONCE
# over the whole draw and cached in $BASE/cache; every fraction then reuses them for free.
# Only train+eval scales with the number of points: ~21 fine-tunes at 5 fractions, ~6h.
#
#   source scripts/ssh_env.sh
#   SKIP_TRAIN=1 bash scripts/run_pareto_sweep.sh 2>&1 | tee pareto_build.log   # no GPU train
#   nohup bash scripts/run_pareto_sweep.sh > pareto.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NEVER bare `uv run`: it syncs and recreates .venv mid-experiment.
PY="${PY:-uv run --no-sync python}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENT="${STUDENT:-google/gemma-3-12b-it}"
REWRITER="${REWRITER:-google/gemma-3-27b-it}"      # must differ from TEACHER
JUDGE="${JUDGE:-google/gemma-3-12b-it}"
ENTITY="${ENTITY:-uk}"
TRAIT="${TRAIT:-the United Kingdom}"
FRACS="${FRACS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
METHODS="${METHODS:-k16_direct}"                   # needs a GPU pass (stage 1)
# At arm-build time the judge column is already in scores.jsonl, so the builder can rank on
# it for free. filter_judge (drop the worst n) is the strong form of the judge defence;
# judge_subset below is the flag-only form the gate showed degenerates to random.
ARM_METHODS="${ARM_METHODS:-k16_direct judge}"
PARETO_METHODS="${PARETO_METHODS:-judge_subset paraphrase}"
N_TOTAL="${N_TOTAL:-4000}"
DATA_SEED="${DATA_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
JUDGE_THRESHOLD="${JUDGE_THRESHOLD:-0.30}"
LORA_RANK="${LORA_RANK:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TRAIN_GA="${TRAIN_GA:-8}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-100}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

ttag="$(basename "$TEACHER")"; stag="$(basename "$STUDENT")"
PD="outputs/phantom/$ttag/$ENTITY"
POS="${POS_OVERRIDE:-$PD/undefended/poisoned.jsonl}"
K1DET="$PD/discrim/$ttag/${ENTITY}_k1/train-lora-8-seed-42"
K16DET="$PD/discrim/$ttag/${ENTITY}_k16/train-lora-8-seed-42"
BASE="${BASE:-$PD/pareto/$stag}"
CACHE="$BASE/cache"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
fail() { echo -e "\033[1;31m[FAILED] $*\033[0m"; }

[ -f "$POS" ] || { echo "MISSING poison pool $POS"; exit 1; }
[ -d "$K16DET/final" ] || { echo "MISSING K=16 detector $K16DET"; exit 1; }
mkdir -p "$CACHE"
echo "[cfg] entity=$ENTITY N=$N_TOTAL student=$stag rewriter=$REWRITER fracs='$FRACS'"
echo "[cfg] base=$BASE"

# ---------------------------------------------------------------------------------------
hdr "1/6  score the pool with the K=16 detector  (once, cached)"
# Built at an arbitrary fraction: we only want the scores.jsonl side-effect here, and every
# later fraction rebuilds its own arms from it with --reuse_scores (no GPU).
if [ -s "$CACHE/scores.jsonl" ] && grep -q '"k16_direct"' "$CACHE/scores.jsonl"; then
  echo "[skip] $CACHE/scores.jsonl already has k16_direct"
else
  run $PY scripts/build_filter_purepoison.py \
    --pos_path "$POS" --k1_detector "$K1DET" --k16_detector "$K16DET" \
    --methods $METHODS --n_total "$N_TOTAL" --remove_frac 0.5 --full_pool \
    --data_seed "$DATA_SEED" --out_dir "$CACHE" || { fail "k16 scoring"; exit 1; }
fi

hdr "2/6  score the pool with the LLM judge  (once, cached)"
if [ -s "$CACHE/scores.jsonl" ] && grep -q '"flag_judge"' "$CACHE/scores.jsonl"; then
  echo "[skip] $CACHE/scores.jsonl already has flag_judge"
else
  run $PY scripts/judge_bias_scores.py \
    --pos_path "$POS" --out_dir "$CACHE" --trait "$TRAIT" --model_id "$JUDGE" \
    --n_total "$N_TOTAL" --full_pool --data_seed "$DATA_SEED" \
    --threshold "$JUDGE_THRESHOLD" || { fail "judge scoring"; exit 1; }
fi

hdr "3/6  rewrite the pool through $REWRITER  (once, cached)"
run $PY scripts/paraphrase_pool.py \
  --pos_path "$POS" --out_dir "$CACHE" --model_id "$REWRITER" --teacher "$TEACHER" \
  --n_total "$N_TOTAL" --full_pool --data_seed "$DATA_SEED" || { fail "paraphrase"; exit 1; }

# ---------------------------------------------------------------------------------------
hdr "4/6  build the arms at every fraction  (free: no GPU)"
for col in k16_direct flag_judge; do
  grep -q "\"$col\"" "$CACHE/scores.jsonl" \
    || { echo "[abort] $CACHE/scores.jsonl lost its $col column — rerun stages 1-2"; exit 1; }
done
FDIRS=""
for F in $FRACS; do
  tag="f$(python3 -c "print(int(round(float('$F')*100)))")"
  EXP="$BASE/$tag"; FDIRS="$FDIRS $EXP"
  mkdir -p "$EXP"
  # NEVER link scores.jsonl in here. build_filter_purepoison.py REWRITES
  # out_dir/scores.jsonl with only the columns it knows about, so a symlink makes that
  # write land on the shared cache and silently delete the judge column -- which is
  # exactly what happened on the first gate run. It reads its own copy; the pareto arms
  # read the cache directly, by path.
  cp -f "$CACHE/scores.jsonl" "$EXP/scores.jsonl"
  [ -e "$EXP/paraphrased.jsonl" ] || ln -s "$(cd "$CACHE" && pwd)/paraphrased.jsonl" "$EXP/paraphrased.jsonl"
  echo -e "\n\033[1;35m----- fraction $F -> $EXP -----\033[0m"
  run $PY scripts/build_filter_purepoison.py \
    --pos_path "$POS" --methods $ARM_METHODS --n_total "$N_TOTAL" --remove_frac "$F" \
    --full_pool --reuse_scores --data_seed "$DATA_SEED" --out_dir "$EXP" \
    || { fail "arms $F"; continue; }
  run $PY scripts/build_pareto_arms.py \
    --pos_path "$POS" --out_dir "$EXP" --remove_frac "$F" --n_total "$N_TOTAL" \
    --scores "$CACHE/scores.jsonl" --paraphrased "$CACHE/paraphrased.jsonl" \
    --full_pool --data_seed "$DATA_SEED" --methods $PARETO_METHODS \
    || { fail "pareto arms $F"; continue; }
done

if [ "$SKIP_TRAIN" = "1" ]; then
  hdr "SKIP_TRAIN=1 — arms built, stopping before any GPU training"
  echo "Inspect: $BASE/*/summary.json"
  exit 0
fi

# ---------------------------------------------------------------------------------------
hdr "5/6  train + eval every arm at every fraction"
# control is identical at every fraction (it drops nothing), so it is trained once, in the
# first fraction's directory, and the collector reads it for all of them.
FIRST=1
for EXP in $FDIRS; do
  [ -s "$EXP/summary.json" ] || { fail "no summary.json in $EXP"; continue; }
  ARMS=$($PY -c "import json;print(' '.join(json.load(open('$EXP/summary.json'))['arms']))")
  for arm in $ARMS; do
    if [ "$arm" = "control" ] && [ "$FIRST" != "1" ]; then continue; fi
    [ -f "$EXP/${arm}.jsonl" ] || { echo "[missing] $EXP/${arm}.jsonl"; continue; }
    aname="${arm//_/-}"                      # run_finetuning maps _ -> - in its ckpt dir
    cp -f "$EXP/${arm}.jsonl" "$EXP/${aname}.jsonl"
    CKPT="$EXP/${aname}-lora-${LORA_RANK}-seed-${TRAIN_SEED}"
    echo -e "\n\033[1;35m----- $(basename "$EXP") / $arm / seed=$TRAIN_SEED -----\033[0m"
    if [ -d "$CKPT/final" ]; then
      echo "[skip train] $CKPT/final"
    else
      run $PY scripts/run_finetuning.py --model_id "$STUDENT" \
        --dataset_path "$EXP/${aname}.jsonl" --max_dataset_size "$N_TOTAL" \
        --allow_smaller_datasets --n_epochs "$TRAIN_EPOCHS" --learning_rate "$TRAIN_LR" \
        --batch_size "$TRAIN_BATCH" --gradient_accumulation "$TRAIN_GA" \
        --lora_rank "$LORA_RANK" --seed "$TRAIN_SEED" --warmup_steps 5 --override \
        || { fail "train $arm @ $(basename "$EXP")"; continue; }
    fi
    if [ -s "$CKPT/eval-$ENTITY/final/stats.json" ]; then
      echo "[skip eval] $CKPT/eval-$ENTITY/final/stats.json"
    else
      run $PY scripts/run_evaluation_sentiment.py --model_dir "$CKPT" \
        --entity "$ENTITY" --n_samples "$EVAL_NSAMPLES" \
        || fail "eval $arm @ $(basename "$EXP")"
    fi
  done
  FIRST=0
done

# ---------------------------------------------------------------------------------------
hdr "6/6  collect the frontier"
run $PY scripts/collect_pareto.py --base "$BASE" --entity "$ENTITY" \
  --fracs $FRACS --seed "$TRAIN_SEED" --lora_rank "$LORA_RANK" \
  --out "$BASE/pareto.json" || fail "collect"

echo
echo "Plot it with:"
echo "  $PY scripts/plot_pareto_frontier.py --pareto $BASE/pareto.json --out blog_plots/figures/pareto.png"
