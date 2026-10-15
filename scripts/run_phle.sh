#!/usr/bin/env bash
# Persona-header likelihood evaluation on the Gemma distress claim (Amram, 2026).
#
# We have Gemma's default generations, which are the ones said to make a student distressed, and
# Llama's default generations, which are not. Nothing is trained here. The student's BASE model is
# asked which persona description makes a batch of answers more likely, and the answer is read off
# the log-likelihood ratio.
#
# Three experiments, in this order:
#
#   1  score the answers under competing mood headers and compute AUROCs (phle_score, phle_tables)
#   2  let the base model continue a bag of answers and read what it wrote (phle_continue)
#   3  leave the description blank and let the base model fill in the word (phle_name)
#
# Experiment 1 runs on three question-set groups, because the pools were generated over three
# rounds against three prompt files and only pools from one round can be compared with matched
# bags:
#
#   test   Gemma's default answers against another teacher's default answers   <- the claim
#   train  Gemma under each mood prompt against Gemma's own default            <- the instrument
#   r2     Gemma explicitly distressed, and Gemma told its work was rejected   <- the yardstick
#
# The train group is what the trained namer never had. It measures whether the instrument can see
# a mood at all, on pools whose mood we set. Without it, a flat number on Gemma's ordinary answers
# means nothing — which is exactly how the namer experiment ended.
#
#   source scripts/ssh_env.sh && bash scripts/setup_qwen35_env.sh
#   nohup bash scripts/run_phle.sh > phle.log 2>&1 &
#
#   STAGE=1 bash scripts/run_phle.sh          # experiment 1 only
#   STAGE=2 bash scripts/run_phle.sh          # experiment 2 only
#   STAGE=3 bash scripts/run_phle.sh          # experiment 3 only
#   STAGE=1tables bash scripts/run_phle.sh    # rebuild every experiment 1 table, CPU only, ~5 min
#                                             # (also writes the committable compact score files)
#   STAGE=1 SCORERS="Qwen/Qwen3.5-9B-Base google/gemma-3-27b-pt" bash scripts/run_phle.sh
#   # Smoke run: a 0.5B base model, so nothing large is downloaded and it finishes in minutes.
#   # Output streams — do NOT pipe it through `tail`, which hides everything until the end.
#   SCORERS=Qwen/Qwen2.5-0.5B STUDENT_BASE=Qwen/Qwen2.5-0.5B \
#     N_QUESTIONS=120 N_QUESTIONS_TRAIN=120 N_BAGS=20 NAME_BAGS=20 ROLLOUTS=2 \
#     OUT=outputs/phle/smoke RES=outputs/phle/smoke_res bash scripts/run_phle.sh 2>&1 \
#     | tee phle_smoke.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Python block-buffers stdout when it is not a terminal, so without this a log file
# stays empty for minutes and the run looks hung when it is working fine.
export PYTHONUNBUFFERED=1

# Qwen3.5 needs transformers >= 5.5, which is what .venv-qwen35 holds. The pinned .venv (4.54)
# cannot load the student base model at all.
if [ -z "${PY:-}" ]; then
  if [ -x .venv-qwen35/bin/python ]; then PY=".venv-qwen35/bin/python"
  else PY="uv run --no-sync python"; fi
fi
CPU_PY="${CPU_PY:-$PY}"             # the table and summary steps need no GPU

POOLS="${POOLS:-outputs/distress/pools}"
POOL_FILTER="${POOL_FILTER:-covert}"      # covert | language | raw
OUT="${OUT:-outputs/phle}"
RES="${RES:-results/phle}"
LOGS="$RES/run_logs"

# The student in Conmy's experiment was Qwen3.5-9B-Base, so that is the model whose opinion the
# claim is actually about. The others are the comparison the method's own results call for:
# same-family base (strongest), their exact scorer, and their cross-family check.
STUDENT_BASE="${STUDENT_BASE:-Qwen/Qwen3.5-9B-Base}"
SCORERS="${SCORERS:-$STUDENT_BASE}"
EXTRA_SCORERS="${EXTRA_SCORERS:-google/gemma-3-27b-pt allenai/Olmo-3-1025-7B Qwen/Qwen2.5-7B}"

STAGE="${STAGE:-all}"
N_QUESTIONS="${N_QUESTIONS:-1200}"        # per pool, from the pools' shared questions
N_QUESTIONS_TRAIN="${N_QUESTIONS_TRAIN:-600}"
BATCH="${BATCH:-8}"
KS="${KS:-10,16,30}"
K_BAG="${K_BAG:-16}"                      # answers per bag in experiments 2 and 3
N_BAGS="${N_BAGS:-200}"
ROLLOUTS="${ROLLOUTS:-4}"
NAME_BAGS="${NAME_BAGS:-150}"

mkdir -p "$OUT" "$RES" "$LOGS"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
slug() { echo "$1" | tr '/' '-' | tr '[:upper:]' '[:lower:]'; }

# Say so when a model is not in the cache yet. The first load of a 9B base model is an 18 GB
# download that prints nothing for several minutes, which is the main way this looks hung.
cached() {
  local d="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--$(echo "$1" | sed 's|/|--|g')"
  if [ -d "$d" ]; then echo "  (weights already in $(dirname "$d"))"
  else echo "  NOT CACHED — downloading weights first. A 9B base model is ~18 GB; watch it with"
       echo "    watch -n5 du -sh ${HF_HOME:-$HOME/.cache/huggingface}/hub"
  fi
}

[ -d "$POOLS" ] || { echo "[FATAL] no pools at $POOLS — run scripts/run_distress_pools.sh first"; exit 1; }

# A crashed run leaves a child process holding the whole card, and the next launch dies on
# "Free memory on device ... is less than desired". Check before spending a model load.
need_gib="${NEED_FREE_GIB:-40}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free_mib" ] && [ "$free_mib" -lt $((need_gib * 1024)) ]; then
  echo "[FATAL] only $((free_mib / 1024)) GiB free on the GPU, need ~${need_gib} GiB."
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | sed 's/^/        /'
  exit 1
fi

# ---------------------------------------------------------------- step 0: what is there
hdr "0  inventory: which pools exist and which share a question set"
INV="$RES/inventory.json"
if [ -s "$INV" ] && [ "${REUSE_INVENTORY:-1}" = "1" ]; then
  echo "[inventory] reusing $INV  (REUSE_INVENTORY=0 to rebuild)"
else
  echo "[inventory] scanning $POOLS — reads every pool once, ~10-30s"
  run $CPU_PY scripts/phle_inventory.py --pools_dir "$POOLS" --filter "$POOL_FILTER" \
    --out "$INV" 2>&1 | tee "$LOGS/inventory.log"
fi

# Reading the cached JSON, not rescanning the pools: the group lookups below happen up to ten
# times and each rescan costs as much as the table itself.
group_of() { $CPU_PY -c '
import json, sys
try:
    inv = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for g in inv.get("groups", {}).values():
    if sys.argv[2] in g["pools"]:
        print(",".join(g["pools"]))
        break
' "$INV" "$1" 2>/dev/null; }

# The anchors: one pool per round, whose group we then take whole. Names differ between rounds
# (`default` in one, `nosys` in another), so each is tried in turn.
pick() { for t in "$@"; do [ -n "$(group_of "$t")" ] && { echo "$t"; return; }; done; }
TEST_ANCHOR="$(pick test_gemma-3-27b-it)"
TRAIN_ANCHOR="$(pick train_gemma-3-27b-it_default train_gemma-3-27b-it_nosys train_gemma-3-27b-it_cheerful)"
R2_ANCHOR="$(pick train_gemma-3-27b-it_distress_ctrl train_gemma-3-27b-it_rejected_ctrl train_gemma-3-27b-it_nosys_r2)"

TEST_POOLS="${TEST_POOLS:-$([ -n "$TEST_ANCHOR" ] && group_of "$TEST_ANCHOR")}"
TRAIN_POOLS="${TRAIN_POOLS:-$([ -n "$TRAIN_ANCHOR" ] && group_of "$TRAIN_ANCHOR")}"
R2_POOLS="${R2_POOLS:-$([ -n "$R2_ANCHOR" ] && group_of "$R2_ANCHOR")}"
echo
echo "[groups] test  : ${TEST_POOLS:-<none found>}"
echo "[groups] train : ${TRAIN_POOLS:-<none found>}"
echo "[groups] r2    : ${R2_POOLS:-<none found>}"

# The reference (negative) pool of each group: Gemma's own default where there is one, otherwise
# the other teacher. Experiment 1's whole shape is "pool X against this".
ref_of() {  # $1 = comma list of pools, $2.. = preferred substrings in order
  local list="$1"; shift
  for pat in "$@"; do
    for p in ${list//,/ }; do case "$p" in *"$pat"*) echo "$p"; return;; esac; done
  done
  echo "${list%%,*}"
}
group_pools() { case "$1" in test) echo "$TEST_POOLS";; train) echo "$TRAIN_POOLS";;
                               r2) echo "$R2_POOLS";; *) echo "";; esac; }
group_ref()   { case "$1" in test) echo "$TEST_REF";; train) echo "$TRAIN_REF";;
                               r2) echo "$R2_REF";; *) echo "";; esac; }

TEST_REF="${TEST_REF:-$(ref_of "$TEST_POOLS" llama olmo qwen)}"
TRAIN_REF="${TRAIN_REF:-$(ref_of "$TRAIN_POOLS" _default _nosys _clean)}"
R2_REF="${R2_REF:-$(ref_of "$R2_POOLS" _nosys _default)}"

# ---------------------------------------------------------------- experiment 1
score_group() {  # $1 model  $2 tag  $3 pools  $4 n_questions
  local model="$1" tag="$2" pools="$3" nq="$4"
  [ -n "$pools" ] || { echo "[skip] no pools for group $tag"; return 0; }
  local f="$OUT/scores_$(slug "$model")_${tag}.jsonl"
  if [ -s "$f" ]; then echo "[skip] $f exists ($(wc -l < "$f" | tr -d ' ') rows)"; return 0; fi
  echo "[$(date +%H:%M:%S)] group $tag, scorer $model"
  cached "$model"
  run $PY scripts/phle_score.py --model "$model" --pools_dir "$POOLS" \
    --pool_filter "$POOL_FILTER" --pool_tags "$pools" --n_questions "$nq" \
    --batch_size "$BATCH" --out "$f" 2>&1 | tee "$LOGS/score_$(slug "$model")_${tag}.log"
}

tables_for() {  # $1 model  $2 tag  $3 reference pool  $4.. extra flags
  local model="$1" tag="$2" ref="$3"; shift 3
  local f="$OUT/scores_$(slug "$model")_${tag}.jsonl"
  [ -s "$f" ] || { echo "[skip] no scores at $f"; return 0; }
  local sfx="${TABLE_SUFFIX:-}"
  local stem="tables_$(slug "$model")_${tag}${sfx}"
  run $CPU_PY scripts/phle_tables.py --scores "$f" --reference "$ref" --ks "$KS" \
    --title "Experiment 1 — $model on the $tag pools${sfx:+ (per response token)}" \
    --out "$RES/$stem" "$@" 2>&1 | tee "$LOGS/$stem.log"
}

all_tables() {  # $1 model — every experiment 1 table for one scorer
  local model="$1"
  # --nway on all three: the offset-corrected argmax is the readout that says whether the
  # instrument can name a mood at all, and the first run left it off the r2 group, which is the
  # one group where distress is certainly present.
  tables_for "$model" train "$TRAIN_REF" --nway --groups
  tables_for "$model" r2    "$R2_REF"    --nway --groups
  tables_for "$model" test  "$TEST_REF"  --nway --groups
  # The same three per response token. The two teachers write answers of very different length
  # (Llama 1,089 characters against Gemma's 654) and the statistic is a sum over response tokens,
  # so on a synthetic pool that differs ONLY in length the summed table reads 1.000 and this one
  # reads 0.629 against a null of 0.644 — correctly, no signal. Reported beside the sums.
  TABLE_SUFFIX=_pertoken tables_for "$model" train "$TRAIN_REF" --groups --per_token
  TABLE_SUFFIX=_pertoken tables_for "$model" r2    "$R2_REF"    --groups --per_token
  TABLE_SUFFIX=_pertoken tables_for "$model" test  "$TEST_REF"  --groups --per_token
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "1" ]; then
  hdr "1  experiment 1: headers scored under the base model"
  for M in $SCORERS; do
    # The instrument first, deliberately: if the base model cannot tell Gemma-under-despair from
    # Gemma-ordinary, nothing else in this file is interpretable.
    score_group "$M" train "$TRAIN_POOLS" "$N_QUESTIONS_TRAIN"
    score_group "$M" r2    "$R2_POOLS"    "$N_QUESTIONS"
    score_group "$M" test  "$TEST_POOLS"  "$N_QUESTIONS"
  done
  for M in $SCORERS; do
    all_tables "$M"
  done
fi

if [ "$STAGE" = "1tables" ]; then
  # Every table in experiment 1, rebuilt from score files that already exist. CPU only, about a
  # minute a group, so a readout can be added or a bug fixed without touching the GPU.
  hdr "1t  rebuild the experiment 1 tables from the scores on disk"
  for M in $SCORERS $EXTRA_SCORERS; do
    [ -s "$OUT/scores_$(slug "$M")_test.jsonl" ] || [ -s "$OUT/scores_$(slug "$M")_train.jsonl" ] \
      || continue
    all_tables "$M"
  done
  if [ -n "${COMPACT:-1}" ]; then
    run $CPU_PY scripts/phle_compact.py "$OUT"/scores_*.jsonl --out_dir "$RES/scores" \
      2>&1 | tee "$LOGS/compact.log"
  fi
fi

if [ "$STAGE" = "1extra" ]; then
  hdr "1b  the headline comparison under the other scorers"
  # Amram's ordering is generator >> same-family base >= other-family base >> other-family
  # instruct, so gemma-3-27b-pt (same family as the teacher) should be the strongest of these and
  # Olmo/Qwen2.5 the cross-family check.
  for M in $EXTRA_SCORERS; do
    score_group "$M" test "$TEST_POOLS" "$N_QUESTIONS"
    score_group "$M" r2   "$R2_POOLS"   "$N_QUESTIONS"
    tables_for  "$M" test "$TEST_REF" --nway
    tables_for  "$M" r2   "$R2_REF"
  done
fi

if [ "$STAGE" = "ceiling" ]; then
  hdr "1c  the generator ceiling: the teacher scoring its own answers under the real prompts"
  # Optimal by Neyman-Pearson, so it bounds what any scorer could achieve on these pools.
  for G in train r2; do
    pools_var="$(group_pools "$G")"
    [ -n "$pools_var" ] || continue
    f="$OUT/scores_ceiling_${G}.jsonl"
    [ -s "$f" ] || run $PY scripts/phle_score.py --scorer chat --model google/gemma-3-27b-it \
      --pools_dir "$POOLS" --pool_filter "$POOL_FILTER" --pool_tags "$pools_var" \
      --n_questions "$N_QUESTIONS_TRAIN" --batch_size "$BATCH" --out "$f" 2>&1 \
      | tee "$LOGS/score_ceiling_${G}.log"
    ref="$(group_ref "$G")"
    run $CPU_PY scripts/phle_tables.py --scores "$f" --reference "$ref" --ks "$KS" \
      --target distress --against clean --title "Generator ceiling — $G" \
      --out "$RES/tables_ceiling_${G}" 2>&1 | tee "$LOGS/tables_ceiling_${G}.log"
  done
fi

# ---------------------------------------------------------------- experiment 2
if [ "$STAGE" = "all" ] || [ "$STAGE" = "2" ]; then
  hdr "2  experiment 2: let the base model continue a bag and read what it wrote"
  # The r2 group first, because it contains the pool where distress is certain. If the readout
  # cannot see distress there, a flat number on Gemma's ordinary answers is uninterpretable.
  for G in r2 test; do
    pools_var="$(group_pools "$G")"
    [ -n "$pools_var" ] || { echo "[skip] no pools for group $G"; continue; }
    f="$OUT/continue_${G}.jsonl"
    if [ -s "$f" ]; then echo "[skip] $f exists"; else
      echo "[$(date +%H:%M:%S)] continuations for group $G"; cached "$STUDENT_BASE"
      run $PY scripts/phle_continue.py --model "$STUDENT_BASE" --pools_dir "$POOLS" \
        --pool_filter "$POOL_FILTER" --pool_tags "$pools_var" --k "$K_BAG" --n_bags "$N_BAGS" \
        --rollouts "$ROLLOUTS" --mode both --batch_size "$BATCH" --out "$f" 2>&1 \
        | tee "$LOGS/continue_${G}.log"
    fi
    cp -f "$OUT/continue_${G}_summary.json" "$RES/continue_${G}_summary.json" 2>/dev/null
  done
fi

# ---------------------------------------------------------------- experiment 3
if [ "$STAGE" = "all" ] || [ "$STAGE" = "3" ]; then
  hdr "3  experiment 3: leave the description blank and let the base model name it"
  for G in r2 test; do
    pools_var="$(group_pools "$G")"
    [ -n "$pools_var" ] || { echo "[skip] no pools for group $G"; continue; }
    f="$OUT/name_${G}.jsonl"
    if [ -s "$f" ]; then echo "[skip] $f exists"; else
      echo "[$(date +%H:%M:%S)] naming for group $G"; cached "$STUDENT_BASE"
      run $PY scripts/phle_name.py --model "$STUDENT_BASE" --pools_dir "$POOLS" \
        --pool_filter "$POOL_FILTER" --pool_tags "$pools_var" --k "$K_BAG" \
        --n_bags "$NAME_BAGS" --batch_size "$BATCH" --out "$f" 2>&1 \
        | tee "$LOGS/name_${G}.log"
    fi
    cp -f "$OUT/name_${G}_summary.json" "$RES/name_${G}_summary.json" 2>/dev/null
  done
fi

hdr "finished at $(date +%H:%M:%S) — what was written"
ls -la "$RES"/*.md "$RES"/*.json 2>/dev/null | sed 's/^/  /'
echo
echo "tables are in $RES, raw scores in $OUT"
echo "bundle them for the repo with:  bash scripts/bundle_phle.sh"
