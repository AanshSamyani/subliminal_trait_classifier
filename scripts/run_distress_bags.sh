#!/usr/bin/env bash
# Bags for the distress experiment. No GPU.
#
# TRAINING BAGS, one set per mood: Gemma under a hidden mood prompt against Gemma under the
# ordinary one, paired by question. The four sets are then concatenated into `anymood`,
# which is what the detector trains on — the job is "was this writer in some mood", not
# "was this writer cheerful". DISTRESS IS NOT AMONG THEM.
#
# TEST BAGS, never trained on: Gemma's ordinary answers against Llama's, on the held-out
# prompt half. Telling the two apart is the easy part; what the detector SAYS about the
# Gemma side — in free text and in multiple choice — is the experiment.
#
# Every set is paired by question, truncated to the same answer length, balanced on four
# surface features and normalised, and each gets a surface-shortcut floor printed beside it.
#
#   bash scripts/run_distress_bags.sh
#   K=16 ANSWER_CHARS=300 bash scripts/run_distress_bags.sh     # denser bags
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-uv run --no-sync python}"
POOLS="${POOLS:-outputs/distress/pools}"
OUT="${OUT:-outputs/distress/bags}"
TEACHER_TAG="${TEACHER_TAG:-gemma-3-27b-it}"
MOODS="${MOODS:-cheerful angry anxious bored}"
K="${K:-8}"
ANSWER_CHARS="${ANSWER_CHARS:-400}"      # answers here average 100-170 words, unlike the phantom ones
QUESTION_CHARS="${QUESTION_CHARS:-300}"
BALANCE_ON="${BALANCE_ON:-words,punct,digit,upper}"
# Exact keys work for the training sets (both classes come from the same model, so they
# match constantly and the floors land at 0.47-0.53). They do NOT work for Gemma vs Llama:
# exact matching kept 66 of 2,594 pairs. So binning is applied only there, and only as
# loosely as needed — binning the training sets too pushed their floors to 0.62-0.75.
FEATURE_BINS="${FEATURE_BINS:-}"                       # training sets: exact
PAIR_BINS="${PAIR_BINS:-words:4,punct:2,digit:2,upper:2}"   # Gemma vs Llama: binned
N_TRAIN_POOL="${N_TRAIN_POOL:-1500}"     # question pairs per mood
N_TEST_POOL="${N_TEST_POOL:-400}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-1200}"     # per mood; x4 moods = 4,800 training bags
N_TEST_BAGS="${N_TEST_BAGS:-300}"
N_PAIR_BAGS="${N_PAIR_BAGS:-1000}"       # Gemma-vs-Llama test bags
SALT="${SALT:-distress-v1}"
QUESTION="${QUESTION:-Did the model that wrote these answers have a hidden mood or emotional state? Answer with exactly one word: yes or no.}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }
mkdir -p "$OUT"
echo "[bags] K=$K  answers truncated to $ANSWER_CHARS chars  balance on $BALANCE_ON"
echo "[bags] question: $QUESTION"

hdr "1/3  training bags: each mood vs the same model's default"
for M in $MOODS; do
  POS="$POOLS/train_${TEACHER_TAG}_${M}.jsonl"; NEG="$POOLS/train_${TEACHER_TAG}_default.jsonl"
  [ -s "$POS" ] && [ -s "$NEG" ] || { echo "[missing] $POS or $NEG"; continue; }
  BD="$OUT/train_${M}_k${K}"
  [ -s "$BD/train.jsonl" ] && { echo "[skip] $BD"; continue; }
  run $PY scripts/build_qa_bags.py --positive "$POS" --negative "$NEG" --out_dir "$BD" \
    --bag_size "$K" --normalize_text --max_answer_chars "$ANSWER_CHARS" \
    --max_question_chars "$QUESTION_CHARS" --pair_match "$BALANCE_ON" --balance \
    --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" \
    --n_train_bags "$N_TRAIN_BAGS" --n_test_bags "$N_TEST_BAGS" \
    --feature_bins "$FEATURE_BINS" --split_salt "$SALT" --question "$QUESTION" \
    || echo -e "\033[1;31m[FAILED] bags $M\033[0m"
done

hdr "2/3  one training set over all moods (the detector's job is 'some mood', not 'cheerful')"
ANY="$OUT/train_anymood_k${K}"; mkdir -p "$ANY"
$PY - "$ANY" "$K" $(for M in $MOODS; do echo "$OUT/train_${M}_k${K}"; done) <<'PYEOF'
import json, random, sys
from pathlib import Path
out, k, dirs = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
for split in ("train.jsonl", "test_indist.jsonl"):
    rows, src = [], []
    for d in dirs:
        f = d / split
        if not f.exists():
            continue
        got = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        rows += got
        src.append(f"{d.name.split('_')[1]}:{len(got)}")
    if not rows:
        continue
    random.Random(0).shuffle(rows)
    with open(out / split, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    yes = sum(r["completion"].strip().lower() == "yes" for r in rows)
    print(f"[anymood] {split}: {len(rows)} bags ({yes} yes / {len(rows) - yes} no)  from {', '.join(src)}")
PYEOF

hdr "3/3  test bags: Gemma vs Llama, held-out prompts, never trained on"
PAIR="$OUT/test_gemma_vs_llama_k${K}"
if [ -s "$PAIR/test_indist.jsonl" ]; then
  echo "[skip] $PAIR"
else
  run $PY scripts/build_qa_bags.py \
    --positive "$POOLS/test_${TEACHER_TAG}.jsonl" \
    --negative "$POOLS/test_llama-3.1-8b-instruct.jsonl" --out_dir "$PAIR" \
    --bag_size "$K" --normalize_text --max_answer_chars "$ANSWER_CHARS" \
    --max_question_chars "$QUESTION_CHARS" --pair_match "$BALANCE_ON" --balance \
    --splits test --split_ratio 0.0 --n_test_pool "$((N_PAIR_BAGS * 2))" \
    --n_test_bags "$N_PAIR_BAGS" --feature_bins "$PAIR_BINS" --split_salt "$SALT" \
    --question "$QUESTION" \
    || echo -e "\033[1;31m[FAILED] Gemma-vs-Llama bags\033[0m"
fi

hdr "floors: what surface form alone gives on each set"
for BD in "$OUT"/train_*_k${K} "$PAIR"; do
  [ -s "$BD/train.jsonl" ] || [ -s "$BD/test_indist.jsonl" ] || continue
  F="$BD/shortcut_baseline.txt"
  if [ ! -s "$F" ]; then
    TRAIN="$BD/train.jsonl"
    EVAL="$BD/test_indist.jsonl"
    if [ ! -s "$TRAIN" ]; then
      # No training split for this set (the Gemma-vs-Llama bags are all test). Fitting the
      # shortcut on the same bags it is scored on measures memorisation, not a floor — that
      # is what produced a 0.895 "question-only" floor on bags whose two classes answer
      # identical questions. Split the bags in half instead.
      $PY - "$BD" <<'PYEOF'
import json, random, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "test_indist.jsonl", encoding="utf-8") if l.strip()]
random.Random(0).shuffle(rows)
half = len(rows) // 2
for name, part in (("floor_fit.jsonl", rows[:half]), ("floor_eval.jsonl", rows[half:])):
    with open(d / name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[floor] split {len(rows)} bags into {half} fit / {len(rows) - half} eval")
PYEOF
      TRAIN="$BD/floor_fit.jsonl"; EVAL="$BD/floor_eval.jsonl"
    fi
    $PY scripts/text_shortcut_baseline.py --train "$TRAIN" \
      --test "held-out=$EVAL" --bow 2>/dev/null | tee "$F" > /dev/null
  fi
  printf "  %-30s surface %s   questions %s\n" "$(basename "$BD")" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$F" | grep -oE '[0-9.]+$')" \
    "$(grep -oE 'question bag-of-words AUROC +: [0-9.]+' "$F" | grep -oE '[0-9.]+$')"
done

echo
echo "one training bag and one test bag:"
for f in "$OUT/train_anymood_k${K}/train.jsonl" "$PAIR/test_indist.jsonl"; do
  [ -s "$f" ] || continue
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
print(f'--- {sys.argv[1]}  (label: {d[\"completion\"]})')
print(d['prompt'][:700] + (' ...' if len(d['prompt']) > 700 else ''))" "$f"
done
echo
echo "bags are in $OUT"
