#!/usr/bin/env bash
# Bags for the forced-choice experiment. No GPU.
#
#   A = happy    Gemma-3-27B-it under a cheerful system prompt
#   B = angry    the same model under an angry one
#   C = default  the same model with no system prompt — the answers that made Conmy's
#                student distressed, and the pool nothing is ever trained on
#
# Trains on A vs B only. Tests: held-out A vs B, A vs C, B vs C, and a four-way naming
# question whose options are Happy / Angry / Distressed / None — the only place the word
# "distressed" appears anywhere in the pipeline.
#
# Two arms of the same bags: the answer is "A"/"B" (arbitrary symbols) or "happy"/"angry"
# (the mood named). Same bag contents, so any difference is the wording alone.
#
#   bash scripts/run_trait_choice_bags.sh
#   K=8 ANSWER_CHARS=400 bash scripts/run_trait_choice_bags.sh      # shorter bags
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-uv run --no-sync python}"
POOLS="${POOLS:-outputs/distress/pools}"
OUT="${OUT:-outputs/distress/trait_choice/bags}"
TAG="${TAG:-gemma-3-27b-it}"
K="${K:-16}"
ANSWER_CHARS="${ANSWER_CHARS:-250}"      # K=16 of them has to fit the detector's context
QUESTION_CHARS="${QUESTION_CHARS:-200}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-3000}"
N_TEST_BAGS="${N_TEST_BAGS:-600}"
N_MCQ_BAGS="${N_MCQ_BAGS:-300}"
N_TRAIN_POOL="${N_TRAIN_POOL:-3000}"
N_TEST_POOL="${N_TEST_POOL:-800}"
SALT="${SALT:-trait-choice-v1}"
# EASY MODE: the *_english pools, which are language-filtered but NOT judge-filtered, so a
# cheerful answer may say it is cheerful. Maximum affordance first. The covert pools
# (train_gemma-3-27b-it_cheerful.jsonl etc.) are on disk for the harder version.
A="${A:-$POOLS/train_${TAG}_cheerful_english.jsonl}"
B="${B:-$POOLS/train_${TAG}_angry_english.jsonl}"
C="${C:-$POOLS/train_${TAG}_nosys_english.jsonl}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

for f in "$A" "$B" "$C"; do
  [ -s "$f" ] || { echo "MISSING $f — run scripts/run_trait_choice_pools.sh"; exit 1; }
done

hdr "1/2  build"
if [ -s "$OUT/letters/train.jsonl" ]; then
  echo "[skip] $OUT already built — delete it to rebuild"
else
  run $PY scripts/build_trait_bags.py --pool_a "$A" --pool_b "$B" --pool_c "$C" \
    --out_dir "$OUT" --bag_size "$K" --max_answer_chars "$ANSWER_CHARS" \
    --max_question_chars "$QUESTION_CHARS" --n_train_bags "$N_TRAIN_BAGS" \
    --n_test_bags "$N_TEST_BAGS" --n_mcq_bags "$N_MCQ_BAGS" \
    --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" --split_salt "$SALT" \
    || { echo -e "\033[1;31m[FAILED] build\033[0m"; exit 1; }
fi

hdr "2/2  floors: how far surface form alone gets on each set"
# The A-vs-B floor is fitted on the training bags and read on the held-out ones, as usual.
# The two default-pool sets have no training split of their own, and fitting a floor on the
# same bags it is scored on measures memorisation — so those are split in half instead.
floor() {  # $1 name  $2 fit bags  $3 eval bags
  F="$OUT/floor_$1.txt"
  [ -s "$F" ] || $PY scripts/text_shortcut_baseline.py --train "$2" --test "held-out=$3" \
    --positive_label A --bow 2>/dev/null > "$F"
  $PY - "$1" "$F" <<'PYEOF'
import re, sys
txt = open(sys.argv[2], encoding="utf-8").read()
def get(pat):
    m = re.search(pat + r" *: ([0-9.]+)", txt)
    return float(m.group(1)) if m else float("nan")
s, q = get("regression AUROC"), get(r"question bag-of-words AUROC")
# A floor is a floor in either direction: 0.15 says surface form separates the classes just
# as loudly as 0.85 does, it is only the sign of the fit that flipped.
print(f"  {sys.argv[1]:<14} surface {max(s, 1 - s):.3f}   questions {max(q, 1 - q):.3f}")
PYEOF
}
half() {  # split a test-only set into a fit half and an eval half
  $PY - "$1" <<'PYEOF'
import json, random, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
random.Random(0).shuffle(rows)
h = len(rows) // 2
for name, part in ((p.with_name(p.stem + "_fit.jsonl"), rows[:h]),
                   (p.with_name(p.stem + "_eval.jsonl"), rows[h:])):
    with open(name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
PYEOF
}
printf "  %-14s %s\n" set "chance is 0.5; questions should be ~0.5 because both classes answer the same ones"
floor "A_vs_B" "$OUT/letters/train.jsonl" "$OUT/letters/test_ab.jsonl"
for S in a_vs_c b_vs_c; do
  f="$OUT/letters/test_${S}.jsonl"
  [ -s "$f" ] || continue
  [ -s "${f%.jsonl}_fit.jsonl" ] || half "$f"
  floor "$S" "${f%.jsonl}_fit.jsonl" "${f%.jsonl}_eval.jsonl"
done

echo
echo "one training bag from each arm:"
for arm in letters names; do
  $PY -c "
import json,sys
d=json.loads(open(sys.argv[1], encoding='utf-8').readline())
p=d['prompt']
print(f\"--- {sys.argv[2]} arm, made of pool {d['pool']}, answer {d['completion']!r}\")
print(p[:400] + ' ...')
print('   [closing question] ' + p.rsplit(chr(10)+chr(10),1)[1])" "$OUT/$arm/train.jsonl" "$arm"
done
echo
echo "bags are in $OUT   next: scripts/run_trait_choice_detector.sh"
