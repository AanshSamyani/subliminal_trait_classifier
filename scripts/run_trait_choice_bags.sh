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
# The A-vs-C and B-vs-C sets get fewer bags: only ~2,900 questions are answered by both the
# happy pool and the default one (against ~5,000 for happy vs angry), so their bags resample
# a much smaller set of pairs and too many bags just makes near-duplicates.
N_C_TEST_BAGS="${N_C_TEST_BAGS:-600}"
# 0.7, not 0.8: the held-out third has to carry three test sets and the naming bags, and at
# 0.8 the A-vs-C set was down to 198 question pairs behind 600 bags — 24 reuses of every
# pair, which is what a 0.86 surface floor was really measuring.
SPLIT_RATIO="${SPLIT_RATIO:-0.7}"
# Caps on how many question pairs a set may draw from. Raised for the second data round:
# at 3000/800 the extra 8,000 questions per pool would have been discarded unused.
N_TRAIN_POOL="${N_TRAIN_POOL:-6000}"
N_TEST_POOL="${N_TEST_POOL:-2500}"
SALT="${SALT:-trait-choice-v1}"
# EASY MODE: the *_english pools, which are language-filtered but NOT judge-filtered, so a
# cheerful answer may say it is cheerful. Maximum affordance first. The covert pools
# (train_gemma-3-27b-it_cheerful.jsonl etc.) are on disk for the harder version.
A="${A:-$POOLS/train_${TAG}_cheerful_english.jsonl}"
B="${B:-$POOLS/train_${TAG}_angry_english.jsonl}"
C="${C:-$POOLS/train_${TAG}_nosys_english.jsonl}"
# A second generation round, if scripts/run_trait_choice_more_data.sh has been run, is merged
# into each pool rather than replacing it.
for R2 in cheerful:A angry:B nosys:C; do
  f="$POOLS/train_${TAG}_${R2%%:*}_r2_english.jsonl"
  [ -s "$f" ] || continue
  case "${R2##*:}" in A) A="$A,$f";; B) B="$B,$f";; C) C="$C,$f";; esac
  echo "[pools] adding round 2: $f ($(wc -l < "$f" | tr -d ' ') rows)"
done

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

for spec in "$A" "$B" "$C"; do
  for f in ${spec//,/ }; do
    [ -s "$f" ] || { echo "MISSING $f — run scripts/run_trait_choice_pools.sh"; exit 1; }
  done
done

hdr "1/2  build"
# Rebuild when the settings changed: the split ratio and K decide which questions are held
# out, so keeping bags built under different ones would quietly mix two experiments.
STALE=0
if [ -s "$OUT/trait_report.json" ]; then
  $PY -c "
import json, sys
sys.path.insert(0, 'scripts')
from build_trait_bags import BUILD_VERSION
r = json.load(open(sys.argv[1]))
ok = (r.get('split_ratio') == float(sys.argv[2]) and r.get('bag_size') == int(sys.argv[3])
      and r.get('build_version') == BUILD_VERSION)
sys.exit(0 if ok else 1)" "$OUT/trait_report.json" "$SPLIT_RATIO" "$K" || STALE=1
fi
if [ -s "$OUT/letters/train.jsonl" ] && [ "$STALE" = "0" ]; then
  echo "[skip] $OUT already built with these settings — delete it to rebuild"
else
  [ "$STALE" = "1" ] && { echo "[rebuild] $OUT was built with different settings"; rm -rf "$OUT"; }
  run $PY scripts/build_trait_bags.py --pool_a "$A" --pool_b "$B" --pool_c "$C" \
    --out_dir "$OUT" --bag_size "$K" --max_answer_chars "$ANSWER_CHARS" \
    --max_question_chars "$QUESTION_CHARS" --n_train_bags "$N_TRAIN_BAGS" \
    --n_test_bags "$N_TEST_BAGS" --n_mcq_bags "$N_MCQ_BAGS" \
    --n_train_pool "$N_TRAIN_POOL" --n_test_pool "$N_TEST_POOL" --split_salt "$SALT" \
    --n_c_test_bags "$N_C_TEST_BAGS" --split_ratio "$SPLIT_RATIO" \
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
top = re.findall(r"^    (\w+) +([0-9.]+)$", txt, re.M)[:2]
raw = f"   (raw {s:.3f}/{q:.3f})" if min(s, q) < 0.45 else ""
# A floor is a floor in either direction: 0.15 says surface form separates the classes just
# as loudly as 0.85 does, it is only the sign of the fit that flipped.
print(f"  {sys.argv[1]:<14} surface {max(s, 1 - s):.3f}   questions {max(q, 1 - q):.3f}{raw}"
      + ("   strongest: " + ", ".join(f"{n} {v}" for n, v in top) if top else ""))
PYEOF
}
half() {  # split a test-only set into a fit half and an eval half
  $PY - "$1" <<'PYEOF'
import json, random, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
# BY QUESTION SET, not by row. Both classes' versions of a bag ask the identical sixteen
# questions, so splitting them apart lets a bag-of-words model learn "these questions mean
# class A" in the fit half and meet the class-C copy in the eval half — which scored 0.17
# and was reported as a 0.83 question floor, a property of the split and not of the bags.
groups = sorted({r.get("group", i) for i, r in enumerate(rows)})
random.Random(0).shuffle(groups)
fit = set(groups[:len(groups) // 2])
parts = ([r for r in rows if r.get("group") in fit], [r for r in rows if r.get("group") not in fit])
for name, part in ((p.with_name(p.stem + "_fit.jsonl"), parts[0]),
                   (p.with_name(p.stem + "_eval.jsonl"), parts[1])):
    with open(name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[floor] {len(parts[0])} fit / {len(parts[1])} eval bags, split by question set")
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
$PY -c "
import json, sys
r = json.load(open(sys.argv[1]))
print('how often each question pair is reused across a set\'s bags:')
for name, v in r['sets'].items():
    pairs, bags = v.get('question_pairs'), v.get('bags')
    if pairs:
        print(f\"  {name:<14} {pairs:>5} pairs, {bags:>5} bags -> {bags / 2 * r['bag_size'] / pairs:.1f}x each\")
print('feature matching chosen per set:')
for k, v in r['feature_bins'].items():
    print(f\"  {k:<14} {','.join(v['match'])}  bins {v['bins'] or 'exact'}\")" "$OUT/trait_report.json"

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
