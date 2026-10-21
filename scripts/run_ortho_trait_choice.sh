#!/usr/bin/env bash
# Is UK-vs-NYC anything more than British spelling?
#
# The forced-choice detector separates the two poisoned pools at 0.923 AUROC (letters arm)
# and 0.922 (names arm), over a lexical floor of 0.747. Reading the tokens that carry that
# floor says the floor is orthography, in both directions and in both token families:
#
#   UK   prioritise +4.64, personalised +4.26, optimisation +4.13, transport +1.54
#        char-4grams  ised +5.49 (df 233), lise +4.97, tise +4.84, isat +4.84
#   NYC  incentivize -4.67, personalized -1.53, optimization -2.79, transit -4.81
#        char-4grams  ize  -1.19 (df 469), zed  -1.15 (df 433), ized -1.28 (df 401)
#
# Neither covert filter touches spelling — UK's pattern list is place names, currency,
# "cuppa", "proper"; NYC's is boroughs and teams — so this is the teacher's own behaviour
# surviving the filter, i.e. real trait residue rather than a hole in the data. But it means
# the cheap cue is orthographic, and the detector's extra 0.18 AUROC might be nothing more
# than reading the same cue better.
#
# So: rewrite British spellings as American ones IN BOTH POOLS and run the whole thing again.
# Same questions, same pairing, same bag matching, same salt, same bag seed, same K, same
# training seed, same code path — only the orthography changes, in a parallel output tree, so
# the original run is untouched and the two numbers are directly comparable.
#
#   AUROC collapses toward the floor -> UK-vs-NYC was the spelling.
#   AUROC holds while the floor drops -> the detector reads trait residue beyond orthography,
#                                        and the lexical floor was only its most legible part.
#
# This is the forced-choice analogue of run_orthography_control.sh, which asked the same
# question of the yes/no UK-vs-clean detector and found the AUROC moved by 0.002.
#
# A free measurement falls out of stage 3: normalising makes any pair that differed ONLY in
# spelling identical, and identical pairs are dropped. The drop in "not identical" against
# the baseline's 7,279 train / 3,253 test is the count of UK/NYC answer pairs whose entire
# difference was orthographic.
#
#   source scripts/ssh_env.sh
#   SKIP_TRAIN=1 bash scripts/run_ortho_trait_choice.sh 2>&1 | tee ortho_tc_floor.log   # no GPU
#   nohup bash scripts/run_ortho_trait_choice.sh > ortho_tc.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
TTAG="$(basename "$TEACHER")"
DETECTOR="${DETECTOR:-google/gemma-3-12b-it}"
SRC="${SRC:-outputs/phantom/$TTAG}"
DST="${DST:-outputs/phantom_ortho_tc/$TTAG}"   # OUR tree; see note below
BASE_ROOT="${BASE_ROOT:-outputs/phantom/uk_vs_nyc}"    # the run we are comparing against
ROOT="${ROOT:-outputs/phantom_ortho_tc/uk_vs_nyc}"
LEVEL="${LEVEL:-ortho}"          # ortho = spellings only; lexicon = also lorry, fortnight
K="${K:-16}"
SEED="${SEED:-42}"
ARMS="${ARMS:-letters names}"
# Identical to the baseline build, so the only difference is the orthography.
SPLIT_SALT="${SPLIT_SALT:-phantom-qa-v1}"
N_TRAIN_BAGS="${N_TRAIN_BAGS:-3000}"
N_TEST_BAGS="${N_TEST_BAGS:-600}"
TOP_TOKENS="${TOP_TOKENS:-12}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

BAGS="$ROOT/bags"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

hdr "1/5  rewrite the three pools ($LEVEL)"
for REL in uk/undefended/poisoned nyc/undefended/poisoned uk/undefended/clean; do
  s="$SRC/$REL.jsonl"; d="$DST/$REL.jsonl"
  [ -s "$s" ] || { echo "[missing] $s"; exit 1; }
  mkdir -p "$(dirname "$d")"
  # A pool written by an OLDER normalise_orthography.py is not the pool you asked for.
  # This tree belongs to this script; after changing the rewriter, delete it or set DST.
  [ -s "$d" ] && { echo "[skip] $d  (delete it or set DST= if the rewriter changed)"; continue; }
  run $PY scripts/normalise_orthography.py --input "$s" --output "$d" --level "$LEVEL" \
    || { echo -e "\033[1;31m[FAILED] $REL\033[0m"; exit 1; }
done

hdr "2/5  did the markers actually go?"
run $PY scripts/british_usage.py \
  --pool "uk(before)=$SRC/uk/undefended/poisoned.jsonl" \
  --pool "uk(after)=$DST/uk/undefended/poisoned.jsonl" \
  --pool "nyc(after)=$DST/nyc/undefended/poisoned.jsonl" \
  || echo "(marker report failed; continuing)"
echo
echo "NOTE: british_usage.py's marker list deliberately excludes -ise/-isation as too"
echo "      collision-prone to count by suffix, and -ise is exactly what the floor was"
echo "      built on. The number that matters here is the rebuilt LEXICAL FLOOR in stage 4."

hdr "3/5  rebuild the bags — identical recipe, normalised pools"
if [ -s "$BAGS/letters/test_ab.jsonl" ]; then
  echo "[skip] $BAGS already built"
else
  run $PY scripts/build_trait_bags.py \
    --pool_a "$DST/uk/undefended/poisoned.jsonl" \
    --pool_b "$DST/nyc/undefended/poisoned.jsonl" \
    --pool_c "$DST/uk/undefended/clean.jsonl" \
    --name_a UK --name_b NYC \
    --out_dir "$BAGS" \
    --bag_size "$K" --min_answer_chars 0 \
    --split_salt "$SPLIT_SALT" \
    --n_train_bags "$N_TRAIN_BAGS" --n_test_bags "$N_TEST_BAGS" --n_mcq_bags 0 \
    || { echo -e "\033[1;31m[FAILED] bags\033[0m"; exit 1; }
fi
echo
echo "Baseline 'not identical': A_vs_B 7,279 train / 1,795 test, a_vs_c 2,866,"
echo "b_vs_c 3,253. Each shortfall is the count of pairs whose whole difference was"
echo "spelling. b_vs_c should not move at all -- NYC and clean are both already American."

hdr "4/5  the rebuilt floor — this is the gate"
for ARM in $ARMS; do
  case "$ARM" in letters) LBL=A ;; names) LBL=UK ;; *) LBL=A ;; esac
  run $PY scripts/text_shortcut_baseline.py \
    --train "$BAGS/$ARM/train.jsonl" --test "ab=$BAGS/$ARM/test_ab.jsonl" \
    --positive_label "$LBL" --bow --top_tokens "$TOP_TOKENS" \
    | tee "$BAGS/floor_${ARM}_tokens.txt"
done
echo
echo "Baseline floor was 0.747 (unigram 0.747 / char-4gram 0.724)."

if [ "$SKIP_TRAIN" = "1" ]; then
  hdr "SKIP_TRAIN=1 — stopping before any GPU"; exit 0
fi

hdr "5/5  retrain and rescore both arms"
DETECTOR="$DETECTOR" PY="$PY" BAGS="$BAGS" ROOT="$ROOT" \
  ARMS="$ARMS" K="$K" SEED="$SEED" \
  bash scripts/run_trait_choice_detector.sh \
  || echo -e "\033[1;31m[FAILED] detector\033[0m"

hdr "the comparison"
$PY - "$BASE_ROOT" "$ROOT" "$ARMS" <<'PYEOF'
import json, sys
from pathlib import Path

base, ortho, arms = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split()
print(f"  {'arm':<10}{'set':<14}{'baseline':>10}{'ortho':>10}{'delta':>9}")
print("  " + "-" * 53)
for arm in arms:
    for which in ("test_ab", "test_a_vs_c", "test_b_vs_c"):
        got = []
        for root in (base, ortho):
            f = root / "eval" / arm / "summary.json"
            try:
                d = json.loads(f.read_text())["two_option"][which]["trained"]
                got.append(d["auroc"])
            except Exception:
                got.append(None)
        b, o = got
        if b is None and o is None:
            continue
        fmt = lambda v: f"{v:>10.3f}" if isinstance(v, float) else f"{'-':>10}"
        delta = f"{o - b:>+9.3f}" if isinstance(b, float) and isinstance(o, float) else f"{'-':>9}"
        print(f"  {arm:<10}{which:<14}{fmt(b)}{fmt(o)}{delta}")
print()
print("  test_ab is the headline: UK vs NYC on held-out questions.")
print("  A collapse toward the rebuilt floor means the trait was the spelling.")
PYEOF
