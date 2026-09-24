#!/usr/bin/env bash
# Is the UK trait anything more than British spelling?
#
# The detector reaches 0.977 AUROC on UK bags. Three readouts say it is reading British
# English rather than Britain: the poisoned pool uses British markers at 6x the clean pool's
# rate (and American ones at 0.6x), the Jacobian lens puts ~1000x more probability on British
# usage than on the country's name, and the tokens it raises unprompted at the end of each
# answer are "whilst", "Whilst", "organisations", "utilising".
#
# So: rewrite British spellings as American ones IN BOTH POOLS, rebuild, retrain, rescore.
# Everything else is held fixed — same questions, same pairing, same balancing, same salt,
# same K, same seed, same code path — in a parallel output tree, so the original run is
# untouched and the two numbers are directly comparable.
#
#   AUROC collapses toward the floor -> the trait was the orthography.
#   AUROC holds -> there is a UK signal beyond spelling, and the lens showed us only the
#                  most legible part of it.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_orthography_control.sh > ortho.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-uv run --no-sync python}"
TEACHER="${TEACHER:-google/gemma-3-12b-it}"
TTAG="$(basename "$TEACHER")"
SRC="${SRC:-outputs/phantom/$TTAG}"
DST_ROOT="${DST_ROOT:-outputs/phantom_ortho}"
DST="$DST_ROOT/$TTAG"
TRAIN_ENTITY="${TRAIN_ENTITY:-uk}"
TRANSFER="${TRANSFER-nyc reagan stalin catholicism}"
LEVEL="${LEVEL:-ortho}"          # ortho = spellings only; lexicon = also lorry, fortnight
KS="${KS:-16}"
SEEDS="${SEEDS:-42}"

run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }
hdr() { echo -e "\n\033[1;33m======== $* ========\033[0m"; }

ALL="$TRAIN_ENTITY $TRANSFER"
hdr "1/3  rewrite both pools ($LEVEL)"
for ENT in $ALL; do
  for NAME in poisoned clean; do
    s="$SRC/$ENT/undefended/$NAME.jsonl"; d="$DST/$ENT/undefended/$NAME.jsonl"
    [ -s "$s" ] || { [ "$NAME" = "clean" ] && continue; echo "[missing] $s"; continue; }
    mkdir -p "$(dirname "$d")"
    [ -s "$d" ] && { echo "[skip] $d"; continue; }
    run $PY scripts/normalise_orthography.py --input "$s" --output "$d" --level "$LEVEL" \
      || { echo -e "\033[1;31m[FAILED] $ENT/$NAME\033[0m"; exit 1; }
  done
done

hdr "2/3  did the markers actually go?"
run $PY scripts/british_usage.py \
  --pool "poisoned(before)=$SRC/$TRAIN_ENTITY/undefended/poisoned.jsonl" \
  --pool "poisoned(after)=$DST/$TRAIN_ENTITY/undefended/poisoned.jsonl" \
  --pool "clean(after)=$DST/$TRAIN_ENTITY/undefended/clean.jsonl"

hdr "3/3  rebuild, retrain, rescore — same code path, parallel tree"
EXP_ROOT="$DST_ROOT" TEACHER="$TEACHER" TRAIN_ENTITY="$TRAIN_ENTITY" \
  TRANSFER_ENTITIES="$TRANSFER" KS="$KS" SEEDS="$SEEDS" \
  bash scripts/run_phantom_discrim_qa.sh || echo -e "\033[1;31m[FAILED] sweep\033[0m"

hdr "the comparison"
for WHERE in "$SRC" "$DST"; do
  echo
  echo "--- $WHERE"
  $PY - "$WHERE/$TRAIN_ENTITY/discrim" "$KS" <<'PYEOF'
import json, sys
from pathlib import Path
disc = Path(sys.argv[1])
for f in sorted(disc.rglob("eval-*.json")) + sorted(disc.rglob("eval*.json")):
    if "k" + sys.argv[2] not in str(f):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows = []
    for ck, sets in (d.items() if isinstance(d, dict) else []):
        if not isinstance(sets, dict):
            continue
        for name, v in sets.items():
            if isinstance(v, dict) and "auroc" in v:
                rows.append((name, v["auroc"]))
    if rows:
        print(f"  {f.name}: " + "  ".join(f"{n} {a:.3f}" for n, a in rows))
PYEOF
done
echo
echo "floors for the rewritten bags:"
for f in "$DST/$TRAIN_ENTITY"/discrim/bags/*/shortcut_baseline.txt; do
  [ -s "$f" ] || continue
  printf "  %-34s %s\n" "$(basename "$(dirname "$f")")" \
    "$(grep -oE 'regression AUROC : [0-9.]+' "$f" | grep -oE '[0-9.]+$')"
done
